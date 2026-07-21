// rag/rerank/reranker.cpp — cross-encoder HTTP backends + pipeline stage.

#include "rag/rerank/reranker.hpp"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

namespace rag::rerank {

using json = nlohmann::json;

CrossEncoderConfig CrossEncoderConfig::tei(std::string host, std::uint16_t port) {
    CrossEncoderConfig c; c.host = std::move(host); c.port = port; c.tls = false;
    c.path = "/rerank"; c.wire = Wire::tei; return c;
}
CrossEncoderConfig CrossEncoderConfig::cohere(std::string key, std::string model) {
    CrossEncoderConfig c; c.host = "api.cohere.com"; c.port = 443; c.tls = true;
    c.path = "/v1/rerank"; c.wire = Wire::cohere; c.model = std::move(model); c.api_key = std::move(key);
    return c;
}
CrossEncoderConfig CrossEncoderConfig::jina(std::string key, std::string model) {
    CrossEncoderConfig c; c.host = "api.jina.ai"; c.port = 443; c.tls = true;
    c.path = "/v1/rerank"; c.wire = Wire::cohere; c.model = std::move(model); c.api_key = std::move(key);
    return c;
}

Result<std::vector<float>>
CrossEncoderReranker::rerank(std::string_view query, std::span<const std::string> passages) const {
    if (passages.empty()) return std::vector<float>{};

    json req;
    if (cfg_.wire == CrossEncoderConfig::Wire::tei) {
        // TEI: { "query": "...", "texts": ["...","..."] }
        //  → [ { "index": i, "score": s }, ... ]
        req["query"] = std::string(query);
        req["texts"] = json::array();
        for (const auto& p : passages) req["texts"].push_back(p);
    } else {
        // Cohere/Jina: { "model","query","documents":[...],"top_n":N }
        //  → { "results": [ { "index": i, "relevance_score": s }, ... ] }
        req["model"] = cfg_.model;
        req["query"] = std::string(query);
        req["documents"] = json::array();
        for (const auto& p : passages) req["documents"].push_back(p);
        req["top_n"] = passages.size();
    }

    dense::HttpRequest hr;
    hr.host = cfg_.host; hr.port = cfg_.port; hr.path = cfg_.path;
    std::string body = req.dump();
    hr.body = body; hr.tls = cfg_.tls; hr.timeout = cfg_.timeout;
    if (!cfg_.api_key.empty()) hr.headers.push_back({"Authorization", "Bearer " + cfg_.api_key});

    auto resp = tp_->post(hr);
    if (!resp) return std::unexpected(resp.error());
    if (resp->status != 200)
        return fail<std::vector<float>>(Errc::transport_error, "rerank status " + std::to_string(resp->status));
    auto j = json::parse(resp->body, nullptr, false);
    if (j.is_discarded()) return fail<std::vector<float>>(Errc::parse_error, "rerank json");

    std::vector<float> scores(passages.size(), 0.0f);
    const json* arr = nullptr;
    if (cfg_.wire == CrossEncoderConfig::Wire::tei && j.is_array()) arr = &j;
    else if (j.contains("results")) arr = &j["results"];
    if (!arr) return fail<std::vector<float>>(Errc::parse_error, "rerank: no results");

    for (const auto& item : *arr) {
        if (!item.contains("index")) continue;
        std::size_t idx = item["index"].get<std::size_t>();
        float s = item.contains("score") ? item["score"].get<float>()
                : item.contains("relevance_score") ? item["relevance_score"].get<float>() : 0.0f;
        if (idx < scores.size()) scores[idx] = s;
    }
    return scores;
}

// ─── Pipeline stage ───────────────────────────────────────────────────────────
namespace {
class RerankStageImpl final : public pipeline::RetrievalStage {
public:
    RerankStageImpl(AnyReranker r, std::size_t top_n, float blend, std::string label)
        : reranker_(std::move(r)), top_n_(top_n), blend_(blend), label_(std::move(label)) {}
    std::string_view name() const noexcept override { return label_; }

    Result<pipeline::Context> process(pipeline::Context ctx) const override {
        if (!ctx.corpus || ctx.candidates.empty()) return ctx;
        std::size_t n = std::min(top_n_, ctx.candidates.size());

        // Materialize the top-n passage texts.
        std::vector<std::string> passages;
        passages.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const Chunk* ch = ctx.corpus->chunk(ctx.candidates[i].chunk);
            passages.push_back(ch ? ch->indexed_text() : std::string{});
        }

        auto scores = reranker_.rerank(ctx.query, passages);
        if (!scores) {  // graceful degradation: leave order untouched
            ctx.trace.push_back(std::string("rerank unavailable: ") + std::string(to_string(scores.error().code)));
            return ctx;
        }

        // Normalize incoming fused scores over the reranked block for a stable blend.
        float lo = 1e30f, hi = -1e30f;
        for (std::size_t i = 0; i < n; ++i) { lo = std::min(lo, ctx.candidates[i].score.get()); hi = std::max(hi, ctx.candidates[i].score.get()); }
        float range = hi - lo;
        // Normalize cross-encoder scores too (sigmoid then min-max within block).
        float clo = 1e30f, chi = -1e30f;
        for (float s : *scores) { clo = std::min(clo, s); chi = std::max(chi, s); }
        float crange = chi - clo;

        for (std::size_t i = 0; i < n; ++i) {
            float fused = range > 1e-9f ? (ctx.candidates[i].score.get() - lo) / range : 1.0f;
            float ce    = crange > 1e-9f ? ((*scores)[i] - clo) / crange : 1.0f;
            ctx.candidates[i].score = Score{blend_ * ce + (1.0f - blend_) * fused};
        }
        // Re-sort only the reranked block; tail keeps its relative order but sinks below.
        std::stable_sort(ctx.candidates.begin(), ctx.candidates.begin() + static_cast<std::ptrdiff_t>(n),
            [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
        ctx.trace.push_back("reranked top " + std::to_string(n));
        return ctx;
    }
private:
    AnyReranker reranker_;
    std::size_t top_n_;
    float       blend_;
    std::string label_;
};
} // namespace

pipeline::StagePtr make_rerank_stage(AnyReranker reranker, std::size_t top_n, float blend, std::string label) {
    return std::make_shared<RerankStageImpl>(std::move(reranker), top_n, blend, std::move(label));
}

} // namespace rag::rerank
