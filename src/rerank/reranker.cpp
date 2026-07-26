// rag/rerank/reranker.cpp — cross-encoder HTTP backends + pipeline stage.

#include "rag/rerank/reranker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <thread>

#include <nlohmann/json.hpp>

#ifdef RAGCPP_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#include "rag/dense/wordpiece.hpp"
#endif

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

// ══════════════════════════════════════════════════════════════════════════
// OnnxReranker — in-process cross-encoder
// ══════════════════════════════════════════════════════════════════════════
struct OnnxReranker::Impl {
    dense::LocalEmbedderConfig cfg;
    std::string id = "onnx-rerank";
#ifdef RAGCPP_WITH_ONNX
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "ragcpp-rerank"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    dense::WordPieceTokenizer     tok;
    bool                has_token_type = false;
#endif
};

OnnxReranker::OnnxReranker() : impl_(std::make_unique<Impl>()) {}
OnnxReranker::OnnxReranker(OnnxReranker&&) noexcept = default;
OnnxReranker& OnnxReranker::operator=(OnnxReranker&&) noexcept = default;
OnnxReranker::~OnnxReranker() = default;

std::string_view OnnxReranker::identity() const { return impl_->id; }

Result<OnnxReranker> OnnxReranker::load(dense::LocalEmbedderConfig cfg) {
#ifndef RAGCPP_WITH_ONNX
    (void)cfg;
    return std::unexpected(Error{Errc::unavailable,
        "ragcpp built without ONNX support (configure -DRAGCPP_WITH_ONNX=ON)"});
#else
    OnnxReranker r;
    auto& im = *r.impl_;
    im.cfg = std::move(cfg);
    if (!im.cfg.identity_tag.empty()) im.id = im.cfg.identity_tag;
    int threads = im.cfg.threads > 0 ? im.cfg.threads
                : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    im.opts.SetIntraOpNumThreads(threads);
    im.opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        auto tk = dense::WordPieceTokenizer::load(im.cfg.tokenizer_path);
        if (!tk) return std::unexpected(tk.error());
        im.tok = std::move(*tk);
#ifdef _WIN32
        std::wstring wpath(im.cfg.model_path.begin(), im.cfg.model_path.end());
        im.session = std::make_unique<Ort::Session>(im.env, wpath.c_str(), im.opts);
#else
        im.session = std::make_unique<Ort::Session>(im.env, im.cfg.model_path.c_str(), im.opts);
#endif
        // A cross-encoder is a sequence-classification head: input_ids +
        // attention_mask, and (for BERT-family) token_type_ids as a third input.
        im.has_token_type = im.session->GetInputCount() >= 3;
    } catch (const std::exception& ex) {
        return std::unexpected(Error{Errc::corrupt_index,
            std::string("onnx reranker load failed: ") + ex.what()});
    }
    return r;
#endif
}

Result<std::vector<float>>
OnnxReranker::rerank(std::string_view query, std::span<const std::string> passages) const {
#ifndef RAGCPP_WITH_ONNX
    (void)query; (void)passages;
    return std::unexpected(Error{Errc::unavailable, "onnx reranker unavailable"});
#else
    if (passages.empty()) return std::vector<float>{};
    auto& im = *impl_;
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<std::string> in_names, out_names;
    for (std::size_t i = 0; i < im.session->GetInputCount(); ++i)
        in_names.push_back(im.session->GetInputNameAllocated(i, alloc).get());
    for (std::size_t i = 0; i < im.session->GetOutputCount(); ++i)
        out_names.push_back(im.session->GetOutputNameAllocated(i, alloc).get());
    std::vector<const char*> in_c, out_c;
    for (auto& s : in_names)  in_c.push_back(s.c_str());
    for (auto& s : out_names) out_c.push_back(s.c_str());

    std::vector<float> scores;
    scores.reserve(passages.size());
    try {
        for (const auto& p : passages) {
            // The query is the FIRST sequence, the passage the SECOND — the order
            // cross-encoders are trained on. encode_pair sets token_type_ids.
            auto enc = im.tok.encode_pair(query, p, im.cfg.max_tokens);
            const std::int64_t L = static_cast<std::int64_t>(enc.ids.size());
            std::array<std::int64_t, 2> shape{1, L};
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
                mem, enc.ids.data(), enc.ids.size(), shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
                mem, enc.mask.data(), enc.mask.size(), shape.data(), 2));
            if (im.has_token_type) {
                inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
                    mem, enc.type_ids.data(), enc.type_ids.size(), shape.data(), 2));
            }
            auto res = im.session->Run(Ort::RunOptions{nullptr},
                in_c.data(), inputs.data(), inputs.size(), out_c.data(), 1);
            // Classifier logits: shape [1, num_labels]. A reranker head has 1
            // label (mono relevance) — take logit[0]. If a model emits 2 labels
            // (irrelevant/relevant, e.g. some ms-marco exports), the positive
            // class is the last logit, which is what ranking should key on.
            auto info = res[0].GetTensorTypeAndShapeInfo();
            auto dims = info.GetShape();
            const std::int64_t n_labels = dims.empty() ? 1 : dims.back();
            const float* logits = res[0].GetTensorData<float>();
            scores.push_back(n_labels >= 2 ? logits[n_labels - 1] : logits[0]);
        }
    } catch (const std::exception& ex) {
        return std::unexpected(Error{Errc::transport_error,
            std::string("onnx reranker run failed: ") + ex.what()});
    }
    return scores;
#endif
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
