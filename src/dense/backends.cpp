// rag/dense/backends.cpp — Ollama / OpenAI / Hash embedder implementations.

#include "rag/dense/backends.hpp"
#include "rag/dense/simd.hpp"

#include <cstdint>
#include <cctype>

#include <nlohmann/json.hpp>

namespace rag::dense {

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Ollama /api/embed  →  { "embeddings": [[...], [...]] }
// ─────────────────────────────────────────────────────────────────────────────
Result<std::vector<Vector>> OllamaEmbedder::embed(std::span<const std::string> texts) const {
    if (texts.empty()) return std::vector<Vector>{};
    json req;
    req["model"] = cfg_.model;
    req["input"] = json::array();
    for (const auto& t : texts) req["input"].push_back(t);

    auto resp = tp_->post_json(cfg_.host, cfg_.port, "/api/embed", req.dump(), cfg_.timeout);
    if (!resp) return std::unexpected(resp.error());
    if (resp->status != 200)
        return fail<std::vector<Vector>>(Errc::transport_error,
            "ollama status " + std::to_string(resp->status));

    auto j = json::parse(resp->body, nullptr, false);
    if (j.is_discarded() || !j.contains("embeddings"))
        return fail<std::vector<Vector>>(Errc::parse_error, "ollama: no embeddings");

    std::vector<Vector> out;
    out.reserve(j["embeddings"].size());
    for (const auto& row : j["embeddings"]) {
        Vector v;
        v.reserve(row.size());
        for (const auto& x : row) v.push_back(x.get<float>());
        normalize(v);
        out.push_back(std::move(v));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAI /v1/embeddings  →  { "data": [ { "embedding": [...] }, ... ] }
// ─────────────────────────────────────────────────────────────────────────────
Result<std::vector<Vector>> OpenAIEmbedder::embed(std::span<const std::string> texts) const {
    if (texts.empty()) return std::vector<Vector>{};
    json req;
    req["model"] = cfg_.model;
    req["input"] = json::array();
    for (const auto& t : texts) req["input"].push_back(t);

    auto resp = tp_->post_json(cfg_.host, cfg_.port, cfg_.path, req.dump(), cfg_.timeout);
    if (!resp) return std::unexpected(resp.error());
    if (resp->status != 200)
        return fail<std::vector<Vector>>(Errc::transport_error,
            "openai status " + std::to_string(resp->status));

    auto j = json::parse(resp->body, nullptr, false);
    if (j.is_discarded() || !j.contains("data"))
        return fail<std::vector<Vector>>(Errc::parse_error, "openai: no data");

    std::vector<Vector> out;
    out.reserve(j["data"].size());
    for (const auto& item : j["data"]) {
        if (!item.contains("embedding")) continue;
        Vector v;
        for (const auto& x : item["embedding"]) v.push_back(x.get<float>());
        normalize(v);
        out.push_back(std::move(v));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// HashEmbedder — deterministic local bag-of-token-hashes. No network.
// ─────────────────────────────────────────────────────────────────────────────
namespace {
std::uint64_t fnv1a(std::string_view s, std::uint64_t seed = 1469598103934665603ull) {
    std::uint64_t h = seed;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}
} // namespace

Result<std::vector<Vector>> HashEmbedder::embed(std::span<const std::string> texts) const {
    std::vector<Vector> out;
    out.reserve(texts.size());
    for (const auto& text : texts) {
        Vector v(dim_, 0.0f);
        // Tokenize on whitespace + fold word + bigram hashes into the bag.
        std::string prev;
        std::string cur;
        auto emit = [&](const std::string& tok) {
            if (tok.empty()) return;
            std::uint64_t h = fnv1a(tok);
            std::size_t idx = h % dim_;
            float sign = (h & (1ull << 63)) ? -1.0f : 1.0f;
            v[idx] += sign;
            if (!prev.empty()) {
                std::uint64_t hb = fnv1a(prev + "_" + tok);
                v[hb % dim_] += (hb & (1ull << 63)) ? -0.5f : 0.5f;
            }
            prev = tok;
        };
        for (char ch : text) {
            unsigned char c = static_cast<unsigned char>(ch);
            if (std::isalnum(c)) cur.push_back(static_cast<char>(std::tolower(c)));
            else { emit(cur); cur.clear(); }
        }
        emit(cur);
        normalize(v);
        out.push_back(std::move(v));
    }
    return out;
}

} // namespace rag::dense
