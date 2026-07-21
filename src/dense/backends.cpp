// rag/dense/backends.cpp — Ollama / OpenAI / llama.cpp / Hash embedders.

#include "rag/dense/backends.hpp"
#include "rag/dense/simd.hpp"

#include <cctype>
#include <cstdint>

#include <nlohmann/json.hpp>

namespace rag::dense {

using json = nlohmann::json;

namespace {
// Parse an array-of-arrays of floats into normalized vectors.
std::vector<Vector> parse_matrix(const json& rows) {
    std::vector<Vector> out;
    out.reserve(rows.size());
    for (const auto& row : rows) {
        Vector v; v.reserve(row.size());
        for (const auto& x : row) v.push_back(x.get<float>());
        normalize(v);
        out.push_back(std::move(v));
    }
    return out;
}
} // namespace

// ─── Ollama /api/embed → { "embeddings": [[...],...] } ────────────────────────
Result<std::vector<Vector>> OllamaEmbedder::embed(std::span<const std::string> texts) const {
    if (texts.empty()) return std::vector<Vector>{};
    json req;
    req["model"] = cfg_.model;
    req["input"] = json::array();
    for (const auto& t : texts) req["input"].push_back(t);

    auto resp = tp_->post_json(cfg_.host, cfg_.port, "/api/embed", req.dump(), cfg_.timeout);
    if (!resp) return std::unexpected(resp.error());
    if (resp->status != 200)
        return fail<std::vector<Vector>>(Errc::transport_error, "ollama status " + std::to_string(resp->status));
    auto j = json::parse(resp->body, nullptr, false);
    if (j.is_discarded() || !j.contains("embeddings"))
        return fail<std::vector<Vector>>(Errc::parse_error, "ollama: no embeddings");
    return parse_matrix(j["embeddings"]);
}

// ─── OpenAI presets ───────────────────────────────────────────────────────────
OpenAIConfig OpenAIConfig::openai(std::string key, std::string model) {
    OpenAIConfig c;
    c.host = "api.openai.com"; c.port = 443; c.tls = true;
    c.path = "/v1/embeddings"; c.model = std::move(model); c.api_key = std::move(key);
    c.dim = c.model.find("large") != std::string::npos ? 3072 : 1536;
    return c;
}
OpenAIConfig OpenAIConfig::together(std::string key, std::string model) {
    OpenAIConfig c;
    c.host = "api.together.xyz"; c.port = 443; c.tls = true;
    c.path = "/v1/embeddings"; c.model = std::move(model); c.api_key = std::move(key);
    c.dim = 768;
    return c;
}
OpenAIConfig OpenAIConfig::local(std::string host, std::uint16_t port, std::string model, std::size_t dim) {
    OpenAIConfig c;
    c.host = std::move(host); c.port = port; c.tls = false;
    c.path = "/v1/embeddings"; c.model = std::move(model); c.dim = dim;
    return c;
}
OpenAIConfig OpenAIConfig::tei(std::string host, std::uint16_t port, std::size_t dim) {
    OpenAIConfig c;
    c.host = std::move(host); c.port = port; c.tls = false;
    c.path = "/embed"; c.model = "tei"; c.dim = dim;
    return c;
}

// ─── OpenAI-compatible /v1/embeddings and TEI /embed ──────────────────────────
Result<std::vector<Vector>> OpenAIEmbedder::embed(std::span<const std::string> texts) const {
    if (texts.empty()) return std::vector<Vector>{};

    const bool is_tei = cfg_.path == "/embed";
    json req;
    if (is_tei) {
        // TEI: { "inputs": ["...", "..."] } → [[...],[...]]
        req["inputs"] = json::array();
        for (const auto& t : texts) req["inputs"].push_back(t);
    } else {
        req["model"] = cfg_.model;
        req["input"] = json::array();
        for (const auto& t : texts) req["input"].push_back(t);
    }

    HttpRequest hr;
    hr.host = cfg_.host; hr.port = cfg_.port; hr.path = cfg_.path;
    std::string body = req.dump();
    hr.body = body; hr.tls = cfg_.tls; hr.timeout = cfg_.timeout;
    if (!cfg_.api_key.empty()) hr.headers.push_back({"Authorization", "Bearer " + cfg_.api_key});

    auto resp = tp_->post(hr);
    if (!resp) return std::unexpected(resp.error());
    if (resp->status != 200)
        return fail<std::vector<Vector>>(Errc::transport_error, "openai status " + std::to_string(resp->status));
    auto j = json::parse(resp->body, nullptr, false);
    if (j.is_discarded()) return fail<std::vector<Vector>>(Errc::parse_error, "openai json");

    if (is_tei && j.is_array()) return parse_matrix(j);
    if (!j.contains("data")) return fail<std::vector<Vector>>(Errc::parse_error, "openai: no data");
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

// ─── llama.cpp server /embedding ──────────────────────────────────────────────
// Newer builds accept {"content": "..."} and return {"embedding":[...]}; batch
// builds return an array of {"embedding":[...]} objects. We handle both, one
// request per text to stay compatible with all server versions.
Result<std::vector<Vector>> LlamaCppEmbedder::embed(std::span<const std::string> texts) const {
    std::vector<Vector> out;
    out.reserve(texts.size());
    for (const auto& t : texts) {
        json req; req["content"] = t;
        auto resp = tp_->post_json(cfg_.host, cfg_.port, cfg_.path, req.dump(), cfg_.timeout);
        if (!resp) return std::unexpected(resp.error());
        if (resp->status != 200)
            return fail<std::vector<Vector>>(Errc::transport_error, "llamacpp status " + std::to_string(resp->status));
        auto j = json::parse(resp->body, nullptr, false);
        if (j.is_discarded()) return fail<std::vector<Vector>>(Errc::parse_error, "llamacpp json");

        const json* emb = nullptr;
        if (j.contains("embedding")) emb = &j["embedding"];
        else if (j.is_array() && !j.empty() && j[0].contains("embedding")) emb = &j[0]["embedding"];
        if (!emb || !emb->is_array()) return fail<std::vector<Vector>>(Errc::parse_error, "llamacpp: no embedding");

        Vector v; v.reserve(emb->size());
        for (const auto& x : *emb) v.push_back(x.get<float>());
        normalize(v);
        if (dim_ == 0) dim_ = v.size();
        out.push_back(std::move(v));
    }
    return out;
}

// ─── HashEmbedder — deterministic local bag-of-token-hashes ───────────────────
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
        std::string prev, cur;
        auto emit = [&](const std::string& tok) {
            if (tok.empty()) return;
            std::uint64_t h = fnv1a(tok);
            v[h % dim_] += (h & (1ull << 63)) ? -1.0f : 1.0f;
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
