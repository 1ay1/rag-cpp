#pragma once
// rag/dense/backends.hpp — concrete Embedder implementations.
//
//   OllamaEmbedder     — POSTs to /api/embed of a local Ollama server.
//   OpenAIEmbedder     — POSTs to /v1/embeddings (OpenAI-compatible: OpenAI,
//                        LM Studio, llama.cpp server, vLLM, together, Groq,
//                        Text-Embeddings-Inference, ...). Bearer-auth + TLS.
//   LlamaCppEmbedder   — POSTs to /embedding of a llama.cpp `server` build.
//   HashEmbedder       — deterministic, local, no network (tests / fallback).
//
// Every network backend takes an injected HttpTransport and degrades to
// Errc::unavailable on failure so callers can fall back to lexical-only search.
// The `RetryingEmbedder` decorator wraps any Embedder with bounded exponential
// backoff, and `FallbackEmbedder` chains a primary → secondary (e.g. hosted →
// local hash) so retrieval never hard-fails.

#include <chrono>
#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"

namespace rag::dense {

// ─── Ollama ───────────────────────────────────────────────────────────────────
struct OllamaConfig {
    std::string   host    = "127.0.0.1";
    std::uint16_t port    = 11434;
    std::string   model   = "nomic-embed-text";
    std::size_t   dim     = 768;
    std::chrono::milliseconds timeout{30'000};
    // In-flight batches. Ollama serializes on one model by default, but the
    // client-side win from pipelining round-trips is real; 4 is a safe default
    // that does not stampede a laptop-hosted server.
    std::size_t   concurrency = 4;
};

class OllamaEmbedder {
public:
    explicit OllamaEmbedder(OllamaConfig cfg,
                            std::shared_ptr<HttpTransport> tp = default_http_transport())
        : cfg_(std::move(cfg)), tp_(std::move(tp)) {}
    [[nodiscard]] std::size_t dimension() const noexcept { return cfg_.dim; }
    [[nodiscard]] std::string_view identity() const noexcept { return cfg_.model; }
    [[nodiscard]] std::size_t max_concurrency() const noexcept { return cfg_.concurrency; }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;
private:
    OllamaConfig cfg_;
    std::shared_ptr<HttpTransport> tp_;
};

// ─── OpenAI-compatible /v1/embeddings ─────────────────────────────────────────
struct OpenAIConfig {
    std::string   host    = "api.openai.com";
    std::uint16_t port    = 443;
    bool          tls     = true;              // requires a TLS transport for the default host
    std::string   path    = "/v1/embeddings";
    std::string   model   = "text-embedding-3-small";
    std::string   api_key;                     // Bearer token
    std::size_t   dim     = 1536;
    std::chrono::milliseconds timeout{30'000};
    // Hosted embedding endpoints are latency-bound and rate-limited per minute,
    // not per connection: 8 concurrent requests is the usual sweet spot before
    // 429s start costing more than they buy.
    std::size_t   concurrency = 8;

    // Presets for common OpenAI-compatible servers.
    static OpenAIConfig openai(std::string key, std::string model = "text-embedding-3-small");
    static OpenAIConfig together(std::string key, std::string model);
    static OpenAIConfig local(std::string host = "127.0.0.1", std::uint16_t port = 8080,
                              std::string model = "default", std::size_t dim = 768);
    // HuggingFace Text-Embeddings-Inference (TEI) speaks /embed with a raw array.
    static OpenAIConfig tei(std::string host, std::uint16_t port, std::size_t dim);
};

class OpenAIEmbedder {
public:
    explicit OpenAIEmbedder(OpenAIConfig cfg,
                            std::shared_ptr<HttpTransport> tp = default_http_transport())
        : cfg_(std::move(cfg)), tp_(std::move(tp)) {}
    [[nodiscard]] std::size_t dimension() const noexcept { return cfg_.dim; }
    [[nodiscard]] std::string_view identity() const noexcept { return cfg_.model; }
    [[nodiscard]] std::size_t max_concurrency() const noexcept { return cfg_.concurrency; }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;
private:
    OpenAIConfig cfg_;
    std::shared_ptr<HttpTransport> tp_;
};

// ─── llama.cpp server /embedding ──────────────────────────────────────────────
struct LlamaCppConfig {
    std::string   host    = "127.0.0.1";
    std::uint16_t port    = 8080;
    std::string   path    = "/embedding";
    std::size_t   dim     = 0;                  // 0 => inferred from first response
    std::chrono::milliseconds timeout{30'000};
    // llama.cpp's server runs the model on all cores per request; concurrent
    // requests mostly queue. Kept at 1 — raise only for a multi-slot server.
    std::size_t   concurrency = 1;
};

class LlamaCppEmbedder {
public:
    explicit LlamaCppEmbedder(LlamaCppConfig cfg,
                              std::shared_ptr<HttpTransport> tp = default_http_transport())
        : cfg_(std::move(cfg)), tp_(std::move(tp)) {}
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::string_view identity() const noexcept { return "llamacpp-embed"; }
    [[nodiscard]] std::size_t max_concurrency() const noexcept { return cfg_.concurrency; }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;
private:
    LlamaCppConfig cfg_;
    std::shared_ptr<HttpTransport> tp_;
    mutable std::size_t dim_ = 0;
};

// ─── Deterministic local hash embedder (no network) ───────────────────────────
class HashEmbedder {
public:
    explicit HashEmbedder(std::size_t dim = 256) : dim_(dim) {}
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::string_view identity() const noexcept { return "hash-embed-v1"; }
    // Pure CPU, no shared state, no internal threading — scales with cores, and
    // NOT beyond them: unlike a network backend there is nothing to overlap, so
    // extra workers would only contend.
    [[nodiscard]] std::size_t max_concurrency() const noexcept {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc ? hc : 1u;
    }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;
private:
    std::size_t dim_;
};

// ─── Decorators ───────────────────────────────────────────────────────────────
// Retry any embedder with bounded exponential backoff on transient failures.
class RetryingEmbedder {
public:
    RetryingEmbedder(AnyEmbedder inner, int max_attempts = 3,
                     std::chrono::milliseconds base_delay = std::chrono::milliseconds(200))
        : inner_(std::move(inner)), attempts_(max_attempts), delay_(base_delay) {}
    [[nodiscard]] std::size_t dimension() const { return inner_.dimension(); }
    [[nodiscard]] std::string_view identity() const { return inner_.identity(); }
    [[nodiscard]] std::size_t max_concurrency() const { return inner_.max_concurrency(); }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const {
        Result<std::vector<Vector>> last = fail<std::vector<Vector>>(Errc::unavailable);
        auto d = delay_;
        for (int i = 0; i < attempts_; ++i) {
            last = inner_.embed(texts);
            if (last) return last;
            if (last.error().code != Errc::transport_error &&
                last.error().code != Errc::unavailable) return last; // non-transient
            if (i + 1 < attempts_) { std::this_thread::sleep_for(d); d *= 2; }
        }
        return last;
    }
private:
    AnyEmbedder inner_;
    int attempts_;
    std::chrono::milliseconds delay_;
};

// Try primary; on unavailable/transport error, fall through to secondary.
// Dimension follows the primary (so both should agree, or the corpus keys on it).
class FallbackEmbedder {
public:
    FallbackEmbedder(AnyEmbedder primary, AnyEmbedder secondary)
        : primary_(std::move(primary)), secondary_(std::move(secondary)) {}
    [[nodiscard]] std::size_t dimension() const { return primary_.dimension(); }
    [[nodiscard]] std::string_view identity() const { return primary_.identity(); }
    // Either path may run, so honour the more conservative of the two.
    [[nodiscard]] std::size_t max_concurrency() const {
        return std::min(primary_.max_concurrency(), secondary_.max_concurrency());
    }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const {
        auto r = primary_.embed(texts);
        if (r) return r;
        if (r.error().code == Errc::transport_error || r.error().code == Errc::unavailable)
            return secondary_.embed(texts);
        return r;
    }
private:
    AnyEmbedder primary_, secondary_;
};

static_assert(Embedder<OllamaEmbedder>);
static_assert(Embedder<OpenAIEmbedder>);
static_assert(Embedder<LlamaCppEmbedder>);
static_assert(Embedder<HashEmbedder>);
static_assert(Embedder<RetryingEmbedder>);
static_assert(Embedder<FallbackEmbedder>);

} // namespace rag::dense
