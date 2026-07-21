#pragma once
// rag/dense/backends.hpp — concrete Embedder implementations.
//
// OllamaEmbedder  — POSTs to /api/embed of a local Ollama server.
// OpenAIEmbedder  — POSTs to /v1/embeddings (OpenAI-compatible: OpenAI, LM
//                   Studio, llama.cpp server, vLLM, together, etc.).
//
// Both take an injected HttpTransport and degrade gracefully: if the server is
// unreachable, embed() returns Errc::unavailable and the caller falls back to
// lexical-only retrieval.

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"

namespace rag::dense {

struct OllamaConfig {
    std::string   host    = "127.0.0.1";
    std::uint16_t port    = 11434;
    std::string   model   = "nomic-embed-text";
    std::size_t   dim     = 768;
    std::chrono::milliseconds timeout{30'000};
};

class OllamaEmbedder {
public:
    explicit OllamaEmbedder(OllamaConfig cfg,
                            std::shared_ptr<HttpTransport> tp = default_http_transport())
        : cfg_(std::move(cfg)), tp_(std::move(tp)) {}

    [[nodiscard]] std::size_t dimension() const noexcept { return cfg_.dim; }
    [[nodiscard]] std::string_view identity() const noexcept { return cfg_.model; }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;

private:
    OllamaConfig cfg_;
    std::shared_ptr<HttpTransport> tp_;
};

struct OpenAIConfig {
    std::string   host    = "api.openai.com";
    std::uint16_t port    = 443;               // NOTE: default transport is plaintext; use a TLS transport for :443
    std::string   path    = "/v1/embeddings";
    std::string   model   = "text-embedding-3-small";
    std::string   api_key;                     // Bearer token (sent via transport if it supports headers)
    std::size_t   dim     = 1536;
    std::chrono::milliseconds timeout{30'000};
};

class OpenAIEmbedder {
public:
    explicit OpenAIEmbedder(OpenAIConfig cfg,
                            std::shared_ptr<HttpTransport> tp = default_http_transport())
        : cfg_(std::move(cfg)), tp_(std::move(tp)) {}

    [[nodiscard]] std::size_t dimension() const noexcept { return cfg_.dim; }
    [[nodiscard]] std::string_view identity() const noexcept { return cfg_.model; }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;

private:
    OpenAIConfig cfg_;
    std::shared_ptr<HttpTransport> tp_;
};

// A deterministic hashing embedder — NO network, purely local. Not semantic,
// but stable and dependency-free: perfect for tests, offline smoke runs, and
// as the guaranteed fallback when no model server is configured. Hashes token
// n-grams into a fixed-dim bag then L2-normalizes.
class HashEmbedder {
public:
    explicit HashEmbedder(std::size_t dim = 256) : dim_(dim) {}
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::string_view identity() const noexcept { return "hash-embed-v1"; }
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;
private:
    std::size_t dim_;
};

static_assert(Embedder<OllamaEmbedder>);
static_assert(Embedder<OpenAIEmbedder>);
static_assert(Embedder<HashEmbedder>);

} // namespace rag::dense
