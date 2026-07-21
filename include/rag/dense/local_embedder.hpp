#pragma once
// rag/dense/local_embedder.hpp — IN-PROCESS embedders (ONNX Runtime + GGUF).
//
// Every other embedder in the library reaches a model over HTTP. These two run
// the model INSIDE the process — no server, no network — which is what most
// production deployments actually want (lower latency, no ops, offline).
//
//   • OnnxEmbedder  — loads a sentence-transformer exported to ONNX and runs it
//     with ONNX Runtime. Handles tokenization (WordPiece) → input_ids/mask →
//     session.Run → mean-pool → L2-normalize. Enabled by RAGCPP_WITH_ONNX.
//
//   • GgufEmbedder  — loads a GGUF embedding model (e.g. nomic-embed, bge) and
//     runs it with llama.cpp's embedding API. Enabled by RAGCPP_WITH_LLAMA.
//
// Both are gated behind CMake options and optional dependencies. When the
// dependency is absent the class is still DECLARED (so callers compile) but its
// factory returns Errc::unavailable — the same graceful-degradation contract
// the HTTP backends honour. This keeps the core dependency-free by default
// while making the in-process path a one-flag opt-in.

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"

namespace rag::dense {

// Pooling strategy over token embeddings → one sentence vector.
enum class Pooling { mean, cls, max };

struct LocalEmbedderConfig {
    std::string model_path;                 // .onnx file or .gguf file
    std::string tokenizer_path;             // ONNX only: HF tokenizer.json / vocab
    Pooling     pooling      = Pooling::mean;
    bool        normalize    = true;        // L2-normalize output (cosine-ready)
    std::size_t max_tokens   = 512;         // truncate long inputs
    int         threads      = 0;           // 0 = auto (hardware_concurrency)
    std::string identity_tag = "local";     // model id for cache keying
};

// ── ONNX Runtime embedder ────────────────────────────────────────────────────
// Models the Embedder concept. PIMPL so the heavy onnxruntime headers never
// leak into ours; when built without RAGCPP_WITH_ONNX the impl is a stub whose
// load() returns unavailable.
class OnnxEmbedder {
public:
    // Load a model. Fails with Errc::unavailable if the library was built
    // without ONNX support, or with a typed error if the file is missing/bad.
    [[nodiscard]] static Result<OnnxEmbedder> load(LocalEmbedderConfig cfg);

    [[nodiscard]] std::size_t      dimension() const;
    [[nodiscard]] std::string_view identity() const;
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;

    // Compile-time capability flag.
    [[nodiscard]] static constexpr bool available() noexcept {
#ifdef RAGCPP_WITH_ONNX
        return true;
#else
        return false;
#endif
    }

    OnnxEmbedder(OnnxEmbedder&&) noexcept;
    OnnxEmbedder& operator=(OnnxEmbedder&&) noexcept;
    ~OnnxEmbedder();

private:
    OnnxEmbedder();
    struct Impl;
    std::shared_ptr<Impl> impl_;   // shared so AnyEmbedder copy-erase is cheap
};

// ── GGUF / llama.cpp embedder ────────────────────────────────────────────────
class GgufEmbedder {
public:
    [[nodiscard]] static Result<GgufEmbedder> load(LocalEmbedderConfig cfg);

    [[nodiscard]] std::size_t      dimension() const;
    [[nodiscard]] std::string_view identity() const;
    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const;

    [[nodiscard]] static constexpr bool available() noexcept {
#ifdef RAGCPP_WITH_LLAMA
        return true;
#else
        return false;
#endif
    }

    GgufEmbedder(GgufEmbedder&&) noexcept;
    GgufEmbedder& operator=(GgufEmbedder&&) noexcept;
    ~GgufEmbedder();

private:
    GgufEmbedder();
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

static_assert(Embedder<OnnxEmbedder>, "OnnxEmbedder must model Embedder");
static_assert(Embedder<GgufEmbedder>, "GgufEmbedder must model Embedder");

} // namespace rag::dense
