// src/plugin/builtins.cpp — register every built-in backend with the plugin
// registry so it is constructible by name from a JSON config.
//
// This is the bridge between the compile-time concept world and the load-time
// config world. After this TU's static initializers run:
//
//   auto emb = Registry<AnyEmbedder>::instance()
//                  .create_from({{"type","ollama"},{"model","nomic-embed-text"}});
//
// works with zero knowledge of OllamaEmbedder at the call site. A config file,
// the CLI, the C ABI, or a REST layer all go through this one door.
//
// Adding a new built-in backend = one RAG_REGISTER block here. Adding an
// out-of-tree backend = the same block in the plugin's own .so (loader.hpp).

#include "rag/plugin/registry.hpp"

#include "rag/bridge/bridge.hpp"
#include "rag/dense/backends.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/dense/local_embedder.hpp"
#include "rag/rerank/reranker.hpp"

namespace rag::plugin {
namespace {

using json = nlohmann::json;
using ::rag::dense::AnyEmbedder;
using ::rag::rerank::AnyReranker;

// Small JSON helpers that never throw — missing/wrong-typed keys fall back to
// the provided default. Keeps factories total.
template <class T>
T get_or(const json& j, std::string_view key, T dflt) {
    if (auto it = j.find(key); it != j.end() && !it->is_null()) {
        try { return it->get<T>(); } catch (...) { return dflt; }
    }
    return dflt;
}

std::chrono::milliseconds get_timeout(const json& j, long dflt_ms = 30'000) {
    return std::chrono::milliseconds(get_or<long>(j, "timeout_ms", dflt_ms));
}

// ── Embedders ────────────────────────────────────────────────────────────────

RAG_REGISTER(AnyEmbedder, "hash", [](const json& c) -> Result<AnyEmbedder> {
    return AnyEmbedder{dense::HashEmbedder{get_or<std::size_t>(c, "dim", 256)}};
});

RAG_REGISTER(AnyEmbedder, "ollama", [](const json& c) -> Result<AnyEmbedder> {
    dense::OllamaConfig cfg;
    cfg.host    = get_or<std::string>(c, "host", cfg.host);
    cfg.port    = static_cast<std::uint16_t>(get_or<int>(c, "port", cfg.port));
    cfg.model   = get_or<std::string>(c, "model", cfg.model);
    cfg.dim     = get_or<std::size_t>(c, "dim", cfg.dim);
    cfg.timeout = get_timeout(c);
    return AnyEmbedder{dense::OllamaEmbedder{std::move(cfg)}};
});

RAG_REGISTER(AnyEmbedder, "openai", [](const json& c) -> Result<AnyEmbedder> {
    dense::OpenAIConfig cfg;
    cfg.host    = get_or<std::string>(c, "host", cfg.host);
    cfg.port    = static_cast<std::uint16_t>(get_or<int>(c, "port", cfg.port));
    cfg.tls     = get_or<bool>(c, "tls", cfg.tls);
    cfg.path    = get_or<std::string>(c, "path", cfg.path);
    cfg.model   = get_or<std::string>(c, "model", cfg.model);
    cfg.api_key = get_or<std::string>(c, "api_key", cfg.api_key);
    cfg.dim     = get_or<std::size_t>(c, "dim", cfg.dim);
    cfg.timeout = get_timeout(c);
    return AnyEmbedder{dense::OpenAIEmbedder{std::move(cfg)}};
});

RAG_REGISTER(AnyEmbedder, "llamacpp", [](const json& c) -> Result<AnyEmbedder> {
    dense::LlamaCppConfig cfg;
    cfg.host    = get_or<std::string>(c, "host", cfg.host);
    cfg.port    = static_cast<std::uint16_t>(get_or<int>(c, "port", cfg.port));
    cfg.path    = get_or<std::string>(c, "path", cfg.path);
    cfg.dim     = get_or<std::size_t>(c, "dim", cfg.dim);
    cfg.timeout = get_timeout(c);
    return AnyEmbedder{dense::LlamaCppEmbedder{std::move(cfg)}};
});

// ── Local (in-process) embedders ─────────────────────────────────────────────
// ONNX Runtime and GGUF/llama.cpp run the model IN this process behind the same
// Embedder concept. They exist as classes already; registering them here is what
// makes them reachable BY NAME from a config file / the CLI / the C ABI, exactly
// like the network backends. When the library was built without the relevant
// feature flag, load() returns Errc::unavailable — so the factory surfaces a
// clear, typed error instead of the name simply not existing. (The alternative,
// not registering them, would make `{"type":"onnx"}` fail with "unknown type"
// even on a build that COULD support it once the flag is flipped — a worse
// diagnostic. Registering always, and letting load() report availability, keeps
// the config surface stable across build configurations.)

dense::LocalEmbedderConfig local_cfg(const json& c) {
    dense::LocalEmbedderConfig lc;
    lc.model_path     = get_or<std::string>(c, "model_path", lc.model_path);
    lc.tokenizer_path = get_or<std::string>(c, "tokenizer_path", lc.tokenizer_path);
    lc.normalize      = get_or<bool>(c, "normalize", lc.normalize);
    lc.max_tokens     = get_or<std::size_t>(c, "max_tokens", lc.max_tokens);
    lc.threads        = get_or<int>(c, "threads", lc.threads);
    lc.identity_tag   = get_or<std::string>(c, "identity_tag", lc.identity_tag);
    std::string pool  = get_or<std::string>(c, "pooling", "mean");
    lc.pooling = (pool == "cls") ? dense::Pooling::cls : dense::Pooling::mean;
    return lc;
}

RAG_REGISTER(AnyEmbedder, "onnx", [](const json& c) -> Result<AnyEmbedder> {
    auto e = dense::OnnxEmbedder::load(local_cfg(c));
    if (!e) return std::unexpected(e.error());   // unavailable if !RAGCPP_WITH_ONNX
    return AnyEmbedder{std::move(*e)};
});

RAG_REGISTER(AnyEmbedder, "gguf", [](const json& c) -> Result<AnyEmbedder> {
    auto e = dense::GgufEmbedder::load(local_cfg(c));
    if (!e) return std::unexpected(e.error());   // unavailable if !RAGCPP_WITH_LLAMA
    return AnyEmbedder{std::move(*e)};
});

// ── Composition decorators ───────────────────────────────────────────────────
// These take NESTED embedder specs and resolve them through the SAME registry,
// so config can express resilience declaratively:
//
//   {"type":"fallback",
//    "primary":  {"type":"onnx", "model_path":"bge.onnx", ...},
//    "secondary":{"type":"hash", "dim":256}}
//
// A model outage then silently degrades to the hash embedder with no code. The
// nested resolution is why Registry invokes factories UNLOCKED (see registry.hpp):
// building a `fallback` recursively builds two more embedders through the same
// singleton, which would otherwise deadlock.

RAG_REGISTER(AnyEmbedder, "retry", [](const json& c) -> Result<AnyEmbedder> {
    auto it = c.find("inner");
    if (it == c.end())
        return fail<AnyEmbedder>(Errc::invalid_argument, "retry: missing \"inner\" embedder spec");
    auto inner = Registry<AnyEmbedder>::instance().create_from(*it);
    if (!inner) return std::unexpected(inner.error());
    int  attempts = get_or<int>(c, "max_attempts", 3);
    auto delay    = std::chrono::milliseconds(get_or<long>(c, "base_delay_ms", 200));
    dense::RetryingEmbedder wrapped(std::move(*inner), attempts, delay);
    return AnyEmbedder{std::move(wrapped)};
});

RAG_REGISTER(AnyEmbedder, "fallback", [](const json& c) -> Result<AnyEmbedder> {
    auto pit = c.find("primary");
    auto sit = c.find("secondary");
    if (pit == c.end() || sit == c.end())
        return fail<AnyEmbedder>(Errc::invalid_argument,
            "fallback: needs both \"primary\" and \"secondary\" embedder specs");
    // The secondary must always be constructible — it is the safety net.
    auto secondary = Registry<AnyEmbedder>::instance().create_from(*sit);
    if (!secondary) return std::unexpected(secondary.error());
    // If the PRIMARY cannot even be constructed (e.g. this build lacks ONNX, or
    // an API key is missing), degrade to the secondary AT CONSTRUCTION rather
    // than failing — that is the whole point of a fallback. A primary that
    // constructs but fails per-request is handled at runtime by FallbackEmbedder.
    auto primary = Registry<AnyEmbedder>::instance().create_from(*pit);
    if (!primary) return AnyEmbedder{std::move(*secondary)};
    dense::FallbackEmbedder wrapped(std::move(*primary), std::move(*secondary));
    return AnyEmbedder{std::move(wrapped)};
});


// ── Rerankers ────────────────────────────────────────────────────────────────

RAG_REGISTER(AnyReranker, "cross_encoder", [](const json& c) -> Result<AnyReranker> {
    std::string wire = get_or<std::string>(c, "wire", "tei");
    if (wire == "cohere") {
        auto cfg = rerank::CrossEncoderConfig::cohere(
            get_or<std::string>(c, "api_key", ""),
            get_or<std::string>(c, "model", "rerank-english-v3.0"));
        return AnyReranker{rerank::CrossEncoderReranker{std::move(cfg)}};
    }
    if (wire == "jina") {
        auto cfg = rerank::CrossEncoderConfig::jina(
            get_or<std::string>(c, "api_key", ""),
            get_or<std::string>(c, "model", "jina-reranker-v2-base-multilingual"));
        return AnyReranker{rerank::CrossEncoderReranker{std::move(cfg)}};
    }
    auto cfg = rerank::CrossEncoderConfig::tei(
        get_or<std::string>(c, "host", "127.0.0.1"),
        static_cast<std::uint16_t>(get_or<int>(c, "port", 8080)));
    return AnyReranker{rerank::CrossEncoderReranker{std::move(cfg)}};
});

} // namespace

// Force the TU to be retained when the library is linked statically into an
// executable. Call this once (e.g. from engine setup or main) to guarantee the
// static registrars above are not stripped by the linker as "unused".
void ensure_builtins_registered() noexcept {
    // Also register the polyglot bridge transports (process/http) so config like
    // {"type":"process", ...} resolves out of the box.
    ::rag::bridge::ensure_bridge_registered();
}

} // namespace rag::plugin
