// src/plugin/builtins.cpp — register every built-in backend with the plugin
// registry so it is constructible by name from a JSON config.
//
// This is the bridge between the compile-time concept world and the load-time
// config world. After the builtins are registered:
//
//   auto emb = plugin::make_embedder({{"type","ollama"},{"model","nomic-embed-text"}});
//
// works with zero knowledge of OllamaEmbedder at the call site. A config file,
// the CLI, the C ABI, or a REST layer all go through this one door.
//
// ADDING A BUILT-IN BACKEND = one register_embedder(...) / register_reranker(...)
// call below. You return your concept type; the AnyX wrap, the description, and
// the not-throwing config parsing are handled by rag/plugin/builder.hpp. Adding
// an OUT-OF-TREE backend = the same call in the plugin's own .so (loader.hpp).
//
// (These are function calls, not static RAG_REGISTER macros, so registration
// happens from register_all() below rather than at static-init time. That makes
// the order explicit and sidesteps the macro comma-in-braces trap entirely.)

#include "rag/plugin/builder.hpp"

#include "rag/bridge/bridge.hpp"
#include "rag/dense/backends.hpp"
#include "rag/dense/local_embedder.hpp"

#include <atomic>
#include <chrono>

namespace rag::plugin {
namespace {

std::chrono::milliseconds timeout_ms(const Config& c, long dflt = 30'000) {
    return std::chrono::milliseconds(c.get<long>("timeout_ms", dflt));
}

dense::LocalEmbedderConfig local_cfg(const Config& c) {
    dense::LocalEmbedderConfig lc;
    lc.model_path     = c.get("model_path", lc.model_path);
    lc.tokenizer_path = c.get("tokenizer_path", lc.tokenizer_path);
    lc.normalize      = c.get("normalize", lc.normalize);
    lc.max_tokens     = c.get<std::size_t>("max_tokens", lc.max_tokens);
    lc.threads        = c.get<int>("threads", lc.threads);
    lc.identity_tag   = c.get("identity_tag", lc.identity_tag);
    lc.pooling = (c.get("pooling", "mean") == "cls") ? dense::Pooling::cls : dense::Pooling::mean;
    return lc;
}

void register_embedders() {
    // ── Network backends ─────────────────────────────────────────────────────
    register_embedder("hash", "deterministic feature-hash vectors, no deps (keys: dim)",
        [](Config c) -> Result<dense::HashEmbedder> {
            return dense::HashEmbedder{c.get<std::size_t>("dim", 256)};
        });

    register_embedder("ollama", "local Ollama server (keys: host, port, model, dim, timeout_ms)",
        [](Config c) -> Result<dense::OllamaEmbedder> {
            dense::OllamaConfig cfg;
            cfg.host    = c.get("host", cfg.host);
            cfg.port    = static_cast<std::uint16_t>(c.get<int>("port", cfg.port));
            cfg.model   = c.get("model", cfg.model);
            cfg.dim     = c.get<std::size_t>("dim", cfg.dim);
            cfg.timeout = timeout_ms(c);
            return dense::OllamaEmbedder{std::move(cfg)};
        });

    register_embedder("openai", "OpenAI-compatible embeddings API (keys: host, port, tls, path, model, api_key, dim)",
        [](Config c) -> Result<dense::OpenAIEmbedder> {
            dense::OpenAIConfig cfg;
            cfg.host    = c.get("host", cfg.host);
            cfg.port    = static_cast<std::uint16_t>(c.get<int>("port", cfg.port));
            cfg.tls     = c.get("tls", cfg.tls);
            cfg.path    = c.get("path", cfg.path);
            cfg.model   = c.get("model", cfg.model);
            cfg.api_key = c.get("api_key", cfg.api_key);
            cfg.dim     = c.get<std::size_t>("dim", cfg.dim);
            cfg.timeout = timeout_ms(c);
            return dense::OpenAIEmbedder{std::move(cfg)};
        });

    // Voyage AI and Together speak the OpenAI /v1/embeddings shape, so they are
    // one-liners over the same backend with different host/model/dim presets.
    // THIS is what "really add more" looks like: a new hosted provider is ~6
    // lines, fully by-name and self-described.
    register_embedder("voyage", "Voyage AI embeddings (keys: api_key, model, dim)",
        [](Config c) -> Result<dense::OpenAIEmbedder> {
            dense::OpenAIConfig cfg;
            cfg.host    = "api.voyageai.com";
            cfg.path    = "/v1/embeddings";
            cfg.model   = c.get("model", "voyage-3");
            cfg.api_key = c.get("api_key", "");
            cfg.dim     = c.get<std::size_t>("dim", 1024);
            cfg.timeout = timeout_ms(c);
            return dense::OpenAIEmbedder{std::move(cfg)};
        });

    register_embedder("together", "Together AI embeddings (keys: api_key, model, dim)",
        [](Config c) -> Result<dense::OpenAIEmbedder> {
            auto key   = c.require<std::string>("api_key");
            if (!key) return std::unexpected(key.error());
            auto cfg   = dense::OpenAIConfig::together(*key, c.get("model", "BAAI/bge-base-en-v1.5"));
            cfg.dim    = c.get<std::size_t>("dim", cfg.dim);
            cfg.timeout = timeout_ms(c);
            return dense::OpenAIEmbedder{std::move(cfg)};
        });

    register_embedder("llamacpp", "llama.cpp server /embedding (keys: host, port, path, dim, timeout_ms)",
        [](Config c) -> Result<dense::LlamaCppEmbedder> {
            dense::LlamaCppConfig cfg;
            cfg.host    = c.get("host", cfg.host);
            cfg.port    = static_cast<std::uint16_t>(c.get<int>("port", cfg.port));
            cfg.path    = c.get("path", cfg.path);
            cfg.dim     = c.get<std::size_t>("dim", cfg.dim);
            cfg.timeout = timeout_ms(c);
            return dense::LlamaCppEmbedder{std::move(cfg)};
        });

    // ── Local (in-process) backends ──────────────────────────────────────────
    // Registered unconditionally; when the build lacks the feature flag, load()
    // returns Errc::unavailable, so the NAME resolves and the factory reports a
    // clear typed error rather than "unknown type". This keeps the config
    // surface stable across build configurations.
    register_embedder("onnx", "in-process ONNX Runtime model (keys: model_path, tokenizer_path, pooling, normalize, max_tokens, threads)",
        [](Config c) -> Result<AnyEmbedder> {
            auto e = dense::OnnxEmbedder::load(local_cfg(c));
            if (!e) return std::unexpected(e.error());
            return AnyEmbedder{std::move(*e)};
        });

    register_embedder("gguf", "in-process GGUF/llama.cpp model (keys: model_path, pooling, normalize, max_tokens, threads)",
        [](Config c) -> Result<AnyEmbedder> {
            auto e = dense::GgufEmbedder::load(local_cfg(c));
            if (!e) return std::unexpected(e.error());
            return AnyEmbedder{std::move(*e)};
        });

    // ── Composition decorators ───────────────────────────────────────────────
    // Take NESTED embedder specs and resolve them through the same registry, so
    // config expresses resilience declaratively. Nesting is unbounded because
    // Registry invokes factories unlocked (see registry.hpp).
    register_embedder("retry", "retry an inner embedder with backoff (keys: inner, max_attempts, base_delay_ms)",
        [](Config c) -> Result<AnyEmbedder> {
            auto spec = c.sub("inner");
            if (!spec) return std::unexpected(spec.error());
            auto inner = resolve<AnyEmbedder>(*spec);
            if (!inner) return std::unexpected(inner.error());
            auto delay = std::chrono::milliseconds(c.get<long>("base_delay_ms", 200));
            dense::RetryingEmbedder wrapped(std::move(*inner), c.get<int>("max_attempts", 3), delay);
            return AnyEmbedder{std::move(wrapped)};
        });

    register_embedder("fallback", "try primary, degrade to secondary (keys: primary, secondary)",
        [](Config c) -> Result<AnyEmbedder> {
            auto pspec = c.sub("primary");
            auto sspec = c.sub("secondary");
            if (!pspec || !sspec)
                return fail<AnyEmbedder>(Errc::invalid_argument,
                    "fallback: needs both \"primary\" and \"secondary\" specs");
            // The secondary is the safety net: it must construct.
            auto secondary = resolve<AnyEmbedder>(*sspec);
            if (!secondary) return std::unexpected(secondary.error());
            // If the primary cannot even be constructed (missing feature flag /
            // key), degrade to the secondary at construction — that is the point
            // of a fallback. A primary that fails per-request degrades at runtime.
            auto primary = resolve<AnyEmbedder>(*pspec);
            if (!primary) return AnyEmbedder{std::move(*secondary)};
            dense::FallbackEmbedder wrapped(std::move(*primary), std::move(*secondary));
            return AnyEmbedder{std::move(wrapped)};
        });
}

void register_rerankers() {
    register_reranker("cross_encoder", "cross-encoder reranker (keys: wire=tei|cohere|jina, host, port, api_key, model)",
        [](Config c) -> Result<AnyReranker> {
            std::string wire = c.get("wire", "tei");
            if (wire == "cohere") {
                auto cfg = rerank::CrossEncoderConfig::cohere(
                    c.get("api_key", ""), c.get("model", "rerank-english-v3.0"));
                return AnyReranker{rerank::CrossEncoderReranker{std::move(cfg)}};
            }
            if (wire == "jina") {
                auto cfg = rerank::CrossEncoderConfig::jina(
                    c.get("api_key", ""), c.get("model", "jina-reranker-v2-base-multilingual"));
                return AnyReranker{rerank::CrossEncoderReranker{std::move(cfg)}};
            }
            auto cfg = rerank::CrossEncoderConfig::tei(
                c.get("host", "127.0.0.1"), static_cast<std::uint16_t>(c.get<int>("port", 8080)));
            return AnyReranker{rerank::CrossEncoderReranker{std::move(cfg)}};
        });
}

} // namespace

// Register every built-in backend. Idempotent (registration is insert-or-assign)
// and thread-safe; runs once behind a flag so repeated make_embedder() calls do
// not re-register on every lookup.
void ensure_builtins_registered() noexcept {
    static std::atomic<bool> done{false};
    bool expected = false;
    if (!done.compare_exchange_strong(expected, true)) return;
    register_embedders();
    register_rerankers();
    // Polyglot bridge transports (process/http/rest) so {"type":"process", ...}
    // resolves out of the box.
    ::rag::bridge::ensure_bridge_registered();
}

} // namespace rag::plugin
