#pragma once
// rag/plugin/plugin.hpp — the extension-point umbrella.
//
// Include this to gain the full runtime-extensibility surface:
//
//   * Registry<Interface>        — name → factory, per component type.
//   * RAG_REGISTER(...)          — self-register a factory at static-init time.
//   * load_plugin / load_plugin_dir — pull in out-of-tree backends from .so's.
//   * ensure_builtins_registered()  — force the built-in registrations to link.
//
// Typical use:
//
//   #include <rag/plugin/plugin.hpp>
//   rag::plugin::ensure_builtins_registered();          // pull in hash/ollama/...
//   rag::plugin::load_plugin_dir("./rag_plugins");       // optional: 3rd-party
//   auto emb = rag::plugin::Registry<rag::AnyEmbedder>::instance()
//                  .create_from(cfg["embedder"]);        // build by name
//
// See docs: PLUGINS.md.

#include "rag/plugin/registry.hpp"
#include "rag/plugin/loader.hpp"
#include "rag/plugin/builder.hpp"

#include "rag/dense/embedder.hpp"
#include "rag/rerank/reranker.hpp"

namespace rag::plugin {

// Defined in src/plugin/builtins.cpp. Referencing it from user code guarantees
// the linker retains that TU (and thus its static registrars) even in a static
// library build. The engine calls this automatically.
void ensure_builtins_registered() noexcept;

// Convenience aliases for the registries every app touches. (AnyEmbedder /
// AnyReranker are defined in builder.hpp, included above.)
using EmbedderRegistry = Registry<AnyEmbedder>;
using RerankerRegistry = Registry<AnyReranker>;

// Build an embedder/reranker directly from a config spec, ensuring built-ins are
// available first. `spec` may be a bare name string or an object with a "type".
[[nodiscard]] inline Result<AnyEmbedder> make_embedder(const Json& spec) {
    ensure_builtins_registered();
    return EmbedderRegistry::instance().create_from(spec);
}
[[nodiscard]] inline Result<AnyReranker> make_reranker(const Json& spec) {
    ensure_builtins_registered();
    return RerankerRegistry::instance().create_from(spec);
}

} // namespace rag::plugin
