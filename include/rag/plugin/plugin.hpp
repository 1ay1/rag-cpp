#pragma once
// rag/plugin/plugin.hpp — the extension-point umbrella.
//
// Include this to gain the full runtime-extensibility surface:
//
//   * Registry<Interface> / Registries — name → factory, per component type.
//   * Registries::builtin()      — the backends this library ships.
//   * load_plugin / load_plugin_dir — add out-of-tree backends from .so's.
//
// Typical use:
//
//   #include <rag/plugin/plugin.hpp>
//   auto regs = rag::plugin::Registries::builtin();      // hash/ollama/...
//   rag::plugin::load_plugin_dir("./rag_plugins", regs); // optional: 3rd-party
//   auto emb = regs.make_embedder(cfg["embedder"]);      // build by name
//
// See docs: PLUGINS.md.

#include "rag/plugin/registry.hpp"
#include "rag/plugin/loader.hpp"
#include "rag/plugin/builder.hpp"

#include "rag/dense/embedder.hpp"
#include "rag/rerank/reranker.hpp"

namespace rag::plugin {

// Build an embedder/reranker from a config spec using the built-in
// backends. `spec` may be a bare name string or an object with a "type".
[[nodiscard]] inline Result<AnyEmbedder> make_embedder(const Json& spec) {
    return Registries::builtin().make_embedder(spec);
}
[[nodiscard]] inline Result<AnyReranker> make_reranker(const Json& spec) {
    return Registries::builtin().make_reranker(spec);
}

} // namespace rag::plugin
