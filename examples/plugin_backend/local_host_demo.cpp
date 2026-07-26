// examples/plugin_backend/local_host_demo.cpp
//
// The HOST side for the local embedder plugin. It shows the full config-driven
// story end to end:
//
//   1. built-in local embedders (onnx/gguf) are reachable BY NAME already;
//   2. a dropped-in plugin adds "local_ngram" at runtime with no host recompile;
//   3. composition decorators (fallback/retry) wire resilience from config —
//      a primary that this build can't construct silently degrades to a backup.
//
// The host was compiled with NO knowledge that "local_ngram" exists; it learns
// the name only by loading the plugin.

#include <rag/rag.hpp>

#include <cstdio>
#include <string>

namespace {
void print_names() {
    std::printf("  embedders: ");
    for (const auto& n : rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().names())
        std::printf("%s ", n.c_str());
    std::printf("\n");
}
} // namespace

int main(int argc, char** argv) {
    std::string plugin_path = argc > 1 ? argv[1] : "./liblocal_ngram_plugin.so";

    rag::plugin::ensure_builtins_registered();
    std::printf("Before loading the plugin:\n");
    print_names();   // note: onnx/gguf are already here (built-in, by name)

    // ── 1. built-in local embedders resolve by name ──────────────────────────
    // On a build without -DRAGCPP_WITH_ONNX this returns a CLEAR typed error
    // rather than "unknown type" — the name exists, the feature does not.
    if (auto e = rag::plugin::make_embedder(
            nlohmann::json{{"type", "onnx"}, {"model_path", "bge-small.onnx"}});
        !e)
        std::printf("\n  onnx by name -> %s\n", e.error().message.c_str());

    // ── 2. load the out-of-tree plugin ───────────────────────────────────────
    auto loaded = rag::plugin::load_plugin(plugin_path);
    if (!loaded) {
        std::printf("\nload failed: %s\n(pass the built plugin path as argv[1])\n",
                    loaded.error().message.c_str());
        return 1;
    }
    loaded->keep();   // keep the .so resident for the process lifetime
    std::printf("\nAfter loading %s:\n", plugin_path.c_str());
    print_names();    // "local_ngram" now appears

    // ── 3. build an engine whose embedder is the plugin's, chosen by config ──
    // The composition here says: "prefer a local ONNX model; if this build can't
    // construct it, fall back to the plugin's local_ngram; retry either on
    // transient failure." All declarative, no backend types named in code.
    nlohmann::json spec = {
        {"type", "fallback"},
        {"primary",   {{"type", "onnx"}, {"model_path", "bge-small.onnx"}}},
        {"secondary", {{"type", "retry"},
                       {"max_attempts", 2},
                       {"inner", {{"type", "local_ngram"}, {"dim", 384}, {"ngram", 3}}}}}};

    rag::Engine engine;
    if (auto r = engine.with_embedder_spec(spec); !r) {
        std::printf("with_embedder_spec failed: %s\n", r.error().message.c_str());
        return 1;
    }
    std::printf("\nWired embedder from config (onnx || retry(local_ngram)).\n");

    engine.add("d1", "The Eiffel Tower is a landmark in Paris.");
    engine.add("d2", "Photosynthesis converts light into chemical energy in leaves.");
    engine.add("d3", "HNSW is a graph index for approximate nearest neighbours.");
    engine.build();

    // Character n-grams give sub-word recall: "landmarks near Paris" matches d1
    // even though the exact phrase never appears.
    for (const char* q : {"landmarks near Paris", "how do plants make energy", "vector search index"}) {
        auto hits = engine.search(q, 1);
        std::printf("  q=%-30s -> %s\n", q,
                    (hits && !hits->empty()) ? (*hits)[0].uri.c_str() : "(none)");
    }
    return 0;
}
