// examples/plugin_backend/host_demo.cpp
//
// The HOST side: load the plugin shared library at runtime, then build its
// embedder purely by name — the host never #includes the plugin's header and
// was compiled with no knowledge that "reverse_hash" exists.

#include <rag/rag.hpp>

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    // Path to the compiled plugin (defaults to the CMake output location).
    std::string plugin_path =
        argc > 1 ? argv[1] : "./libreverse_hash_plugin.so";

    rag::plugin::ensure_builtins_registered();

    std::printf("embedders before load: ");
    for (const auto& n : rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().names())
        std::printf("%s ", n.c_str());
    std::printf("\n");

    auto loaded = rag::plugin::load_plugin(plugin_path);
    if (!loaded) {
        std::printf("load failed: %s\n", loaded.error().message.c_str());
        std::printf("(pass the path to the built plugin as argv[1])\n");
        return 1;
    }
    loaded->keep(); // keep the library resident for the process lifetime

    std::printf("embedders after load:  ");
    for (const auto& n : rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().names())
        std::printf("%s ", n.c_str());
    std::printf("\n");

    // Build the plugin-provided embedder BY NAME and drive the engine with it.
    rag::Engine engine;
    if (auto e = engine.with_embedder_spec(
            nlohmann::json{{"type", "reverse_hash"}, {"dim", 384}});
        !e) {
        std::printf("with_embedder_spec failed: %s\n", e.error().message.c_str());
        return 1;
    }

    engine.add("d1", "The Eiffel Tower is in Paris.");
    engine.add("d2", "Photosynthesis happens in plant leaves.");
    engine.build();

    auto hits = engine.search("landmark in Paris", 2);
    if (hits) {
        std::printf("search ok, %zu results, top = %s\n",
                    hits->size(), hits->empty() ? "(none)" : (*hits)[0].uri.c_str());
    }
    return 0;
}
