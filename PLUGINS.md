# Extending rag-cpp — the three axes

rag-cpp is designed to be extensible **to everything** without forking the core.
There are three composable extension axes, from cheapest/most-static to most
dynamic. Pick the lowest one that fits.

## 1. Compile-time: model a concept

Every algorithmic role in the framework is a C++20 **concept**, not a base
class. Write a struct whose shape matches and the generic templates accept it
for free — no inheritance, no vtable on the hot path.

```cpp
struct MyEmbedder {                      // models rag::Embedder
    std::size_t dimension() const noexcept { return 384; }
    std::string_view identity() const noexcept { return "my-embed-v1"; }
    rag::Result<std::vector<rag::Vector>>
    embed(std::span<const std::string> texts) const { /* ... */ }
};
static_assert(rag::Embedder<MyEmbedder>);
```

The concepts (see `rag/core/concepts.hpp`): `Embedder`, `Retriever`, `Ranker`,
`Reranker`, `Tokenizer`. Anything satisfying one is a drop-in.

## 2. Link-time: type-erase into an `AnyX`

For a runtime pipeline of heterogeneous stages, wrap your concept model in the
matching type-eraser — a copyable shared handle:

```cpp
rag::dense::AnyEmbedder e{MyEmbedder{}};   // erases the concrete type
engine.with_embedder(std::move(e));
```

`AnyEmbedder`, `AnyReranker` exist today; the pattern is identical for the rest.

## 3. Load-time: register by name (the plugin registry)

This is what makes rag-cpp extensible **to everything from config or a shared
library**. A backend registers a factory under a string name; anything
downstream (a config file, the CLI, the C ABI, a REST server) builds it by name
from a JSON blob — with **zero compile-time knowledge** of the backend.

### Register a factory

```cpp
#include <rag/plugin/plugin.hpp>

RAG_REGISTER(rag::plugin::AnyEmbedder, "my_embed",
    [](const nlohmann::json& cfg) -> rag::Result<rag::plugin::AnyEmbedder> {
        auto dim = cfg.value("dim", 384);
        return rag::plugin::AnyEmbedder{MyEmbedder{static_cast<std::size_t>(dim)}};
    });
```

### Build by name

```cpp
rag::plugin::ensure_builtins_registered();          // hash/ollama/openai/llamacpp
auto emb = rag::plugin::make_embedder(              // or Registry<...>::create_from
    nlohmann::json{{"type","my_embed"},{"dim",512}});
engine.with_embedder(std::move(*emb));

// or, straight from config, on the Engine:
engine.with_embedder_spec(config["embedder"]);       // {"type": "ollama", ...}
```

Built-in embedder names: `hash`, `ollama`, `openai`, `llamacpp`.
Built-in reranker names: `cross_encoder` (with `wire` = `tei` | `cohere` | `jina`).

### Ship a plugin as a shared library (no recompile of the app)

Compile a `.so`/`.dylib`/`.dll` containing your `RAG_REGISTER` blocks (and
optionally an `extern "C" void rag_plugin_register()` hook), then at runtime:

```cpp
auto p = rag::plugin::load_plugin("./rag_plugins/libmy_embed.so");   // one file
auto ps = rag::plugin::load_plugin_dir("./rag_plugins");             // whole dir
// ...names are now in every registry. Keep the handles alive.
```

Build your plugin with the **same compiler and rag headers** as the host so both
see the same registry singletons. See `examples/plugin_backend/` for a complete,
buildable example.

## Why one registry serves everything

`Registry<Interface>` is a single template keyed on the interface type. The same
mechanism registers embedders, rerankers, retrievers, tokenizers, generators,
summarizers — any `AnyX`. Adding a *new kind* of extension point is one line:
`using MyRegistry = rag::plugin::Registry<AnyMyThing>;`. There is no per-kind
boilerplate to maintain.
