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

### Register a backend — one function, no boilerplate

The ergonomic way (`rag/plugin/builder.hpp`): return your **concept type** from a
`Config -> Result<T>` lambda. The `AnyEmbedder` wrap, the non-throwing config
parsing, and the `describe()` help text are all handled. No macro, so no
comma-in-braces trap.

```cpp
#include <rag/plugin/plugin.hpp>   // pulls in builder.hpp

rag::plugin::register_embedder(
    "my_embed",                                         // name used in config
    "my custom embedder (keys: dim)",                   // shown by `ragcpp list`
    [](rag::plugin::Config c) -> rag::Result<MyEmbedder> {
        return MyEmbedder{c.get<std::size_t>("dim", 384)};
    });
```

`Config` is a total, non-throwing view over the spec:

| Call | Behaviour |
|------|-----------|
| `c.get("model", "bge")` | value or default; wrong type falls back to default |
| `c.require<std::string>("api_key")` | `Result<T>` — typed error if absent; just propagate it |
| `c.has("dim")` | presence test |
| `c.sub("primary")` | a nested component spec (feed to `resolve<>()`) |

Building a nested sub-component (what decorators do) is one call:
`rag::plugin::resolve<AnyEmbedder>(spec)`.

The same shape works for rerankers via `register_reranker(...)`.

<details><summary>The raw macro (escape hatch)</summary>

If you want to bypass the helper and hand-roll the factory (returning `AnyEmbedder`
directly, no description), the `RAG_REGISTER` macro is still there. Note commas
inside `{}`-brace-init confuse the preprocessor — construct via a named local.

```cpp
RAG_REGISTER(rag::plugin::AnyEmbedder, "my_embed",
    [](const nlohmann::json& cfg) -> rag::Result<rag::plugin::AnyEmbedder> {
        return rag::plugin::AnyEmbedder{MyEmbedder{cfg.value("dim", 384)}};
    });
```
</details>

### See everything that's registered

`describe()` returns `(name, help)` for every backend, which the CLI surfaces:

```sh
ragcpp list                       # all embedders + rerankers, with config keys
ragcpp list embedders
ragcpp list embedders --plugins=./plugins   # include third-party .so backends
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

Built-in embedder names: `hash`, `ollama`, `openai`, `voyage`, `together`,
`llamacpp`, and the in-process **local** models `onnx` and `gguf` (see below).
Composition decorators: `retry`, `fallback`. Polyglot transports: `process`,
`http`, `rest`. Built-in reranker names: `cross_encoder` (with `wire` = `tei` |
`cohere` | `jina`). Run `ragcpp list` to see them all with their config keys.

Adding a hosted provider that speaks the OpenAI `/v1/embeddings` shape (Voyage,
Together, …) is ~6 lines over the `openai` backend — see `voyage` / `together`
in `src/plugin/builtins.cpp` for the pattern.

### Local (in-process) embedders are just names too

The ONNX Runtime and GGUF/llama.cpp embedders run the model **inside your
process** — no server, no HTTP hop. They are reachable by name exactly like the
network backends:

```cpp
engine.with_embedder_spec({
    {"type", "onnx"},                       // or "gguf"
    {"model_path", "bge-small-en.onnx"},
    {"tokenizer_path", "tokenizer.json"},   // ONNX only
    {"pooling", "mean"}, {"normalize", true}, {"max_tokens", 512}});
```

These require the corresponding build flag (`-DRAGCPP_WITH_ONNX=ON` /
`-DRAGCPP_WITH_LLAMA=ON`). On a build **without** the flag the name still
resolves, but the factory returns a clear `Errc::unavailable` ("built without
ONNX support") instead of "unknown type" — the config surface is stable across
build configurations, so the same config file works everywhere and tells you
precisely what to flip.

### Compose resilience from config

Two decorators take **nested** embedder specs and resolve them through the same
registry, so a config file can express retry-then-degrade declaratively — no
code:

```jsonc
{ "type": "fallback",
  "primary":   { "type": "onnx", "model_path": "bge-small-en.onnx" },
  "secondary": { "type": "retry", "max_attempts": 3,
                 "inner": { "type": "ollama", "model": "nomic-embed-text" } } }
```

- **`retry`** — wraps `inner` with bounded exponential backoff on transient
  failures (`max_attempts`, `base_delay_ms`).
- **`fallback`** — tries `primary`, then `secondary`. If the primary cannot even
  be *constructed* (missing feature flag, missing key), it degrades to the
  secondary at construction time — so a config naming a local model this build
  can't load still yields a working embedder. A primary that constructs but
  fails per-request degrades at runtime.

Composition **nests arbitrarily** (`retry(fallback(a, b))`), because the registry
invokes factories *unlocked* — building a composite re-enters the same singleton
several times on one thread, which a naive lock-around-factory design would
deadlock.

### Ship a plugin as a shared library (no recompile of the app)

Compile a `.so`/`.dylib`/`.dll` containing your `RAG_REGISTER` blocks (and
optionally an `extern "C" void rag_plugin_register()` hook), then at runtime:

```cpp
auto p = rag::plugin::load_plugin("./rag_plugins/libmy_embed.so");   // one file
auto ps = rag::plugin::load_plugin_dir("./rag_plugins");             // whole dir
// ...names are now in every registry. Keep the handles alive.
```

Build your plugin with the **same compiler and rag headers** as the host so both
see the same registry singletons. `examples/plugin_backend/` has two complete,
buildable examples: `my_embedder_plugin.cpp` (a minimal toy) and
`local_embedder_plugin.cpp` (a real in-process character-n-gram LOCAL embedder,
no external deps), each with a host demo that loads the `.so` and builds the
backend by name.

## Polyglot backends — engines/retrievers/graphs in ANY language

A backend does not have to be C++, and does not even have to live in your
process. The `rag/bridge/` module lets a component written in **Python, Node,
Rust, Go — anything** — plug in over a subprocess pipe or an HTTP endpoint. It
models the same concepts as a native backend, so a remote Python engine drops
into the Corpus / Pipeline / Engine with no special casing.

### The wire protocol (tiny on purpose)

One compact JSON object per request, one per reply:

```
--> {"method":"embed",   "params":{"texts":["..."]}}
<-- {"ok":true,"result":{"vectors":[[...],[...]]}}

--> {"method":"retrieve","params":{"query":"...","k":5}}
<-- {"ok":true,"result":{"hits":[{"id":"d1","score":0.9,"text":"..."}]}}
```

Methods: `embed`, `rerank`, `retrieve`, `graph` (an escape hatch for GraphRAG
local/global ops). Errors come back as `{"error":{"message":"..."}}`.

### Two transports

- **subprocess** (`rag::bridge::ProcessChannel`): the host spawns your program
  (`python3 my_server.py`) and speaks newline-delimited JSON over stdin/stdout.
  Zero dependencies; the universal bridge.
- **HTTP/REST** (`rag::bridge::HttpChannel`): POST the same envelope to
  `<base>/<method>` using the library's injectable `HttpTransport`.

### From config (registered as `process` / `http`)

```cpp
engine.with_embedder_spec({
    {"type", "process"},                       // or "http"
    {"argv", {"python3", "ragcpp_server.py"}},  // or {host,port,base_path,...}
    {"dim", 384},
});
```

### As a self-contained remote engine

When the far side owns its OWN index (a Python FAISS store, Elasticsearch, a
graph engine), use `RemoteRetriever` — it answers full queries and exposes
`op("local"/"global", ...)` for graph operations:

```cpp
auto ch = rag::bridge::open_channel({{"transport","process"},
                                     {"argv",{"python3","ragcpp_server.py"}}});
rag::bridge::RemoteRetriever engine{*ch, "python:engine"};
auto hits = engine.retrieve("great wall", 5);
auto summary = engine.op("global");
```

A complete, runnable reference server is in `examples/polyglot/ragcpp_server.py`
(pure stdlib — swap the toy bodies for sentence-transformers / FAISS / networkx /
an LLM), driven by `examples/polyglot/python_backend.cpp`.

## Why one registry serves everything

`Registry<Interface>` is a single template keyed on the interface type. The same
mechanism registers embedders, rerankers, retrievers, tokenizers, generators,
summarizers — any `AnyX`. Adding a *new kind* of extension point is one line:
`using MyRegistry = rag::plugin::Registry<AnyMyThing>;`. There is no per-kind
boilerplate to maintain.
