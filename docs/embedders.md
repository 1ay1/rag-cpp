# Embedders

An **embedder** turns text into a dense vector. rag-cpp needs one only for the
*dense* half of hybrid retrieval — with none attached, retrieval is pure BM25 and
everything still works. When you want semantic search, attach a backend.

## The concept

Any type that satisfies the `Embedder` concept is an embedder — no inheritance:

```cpp
struct MyEmbedder {                                   // models rag::dense::Embedder
    std::size_t dimension() const noexcept;
    rag::Result<std::vector<rag::Vector>>
        embed(std::span<const std::string> texts) const;
};
```

To carry one around type-erased (as the Engine does), wrap it in `AnyEmbedder`.

## Built-in backends

| Backend | Type name (for spec) | Notes |
|---------|----------------------|-------|
| `HashEmbedder` | `"hash"` | Deterministic feature-hash vectors. **The default.** Zero setup, no network — makes hybrid work out of the box and is ideal for tests. |
| `OllamaEmbedder` | `"ollama"` | Local [Ollama](https://ollama.com) server (`nomic-embed-text`, etc.). |
| `OpenAIEmbedder` | `"openai"` | OpenAI (and any OpenAI-compatible) embeddings endpoint. |
| (OpenAI-compatible) | `"voyage"`, `"together"` | Voyage AI / Together AI, presets over the OpenAI shape. Adding another hosted provider is ~6 lines — see [`PLUGINS.md`](../PLUGINS.md). |
| `LlamaCppEmbedder` | `"llamacpp"` | A running `llama.cpp` server's `/embedding` endpoint. |
| `OnnxEmbedder` | `"onnx"` | ONNX Runtime, **in-process**. Needs `-DRAGCPP_WITH_ONNX=ON`. |
| `GgufEmbedder` | `"gguf"` | GGUF via llama.cpp, **in-process**. Needs `-DRAGCPP_WITH_LLAMA=ON`. |

Network backends (Ollama / OpenAI / llama.cpp) all take an **injected
`HttpTransport`** and degrade gracefully when the endpoint is unavailable.

## Attaching one

Directly:

```cpp
engine.with_embedder(rag::dense::AnyEmbedder{
    rag::dense::OllamaEmbedder{{.model = "nomic-embed-text"}}});
```

Or config-driven, by name, with zero backend knowledge at the call site — this
is what lets the whole engine be wired from a JSON config file, and it also
picks up any embedder registered by a loaded [plugin](../PLUGINS.md):

```cpp
engine.with_embedder_spec({{"type", "ollama"}, {"model", "nomic-embed-text"}});
engine.with_embedder_spec({{"type", "openai"}, {"model", "text-embedding-3-small"}});
engine.with_embedder_spec({{"type", "hash"},   {"dim", 256}});
```

## Resilience decorators

Two decorators wrap **any** `AnyEmbedder` (they model `Embedder` themselves, so
they compose):

- **`RetryingEmbedder`** — bounded exponential backoff on transient failures.

  ```cpp
  auto robust = rag::dense::AnyEmbedder{
      rag::dense::RetryingEmbedder{primary, /*max_attempts*/ 3}};
  ```

- **`FallbackEmbedder`** — a primary → secondary chain (e.g. hosted → local, so a
  provider outage silently downgrades instead of failing the query):

  ```cpp
  auto ha = rag::dense::AnyEmbedder{rag::dense::FallbackEmbedder{
      hosted,               // try this first
      local_fallback}};     // fall back to this
  ```

Compose them: `Fallback{ Retrying{hosted}, local }` gives you retry-then-degrade.

### Composing from config (no code)

The decorators are also **registered by name**, taking nested embedder specs, so
resilience is expressible in a config file:

```jsonc
{ "type": "fallback",
  "primary":   { "type": "onnx",  "model_path": "bge-small-en.onnx" },
  "secondary": { "type": "retry", "max_attempts": 3,
                 "inner": { "type": "ollama", "model": "nomic-embed-text" } } }
```

`fallback` even degrades when the **primary can't be constructed** (e.g. this
build lacks ONNX) — the same config then works on every build. Composition nests
arbitrarily. See [`PLUGINS.md`](../PLUGINS.md#compose-resilience-from-config).

## The `HttpTransport` seam

Every network backend takes a `std::shared_ptr<HttpTransport>`. The library ships
a default socket-based transport (`default_http_transport()`), but the interface
is injectable, which is why the network backends are fully testable **without a
network** — tests inject a fake transport that returns canned responses:

```cpp
struct HttpTransport {
    virtual ~HttpTransport() = default;
    virtual Result<HttpResponse> post(std::string_view url,
                                       std::string_view body,
                                       const Headers& headers) const = 0;
    // ...
};

rag::dense::OllamaEmbedder e{{.model = "x"}, my_transport};
```

## In-process backends

`OnnxEmbedder` and `GgufEmbedder` run the model **in your process** — no server,
no HTTP hop — behind the same `Embedder` concept, and are selectable by name
(`"onnx"` / `"gguf"`) like any other backend. They are opt-in at build time
because they pull in a heavy dependency:

```sh
cmake -B build -DRAGCPP_WITH_ONNX=ON     # or -DRAGCPP_WITH_LLAMA=ON
```

Without the flag the code isn't compiled in, and everything degrades to the
network / hash backends. See `LocalEmbedderConfig` in
`include/rag/dense/local_embedder.hpp` for the model-path / threads / tag knobs.
A build without the flag still **resolves** `"onnx"`/`"gguf"` by name but returns
a clear `unavailable` error — so a `fallback` naming a local primary degrades to
its secondary instead of breaking.

## Remote embedders over a bridge

The plugin bridge registers `"process"`, `"http"`, and `"rest"` embedder types
that drive an out-of-process backend (e.g. a Python model server) over a channel.
See [`PLUGINS.md`](../PLUGINS.md) and `examples/polyglot/` for the pattern.

## Batching

Embedding is batched — `CorpusConfig::embed_batch` (default `32`) controls how
many texts go to the backend per call at ingest. At query time, HyDE and
multi-query batch their hypotheticals into a single `embed()` call, and on Apple
hardware that batch can be scored on the [GPU](gpu.md).
