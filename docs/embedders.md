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

### Running a real model in process

End to end, from nothing to hybrid retrieval with a real transformer. This is the
exact path the [hybrid BEIR numbers](../BENCHMARKS.md#hybrid-retrieval-with-a-neural-embedder)
were measured on.

**1. Install ONNX Runtime.**

```sh
brew install onnxruntime            # macOS
# Linux: apt-get install libonnxruntime-dev, or unpack a release tarball
```

**2. Get a model.** Any sentence-transformer with an exported ONNX graph works.
`all-MiniLM-L6-v2` is the reasonable default — 384-dim, 86 MB, fast on CPU:

```sh
mkdir -p minilm && cd minilm
B=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main
curl -L -o model.onnx     $B/onnx/model.onnx
curl -L -o tokenizer.json $B/tokenizer.json
```

> Use the `resolve/main/...` URL, not `blob/main/...`. The `blob` path returns an
> HTML page or a 15-byte `Entry not found` stub, and the failure surfaces much
> later as a tokenizer parse error.

**3. Build with the flag and attach it:**

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRAGCPP_WITH_ONNX=ON && cmake --build build -j
```

```cpp
#include "rag/dense/local_embedder.hpp"

rag::dense::LocalEmbedderConfig cfg;
cfg.model_path     = "minilm/model.onnx";
cfg.tokenizer_path = "minilm/tokenizer.json";
cfg.normalize      = true;    // cosine == dot product afterwards
cfg.max_tokens     = 256;     // MiniLM's window is 512; shorter is faster

auto emb = rag::dense::OnnxEmbedder::load(cfg);
if (!emb) { /* handle rag::Error — bad path, bad graph, no runtime */ }
corpus.set_embedder(rag::dense::AnyEmbedder{std::move(*emb)});
```

That is the whole integration. From here `corpus.dense_search` works, and
`Pipeline::standard()` fuses it with BM25 automatically — nothing else to enable.

**Cost, measured on an M1 (CPU, no GPU):** indexing 5183 SciFact documents takes
~185 s, i.e. ~28 docs/s single-model. Indexing is the expensive part and happens
once; queries embed a single short string and are sub-millisecond against the
resulting index. Raise `LocalEmbedderConfig::threads` and
`CorpusConfig::embed_batch` if the ingest wall-clock matters to you.

**Choosing a model.** Anything in the sentence-transformers / BGE / E5 family
with an ONNX export drops into the same three lines; only `model_path` changes.
MiniLM-L6 is the floor, not the ceiling — it is the model the benchmarks use
precisely because winning with the *small* one is the stronger claim. Going
bigger is measured, not hand-waved: **`bge-base-en-v1.5`** (dim 768) in the same
slot lifts SciFact nDCG@10 0.7347→0.7563 and ArguAna 0.3848→0.4455, and —
usefully — makes pure `dense` overtake hybrid, because a strong embedder no
longer needs BM25 fused in. The comparison and its caveats are in
[BENCHMARKS.md](../BENCHMARKS.md#scaling-the-embedder-bge-base-vs-minilm-l6).

## Remote embedders over a bridge

The plugin bridge registers `"process"`, `"http"`, and `"rest"` embedder types
that drive an out-of-process backend (e.g. a Python model server) over a channel.
See [`PLUGINS.md`](../PLUGINS.md) and `examples/polyglot/` for the pattern.

## Batching

Embedding is batched — `CorpusConfig::embed_batch` (default `32`) controls how
many texts go to the backend per call at ingest. At query time, HyDE and
multi-query batch their hypotheticals into a single `embed()` call, and on Apple
hardware that batch can be scored on the [GPU](gpu.md).
