# rag-cpp

**A type-theoretic, production-grade retrieval (RAG) engine in modern C++23.**

One library. No vector-DB dependency. Hybrid lexical + dense retrieval, HNSW
ANN with filtered search, cross-encoder reranking, pluggable embedders, a
composable stage pipeline, source loaders (files / HTML / PDF / code), a stable
versioned on-disk format, and a C ABI for Python/Rust/Go — on a foundation that
makes illegal states unrepresentable.

```cpp
#include <rag/rag.hpp>

rag::Engine engine;
engine.with_embedder(rag::dense::AnyEmbedder{
    rag::dense::OllamaEmbedder{{.model = "nomic-embed-text"}}});   // or Hash / OpenAI / llama.cpp

engine.add("intro.md", "rag-cpp fuses BM25 with dense vector search...");
engine.build();

for (auto& r : *engine.search("how does hybrid retrieval work?", 5))
    std::printf("[%.3f] %s\n", r.score.get(), r.uri.c_str());
```

## Why "type-theoretic"?

The library treats the type system as a proof assistant:

- **Strong / phantom types.** `DocId`, `ChunkId`, `TermId` are nominally
  distinct even though all wrap `uint32` — you cannot pass one where another is
  wanted. `Score` and `Similarity` are distinct float newtypes.
- **Total functions.** Every fallible op returns `Result<T> = std::expected<T,
  Error>`; no exceptions for control flow. Errors are a closed `Errc` sum type.
- **Structural interfaces via concepts.** `Embedder`, `Retriever`, `Ranker`,
  `Reranker`, `Tokenizer` are concepts — plug in a backend by matching a shape,
  zero vtables on the hot path. Runtime polymorphism (config-assembled
  pipelines) rides separate type-erased `AnyX` adapters.

## Features

### Retrieval
- **Hybrid** BM25 (Okapi, smoothed idf) + dense cosine, fused with **RRF**
  (weighted) or **Relative Score Fusion**.
- **HNSW** ANN (Malkov & Yashunin) with **Matryoshka truncation** + **binary
  quantization**, and **filtered-HNSW**: a metadata predicate pushed *into* the
  graph walk (pre-filter) so selective filters still return k results.
- **Cross-encoder reranking** as a first-class stage — TEI `/rerank`,
  Cohere/Jina `/v1/rerank`, or any in-process scorer.
- **Query expansion** (RM3-lite PRF) and **parent-document stitching**
  (small-to-big) pipeline stages.

### Embedders (pluggable)
`Ollama` · `OpenAI`-compatible (+ **Together**, **TEI** presets) · `llama.cpp`
server · deterministic local `Hash` (no network). Decorators: `RetryingEmbedder`
(exponential backoff), `FallbackEmbedder` (primary → secondary). All behind an
injectable **`HttpTransport`** seam — bring your own TLS/gRPC/mock.

### Source loaders
Filesystem directory walker (include/exclude globs) · HTML → text · PDF (via
`pdftotext`) · **code-aware chunker** that splits on function/class boundaries
for C-like, Python, Ruby, Go, and Rust.

### Persistence
A **stable, versioned, CRC-checked `.ragdb` container** (documented in
[`FORMAT.md`](FORMAT.md)) — a public contract, not an internal cache. Reopening
never rebuilds.

### C API / bindings
A flat opaque-handle C ABI ([`include/rag/c/rag.h`](include/rag/c/rag.h)) drives
the whole engine from any language. See [`examples/ragcpp.py`](examples/ragcpp.py)
for ctypes bindings; Rust `bindgen` / Go `cgo` work the same way.

## Build

Requires a C++23 compiler (GCC 13+, Clang 17+) and CMake 3.24+.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                     # C++ + C-API suites
./build/examples/ragcpp_full_pipeline      # the maximal funnel
./build/bench/ragcpp_bench 5000            # ablation + latency
```

For the shared library (Python/Rust FFI): `-DBUILD_SHARED_LIBS=ON`. The only
dependency (nlohmann/json) is fetched via `FetchContent`.

## Performance (5000 chunks, Apple M-series, NEON)

```
embed+build:   ~3.7 s    (one-time)
hybrid query:  ~2.1 ms/query
bm25 only:     ~0.18 ms/query
dense (HNSW):  ~0.14 ms/query
```

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layers, seams, invariants.
- [`FORMAT.md`](FORMAT.md) — the `.ragdb` on-disk contract.

## License

MIT.
