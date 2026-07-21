# rag-cpp

**A type-theoretic, production-grade retrieval (RAG) engine in modern C++23.**

One static library. No vector-DB dependency. Hybrid lexical + dense retrieval,
HNSW ANN, reciprocal rank fusion, a composable stage pipeline, graceful
degradation, and on-disk persistence — built on a foundation that makes illegal
states unrepresentable.

```cpp
#include <rag/rag.hpp>

rag::Engine engine;
engine.with_embedder(rag::dense::AnyEmbedder{
    rag::dense::OllamaEmbedder{{.model = "nomic-embed-text"}}});  // or HashEmbedder (no network)

engine.add("intro.md", "rag-cpp fuses BM25 with dense vector search...");
engine.build();

auto results = engine.search("how does hybrid retrieval work?", 5);
for (auto& r : *results)
    std::printf("[%.3f] %s\n", r.score.get(), r.uri.c_str());
```

## Why "type-theoretic"?

The library treats the type system as a proof assistant:

- **Strong types (newtype pattern).** `DocId`, `ChunkId`, `TermId` are phantom-
  tagged `StrongId`s — nominally distinct even though all wrap `uint32`. You
  cannot pass a `DocId` where a `ChunkId` is wanted. `Score` and `Similarity`
  are distinct float newtypes.
- **Total functions.** Every fallible operation returns `Result<T> =
  std::expected<T, Error>`; no exceptions for control flow. Errors are a closed
  sum type (`Errc`).
- **Structural interfaces via concepts.** `Embedder`, `Retriever`, `Ranker`,
  `Tokenizer` are C++20 concepts — plug in a new backend by matching a shape,
  no inheritance, no vtable on the hot path. Runtime polymorphism (for
  config-assembled pipelines) is provided separately via type-erased `AnyX`
  adapters, so the generic and dynamic paths stay cleanly separated.
- **Algebraic domain model.** Product types (`Document`, `Chunk`) and sum types
  (`Errc`, fusion strategy) describe the domain precisely.

## What's inside

| Layer | Component |
|---|---|
| **core** | strong types, `Result`, concepts, `Document`/`Chunk` records |
| **text** | Unicode-lite tokenizer, full Porter stemmer, stopwords, semantic line-aligned chunker with heading-breadcrumb *contextual retrieval* |
| **lexical** | Okapi **BM25** over an inverted index (smoothed idf, binary serialization) |
| **dense** | `Embedder` concept + injectable `HttpTransport` seam; **Ollama** / **OpenAI**-compatible / deterministic **Hash** embedders; SIMD (AVX2 / NEON / scalar) dot, cosine, sign-packing, Hamming |
| **index** | **HNSW** ANN (Malkov & Yashunin 2016) with **Matryoshka truncation** + **binary quantization**; `Corpus` (hybrid store, incremental ingest, persistence) |
| **fusion** | **Reciprocal Rank Fusion** (weighted) + **Relative Score Fusion** |
| **pipeline** | composable `RetrievalStage` funnel: hybrid retrieve → filter → feature rerank → top-k; deterministic lexical-coverage reranker |
| **engine** | one-call facade: `add` → `build` → `search` |

## Design principles

1. **No hot-path dependency on the network.** The only external I/O is the
   embedding call, behind an injectable `HttpTransport`. Ships a socket-based
   default; inject your own (TLS stack, gRPC, mock) for anything else.
2. **Graceful degradation.** If the embedder is offline, hybrid search falls
   back to pure BM25 — the query still returns results.
3. **Everything is serializable.** BM25 and HNSW have versioned binary formats;
   a `Corpus` round-trips to a single file. Re-opening never rebuilds.
4. **Deterministic by default.** The `HashEmbedder` needs no network, so tests,
   CI, and offline smoke runs are fully reproducible.

## Build

Requires a C++23 compiler (GCC 13+, Clang 17+) and CMake 3.24+.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # 14 test cases
./build/examples/ragcpp_quickstart
./build/bench/ragcpp_bench 5000 # ablation + latency harness
```

The only dependency (nlohmann/json) is fetched automatically via
`FetchContent`.

## Performance (5000 chunks, Apple M-series, NEON)

```
embed+build:   3733 ms   (one-time)
hybrid query:  2.1 ms/query
bm25 only:     0.18 ms/query
dense (HNSW):  0.14 ms/query
```

## Roadmap

- Cross-encoder reranker stage (bge-reranker via the transport seam)
- Query expansion (PRF/RM3) and parent-document stitching stages
- GraphRAG-lite (link graph + community summaries)
- Learned per-passage priors from click/use feedback
- PDF / HTML source loaders
- C API + Python bindings

## License

MIT.
