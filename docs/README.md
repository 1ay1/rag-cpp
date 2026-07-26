# rag-cpp documentation

Task- and subsystem-oriented guides for [rag-cpp](../README.md) — a
type-theoretic, production-grade retrieval (RAG) engine in modern C++23.

New here? Read [**Getting Started**](getting-started.md) first, then reach for
the guide that matches what you're building.

## Guides

| Guide | What it covers |
|-------|----------------|
| [Getting Started](getting-started.md) | Install, build, first index, first query — both the CLI and the C++ library. |
| [Retrieval](retrieval.md) | Hybrid BM25 + dense, HNSW ANN, score fusion, metadata filtering, quantization. The scoring core. |
| [Vector store](vector-store.md) | Using rag-cpp as a plain vector database: you own the ids and vectors, no text pipeline. |
| [The Pipeline](pipeline.md) | Composable retrieval stages, the `standard` / `quality` / `context` factories, and writing your own stage. |
| [Embedders](embedders.md) | Every embedding backend, the retry/fallback decorators, the `HttpTransport` seam, and the in-process ONNX / GGUF paths. |
| [Advanced Retrieval](advanced-retrieval.md) | SPLADE, ColBERT, RAPTOR, HyDE / multi-query, Corrective RAG, and GraphRAG. |
| [Configuration](configuration.md) | Every tunable: `CorpusConfig`, `HnswConfig`, chunking, contextual retrieval, thresholds. |
| [The CLI](cli.md) | Full `ragcpp` command reference — `index`, `query`, `eval`, `info`, `serve`. |
| [Serving over RCP](rcp-server.md) | Turning an `Engine` into a conformant RCP/1 server; capability advertisement and conformance. |
| [Persistence](persistence.md) | The `.ragdb` container, the write-ahead log, and the durability model. |
| [The C API & bindings](c-api.md) | The flat C ABI and how to drive the engine from Python, Rust, or Go. |
| [GPU acceleration](gpu.md) | The optional Metal batch-scoring backend: when it engages and what it buys. |

## Reference docs (repository root)

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — the layer diagram, seams, and design invariants.
- [`PLUGINS.md`](../PLUGINS.md) — the three extension axes (concepts, `AnyX`, the load-time registry).
- [`FORMAT.md`](../FORMAT.md) — the byte-level `.ragdb` on-disk contract.

## The shape of the library in one paragraph

You build an [`Engine`](getting-started.md#the-engine-facade), attach an optional
[embedder](embedders.md), `add()` documents, `build()` the indexes, and
`search()`. Under the facade sits a [`Corpus`](retrieval.md) (ground truth + BM25
+ HNSW + dense vectors) and a [`Pipeline`](pipeline.md) of composable stages.
Everything fallible returns `Result<T> = std::expected<T, Error>` — there are no
exceptions for expected failure. Every extension point is a **concept**, so you
plug in a backend by matching a shape, not by inheriting. The whole thing
persists to a single [versioned file](persistence.md) and can be
[served over RCP](rcp-server.md) with no glue code.
