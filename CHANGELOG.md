# Changelog

All notable changes to **rag-cpp** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2025-07-26

The first tagged release. rag-cpp is a type-theoretic, production-grade retrieval
(RAG) engine in modern C++23: one library, no vector-DB dependency, with a
conformant RCP/1 protocol server and a C ABI. CI is green on macOS (clang, with
and without Metal) and Linux (gcc); the RCP server is certified **conformance
level L2**.

> Versioned `0.1.0` — the library is stable and measured, but the C++ API is not
> yet frozen at 1.0. The **C ABI** (`rag/c/rag.h`) is already versioned at 1.0
> and evolves compatibly. The on-disk `.ragdb` format is versioned and
> CRC-checked (see `FORMAT.md`).

### Retrieval core

- **Hybrid retrieval** — BM25 (inverted index) and dense (vector) retrievers run
  concurrently so hybrid latency is `max`, not the sum.
- **Score fusion** — `convex` (default), `rrf`, and `rsf`; the convex default was
  measured at +18% nDCG@10 over RRF on the eval harness.
- **HNSW ANN** with filtered search, tunable `ef` as a per-query dial, and
  measured recall/QPS curves on real datasets (GloVe, SIFT).
- **Quantization** — SQ8-quantized graph walk with full-precision rescoring
  (3.2× smaller at 0.93 recall) and optional PQ for very large corpora.
- **Metadata filtering** evaluated during the walk, not as a post-filter.

### Advanced retrieval

- **SPLADE** (learned sparse), **ColBERT** (late-interaction MaxSim),
  **RAPTOR** (hierarchical summary tree), **HyDE** and multi-query rewriting,
  **Corrective RAG (CRAG)**, a budgeted **cascade**, and **GraphRAG**
  (`graph_local` / `graph_global`).
- **Contextual Retrieval** (Anthropic 2024) as an ingest transform, reachable via
  `CorpusConfig::contextual` and `ragcpp index --contextual`.

### Pipeline

- Composable `RetrievalStage`s with three pre-assembled factories:
  `Pipeline::standard()` (the default), `Pipeline::quality()` (+MMR diversity),
  and `Pipeline::context()` (+ParentStitch small-to-big). The two extras are
  opt-in and carry measured justifications.

### Embedders & plugins

- Backends: `hash`, `ollama`, `openai`, `voyage`, `together`, `llamacpp`, plus
  in-process `onnx` and `gguf`.
- **Composition from config** — `retry` and `fallback` decorators resolve nested
  specs, so resilience is declarative and `fallback` degrades even when a primary
  can't be constructed.
- **Plugin registry** — register a backend by name; a one-function authoring API
  (`register_embedder` / `Config` / `resolve`) makes adding one trivial. Ship a
  backend as a shared library and load it at runtime with no host recompile.
- **Polyglot bridge** — drive an engine / retriever / graph written in any
  language over a process or HTTP channel.
- `ragcpp list` prints every registered backend with its config keys.

### Performance

Measured, not assumed: parallel HNSW build, allocation-free search, overlapped
retrievers, memoized Porter stemmer (3.1× faster ingest), precomputed BM25
postings, and a **Metal GPU** batch-scoring backend (6–9× on large batches where
it wins, never slower otherwise).

### Durability & serving

- **Write-ahead log** — an acknowledged write is `O(record)`, not `O(corpus)`
  (25 ms → 1.35 ms at 20k docs); opening the WAL performs crash recovery.
- Stable, versioned, CRC-checked **`.ragdb`** container; reopening never rebuilds.
- **RCP/1 server** (stdio + HTTP) consuming the pinned RCP C++ SDK; certified
  conformance **L2**. Crash-safe writes, per-request pipeline overrides, and
  exact citations (`startLine`/`endLine`) for grounded generation.

### C ABI & bindings

- Flat opaque-handle C API (`rag/c/rag.h`, version 1.0) drives the whole engine
  from Python/Rust/Go; errors cross as status codes, never exceptions.

### Tooling & quality

- **CI** across macOS (clang ± Metal), Linux (gcc), a `-Werror` job, and a
  pinned-SDK "clean clone" job that verifies the build a stranger gets.
- Comprehensive docs in [`docs/`](docs/), plus `ARCHITECTURE.md`, `PLUGINS.md`,
  and `FORMAT.md`.

[0.1.0]: https://github.com/1ay1/rag-cpp/releases/tag/v0.1.0
