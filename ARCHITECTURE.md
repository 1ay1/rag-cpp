# Architecture

`rag-cpp` is layered so that each concern is a swappable seam. The type system
enforces the boundaries: nominal strong types for identity, structural concepts
for extension points, and `Result<T>` for totality.

## The dependency graph

```
                         ┌───────────┐
                         │  Engine   │   one-call facade: add · build · search
                         └─────┬─────┘
                   ┌───────────┴───────────┐
                   ▼                       ▼
             ┌──────────┐           ┌─────────────┐
             │ Pipeline │           │   Corpus    │  ground truth + indexes
             └────┬─────┘           └──────┬──────┘
       ┌──────────┼──────────┐       ┌─────┼───────┬─────────┐
       ▼          ▼          ▼       ▼     ▼       ▼         ▼
   Retrieve   Rerank    Expand/    BM25  HNSW   Embedder   store
   (hybrid)  (crossenc) Stitch   (lexical)(ANN) (dense)  (.ragdb)
       │          │                       │        │
       └── fusion (RRF/RSF) ──────────────┘        ▼
                                              HttpTransport ── network seam
```

## Layers

### core — the type foundation
`StrongId<Tag>` (phantom-typed `DocId`/`ChunkId`/`TermId`), `Score`/`Similarity`
newtypes, `Result<T> = expected<T, Error>` with a closed `Errc` sum type, and
the algebraic records `Document`/`Chunk`/`SearchResult`. The concepts
`Embedder`, `Retriever`, `Ranker`, `Tokenizer` define the extension points
structurally — implement the shape, get accepted by the templates.

### text — ingestion
Tokenizer (lowercase, stopwords, Porter stemmer), a prose chunker (semantic
line-aligned with heading breadcrumbs for contextual retrieval), and — in
`loaders/` — a code-aware chunker that splits on definition boundaries.

### lexical — BM25
Okapi BM25 over an inverted index with smoothed idf. Serializes to a versioned
blob. Always available (no dependencies), so retrieval never hard-fails.

### dense — embeddings
The `Embedder` concept + the injectable `HttpTransport` seam (the library's only
outbound I/O). Backends: Ollama, OpenAI-compatible (+ Together/TEI presets),
llama.cpp, and a deterministic local Hash embedder. Decorators `RetryingEmbedder`
(exponential backoff) and `FallbackEmbedder` (primary → secondary) compose over
any embedder. SIMD kernels (AVX2 / NEON / scalar) for dot, cosine, sign-packing,
Hamming.

### index — HNSW + Corpus
HNSW ANN (Malkov & Yashunin) with Matryoshka truncation and binary quantization,
plus **filtered search** — a metadata predicate pushed into the graph walk
(pre-filter) so a selective filter still returns k results. `Corpus` owns the
chunks + both indexes, does incremental ingest, and persists via `store/`.

### fusion — combine retrievers
Reciprocal Rank Fusion (weighted) and Relative Score Fusion.

### rerank — the accuracy ceiling
Cross-encoder reranking over HTTP (TEI `/rerank` and Cohere/Jina `/v1/rerank`
wire formats) and a local `ScoreFnReranker` for in-process models. Adapts into a
pipeline stage via `make_rerank_stage()`, blending cross-encoder score with the
fused score.

### pipeline — the funnel
Composable `RetrievalStage`s: `PrfExpand` (RM3-lite query expansion) →
`HybridRetrieve` (BM25 + dense + fusion) → `Filter` → cross-encoder rerank →
`ParentStitch` (small-to-big) → `TopK`. Every stage has the same interface, so
they compose in any order. Stages are runtime-polymorphic; the hot scoring loops
they call stay non-virtual inside `Corpus`.

### store — persistence
The stable, versioned, CRC-checked `.ragdb` container (see `FORMAT.md`).

### c — the ABI
A flat opaque-handle C API (`rag/c/rag.h`) so Python/Rust/Go/any language can
drive the engine. Errors cross as status codes, never exceptions.

## Design invariants

1. **The network is optional and injected.** `HttpTransport` is the single
   outbound seam; everything degrades to BM25 when it's unavailable.
2. **Generic hot path, dynamic cold path.** Concepts constrain the templated
   scoring code (no vtables per token); type-erased `AnyEmbedder`/`AnyReranker`/
   `RetrievalStage` carry the runtime-assembled pipeline.
3. **Totality.** No exceptions for expected failure. Every fallible call returns
   `Result<T>` and composes via `and_then`/`transform`.
4. **Everything serializes.** BM25, HNSW, and the whole corpus round-trip to a
   documented on-disk contract; reopening never rebuilds.
