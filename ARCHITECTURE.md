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

### graph — GraphRAG
An explicit **document graph** (nodes = documents) with two edge kinds — **link
edges** (markdown `[..](..)`, bare URLs, `[[wikilinks]]` mined from the text and
resolved against document URIs/titles) and **similarity edges** (dense centroid
cosine when embedded, else lexical Jaccard; k-NN sparsified). Communities are
found by deterministic **synchronous label propagation** (Raghavan 2007) and
summarized **extractively** by default (highest-centrality sentences — free, no
model) with an optional abstractive `Summarizer` seam. Two retrieval entries:
**local** (hybrid seed → Personalized-PageRank expansion over the graph →
chunks re-scored by their document's restart-biased centrality) and **global**
(rank community summaries against the query — GraphRAG's map-reduce over
community reports, reduced to its retrieval core).

### ralm — retrieval-augmented LM assembly
The retrieval-side machinery of four landmark recipes, generator-agnostic (the
LM call is an injected seam): **RAG/REPLUG** ensemble weights (temperature-scaled
softmax `p(z|x)` over retrieval scores) + a `replug_combine` distribution mixer;
**RETRO** chunked-neighbour retrieval (per-stride neighbours with their
*continuation* — the following chunk); **In-Context RALM** stride schedule with a
rerank hook; and grounded, source-attributed prompt assembly.

### sparse — learned-sparse (SPLADE-style)
An inverted-index retriever with **saturated impact weighting** (`log(1+tf)·idf`)
and **query-time term EXPANSION** from a corpus co-occurrence graph — SPLADE's
two ideas (learned weights + expansion) approximated model-free, or driven by a
real SPLADE model through the encoder seam. BM25 speed, dense-like recall.

### late — ColBERT late interaction
Token-level **MaxSim** scoring: every query token softly matches its best
document token and the matches sum. Precomputable document token embeddings make
it a fast reranker between bi-encoders and cross-encoders. Ships a deterministic
hashed-n-gram token embedder; a real ColBERT checkpoint plugs into the seam.

### raptor — recursive tree retrieval
**RAPTOR** (Sarthi 2024): recursively cluster → summarize chunks bottom-up into a
tree of increasing abstraction, then retrieve over the **collapsed tree** (all
levels pooled) so a query matches a fine leaf or a high-level synopsis. Dense or
lexical agglomerative clustering; extractive summaries with an abstractive seam.

### query — HyDE + multi-query
**HyDE** (Gao 2022): embed an LLM-hallucinated *hypothetical* answer document to
close the query-document distribution gap. **Multi-query / RAG-Fusion**: retrieve
for several paraphrases and RRF-fuse. Both degrade to plain dense search with no
generator.

### crag — Corrective RAG + Self-RAG
A **retrieval evaluator** grades the retrieved set's confidence and triggers
CRAG's three actions (correct / ambiguous / incorrect) with decompose-recompose
knowledge strips and an external-source fallback seam. Self-RAG **reflection**:
per-passage relevance filtering and an answer **groundedness** (`IsSupported`)
score. Model-free graders by default; learned graders via seams.

### dense/local_embedder — in-process ONNX + GGUF
**OnnxEmbedder** (ONNX Runtime + WordPiece) and **GgufEmbedder** (llama.cpp) run
embedding models INSIDE the process — no server, no network. Gated behind
`RAGCPP_WITH_ONNX` / `RAGCPP_WITH_LLAMA`; absent the dep, `load()` returns
`unavailable` (the same graceful-degradation contract as the HTTP backends).

### eval — BEIR harness
Loads the standard BEIR format (corpus/queries jsonl + qrels tsv) and computes
**nDCG@k, Recall@k, Precision@k, MAP, MRR** — the measurement that turns "SOTA"
from a claim into a number you can diff.

### rerank/mmr — diversity
**Maximal Marginal Relevance** (Carbonell & Goldstein 1998): greedily trade
relevance against novelty (`λ·rel − (1−λ)·max-sim-to-chosen`) so the top-k stop
being k paraphrases of the same passage. Cosine or lexical similarity; a pipeline
stage too.

### index/pq — Product Quantization
Compress embeddings 4–64×: split each vector into `m` sub-vectors, k-means each
subspace to 256 centroids, store one byte per sub-vector. Scoring is **ADC**
(Asymmetric Distance Computation): a per-subspace query·centroid lookup table,
then `m` table lookups per vector — no decompression. Serializable.

### cascade — telescoping retrieval
The production funnel as one call: hybrid retrieve (top-N₀) → ColBERT late
interaction (top-N₁) → cross-encoder rerank (top-N₂) → top-k, each stage with a
candidate BUDGET so the expensive stages never blow up. Stages are optional and
degrade gracefully (an unavailable stage is skipped).

### cache — embedding + query memoization
Bounded thread-safe LRU caches. **EmbeddingCache** keys on `(embedder identity,
text)` so a model swap never returns a stale vector; **QueryCache** memoizes
`(query, k)` → hits (clear on corpus mutation). Both are the cheapest latency win
in a RAG loop.

### text/semantic_chunker — meaning-aware splitting
**Semantic chunking**: embed sentences and break where consecutive-sentence
similarity drops below a percentile — chunks are topically coherent runs.
**Proposition chunking** (dense-x): atomic self-contained statements. Lexical
fallback with no embedder.

### text/contextual — Contextual Retrieval (Anthropic 2024)
Prepend to each chunk a short blurb SITUATING it in its document before indexing,
so both BM25 and dense representations carry the disambiguating context.
LLM-generated via a seam, or a deterministic extractive default (title + best-
overlap sentence).

### Incremental delete
HNSW gains **tombstone soft-delete** (`remove`/`is_deleted`/`compact`): deleted
nodes stay in the graph for connectivity but never surface in results; re-adding
an id resurrects it; `compact()` rebuilds without tombstones. `Corpus::
remove_document` tombstones a doc's chunks across BM25 + HNSW with stable ids
(so the chunk meta-pointer invariant holds).

### cli — the `ragcpp` tool
A turnkey binary: `ragcpp index <dir> <out.ragdb>`, `query <db> "..."`,
`eval <beir-dir>`, `info <db>` — build and search a corpus with no C++.

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
they call stay non-virtual inside `Corpus`. Three pre-assembled factories ship:
`Pipeline::standard()` (hybrid → filter → feature-rerank → top-k, the default),
`Pipeline::quality()` (+MMR diversity), and `Pipeline::context()` (+ParentStitch
small-to-big). The two extras are opt-in because their good is coverage, not
accuracy, and each carries a measured justification at its declaration.

### store — persistence
The stable, versioned, CRC-checked `.ragdb` container (see `FORMAT.md`).

### c — the ABI
A flat opaque-handle C API (`rag/c/rag.h`) so Python/Rust/Go/any language can
drive the engine. Errors cross as status codes, never exceptions.

### rcp — the protocol front-end
rag-cpp as a conformant [**RCP/1**](https://github.com/1ay1/rcp) server. A
header-only framework layer (`rag/rcp/`) over the RCP C++ SDK, layered like the
rest of the library:

- `error.hpp` — the one total map between `rag::Error` (Errc) and the wire's
  JSON-RPC codes (`rcp::errc`), exhaustively switched.
- `convert.hpp` — pure wire↔engine translation: `retrieve` param parsing with the
  funnel invariant (§3.3), `SearchResult`→`Hit` (id/score/citation/trust/vector),
  and the §8 filter tree → a `MetaFilter` predicate (validated against advertised
  fields, precise `-32602` otherwise).
- `handler.hpp` — `EngineHandler` satisfies the SDK `Handler` concept over an
  `Engine&`. Capabilities are **data** (`Options`), advertised only when honestly
  backable (e.g. `embed` iff an embedder is attached); disabled methods are
  unreachable via `-32003` before the hook runs. `Hooks` lets a host override or
  extend any method with a `std::function` — no subclassing.
- `server.hpp` — `ServerBuilder` + `serve_stdio`/`serve_http` free functions.

The engine is borrowed by reference (host owns lifetime + ingestion); the server
reads the live corpus. Certified at conformance level L2.

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
