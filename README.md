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

### GraphRAG
- An explicit **document graph**: **link edges** (markdown `[..](..)`, bare
  URLs, `[[wikilinks]]`) + **similarity edges** (dense centroid cosine, else
  lexical Jaccard; k-NN sparsified).
- **Communities** via deterministic label propagation, **summarized
  extractively** (free, no model) with an optional abstractive summarizer seam.
- **Local search** (hybrid seed → **Personalized-PageRank** graph expansion) and
  **Global search** (rank community summaries) — the full GraphRAG recipe with
  the expensive LLM ingredients made deterministic by default.

### Cutting-edge retrieval
- **SPLADE-style learned sparse** — saturated impact weighting + query-time term
  expansion over an inverted index (BM25 speed, dense-like recall).
- **ColBERT late interaction** — token-level **MaxSim** reranking between
  bi-encoders and cross-encoders.
- **RAPTOR** — recursive cluster-summarize tree; retrieve the **collapsed tree**
  across abstraction levels (Sarthi et al., ICLR 2024).
- **HyDE** — hypothetical-document embeddings + **multi-query / RAG-Fusion**
  (Gao et al. 2022).
- **Corrective RAG + Self-RAG** — a retrieval evaluator grading confidence into
  correct / ambiguous / incorrect actions, decompose-recompose knowledge strips,
  external-source fallback, and an answer **groundedness** score (Yan et al.
  2024; Asai et al. 2024).

### Evaluation
A **BEIR-format harness** (corpus/queries jsonl + qrels tsv) computing
**nDCG@k, Recall@k, Precision@k, MAP, MRR** — so SOTA is a measured number, not
a claim. See [`examples/beir_eval.cpp`](examples/beir_eval.cpp).

### Production
- **Product Quantization** — 4–64× embedding compression with ADC scoring.
- **MMR** diversity reranking — stop returning k paraphrases of one passage.
- **Retrieval cascade** — hybrid → ColBERT → cross-encoder with per-stage budgets.
- **Semantic + proposition chunking** and **Contextual Retrieval** (Anthropic
  2024) situating context.
- **Incremental delete** — HNSW tombstones + `Corpus::remove_document` with
  stable ids and `compact()`.
- **Embedding + query LRU caches** (thread-safe, identity-keyed).
- **`ragcpp` CLI** — `index` / `query` / `eval` / `info` with no C++.

### RALM assemblies (retrieval-augmented language modeling)
Generator-agnostic retrieval frontends for four landmark recipes (the LM call is
your seam):
- **RAG / REPLUG** — temperature-scaled softmax ensemble weights `p(z|x)` over
  retrieval scores + a distribution-mixing combinator.
- **RETRO** — chunked-neighbour retrieval with **continuations** (the chunk that
  followed each neighbour).
- **In-Context RALM** — a retrieval **stride schedule** with a rerank hook.
- **Grounded prompt assembly** — numbered, source-attributed context.

### Embedders (pluggable)
`Ollama` · `OpenAI`-compatible (+ **Together**, **TEI** presets) · `llama.cpp`
server · deterministic local `Hash` (no network). Decorators: `RetryingEmbedder`
(exponential backoff), `FallbackEmbedder` (primary → secondary). All behind an
injectable **`HttpTransport`** seam — bring your own TLS/gRPC/mock.

**In-process** (no server): **ONNX Runtime** (`OnnxEmbedder`, with WordPiece
tokenization + mean/CLS/max pooling) and **GGUF via llama.cpp** (`GgufEmbedder`).
Opt in with `-DRAGCPP_WITH_ONNX=ON` / `-DRAGCPP_WITH_LLAMA=ON`; without the dep
they report `unavailable` and the core stays dependency-free.

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
./build/examples/ragcpp_graphrag           # GraphRAG + RALM assemblies
./build/examples/ragcpp_advanced           # SPLADE, ColBERT, RAPTOR, HyDE, CRAG
./build/examples/ragcpp_beir_eval <dir>    # BEIR nDCG/Recall/MAP/MRR
./build/bench/ragcpp_bench 5000            # ablation + latency
```

For the shared library (Python/Rust FFI): `-DBUILD_SHARED_LIBS=ON`. The only
dependency (nlohmann/json) is fetched via `FetchContent`.

## CLI

The `ragcpp` binary builds, searches, and **serves** a corpus with no code:

```sh
ragcpp index ./docs corpus.ragdb      # walk a directory, chunk, index, save
ragcpp info  corpus.ragdb             # document/chunk counts
ragcpp query corpus.ragdb "my question" -k 5 --mmr
ragcpp serve corpus.ragdb             # a conformant RCP/1 server over stdio
ragcpp serve corpus.ragdb --http 8000 --write --graph   # HTTP + writable index + GraphRAG
ragcpp eval  ./nfcorpus --split=test  # BEIR metrics
```

`ragcpp serve` opens a saved `.ragdb` and brings up an [RCP/1](https://github.com/1ay1/rcp)
endpoint with **zero C++** — L2-certified, driveable by any RCP client. `--write`
advertises a writable index (`index/add`+`index/delete` with upsert); `--graph`
advertises GraphRAG local/global search.

## Use rag-cpp as a library

rag-cpp installs a relocatable CMake package, so consuming it from another
project is one `find_package` away — the dependency on nlohmann/json is resolved
for you:

```cmake
find_package(ragcpp CONFIG REQUIRED)
target_link_libraries(app PRIVATE ragcpp::ragcpp)
```

```sh
cmake --install build --prefix /usr/local     # or any prefix on CMAKE_PREFIX_PATH
```

Or vendor it directly with FetchContent / `add_subdirectory` — the same
`ragcpp::ragcpp` target name works in-tree and installed.


## Serve over RCP (the Retrieval Context Protocol)

rag-cpp speaks [**RCP/1**](https://github.com/1ay1/rcp) — an open JSON-RPC
protocol for retrieval engines (what MCP is to tools, RCP is to grounding
context). Any RCP client can drive the engine's hybrid retrieval, embedding,
GraphRAG, and index over stdio or HTTP. The front-end is a thin, header-only
framework layer (`rag/rcp/`) over the same live `Engine` your app already builds:

```cpp
#include <rag/rcp/rcp.hpp>

int main() {
    rag::Engine engine;                       // your engine, your corpus
    engine.with_embedder(/* … */);
    engine.add("doc://1", "…"); engine.build();

    rag::rcp::serve_stdio(engine);            // now a conformant RCP/1 server
}
```

Need more surface? The fluent `ServerBuilder` advertises exactly the
capabilities you turn on. Because rag-cpp already *has* the machinery, you
attach the engine's own advanced components and the matching RCP methods light
up **natively** — no host glue:

```cpp
rag::rcp::ServerBuilder(engine)
    .named("docs", "1.0")
    .with_index(/*writable=*/true)                 // index/add + index/delete (upsert)
    .with_graph().with_memory()                    // GraphRAG + HippoRAG-style memory
    .with_splade(rag::sparse::SpladeIndex::build(engine.corpus()).value())
    .with_colbert(rag::late::hashed_token_embedder(64))   // embed/multi + colbert rerank
    .with_reranker(my_cross_encoder)               // rerank + retrieve.rerank
    .with_generator(my_llm)                        // query/transform + retrieve.rewrite (HyDE)
    .filter_on("lang", "keyword")                  // §8 metadata filtering
    .serve_http(8000);
```

The engine is held by **reference** — the host keeps ingesting/persisting
through its own handle while the server reads the same live corpus. Every
capability is advertised **honestly**: `embed` only with an embedder, `sparse`
mode + `embed/sparse` only with a SPLADE index, `rerank`/`transform` only when a
reranker/generator is attached. The funnel invariant `candidateK ≥ topN ≥ k` is
wire-enforced, and every hit carries a citation for grounded generation.

The bundled `ragcpp_rcp_server` demo wires **all twelve RCP methods** to real
engine machinery — hybrid/dense/sparse `retrieve`, cross-encoder + ColBERT
`rerank`, HyDE/multi-query `query/transform`, `embed`/`embed/sparse`/`embed/multi`,
GraphRAG `graph`, `index/add`+`index/delete`, and `memory/build`+`memory/recall`
over the community graph.

Build and certify:

```sh
cmake --build build --target ragcpp_rcp_server        # RAGCPP_WITH_RCP=ON (default)
./build/examples/ragcpp_rcp_server                     # stdio
./build/examples/ragcpp_rcp_server --http 8000         # HTTP
python3 ~/projects/rcp/conformance/check.py -- ./build/examples/ragcpp_rcp_server
#  → CERTIFIED LEVEL: L2
```

## Performance (5000 chunks, Apple M-series, NEON)

```
embed+build:   ~3.7 s    (one-time)
hybrid query:  ~2.1 ms/query
bm25 only:     ~0.18 ms/query
dense (HNSW):  ~0.14 ms/query
```

## Quality (BEIR, measured)

Speed is only half of it. The `standard` pipeline with `all-MiniLM-L6-v2`
attached in-process, evaluated through `Pipeline::run` — the same code path
`Engine::search` executes:

| SciFact nDCG@10 | |
|---|---|
| BM25 (BEIR paper) | 0.665 |
| rag-cpp, no model at all | 0.6809 |
| ColBERTv2 | 0.693 |
| SPLADE++ | 0.710 |
| **rag-cpp hybrid (MiniLM-L6)** | **0.7347** |

Hybrid beats **both of its own halves** — BM25 alone scores 0.6800, MiniLM alone
0.6518 — on all three datasets tested, which is the fusion earning its keep
rather than a better retriever being picked. Full tables, the NFCorpus/ArguAna
runs, and the reproduction commands are in
[`BENCHMARKS.md`](BENCHMARKS.md#hybrid-retrieval-with-a-neural-embedder).

## Documentation

Full guides live in [**`docs/`**](docs/README.md). Start there if you're new.

**Guides**

- [Getting Started](docs/getting-started.md) — install, first index, first query (CLI + library).
- [Retrieval](docs/retrieval.md) — hybrid BM25 + dense, HNSW, fusion, filtering, quantization.
- [Vector store](docs/vector-store.md) — using rag-cpp as a plain vector DB (your ids, your vectors).
- [The Pipeline](docs/pipeline.md) — composable stages; `standard` / `quality` / `context` factories; custom stages.
- [Embedders](docs/embedders.md) — every backend, the retry/fallback decorators, the `HttpTransport` seam.
- [Advanced Retrieval](docs/advanced-retrieval.md) — SPLADE, ColBERT, RAPTOR, HyDE, CRAG, GraphRAG.
- [Configuration](docs/configuration.md) — every `CorpusConfig` / `HnswConfig` tunable.
- [The CLI](docs/cli.md) — full `ragcpp` command reference.
- [Serving over RCP](docs/rcp-server.md) — turning an `Engine` into a conformant RCP/1 server.
- [Persistence](docs/persistence.md) — the `.ragdb` container and the write-ahead log.
- [The C API & bindings](docs/c-api.md) — the flat C ABI and driving it from Python/Rust/Go.
- [GPU acceleration](docs/gpu.md) — the optional Metal batch-scoring backend.

**Reference**

- [`BENCHMARKS.md`](BENCHMARKS.md) — measured BEIR / ANN / throughput numbers, reproducible.
- [`CHANGELOG.md`](CHANGELOG.md) — release notes.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layers, seams, invariants.
- [`PLUGINS.md`](PLUGINS.md) — the three extension axes: concepts, `AnyX`, and the
  load-time plugin registry (register backends by name / from a shared library).
- [`FORMAT.md`](FORMAT.md) — the `.ragdb` on-disk contract.

## References

The design draws on the retrieval-augmentation literature:

- Lewis et al., **Retrieval-Augmented Generation** (NeurIPS 2020) — marginalize
  the generator over retrieved documents weighted by `p(z|x)`.
- Borgeaud et al., **RETRO** (ICML 2022) — chunked cross-attention over
  retrieved neighbours and their continuations.
- Ram et al., **In-Context RALM** (TACL 2023) — retrieve at a generation stride,
  rerank, prepend; leave the LM untouched.
- Shi et al., **REPLUG** (NAACL 2024) — black-box LM ensembling over retrieved
  documents.
- Edge et al., **GraphRAG** (Microsoft 2024) — entity/document graph →
  communities → community summaries → graph-aware local + global search.
- Sarthi et al., **RAPTOR** (ICLR 2024) — recursive cluster-summarize tree,
  collapsed-tree retrieval across abstraction levels.
- Khattab & Zaharia, **ColBERT** (SIGIR 2020) — late interaction (MaxSim) over
  token embeddings.
- Formal et al., **SPLADE** (SIGIR 2021) — learned sparse retrieval with term
  expansion.
- Gao et al., **HyDE** (2022) — precise zero-shot dense retrieval via
  hypothetical documents.
- Yan et al., **Corrective RAG** (2024) and Asai et al., **Self-RAG** (ICLR
  2024) — retrieval evaluation, correction, and reflective critique.
- Thakur et al., **BEIR** (NeurIPS 2021) — the zero-shot IR benchmark this
  library's eval harness targets.

## License

MIT.
