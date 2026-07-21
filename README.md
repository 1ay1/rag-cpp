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
