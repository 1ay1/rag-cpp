# Configuration

Every tunable lives in `CorpusConfig` (and the structs it nests). Each field is
documented at its declaration in `include/rag/index/corpus.hpp` — this guide is
the map. Construct a corpus (or Engine) with a config:

```cpp
rag::index::CorpusConfig cfg;
cfg.chunk.max_lines = 20;
cfg.hnsw.ef_search  = 128;
rag::Engine engine{cfg};
```

## Top-level `CorpusConfig`

| Field | Type / default | What it controls |
|-------|----------------|------------------|
| `chunk` | `ChunkOptions` | How documents are split into chunks. See [Chunking](#chunking). |
| `bm25` | `Bm25Params` | BM25 `k1` / `b` term-weighting parameters. |
| `tokenize` | `TokenizeOptions` | Tokenization (casefolding, stopwords, etc.). |
| `hnsw` | `HnswConfig` | The ANN graph. See [HNSW](#hnsw). |
| `hnsw_threshold` | `size_t` = `2000` | Build HNSW once chunk count crosses this; below it, brute-force cosine is exact and faster. |
| `embed_batch` | `size_t` = `32` | Texts per embedding backend call at ingest. |
| `chunking` | `Chunking` = `fixed` | `fixed` (structural) or `semantic` (topic-drift boundaries). |
| `semantic` | `SemanticChunkOptions` | Knobs for the semantic chunker (used when `chunking = semantic`). |
| `contextual` | `bool` = `false` | [Contextual Retrieval](#contextual-retrieval). |

## Chunking

**`fixed`** (default) — the structural chunker: headings + token windows. Fast,
deterministic, the right default. Key `ChunkOptions` fields:

- `max_lines` — chunk size in lines.
- `overlap_lines` — overlap between adjacent chunks (0 = none).
- `heading_context` — prepend the enclosing heading breadcrumb into each chunk's
  `context` so a fragment that lost its heading still ranks.

**`semantic`** (`chunking = Chunking::semantic`, or CLI `--semantic`) — places
boundaries where the topic actually drifts, producing more self-contained chunks
on prose at the cost of a similarity pass over adjacent sentences. It uses the
embedder when one is attached and falls back to lexical (Jaccard) drift
otherwise, so it never becomes a hard dependency on an embedding backend.

> Chunk geometry is **persisted**: a corpus built at `max_lines = 3` reopens
> chunking new documents the same way. See [Persistence](persistence.md).

## HNSW

`CorpusConfig::hnsw` (`HnswConfig`), only consulted above `hnsw_threshold` chunks:

| Field | Default | Meaning |
|-------|---------|---------|
| `M` | `16` | Max neighbours per node (2·M on base layer). ↑ recall, ↑ memory. |
| `ef_construction` | `200` | Beam width during insert. ↑ graph quality, ↓ build speed. |
| `ef_search` | `64` | Beam width during query — the main recall/latency dial. |
| `ml` | `0.0` | Level multiplier; `0` ⇒ `1/ln(M)`. |
| `matryoshka_dim` | `0` | `>0`: walk on the leading-dim prefix, rescore on full. |
| `binary` | `false` | 1-bit sign codes for the walk (rescored on full precision). |
| `drop_floats` | — | Keep only SQ8 codes (1 byte/dim) — the default compression. |
| `pq_codes` | `0` | `>0`: product-quantize to N codes (large-scale-only; SQ8 mirror kept for rescoring). |

Measured recall/QPS curves for these are tabulated in the `HnswConfig` comment
and reproducible via `build/bench/ragcpp_ann_bench`. Details in
[Retrieval § HNSW](retrieval.md#hnsw--approximate-nearest-neighbours).

## Fusion

Selected per query via `HybridRetrieveConfig` (a pipeline concern, not a corpus
one): `convex` (default, `alpha = 0.8`), `rrf`, or `rsf`. See
[Retrieval § Fusion](retrieval.md#fusion--combining-the-two-rankings) and
[The Pipeline](pipeline.md).

## Contextual Retrieval

`contextual = true` (or CLI `--contextual`). Before indexing, prepend to each
chunk a short blurb **situating** it in its source document, so a fragment that
says "revenue grew 3%" still carries the company it's about. Both the BM25
postings and the dense vector are built from the situated text.

This lives in the **ingest** config, not a pipeline stage, on purpose: the method
rewrites what is indexed, so it must run before the indexes are built — a
query-time stage cannot reproduce it.

**It is not free.** Every chunk grows and the deterministic backend tokenizes
each document once, for a measured **~3× ingest cost** (`bench/contextual_bench.cpp`,
2000 docs). The recall gain is large *on a corpus where chunks are genuinely
unattributable* and **zero on a corpus where every chunk already repeats its
subject** — a document-level metric can't even see the feature. Measure on your
corpus before enabling. Full cost/benefit in the `CorpusConfig::contextual`
comment.

## Wiring from a file

Because embedders attach by name (`with_embedder_spec`) and every algorithmic
role is a concept resolved through the [plugin registry](../PLUGINS.md), the whole
engine can be configured from JSON with no backend knowledge at the call site.
That's the point of the registry: a config file, not a recompile, chooses the
backends.
