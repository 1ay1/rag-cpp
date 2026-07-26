# Retrieval

This is what happens inside `search()`: how candidates are found, scored, fused,
filtered, and ranked. If you only read one deep-dive, read this one.

## The two retrievers

rag-cpp is **hybrid** by construction. Every query runs against two independent
index structures whose failure modes don't overlap:

- **Lexical (BM25)** — an inverted index over tokens. Exact term matching, great
  for names, codes, rare words, and anything the user typed verbatim. Pointer-
  chasing over postings lists.
- **Dense (vectors)** — cosine similarity over embedding vectors. Captures
  meaning and paraphrase, finds relevant passages that share no words with the
  query. Bandwidth-bound sequential scan (or an HNSW graph walk).

They read disjoint structures and neither observes the other's output, so the
pipeline runs them **concurrently** — hybrid latency is `max(lexical, dense)`,
not the sum. The dense retriever only spins up a thread when an embedder is
attached; with none, retrieval is pure BM25 and everything still works.

## Fusion — combining the two rankings

BM25 scores and cosine similarities are **not commensurable** — you can't just
add them. Three fusion methods bridge them, selectable per query via
`HybridRetrieveConfig::fusion`:

| Method | How it combines | When to use |
|--------|-----------------|-------------|
| **`convex`** (default) | A convex combination `α·list₁ + (1−α)·list₂` of min-max-normalized scores. `α` (default `0.8`) transfers across domains and is sample-efficient to tune. | The default. Keeps the *margin* information RRF throws away. |
| **`rrf`** | Reciprocal Rank Fusion: sum of `1/(k + rank)`. Ignores scores, uses rank only. | Robust when the two score distributions are wildly different or untrustworthy. |
| **`rsf`** | Relative Score Fusion (weighted min-max normalized sum). | A middle ground; weights `bm25_weight` / `dense_weight`. |

The default is `convex` because rank-only fusion (RRF) discards how *confident*
each retriever was: two hits at ranks 1 and 2 look identical to RRF whether their
scores were 0.9 / 0.89 or 0.9 / 0.02, and that margin is exactly the signal a
good combiner should keep. Note `bm25_weight` / `dense_weight` are ignored under
`convex`, which uses `convex.alpha` instead.

```cpp
rag::pipeline::HybridRetrieveConfig hc;
hc.fusion       = rag::pipeline::HybridRetrieveConfig::Fusion::convex;
hc.convex.alpha = 0.7f;                       // lean slightly more on the dense list
auto pipe = rag::pipeline::Pipeline::standard_with(hc);
```

Over the wire, an RCP client picks the method with `fusion_method` = `"rrf"` /
`"weighted"` / (default) — see [Serving over RCP](rcp-server.md).

## HNSW — approximate nearest neighbours

Below `CorpusConfig::hnsw_threshold` chunks (default **2000**), the dense scan is
brute-force cosine: exact, and faster than any graph at that size. Above it,
`build()` constructs an **HNSW** graph and the dense walk becomes sub-linear.

The knobs (`CorpusConfig::hnsw`, all documented at their declaration in
`include/rag/index/hnsw.hpp`):

| Field | Default | Meaning |
|-------|---------|---------|
| `M` | `16` | Max neighbours per node (2·M on the base layer). Higher = better recall, more memory. |
| `ef_construction` | `200` | Beam width during insertion. Higher = better graph, slower build. |
| `ef_search` | `64` | Beam width during query. The main recall/latency dial at query time. |
| `matryoshka_dim` | `0` | `>0`: walk on the leading-dim prefix of Matryoshka embeddings, rescore on full. |
| `binary` | `false` | 1-bit sign codes for the walk (rescored on full precision). |

Measured recall/QPS curves for these settings live in the `HnswConfig` comment
block — they are reproducible via `build/bench/ragcpp_ann_bench` against standard
datasets (GloVe, SIFT).

### Quantization

The graph can store compressed codes to shrink memory while keeping accuracy,
because **codes only order the walk — the returned top-k is always rescored on
the more precise vectors:**

- **SQ8** (`drop_floats`) — 1 byte per dimension (scalar quantization). Accurate
  enough (~1/127 error per axis) that it's the default compression story.
- **PQ** (`pq_codes = N`) — product quantization to `N` codes. Only pays off once
  the SQ8 mirror itself no longer fits in RAM; the SQ8 mirror is kept for
  rescoring regardless, so PQ is a large-scale-only option. Recall/QPS tradeoffs
  are tabulated in the `HnswConfig` comment.

## Metadata filtering

Every document carries a `Metadata` map, and search can be constrained to
documents matching a predicate — evaluated **during** the graph walk / scan, not
as a post-filter, so a filter that excludes most of the corpus doesn't waste the
candidate budget:

```cpp
rag::index::MetaFilter only_public = /* field == value, ranges, sets, boolean trees */;
auto hits = engine.search("quarterly revenue", 10, only_public);
```

Over RCP the filter arrives as a JSON tree (§8 of the spec), is **validated by
the RCP SDK** against the fields the server advertised, and is translated to a
`MetaFilter` predicate. Only the *evaluation* is rag-cpp's own code — see
[Serving over RCP](rcp-server.md).

## The result

`search()` returns `Result<vector<SearchResult>>`. Each `SearchResult` carries:

- `uri`, `title`, chunk `text` and `context`
- `score` (the fused, reranked final score)
- `start_line` / `end_line` — exact provenance for grounded generation and audit

## Beyond the default

Everything above is the `standard()` path. To reorder, diversify, or fold the
candidate set, see [The Pipeline](pipeline.md). For learned-sparse (SPLADE),
late-interaction (ColBERT), hierarchical (RAPTOR), query-rewriting (HyDE),
corrective (CRAG), or graph (GraphRAG) retrieval, see
[Advanced Retrieval](advanced-retrieval.md).
