# Using rag-cpp as a vector database

`Corpus` is the text-RAG engine: hand it documents, it chunks, embeds, and
indexes them. But sometimes you **already have vectors** — from your own model, a
batch job, another service — and you just want a fast, persistent
approximate-nearest-neighbour store keyed by your own ids.

That's `rag::index::VectorStore`. It's a thin façade over the same HNSW engine the
`Corpus` uses, so it inherits the parallel graph build, SQ8/PQ compression, SIMD
distance kernels, per-query `ef`, filtered search, and versioned on-disk format —
behind a five-method surface with no chunks, no documents, no embedder.

## The whole API

```cpp
#include <rag/rag.hpp>
using rag::index::VectorStore;

VectorStore store{384};                       // dimension is fixed for its life

store.add(id, vec);                           // your uint32 id, your float vector
store.build();                                // once, after a batch
for (auto& [id, score] : store.search(query, 10)) { /* ... */ }
store.remove(id);
store.save("vectors.ragvec");                 // and VectorStore::load(...)
```

That's it. Vectors are unit-normalized on insert, so `score` is **cosine
similarity in `[-1, 1]`, descending**.

## A complete example

```cpp
#include <rag/rag.hpp>
#include <cstdio>

int main() {
    rag::index::VectorStore store{128, rag::index::HnswConfig::accurate()};

    for (std::uint32_t i = 0; i < 100'000; ++i)
        store.add(i, my_vector_for(i));       // returns Result<void>

    store.build();                            // parallel HNSW build

    auto hits = store.search(my_query_vector(), 10);
    for (const auto& h : hits)
        std::printf("id=%u  cos=%.4f\n", h.id, h.score);

    store.save("vectors.ragvec");
}
```

Reopening never rebuilds:

```cpp
auto store = rag::index::VectorStore::load("vectors.ragvec");
if (!store) { /* typed error: corrupt_index / io_error */ }
auto hits = store->search(q, 10);
```

## Choosing an index configuration

You don't have to reason about nine HNSW knobs. Start from a **preset**:

| Preset | `M` / `ef_construction` / `ef_search` | Use when |
|--------|---------------------------------------|----------|
| `HnswConfig::fast()` | 16 / 100 / 32 | Interactive latency matters most (~0.94 recall@10 on SIFT). |
| `HnswConfig::balanced()` | 16 / 200 / 64 | **The default.** ~0.97 recall at ~2× the throughput of `accurate()`. |
| `HnswConfig::accurate()` | 32 / 400 / 128 | Recall first (~0.99+): offline eval, legal/medical retrieval. |
| `HnswConfig::compact()` | balanced + `drop_floats` | Memory first: ~1 byte/dim (a ~3× saving) at ~0.94 recall. |
| `HnswConfig::for_scale(n)` | picks one of the above | You just want a sane default for `n` vectors. |

```cpp
VectorStore store{384, rag::index::HnswConfig::for_scale(5'000'000)};
```

These are named points on the recall/latency/memory curves **measured** on GloVe
and SIFT — the numbers are tabulated in the `HnswConfig` comment block and
reproducible with `build/bench/ragcpp_ann_bench`.

## The recall/latency dial, per query

`ef` is a pure search-time beam width. It costs nothing to change and belongs to
the **request**, not the index — an autocomplete wants a narrow beam, an offline
evaluation wants a wide one:

```cpp
auto quick    = store.search(q, 10, /*ef*/ 32);    // lower latency
auto thorough = store.search(q, 10, /*ef*/ 256);   // higher recall
```

Passing `0` (the default) uses the configured `ef_search`.

## Filtered search

Restrict results to ids your predicate accepts. The filter is evaluated **during**
the graph walk, not as a post-filter, so a selective predicate still returns `k`
results instead of running out:

```cpp
auto hits = store.search(q, 10, [&](std::uint32_t id) {
    return is_visible_to_current_user(id);
});
```

## Behaviour worth knowing

- **Upsert.** Re-adding an id replaces its vector; an id appears **at most once**
  in any result set.
- **Remove** is a tombstone — the vector stops being returned immediately, and
  `save()` compacts it away physically so it never comes back on reload.
- **Small stores skip the graph.** Below `brute_force_below` (default 2000)
  vectors, search is an exact brute-force scan — faster *and* exact at that size.
  `build()` is a no-op there; it builds the graph once you cross the threshold.
- **Dimension is fixed.** `add()` returns `Errc::invalid_argument` on a mismatch
  rather than corrupting the index.
- **Persistence is CRC-checked.** A corrupted or truncated `.ragvec` fails with a
  typed error, never a garbage read.

## Introspection

```cpp
store.size();             // live vector count
store.dimension();
store.is_graph_built();   // false while below the brute-force threshold
store.memory_bytes();     // what your compression setting actually bought
store.config();           // the HnswConfig in force
```

## When to use which

- **`VectorStore`** — you own the vectors and the ids. A recommendation index,
  an image/audio embedding store, a cache of precomputed embeddings.
- **[`Corpus` / `Engine`](getting-started.md)** — you have *text* and want
  chunking, BM25, hybrid fusion, reranking, and citations. See
  [Retrieval](retrieval.md).

Both sit on the same HNSW core, so the performance characteristics — and the
[GPU batch-scoring path](gpu.md) — are the same.
