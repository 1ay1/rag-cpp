# GPU acceleration

rag-cpp has an optional **Metal** backend that scores dense-retrieval batches on
the GPU. It is on by default on Apple platforms (`-DRAGCPP_WITH_METAL=ON`) and
absent elsewhere. It is a **throughput optimization for batch scoring**, not a
requirement — every path has a CPU fallback and the GPU is used only when it is
provably faster.

## What it accelerates

The GPU scores a *batch* of query vectors against the corpus matrix in one
dispatch. The single entry point that reaches it in production is:

```cpp
index::Corpus::dense_search_batch(queries, k, filter);
```

This is called from **HyDE** and **multi-query** search, which each produce
several hypotheticals/paraphrases and now hand the whole set to
`dense_search_batch` instead of looping one query at a time. See
[Advanced Retrieval](advanced-retrieval.md#hyde-and-multi-query).

## When it engages (and when it doesn't)

Routing is conservative and lives in `Corpus`, not the GPU module. The batch is
offloaded **only** when all of these hold:

- there is **no HNSW graph** (the GPU does a flat scan; a graph walk is already
  sub-linear),
- there is **no metadata filter**,
- the batch clears `gpu::min_batch_work()` (a minimum `nq · n · dim` multiply-add
  count — currently ~2 G MACs),
- a GPU is actually present (`gpu::available()`).

Otherwise it falls back to the per-query CPU loop — so it is **never slower**.
Below the threshold the measured GPU/CPU ratio is ~1.0×; the win only shows up on
large batches over a large flat corpus.

## Measured

Through the real `dense_search_batch` API (dim 384, k=10, on an M-series Mac):

| batch | corpus | CPU | GPU | speedup |
|-------|--------|-----|-----|---------|
| 32 queries | 200k chunks | 282 ms | 45 ms | 6.2× |
| 128 queries | 200k chunks | 1154 ms | 123 ms | 9.4× |
| 128 queries | 50k chunks | — | — | 6.3× |

Reproduce with `build/bench/ragcpp_gpu_bench` (and `ragcpp_batch_bench`, which
also checks the GPU and CPU paths return **identical** rankings).

### The packed mirror

A packed contiguous matrix is kept as an epoch-keyed, lazily-built mirror of the
corpus vectors — it is **mandatory, not an optimization**. Packing per call
measured 37 ms at n=200k against a 46.7 ms CPU scan, which turned a 1.7× win into
a 0.73× *loss*. The mirror is invalidated on mutation and rebuilt on demand.

## Determinism

Batched and per-query scans return the **same ranking**, including under tied
scores: the dense comparator is a total order (score descending, then chunk id
ascending). Ties are common with duplicate passages or quantized vectors, and
without the id tiebreak `std::partial_sort` (not stable) would order them
arbitrarily — so the two code paths could return the same *set* in a different
*order*. The tiebreak makes both paths agree exactly.

## Building without it

```sh
cmake -B build -DRAGCPP_WITH_METAL=OFF
```

The Metal code is then not compiled in, `gpu::available()` is always false, and
every batch runs on the CPU. On non-Apple platforms this is automatic.

## Testing note

`gpu::disable()` is a one-way, process-global latch used by tests to force the
CPU path. On CI / non-Apple hosts the GPU kernel itself isn't exercised (the
equivalence test's GPU arm returns early when `gpu::available()` is false); the
CPU-equivalence arm still runs everywhere. The kernel is exercised on Apple
developer hardware.
