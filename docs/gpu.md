# GPU acceleration

rag-cpp scores dense-retrieval batches on the GPU through **two interchangeable
backends**, so acceleration is available on every major vendor — not just Apple:

| Backend | Runs on | Built when |
|---------|---------|------------|
| **Metal** | Apple silicon / macOS | `-DRAGCPP_WITH_METAL=ON` (default on Apple) |
| **OpenCL** | **NVIDIA, AMD, Intel**, and Apple as a fallback | `-DRAGCPP_WITH_OPENCL=ON` (auto-detected) |

The OpenCL kernel is compiled **at runtime** by whichever driver is installed, so
one binary runs on any of those vendors with no vendor SDK at build time.

GPU support is a **throughput optimization for batch scoring**, not a
requirement — every path has a CPU fallback and the GPU is used only when it is
provably faster.

## Which backend gets used

Selection happens once, at first use:

1. `RAGCPP_GPU_BACKEND=metal|opencl|none` overrides everything, if set.
2. **Metal**, when compiled in and a device initialises.
3. **OpenCL**, otherwise.

A backend that fails to initialise falls through to the next rather than
disabling the GPU entirely — a broken OpenCL ICD should not cost an Apple user
their Metal path.

```cpp
for (auto b : rag::gpu::compiled_backends())   // what this BINARY can use
    std::printf("%s\n", rag::gpu::backend_name(b));

if (rag::gpu::available())                      // what this MACHINE has now
    std::printf("active: %s (%s)\n",
                rag::gpu::backend_name(rag::gpu::device_info().backend),
                rag::gpu::device_info().name.c_str());
```

Those are deliberately two different questions: "this build cannot use your GPU"
and "this machine has no usable GPU" are very different problems.

### Why Metal is preferred on Apple

Measured on an M1, 200k × 384 corpus, 64 queries, against an 8-thread CPU scan:

| Backend | Time | vs CPU |
|---------|------|--------|
| CPU (8 threads) | 482 ms | 1.0× |
| **Metal** | **41 ms** | **11.6×** |
| OpenCL | 692 ms | 0.7× |

The OpenCL result is **not** the kernel's fault. Profiling with
`CL_QUEUE_PROFILING_ENABLE` separates the two:

```
kernel execution (device timestamps)   16.1 ms   <- faster than Metal
readback of the 48 MB result            4.3 ms
wall time of the same dispatch         672   ms   <- Apple's ICD
```

Apple's OpenCL driver is deprecated and adds ~650 ms of per-dispatch latency,
reproducibly, on every dispatch and at every work-group size. On NVIDIA/AMD/Intel
ICDs that overhead is absent — which is exactly why Metal is preferred on Apple
and OpenCL is the path that matters everywhere else.

## Correctness across backends

A GPU accumulates a dot product in a different **order** than the CPU, so results
can differ by a float ULP. What is guaranteed, and tested
(`gpu_active_backend_matches_cpu_whichever_it_is`), is that **scores** match the
CPU to within `1e-4` on unit vectors — measured max error `2.4e-07` across 12.8M
values, with zero exceedances on either backend.

Ranked **ids** match too, except that the last slot may swap when two distinct
documents sit within a ULP of each other astride the `k` boundary. Both answers
are equally correct; demanding otherwise would require every vendor's FPU to
reassociate exactly like the host's.

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
