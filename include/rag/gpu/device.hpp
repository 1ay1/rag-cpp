#pragma once
// rag/gpu/device.hpp — optional GPU acceleration for BATCH scoring.
//
// WHAT THIS IS FOR, and what it deliberately is not.
//
// The decision of which stages to offload was made by measurement against the
// CPU path that ALREADY SHIPS — NEON dot products across 8 threads — not
// against a naive loop. That distinction is the whole story here. On an Apple
// M1 (8 CPU cores, 8 GPU cores, unified memory), scoring 200k candidates:
//
//   workload                          vs naive CPU   vs the CPU we ship
//   -------------------------------   ------------   ------------------
//   batched scan, 128 queries              10.2x            3.1x
//   batched scan,  32 queries               6.3x            1.6x
//   batched scan,   8 queries               1.0x            0.3x  (declined)
//   PQ k-means assignment                    —               0.7x
//   HNSW graph walk                          —                n/a
//
// Two things had to be true before any of this was worth shipping.
//
// FIRST, the work must be BATCHED. A single-query scan reads n*dim*4 bytes to
// do 2*n*dim flops — half a flop per byte — so it is limited by memory
// bandwidth, and on unified memory both processors share the same bus.
// Measured directly: an f32 scan and an f16 scan of the same vectors both land
// at ~47 GB/s despite the f16 kernel moving half the bytes, which is the
// signature of a bandwidth ceiling that no kernel tuning can lift. Batching Q
// queries into one dispatch raises intensity to ~Q flops/byte. Hence this API
// is BATCH-ONLY: there is deliberately no single-vector entry point, because
// offering one would invite exactly the use that cannot pay, and a 0.2 ms
// dispatch round trip would swamp the 0.07 ms an HNSW query costs today.
//
// SECOND, the kernel must reuse each candidate row across many queries, the
// same way the CPU's blocked loop keeps that row in L1. The obvious kernel
// — one thread per (query, candidate) — reaches only 105 GFLOP/s and LOSES to
// the blocked CPU (0.9x). Register-tiling over 8 queries reaches 320 GFLOP/s
// and wins by 3.1x. Same hardware, same arithmetic, same dispatch.
//
// The HNSW walk is not offloaded and should not be: it is a serial dependent
// chain of ef-wide beam steps over scattered memory — the workload GPUs are
// worst at, and the one this library spends most of its query time in. GPU
// acceleration here is for BULK work (offline eval, reindexing, clustering,
// large reranks), not for the interactive query path.
//
// CONTRACT
//   * Availability is a RUNTIME question. A build with GPU support still runs
//     on a machine with no device; every entry point degrades to the CPU path
//     and callers cannot tell the difference except in timing.
//   * Results are numerically equivalent to the CPU, not bit-identical. Lanes
//     accumulate in a different order. Measured max absolute error against a
//     double-precision reference is <1e-7 for dim=384 unit vectors — smaller
//     than the CPU's own scalar-loop error, since the GPU sums in 4 partial
//     lanes rather than one long chain.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rag::gpu {

// WHO CALLS THIS IN PRODUCTION.
//
// index::Corpus::dense_search_batch() is the entry point this module exists to
// serve, and query::hyde_search / query::multi_query_search reach it by handing
// their whole set of hypotheticals/paraphrases to the corpus at once instead of
// looping. Routing is decided there, not here: the GPU is used only for an
// unfiltered, HNSW-less scan whose batch clears min_batch_work(), and every
// other shape runs the same threaded NEON path as before.
//
// Measured end-to-end through the real Corpus API (M1, dim 384, 200k chunks,
// hash embedder, k=10) against the per-query loop the caller would otherwise
// have written:
//
//   32 queries   268 ms -> 44 ms   6.2x
//   64 queries   535 ms -> 65 ms   8.2x
//  128 queries  1084 ms -> 128 ms  8.5x
//
// Below the threshold the batch entry point measures 0.97-1.00x versus the
// loop, i.e. it costs nothing to call when it cannot help.
//
// One correctness note that cost real debugging time: the packed matrix this
// module needs must be AMORTIZED. Packing per call was measured at n=200k as
// 37 ms against a 46.7 ms CPU scan, turning a 1.70x win into 0.73x. Corpus
// keeps an epoch-keyed mirror instead.

// What the active backend is, for diagnostics and for tests that must assert
// they exercised the path they think they did.
enum class Backend { none, metal };

[[nodiscard]] const char* backend_name(Backend b) noexcept;

// Description of the device actually in use.
struct DeviceInfo {
    Backend     backend = Backend::none;
    std::string name    = "cpu";
    bool        unified_memory = false;   // no host<->device copy needed
    std::size_t max_buffer_bytes = 0;
};

// Is a GPU usable right now? False when built without GPU support, when no
// device exists, or when shader compilation failed at startup. Cheap after the
// first call (the answer is memoized along with the compiled pipelines).
[[nodiscard]] bool available() noexcept;

// Info for the active device; `backend == none` when running on CPU.
[[nodiscard]] const DeviceInfo& device_info() noexcept;

// Force the GPU off for this process. Intended for benchmarks and for tests
// that need to compare the two paths; also a safety valve for a host that hits
// a driver problem in the field. Cannot be undone — a one-way switch is much
// easier to reason about than a toggle racing with in-flight work.
void disable() noexcept;

// ── Batched scoring ─────────────────────────────────────────────────────────
//
// Score every one of `nq` queries against every one of `n` candidates:
//
//     out[q * n + i] = dot(queries[q], corpus[i])
//
// `corpus` is n*dim floats, row-major; `queries` is nq*dim floats, row-major;
// `out` must have room for nq*n floats. Vectors are used as given \u2014 this does
// not normalize, matching rag::dense::dot, which assumes unit vectors.
//
// Returns false if the work was NOT done on the GPU (no device, batch too
// small to be worth a dispatch, buffer larger than the device allows). A false
// return means `out` is untouched and the caller must run its own CPU path;
// it is a routing answer, not an error.
[[nodiscard]] bool score_batch(std::span<const float> corpus,
                               std::span<const float> queries,
                               std::size_t dim,
                               std::span<float> out) noexcept;

// The smallest total work (nq * n * dim multiply-adds) for which score_batch
// will accept a job. Below this the dispatch round trip dominates and the CPU
// is faster; exposed so callers can make the same routing decision without
// speculatively calling in, and so the threshold is testable rather than a
// magic number buried in a .cpp.
[[nodiscard]] std::size_t min_batch_work() noexcept;

} // namespace rag::gpu
