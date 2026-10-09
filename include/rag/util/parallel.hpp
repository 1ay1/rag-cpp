#pragma once
// rag/util/parallel.hpp — how the engine's parallel regions are split, as a
// value the caller hands in. rag-cpp owns no threads and no executor.
//
// A Splitter runs fn(i) for every i in [0, n) and returns when all have
// finished; the calls may run concurrently. The host builds one on its own
// runtime and puts it in the configs of the objects that want it (HNSW
// build, PQ training, corpus embedding, hybrid retrieval). An empty Splitter
// runs everything inline, on the caller's thread, which is also what a host
// that wants no parallelism passes.
//
// The primitives below split [0, n) into contiguous blocks, one per share,
// rather than per-item tasks: retrieval work is uniform-cost, so static
// blocking beats a queue and needs no shared counter. Below kParallelMin
// items they run inline, so small corpora pay nothing.

#include <cstddef>
#include <functional>

namespace rag::util {

// Below this many items, parallel primitives run serially: the fork/join
// overhead dominates and the cache stays warm on one core.
inline constexpr std::size_t kParallelMin = 512;

class Splitter {
public:
    using Run = std::function<void(std::size_t n, const std::function<void(std::size_t)>& fn)>;

    Splitter() = default;   // inline
    // `run` must not deadlock when called from inside one of its own fn(i):
    // rag regions nest (a query inside a server worker). `width` is how many
    // shares are worth making.
    Splitter(Run run, std::size_t width) : run_(std::move(run)), width_(width ? width : 1) {}

    [[nodiscard]] std::size_t width() const noexcept { return run_ ? width_ : 1; }
    [[nodiscard]] bool inline_only() const noexcept { return !run_; }

    void operator()(std::size_t n, const std::function<void(std::size_t)>& fn) const {
        if (run_ && n > 1) { run_(n, fn); return; }
        for (std::size_t i = 0; i < n; ++i) fn(i);
    }

private:
    Run         run_;
    std::size_t width_ = 1;
};

// How many contiguous blocks parallel_blocks(n) will produce. Callers that
// need one private accumulator per block can size their vector up front.
[[nodiscard]] inline std::size_t block_count(const Splitter& s, std::size_t n) noexcept {
    if (n == 0) return 0;
    const std::size_t w = s.width();
    if (n < kParallelMin || w <= 1) return 1;
    return w < n ? w : n;
}

// Run `body(i)` for i in [0,n), in contiguous blocks.
template <class F>
void parallel_for(const Splitter& s, std::size_t n, F&& body) {
    const std::size_t blocks = block_count(s, n);
    if (blocks <= 1) { for (std::size_t i = 0; i < n; ++i) body(i); return; }
    const std::size_t chunk = (n + blocks - 1) / blocks;
    s(blocks, [&](std::size_t b) {
        const std::size_t lo = b * chunk;
        const std::size_t hi = lo + chunk < n ? lo + chunk : n;
        for (std::size_t i = lo; i < hi; ++i) body(i);
    });
}

// Run `body(lo, hi, block_index)` once per contiguous block. Use when a worker
// wants to amortize setup (scratch buffers, a local top-k heap) across its
// whole range; `block_index` in [0, block_count(n)) indexes per-block state.
template <class F>
void parallel_blocks(const Splitter& s, std::size_t n, F&& body) {
    const std::size_t blocks = block_count(s, n);
    if (blocks == 0) return;
    if (blocks == 1) { body(std::size_t{0}, n, std::size_t{0}); return; }
    const std::size_t chunk = (n + blocks - 1) / blocks;
    s(blocks, [&](std::size_t b) {
        const std::size_t lo = b * chunk;
        const std::size_t hi = lo + chunk < n ? lo + chunk : n;
        body(lo, hi, b);
    });
}

// Run `body(i)` for i in [0,n) over `workers` strided shares — for coarse,
// non-uniform items such as embedding batches, where each item costs
// milliseconds to seconds. Unlike parallel_for there is no kParallelMin gate
// (a handful of network round-trips is already worth it) and `workers` is
// the caller's, not clamped to the splitter's width: IO-bound items want
// more in flight than there are cores. Share k takes items k, k+w, k+2w, …
template <class F>
void parallel_for_strided(const Splitter& s, std::size_t n, std::size_t workers, F&& body) {
    if (n == 0) return;
    std::size_t w = workers ? workers : 1;
    if (w > n) w = n;
    if (w <= 1) { for (std::size_t i = 0; i < n; ++i) body(i); return; }
    s(w, [&](std::size_t k) {
        for (std::size_t i = k; i < n; i += w) body(i);
    });
}

} // namespace rag::util
