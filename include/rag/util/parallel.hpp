#pragma once
// rag/util/parallel.hpp — the parallel primitives the engine needs, run on an
// Executor the host installs. rag-cpp owns no threads.
//
// Design notes:
//   • One process executor, installed by the host (set_executor). The engine
//     is a library: the program decides what threads run its work.
//   • `parallel_for` splits [0,n) into contiguous blocks — one per share —
//     rather than per-item tasks. Retrieval workloads are uniform-cost, so
//     static blocking beats a task queue and costs no atomics in the hot loop.
//   • Exceptions are not used for control flow anywhere in rag-cpp; worker
//     bodies must be noexcept in spirit (all engine bodies return Result).
//
// Threading is OFF by default for tiny inputs: below `kParallelMin` items the
// primitives run inline, so small corpora pay zero synchronization cost.

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>   // the fallback executor only
#include <vector>

namespace rag::util {

// Below this many items, parallel primitives run serially: the fork/join
// overhead dominates and the cache stays warm on one core.
inline constexpr std::size_t kParallelMin = 512;

// ── The host's executor ─────────────────────────────────────────────────
// rag-cpp owns no threads. Every parallel region below hands its blocks to
// an Executor the HOST installs (rag::util::set_executor), so the program
// decides what runs where — agentty runs them on its own runtime. A host that
// installs nothing gets the fallback: one std::jthread per extra block,
// joined before the region returns. It is the only place rag-cpp names a
// thread.
class Executor {
public:
    virtual ~Executor() = default;
    // Run fn(i) for every i in [0, n); return when all have finished. Calls
    // may run concurrently. Must not deadlock when called from inside one of
    // its own fn(i) (rag regions nest: a query inside a server worker).
    virtual void parallel_for(std::size_t n,
                              const std::function<void(std::size_t)>& fn) = 0;
    // How many concurrent shares a region should be split into.
    [[nodiscard]] virtual std::size_t width() const noexcept = 0;
};

class ThreadExecutor final : public Executor {
public:
    void parallel_for(std::size_t n,
                      const std::function<void(std::size_t)>& fn) override {
        if (n <= 1) { if (n) fn(0); return; }
        std::vector<std::jthread> ts;
        ts.reserve(n - 1);
        for (std::size_t i = 1; i < n; ++i) ts.emplace_back([&fn, i] { fn(i); });
        fn(0);
    }   // joined here
    [[nodiscard]] std::size_t width() const noexcept override {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc ? hc : 1;
    }
};

namespace detail {
inline std::shared_ptr<Executor>& installed_executor() {
    static std::shared_ptr<Executor> e;
    return e;
}
} // namespace detail

// Install the process executor, once, at startup. nullptr restores the
// fallback. Not for swapping while a region runs.
inline void set_executor(std::shared_ptr<Executor> e) {
    detail::installed_executor() = std::move(e);
}

inline Executor& executor() {
    static ThreadExecutor fallback;
    auto& e = detail::installed_executor();
    return e ? *e : fallback;
}

// How many workers a parallel region will use (including the caller).
[[nodiscard]] inline std::size_t max_workers() noexcept {
    const std::size_t w = executor().width();
    return w ? w : 1;
}

// How many contiguous blocks parallel_blocks(n) will produce. Callers that
// need one private accumulator per block can size their vector up front.
[[nodiscard]] inline std::size_t block_count(std::size_t n) noexcept {
    if (n == 0) return 0;
    const std::size_t workers = max_workers();
    if (n < kParallelMin || workers <= 1) return 1;
    return workers < n ? workers : n;
}

// Run `body(i)` for i in [0,n). Contiguous static blocking; the calling thread
// takes the last block. Serial below kParallelMin or when the pool is empty.
template <class F>
void parallel_for(std::size_t n, F&& body) {
    if (n == 0) return;
    const std::size_t workers = max_workers();
    if (n < kParallelMin || workers <= 1) {
        for (std::size_t i = 0; i < n; ++i) body(i);
        return;
    }

    const std::size_t blocks = workers < n ? workers : n;
    const std::size_t chunk  = (n + blocks - 1) / blocks;
    executor().parallel_for(blocks, [&](std::size_t b) {
        const std::size_t lo = b * chunk;
        const std::size_t hi = lo + chunk < n ? lo + chunk : n;
        for (std::size_t i = lo; i < hi; ++i) body(i);
    });
}

// Run `body(lo, hi, block_index)` once per contiguous block. Use when a worker
// wants to amortize setup (scratch buffers, a local top-k heap) across its
// whole range; `block_index` in [0, block_count(n)) indexes per-block state
// without any atomics.
template <class F>
void parallel_blocks(std::size_t n, F&& body) {
    if (n == 0) return;
    const std::size_t workers = max_workers();
    if (n < kParallelMin || workers <= 1) { body(std::size_t{0}, n, std::size_t{0}); return; }

    const std::size_t blocks = workers < n ? workers : n;
    const std::size_t chunk  = (n + blocks - 1) / blocks;
    executor().parallel_for(blocks, [&](std::size_t b) {
        const std::size_t lo = b * chunk;
        const std::size_t hi = lo + chunk < n ? lo + chunk : n;
        body(lo, hi, b);
    });
}

// Run `body(i)` for i in [0,n) with an EXPLICIT worker count and DYNAMIC
// (atomic-counter) scheduling — the counterpart to parallel_for for coarse,
// non-uniform items such as embedding batches, where each item costs
// milliseconds-to-seconds and one straggler must not stall a whole block.
//
// Differences from parallel_for:
//   • no kParallelMin gate — the caller has already decided this is worth
//     parallelizing (a handful of network round-trips easily is);
//   • `workers` is set by the caller (e.g. an embedder's concurrency hint) and
//     is NOT clamped to the core count. These items are typically IO-bound —
//     eight in-flight HTTP requests are useful on a two-core box, where a
//     core-clamped pool would leave 6/8 of the throughput on the table;
//   • it runs on the same Executor; the Executor contract (no deadlock when
//     nested) is what makes calling it from inside another region safe;
//   • work is claimed one index at a time, so uneven items self-balance.
template <class F>
void parallel_for_dynamic(std::size_t n, std::size_t workers, F&& body) {
    if (n == 0) return;
    std::size_t w = workers ? workers : 1;
    if (w > n) w = n;
    if (w <= 1) {
        for (std::size_t i = 0; i < n; ++i) body(i);
        return;
    }

    std::atomic<std::size_t> next{0};
    auto drain = [&] {
        for (;;) {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) return;
            body(i);
        }
    };

    // w drainers on the executor, all finished before this returns.
    executor().parallel_for(w, [&](std::size_t) { drain(); });
}

} // namespace rag::util
