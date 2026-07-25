#pragma once
// rag/util/parallel.hpp — a minimal, dependency-free work-stealing-free thread
// pool plus the two parallel primitives the engine actually needs.
//
// Design notes:
//   • One global pool, lazily constructed, sized to hardware_concurrency. The
//     engine is a library: we must not spawn a pool per index build, and we
//     must not fight the host application for cores on every call.
//   • `parallel_for` splits [0,n) into contiguous blocks — one per worker —
//     rather than per-item tasks. Retrieval workloads are uniform-cost, so
//     static blocking beats a task queue and costs no atomics in the hot loop.
//   • The calling thread participates. With a pool of P threads we run P+1
//     workers and never idle the caller.
//   • Exceptions are not used for control flow anywhere in rag-cpp; a worker
//     that throws would terminate, so worker bodies must be noexcept in spirit
//     (all engine bodies return Result and never throw).
//
// Threading is OFF by default for tiny inputs: below `kParallelMin` items the
// primitives run inline, so small corpora pay zero synchronization cost.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace rag::util {

// Below this many items, parallel primitives run serially: the fork/join
// overhead dominates and the cache stays warm on one core.
inline constexpr std::size_t kParallelMin = 512;

class ThreadPool {
public:
    explicit ThreadPool(unsigned n) {
        workers_.reserve(n);
        for (unsigned i = 0; i < n; ++i)
            workers_.emplace_back([this] { run(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

    // Enqueue a job. Fire-and-forget; completion is tracked by the caller's
    // own latch (see parallel_for), which keeps the pool free of futures.
    void submit(std::function<void()> job) {
        {
            std::lock_guard lk(mu_);
            jobs_.push(std::move(job));
        }
        cv_.notify_one();
    }

private:
    void run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop();
            }
            job();
        }
    }

    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> jobs_;
    std::mutex                        mu_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
};

// The process-wide pool. Sized to hardware_concurrency-1 because the calling
// thread participates in every parallel_for.
inline ThreadPool& pool() {
    static ThreadPool p{[] {
        unsigned hc = std::thread::hardware_concurrency();
        return hc > 1 ? hc - 1 : 0u;
    }()};
    return p;
}

// How many workers a parallel region will use (including the caller).
[[nodiscard]] inline std::size_t max_workers() noexcept { return pool().size() + 1; }

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

    std::atomic<std::size_t> remaining{blocks - 1};
    std::mutex              done_mu;
    std::condition_variable done_cv;

    // Dispatch all but the final block to the pool.
    for (std::size_t b = 0; b + 1 < blocks; ++b) {
        const std::size_t lo = b * chunk;
        const std::size_t hi = lo + chunk < n ? lo + chunk : n;
        pool().submit([lo, hi, &body, &remaining, &done_mu, &done_cv] {
            for (std::size_t i = lo; i < hi; ++i) body(i);
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lk(done_mu);
                done_cv.notify_one();
            }
        });
    }

    // The caller runs the tail block itself.
    for (std::size_t i = (blocks - 1) * chunk; i < n; ++i) body(i);

    std::unique_lock lk(done_mu);
    done_cv.wait(lk, [&] { return remaining.load(std::memory_order_acquire) == 0; });
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

    std::atomic<std::size_t> remaining{blocks - 1};
    std::mutex              done_mu;
    std::condition_variable done_cv;

    for (std::size_t b = 0; b + 1 < blocks; ++b) {
        const std::size_t lo = b * chunk;
        const std::size_t hi = lo + chunk < n ? lo + chunk : n;
        pool().submit([lo, hi, b, &body, &remaining, &done_mu, &done_cv] {
            body(lo, hi, b);
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lk(done_mu);
                done_cv.notify_one();
            }
        });
    }
    body((blocks - 1) * chunk, n, blocks - 1);

    std::unique_lock lk(done_mu);
    done_cv.wait(lk, [&] { return remaining.load(std::memory_order_acquire) == 0; });
}

} // namespace rag::util
