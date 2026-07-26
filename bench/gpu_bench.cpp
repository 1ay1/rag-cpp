// gpu_bench.cpp — is the GPU actually worth it, on this machine, right now?
//
// Prints the recall-free half of the question: batch scoring throughput,
// GPU against the CPU path the library actually uses. The CPU baseline here is
// deliberately the STRONG one — NEON dot products across every core, iterating
// candidates outer and queries inner so each candidate row stays in L1 across
// the whole query batch. Benchmarking against a naive loop instead inflates the
// GPU's apparent win by 3-4x and hides the fact that a naive GPU kernel loses.
//
// Usage: ragcpp_gpu_bench [dim]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "rag/dense/simd.hpp"
#include "rag/gpu/device.hpp"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

int main(int argc, char** argv) {
    const std::size_t dim = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 384;

    const auto& info = rag::gpu::device_info();
    std::printf("gpu       : %s\n", rag::gpu::available() ? "available" : "NOT available (CPU only)");
    std::printf("backend   : %s\n", rag::gpu::backend_name(info.backend));
    std::printf("device    : %s%s\n", info.name.c_str(),
                info.unified_memory ? "  (unified memory)" : "");
    std::printf("threshold : %.2f G MACs before a batch is sent to the GPU\n",
                rag::gpu::min_batch_work() / 1e9);
    std::printf("dim       : %zu\n\n", dim);

    std::mt19937_64 rng(7);
    std::normal_distribution<float> g(0.f, 1.f);
    const unsigned T = std::max(1u, std::thread::hardware_concurrency());

    std::printf("%-9s %-7s %12s %12s %9s %11s %10s\n",
                "n", "queries", "CPU ms", "GPU ms", "speedup", "GPU GF/s", "max err");

    for (auto [n, nq] : {std::pair<std::size_t, std::size_t>{50000, 8},
                         {50000, 32}, {200000, 32}, {200000, 128}, {500000, 128}}) {
        std::vector<float> corpus(n * dim), queries(nq * dim);
        for (auto& x : corpus)  x = g(rng);
        for (auto& x : queries) x = g(rng);
        for (std::size_t i = 0; i < n; ++i)
            rag::dense::normalize(std::span<float>(corpus.data() + i * dim, dim));
        for (std::size_t i = 0; i < nq; ++i)
            rag::dense::normalize(std::span<float>(queries.data() + i * dim, dim));

        // CPU: cache-blocked, all cores.
        std::vector<float> ref(n * nq);
        auto t0 = clk::now();
        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < T; ++t)
                ts.emplace_back([&, t] {
                    for (std::size_t i = n * t / T; i < n * (t + 1) / T; ++i)
                        for (std::size_t q = 0; q < nq; ++q)
                            ref[q * n + i] = rag::dense::dot(
                                std::span<const float>(queries.data() + q * dim, dim),
                                std::span<const float>(corpus.data() + i * dim, dim));
                });
            for (auto& t : ts) t.join();
        }
        const double cpu_ms = ms(clk::now() - t0);

        std::vector<float> out(n * nq);
        (void)rag::gpu::score_batch(corpus, queries, dim, out);          // warm
        auto t1 = clk::now();
        const bool ran = rag::gpu::score_batch(corpus, queries, dim, out);
        const double gpu_ms = ms(clk::now() - t1);

        if (!ran) {
            std::printf("%-9zu %-7zu %12.1f %12s %9s %11s %10s\n",
                        n, nq, cpu_ms, "declined", "-", "-", "(cpu path)");
            continue;
        }
        double worst = 0;
        for (std::size_t i = 0; i < out.size(); ++i)
            worst = std::max(worst, (double)std::fabs(out[i] - ref[i]));
        std::printf("%-9zu %-7zu %12.1f %12.1f %8.1fx %11.1f %10.1e\n",
                    n, nq, cpu_ms, gpu_ms, cpu_ms / gpu_ms,
                    (2.0 * n * nq * dim) / (gpu_ms * 1e6), worst);
    }
    return 0;
}
