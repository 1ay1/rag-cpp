// ann_bench.cpp — recall/QPS curve against a STANDARD ANN benchmark dataset.
//
// The in-tree unit tests gate recall on synthetic clusters, which is fine for
// catching regressions but says nothing about how this index compares to
// hnswlib/FAISS. This measures the numbers the ANN literature reports, on the
// files everyone reports them on, so the claims in HnswConfig are reproducible
// rather than folklore.
//
// Usage:
//   ragcpp_ann_bench <base.fvecs> [query.fvecs] [max_vectors]
//
// Get data (both are the standard files used by ann-benchmarks.com):
//   SIFT1M   curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
//   GloVe    http://ann-benchmarks.com/glove-25-angular.hdf5  (convert to fvecs)
//
// NOTE ON GROUND TRUTH: this index is cosine-only and normalizes on insert, so
// the published L2 ground-truth files do NOT apply. Truth is recomputed here as
// the exact COSINE top-k of the normalized corpus. What the datasets contribute
// is realistic vector GEOMETRY — which is the thing synthetic data gets wrong.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "rag/dense/simd.hpp"
#include "rag/index/hnsw.hpp"

using clk = std::chrono::steady_clock;
static double us(clk::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }

// .fvecs: repeated [int32 dim][float32 x dim].
static std::vector<std::vector<float>> read_fvecs(const std::string& path, std::size_t limit = 0) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::vector<float>> out;
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return out; }
    while (in && (limit == 0 || out.size() < limit)) {
        std::int32_t d = 0;
        if (!in.read(reinterpret_cast<char*>(&d), 4)) break;
        if (d <= 0 || d > 4096) break;
        std::vector<float> v(static_cast<std::size_t>(d));
        if (!in.read(reinterpret_cast<char*>(v.data()), d * 4)) break;
        rag::dense::normalize(v);
        out.push_back(std::move(v));
    }
    return out;
}

// Intrinsic dimensionality (TwoNN; Facco et al., Sci. Rep. 2017). Reported
// because it, not the ambient dimension, predicts how hard a dataset is for a
// graph index: SIFT is 128-d ambient but ~3-d intrinsic, while isotropic
// Gaussian noise in 128-d is ~59-d intrinsic and defeats every ANN index.
static double intrinsic_dim(const std::vector<std::vector<float>>& v, std::size_t sample = 400) {
    if (v.size() < 3) return 0;
    const std::size_t n = v.size(), step = std::max<std::size_t>(1, n / sample);
    double acc = 0; std::size_t used = 0;
    for (std::size_t i = 0; i < n && used < sample; i += step) {
        double r1 = 1e30, r2 = 1e30;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const double d = std::sqrt(std::max(0.0, 2.0 - 2.0 * rag::dense::dot(v[i], v[j])));
            if (d < r1) { r2 = r1; r1 = d; } else if (d < r2) { r2 = d; }
        }
        if (r1 > 1e-12 && r2 > r1) { acc += std::log(r2 / r1); ++used; }
    }
    return used && acc > 0 ? static_cast<double>(used) / acc : 0.0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <base.fvecs> [query.fvecs] [max_vectors]\n", argv[0]);
        return 2;
    }
    const std::string base_path  = argv[1];
    const std::string query_path = argc > 2 ? argv[2] : argv[1];
    const std::size_t limit      = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 0;
    constexpr std::size_t k = 10;

    auto base = read_fvecs(base_path, limit);
    if (base.empty()) return 1;
    auto query = read_fvecs(query_path, 200);
    if (query.empty()) return 1;

    const std::size_t n = base.size(), dim = base[0].size(), nq = query.size();
    std::printf("dataset : %s\n", base_path.c_str());
    std::printf("vectors : %zu x %zu   queries: %zu\n", n, dim, nq);

    {
        std::vector<std::vector<float>> head(base.begin(),
                                             base.begin() + std::min<std::size_t>(n, 20000));
        std::printf("geometry: intrinsic dim ~%.1f (ambient %zu)\n", intrinsic_dim(head), dim);
    }

    // Exact cosine ground truth, in parallel — otherwise the slowest step here.
    std::vector<std::vector<std::uint32_t>> truth(nq);
    {
        auto t0 = clk::now();
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> ts;
        const unsigned T = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned t = 0; t < T; ++t)
            ts.emplace_back([&] {
                std::vector<std::pair<float, std::uint32_t>> all(n);
                for (std::size_t q = next++; q < nq; q = next++) {
                    for (std::size_t i = 0; i < n; ++i)
                        all[i] = {rag::dense::dot(query[q], base[i]), static_cast<std::uint32_t>(i)};
                    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });
                    truth[q].resize(k);
                    for (std::size_t i = 0; i < k; ++i) truth[q][i] = all[i].second;
                }
            });
        for (auto& t : ts) t.join();
        std::printf("truth   : exact cosine top-%zu in %.1f s\n", k, us(clk::now() - t0) / 1e6);
    }

    rag::index::HnswConfig cfg;
    rag::index::HnswIndex idx(cfg);
    {
        auto t0 = clk::now();
        idx.build_batch(n,
            [&](std::size_t i) { return std::span<const float>(base[i]); },
            [&](std::size_t i) { return static_cast<std::uint32_t>(i); });
        std::printf("build   : M=%zu efc=%zu -> %.1f s, %zu B/vector\n\n",
                    cfg.M, cfg.ef_construction, us(clk::now() - t0) / 1e6, idx.memory_bytes() / n);
    }

    std::printf("%-6s %11s %11s %11s %12s\n", "ef", "recall@10", "us/query", "QPS(1t)", "QPS(all)");
    for (std::size_t ef : {10u, 16u, 32u, 64u, 128u, 256u, 512u}) {
        (void)idx.search(query[0], k, ef);                 // warm

        double hit = 0;
        auto t0 = clk::now();
        for (std::size_t q = 0; q < nq; ++q)
            for (const auto& h : idx.search(query[q], k, ef))
                if (std::find(truth[q].begin(), truth[q].end(), h.chunk.get()) != truth[q].end()) ++hit;
        const double per_q = us(clk::now() - t0) / static_cast<double>(nq);

        const unsigned T = std::max(1u, std::thread::hardware_concurrency());
        std::atomic<long> done{0};
        auto m0 = clk::now();
        {
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < T; ++t)
                ts.emplace_back([&] {
                    for (int rep = 0; rep < 5; ++rep)
                        for (std::size_t q = 0; q < nq; ++q) {
                            (void)idx.search(query[q], k, ef);
                            done.fetch_add(1, std::memory_order_relaxed);
                        }
                });
            for (auto& t : ts) t.join();
        }
        const double mt_qps = static_cast<double>(done.load()) / (us(clk::now() - m0) / 1e6);

        std::printf("%-6zu %11.4f %11.1f %11.0f %12.0f\n",
                    ef, hit / static_cast<double>(nq * k), per_q, 1e6 / per_q, mt_qps);
    }
    return 0;
}
