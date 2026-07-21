// bench/bench.cpp — an ablation harness: measure each retrieval stage's
// contribution and the latency of the index primitives. Uses a synthetic
// corpus so it runs with zero external services.

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <rag/rag.hpp>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

std::string synth_doc(std::mt19937& rng, std::size_t words) {
    static const char* vocab[] = {
        "vector","index","query","retrieval","embedding","token","rank","fusion",
        "corpus","semantic","lexical","dense","sparse","graph","neural","search",
        "document","chunk","cosine","distance","cluster","model","score","hybrid",
    };
    std::uniform_int_distribution<int> pick(0, 23);
    std::string s;
    for (std::size_t i = 0; i < words; ++i) { s += vocab[pick(rng)]; s += ' '; }
    return s;
}
} // namespace

int main(int argc, char** argv) {
    std::size_t n = argc > 1 ? std::stoul(argv[1]) : 5000;
    std::printf("rag-cpp bench — SIMD tier: %s\n", rag::dense::simd_tier());
    std::printf("Building synthetic corpus of %zu docs...\n", n);

    std::mt19937 rng(42);
    rag::index::CorpusConfig cfg;
    cfg.hnsw_threshold = 1000;  // force HNSW build
    rag::Engine engine(cfg);
    engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{256}});

    auto t0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i)
        engine.add("doc" + std::to_string(i), synth_doc(rng, 60));
    auto t1 = Clock::now();
    auto b = engine.build();
    auto t2 = Clock::now();

    std::printf("  ingest+chunk:  %8.1f ms\n", ms(t1 - t0));
    std::printf("  embed+build:   %8.1f ms  (%s)\n", ms(t2 - t1), b ? "ok" : "FAILED");
    std::printf("  chunks:        %zu\n", engine.corpus().chunk_count());

    const int Q = 200;
    std::vector<std::string> queries;
    for (int i = 0; i < Q; ++i) queries.push_back(synth_doc(rng, 5));

    auto q0 = Clock::now();
    std::size_t total_hits = 0;
    for (auto& q : queries) {
        auto r = engine.search(q, 10);
        if (r) total_hits += r->size();
    }
    auto q1 = Clock::now();
    std::printf("  hybrid query:  %8.3f ms/query  (%d queries, %zu total hits)\n",
                ms(q1 - q0) / Q, Q, total_hits);

    auto& corpus = engine.corpus();
    auto l0 = Clock::now();
    for (auto& q : queries) (void)corpus.lexical_search(q, 10);
    auto l1 = Clock::now();
    std::printf("  bm25 only:     %8.3f ms/query\n", ms(l1 - l0) / Q);

    auto d0 = Clock::now();
    for (auto& q : queries) (void)corpus.dense_search(q, 10);
    auto d1 = Clock::now();
    std::printf("  dense(HNSW):   %8.3f ms/query\n", ms(d1 - d0) / Q);

    return 0;
}
