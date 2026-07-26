// bench/batch_bench.cpp — dense_search_batch vs the per-query loop, end to end.
//
// This exists because the GPU unit tests can pass without ever reaching the
// GPU: score_batch() declines below min_batch_work(), so a small fixture
// silently asserts that the CPU path equals itself. This drives the REAL
// Corpus API at a size that clears the threshold, and checks the batch result
// against the loop the caller would otherwise have written.
//
// It earned its place: the first run reported *** MISMATCH ***, which turned
// out to be tied scores ordered arbitrarily by an unstable partial_sort. Same
// set, different order, on two paths that must be interchangeable. Dense
// ranking now has a total order (score desc, chunk id asc).
//
// Runs on any machine: without a GPU every row simply reports ~1.0x, which is
// itself the useful claim (the batch entry point never costs anything).

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rag/gpu/device.hpp"
#include "rag/index/corpus.hpp"
#include "rag/plugin/plugin.hpp"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

int main() {
    std::printf("gpu available: %s\n\n", rag::gpu::available() ? "yes" : "no");

    for (std::size_t docs : {50000ul, 200000ul}) {
        for (std::size_t nq : {8ul, 32ul, 128ul}) {
            rag::index::CorpusConfig cfg;
            cfg.hnsw_threshold = 1'000'000;          // force the scan path
            rag::index::Corpus c{cfg};
            auto emb = rag::plugin::make_embedder(nlohmann::json{{"type", "hash"}, {"dim", 384}});
            c.set_embedder(std::move(*emb));
            for (std::size_t i = 0; i < docs; ++i)
                c.add_document("d" + std::to_string(i),
                               "retrieval document " + std::to_string(i) +
                               " about vectors graphs indexes ranking and search");
            (void)c.build();

            std::vector<std::string> qs;
            for (std::size_t i = 0; i < nq; ++i) qs.push_back("vectors ranking " + std::to_string(i));

            (void)c.dense_search_batch(qs, 10);      // warm the packed mirror

            auto t0 = clk::now();
            std::vector<std::vector<rag::Hit>> loop;
            for (const auto& q : qs) loop.push_back(*c.dense_search(q, 10));
            const double loop_ms = ms(clk::now() - t0);

            auto t1 = clk::now();
            auto batch = c.dense_search_batch(qs, 10);
            const double batch_ms = ms(clk::now() - t1);

            bool same = batch && batch->size() == loop.size();
            if (same)
                for (std::size_t q = 0; q < loop.size() && same; ++q) {
                    if ((*batch)[q].size() != loop[q].size()) { same = false; break; }
                    for (std::size_t i = 0; i < loop[q].size(); ++i)
                        if ((*batch)[q][i].chunk.get() != loop[q][i].chunk.get()) { same = false; break; }
                }

            std::printf("chunks=%-7zu nq=%-4zu loop %8.2f ms | batch %8.2f ms | %5.2fx | %s\n",
                        c.chunk_count(), nq, loop_ms, batch_ms, loop_ms / batch_ms,
                        same ? "identical" : "*** MISMATCH ***");
        }
    }
    return 0;
}
