// bench/stitch_bench.cpp — what ParentStitch (small-to-big) costs and buys.
//
// Small-to-big / parent-document retrieval (LlamaIndex, LangChain) indexes
// documents as many small chunks for precise matching, then, at retrieval,
// folds adjacent matching fragments of the SAME document back together so the
// caller gets one coherent window per location instead of three overlapping
// slivers of it. The good it produces is NOT better ranking — it is that the
// top-k stops spending several of its slots on near-adjacent fragments of one
// passage, and so COVERS MORE DISTINCT LOCATIONS. That is the same kind of good
// MMR produces (coverage, not accuracy), and it is measured the same way.
//
// This bench answers, on a corpus built to make the effect real:
//   1. how many candidates does stitch fold away,
//   2. does the top-k end up covering more distinct documents / locations,
//   3. what does the extra stage cost per query.
//
// The corpus is built so the failure mode is REAL rather than assumed: every
// document is long and split into many small adjacent chunks (max_lines=3), and
// the query term appears in a RUN of consecutive chunks within each document.
// Without stitching a single document's adjacent fragments crowd the top-k; on
// a corpus whose documents each produce one chunk, stitch has nothing to fold
// and the bench would (correctly) show no change — which is the honest result
// to report for such corpora.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"

using namespace std::chrono;

namespace {

// N documents. A FEW documents (n is small) are very long and mention the query
// topic in a long RUN of consecutive paragraphs, so each fragments into MANY
// adjacent matching chunks. Because there are few documents, one document's
// adjacent fragments genuinely crowd the top-k: without stitching the top-10
// can be five slivers of doc0's topic run plus five of doc1's, covering only 2
// documents; stitching folds each run to its best sibling and frees the slots.
std::vector<std::pair<std::string, std::string>> make_corpus(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(n);
    static const char* kFillLines[] = {
        "The design review covered failure domains and rollout ordering.",
        "An on-call rotation was established for the first two weeks.",
        "Dashboards tracked the p99 across every shard in the fleet.",
        "The rollback plan was rehearsed against a staging replica.",
        "Capacity headroom was confirmed before the cutover began.",
        "A postmortem template was prepared in case of regressions.",
    };
    std::uniform_int_distribution<int> fill(0, 5);
    for (std::size_t i = 0; i < n; ++i) {
        std::string uri = "doc" + std::to_string(i) + ".md";
        std::string body = "Engineering note " + std::to_string(i) + "\n\n";
        for (int p = 0; p < 2; ++p) { body += kFillLines[fill(rng)]; body += "\n\n"; }
        // A LONG run of consecutive topic paragraphs — each becomes an adjacent
        // matching chunk at max_lines=3, so this one document alone can supply
        // more than k matching fragments.
        for (int p = 0; p < 12; ++p) {
            body += "The migration to the new storage engine reduced replication lag "
                    "and compaction backlog under peak write load (step " + std::to_string(p) + ").";
            body += "\n\n";
        }
        for (int p = 0; p < 2; ++p) { body += kFillLines[fill(rng)]; body += "\n\n"; }
        out.emplace_back(std::move(uri), std::move(body));
    }
    return out;
}

rag::index::CorpusConfig config() {
    rag::index::CorpusConfig cfg;
    // Small chunks, so each document's topic run fragments into several ADJACENT
    // matching chunks. With the default 40-line window a whole document is one
    // chunk, there is nothing adjacent to fold, and the bench measures nothing.
    cfg.chunk.max_lines       = 3;
    cfg.chunk.overlap_lines   = 0;
    cfg.chunk.heading_context = false;
    return cfg;
}

struct Run {
    double      query_ms       = 0;   // mean per-query wall time
    double      distinct_docs  = 0;   // mean distinct documents in top-k
    double      results        = 0;   // mean hits returned (<= k)
};

// distinct documents covered by the top-k — the coverage metric. A top-k full
// of one document's adjacent fragments scores low here; spreading across
// documents scores high. `results` reports how many hits came back, since
// stitch can return fewer than k after folding.
Run measure(const rag::index::Corpus& c, const rag::pipeline::Pipeline& pipe,
            const std::vector<std::string>& queries, std::size_t k) {
    Run r;
    double docs_sum = 0, res_sum = 0, ms_sum = 0;
    for (const auto& q : queries) {
        auto t0 = steady_clock::now();
        auto hits = pipe.run(c, q, k);
        ms_sum += duration<double, std::milli>(steady_clock::now() - t0).count();
        if (!hits) continue;
        std::unordered_set<std::uint32_t> docs;
        for (const auto& h : *hits) {
            if (const auto* ch = c.chunk(h.chunk)) docs.insert(ch->doc.get());
        }
        docs_sum += double(docs.size());
        res_sum  += double(hits->size());
    }
    const double nq = double(queries.size());
    r.query_ms      = ms_sum / nq;
    r.distinct_docs = docs_sum / nq;
    r.results       = res_sum / nq;
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::stoul(argv[1]) : 5;
    const std::size_t k = argc > 2 ? std::stoul(argv[2]) : 10;

    auto docs = make_corpus(n, 42);
    rag::index::Corpus c{config()};
    for (const auto& [uri, body] : docs) (void)c.add_document(uri, body);
    (void)c.build();

    std::size_t chunk_count = 0;
    { auto lease = c.chunks(); chunk_count = lease.size(); }

    // The query hits the topic run that every document shares, so every
    // document contributes a cluster of adjacent matching chunks — exactly the
    // situation stitch exists for.
    std::vector<std::string> queries = {
        "migration storage engine latency",
        "replication lag write load",
        "compaction backlog after migration",
    };

    // standard(): hybrid → filter → feature_rerank → topk (no stitch).
    auto plain = rag::pipeline::Pipeline::standard();

    std::printf("ParentStitch (small-to-big) — %zu documents, %zu chunks, k=%zu\n\n",
                n, chunk_count, k);

    Run off = measure(c, plain, queries, k);

    // Apply stitch as a post-step on the SAME candidate lists standard() would
    // return, by running standard() then folding — this isolates exactly what
    // the stage changes without re-plumbing the whole pipeline in the bench.
    rag::pipeline::ParentStitchStage stitch(1);
    Run on;
    {
        double docs_sum = 0, res_sum = 0, ms_sum = 0;
        for (const auto& q : queries) {
            // Pull a larger candidate pool so folding has neighbours to remove,
            // then stitch, then trim to k — the stage's intended position.
            auto hits = plain.run(c, q, /*k*/k * 4);
            if (!hits) continue;
            rag::pipeline::Context ctx;
            ctx.corpus = &c; ctx.candidates = *hits; ctx.k = k;
            auto t0 = steady_clock::now();
            auto out = stitch.process(std::move(ctx));
            ms_sum += duration<double, std::milli>(steady_clock::now() - t0).count();
            if (!out) continue;
            auto cands = std::move(out->candidates);
            if (cands.size() > k) cands.resize(k);
            std::unordered_set<std::uint32_t> ds;
            for (const auto& h : cands)
                if (const auto* ch = c.chunk(h.chunk)) ds.insert(ch->doc.get());
            docs_sum += double(ds.size());
            res_sum  += double(cands.size());
        }
        const double nq = double(queries.size());
        on.query_ms      = ms_sum / nq;
        on.distinct_docs = docs_sum / nq;
        on.results       = res_sum / nq;
    }

    std::printf("%-26s %14s %14s %10s\n", "", "standard", "+ stitch", "delta");
    std::printf("%-26s %14.4f %14.4f %+9.4f\n", "distinct docs in top-k",
                off.distinct_docs, on.distinct_docs, on.distinct_docs - off.distinct_docs);
    std::printf("%-26s %14.2f %14.2f %+9.2f\n", "hits returned (<= k)",
                off.results, on.results, on.results - off.results);
    std::printf("%-26s %14.4f %14.4f %+9.4f\n", "stitch stage (ms/query)",
                0.0, on.query_ms, on.query_ms);

    std::printf("\nRead this as coverage, not accuracy: stitch does not reorder by relevance,\n"
                "it removes adjacent fragments a higher-ranked sibling already represents, so\n"
                "the freed slots go to distinct locations. On a corpus whose documents each\n"
                "produce a single chunk there is nothing adjacent to fold and the delta is 0 —\n"
                "which is the correct, honest result for such corpora.\n");
    return 0;
}
