// bench/contextual_bench.cpp — what Contextual Retrieval costs at ingest, and
// what it buys at retrieval.
//
// Anthropic's Contextual Retrieval (2024) prepends to each chunk a short blurb
// situating it in its source document before indexing. The claim is a large cut
// in retrieval failures. The claim is NOT free: every chunk grows, and the
// deterministic backend tokenizes every document sentence once per chunk.
//
// This bench answers three questions the feature's config comment must not
// answer from memory:
//   1. how much slower is ingest,
//   2. how much bigger is the index,
//   3. does recall actually go up, on a corpus where it plausibly could.
//
// The corpus is built to make the failure mode REAL rather than assumed: each
// document names its subject once, in the title, and never again. Its body
// paragraphs are near-identical across documents ("revenue grew", "headcount
// was flat"), so a chunk torn out of its document is genuinely unattributable
// — which is precisely the situation the paper describes. On a corpus where
// every chunk already repeats the subject, this feature can only cost you
// something, and the bench would (correctly) show no gain.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "rag/index/corpus.hpp"

using namespace std::chrono;

namespace {

struct Doc {
    std::string uri;
    std::string subject;
    std::string body;
};

// N documents, each about a distinct fictional company, whose paragraphs are
// drawn from a shared pool of subject-free sentences.
std::vector<Doc> make_corpus(std::size_t n, std::uint32_t seed) {
    static const char* kSubjectFree[] = {
        "The quarter closed ahead of the internal plan.",
        "Revenue grew three percent year over year.",
        "Headcount was flat across all divisions.",
        "Operating margin improved by forty basis points.",
        "The board approved a buyback of ten million shares.",
        "Free cash flow was positive for the sixth consecutive quarter.",
        "Deferred revenue rose on strength in multi-year contracts.",
        "Gross margin compressed slightly on hardware mix.",
        "Churn in the small business segment remained elevated.",
        "The company reiterated its full year guidance.",
    };
    static const char* kStems[] = {"Acme", "Globex", "Initech", "Umbrella", "Soylent",
                                   "Hooli", "Vehement", "Massive", "Wonka", "Tyrell"};

    std::mt19937 rng(seed);
    std::vector<Doc> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Doc d;
        d.subject = std::string(kStems[i % 10]) + std::to_string(i);
        d.uri     = d.subject + ".md";
        // The subject appears ONCE, in the title line.
        d.body    = d.subject + " Corporation\n";
        std::uniform_int_distribution<int> pick(0, 9);
        for (int p = 0; p < 8; ++p) {
            d.body += "\n";
            d.body += kSubjectFree[pick(rng)];
            d.body += "\n";
        }
        out.push_back(std::move(d));
    }
    return out;
}

rag::index::CorpusConfig config(bool contextual) {
    rag::index::CorpusConfig cfg;
    cfg.contextual          = contextual;
    // Small chunks, so most chunks genuinely lose the subject. With the default
    // 40-line window a whole document fits in one chunk and nothing is lost —
    // and the bench would measure nothing.
    cfg.chunk.max_lines     = 4;
    cfg.chunk.overlap_lines = 0;
    cfg.chunk.heading_context = false;   // isolate the contextual signal
    return cfg;
}

struct Run {
    double      ingest_ms  = 0;
    std::size_t chunks     = 0;
    std::size_t indexed_bytes = 0;
    double      recall     = 0;
    double      mrr        = 0;
};

// A query per document, of the form "<subject> <topic>", where <topic> is a
// phrase the document's OTHER chunks contain. The relevant set is every chunk
// of that document that mentions the topic.
//
// Scoring is CHUNK-level on purpose. Document-level recall on this corpus is
// trivially 1.0 for both arms — the subject token is unique, so the title chunk
// always matches and the document is always "found". That is the same
// saturation trap the MMR benchmark hit: a metric that cannot go up cannot
// tell you anything. What contextual retrieval actually changes is WHICH
// chunks are reachable, so that is what is measured.
Run measure(const std::vector<Doc>& docs, bool contextual, std::size_t k) {
    Run r;
    rag::index::Corpus c{config(contextual)};

    auto t0 = steady_clock::now();
    for (const auto& d : docs) (void)c.add_document(d.uri, d.body);
    (void)c.build();
    r.ingest_ms = duration<double, std::milli>(steady_clock::now() - t0).count();

    {
        auto lease = c.chunks();
        r.chunks = lease.size();
        for (const auto& ch : lease) r.indexed_bytes += ch.indexed_text().size();
    }

    // Build the ground truth from the store itself: for each document, the
    // chunks whose BODY contains the topic phrase. This is independent of how
    // the chunk was indexed, so it is the same relevant set for both arms.
    static const char* kTopic = "headcount";
    std::vector<std::vector<std::uint32_t>> relevant(docs.size());
    {
        auto lease = c.chunks();
        for (const auto& ch : lease) {
            std::string body = ch.text;
            for (auto& x : body) x = static_cast<char>(std::tolower((unsigned char)x));
            if (body.find(kTopic) != std::string::npos)
                relevant[ch.doc.get()].push_back(ch.id.get());
        }
    }

    double recall_sum = 0, mrr_sum = 0;
    std::size_t scored = 0;
    for (std::size_t di = 0; di < docs.size(); ++di) {
        const auto& rel = relevant[di];
        if (rel.empty()) continue;          // no relevant chunk: unscoreable
        ++scored;
        auto res = c.lexical_search(docs[di].subject + " " + kTopic, k);
        std::size_t found = 0;
        bool first = true;
        for (std::size_t i = 0; i < res.size(); ++i) {
            bool is_rel = std::find(rel.begin(), rel.end(), res[i].chunk.get()) != rel.end();
            if (!is_rel) continue;
            ++found;
            if (first) { mrr_sum += 1.0 / double(i + 1); first = false; }
        }
        recall_sum += double(found) / double(std::min(rel.size(), k));
    }
    r.recall = scored ? recall_sum / double(scored) : 0.0;
    r.mrr    = scored ? mrr_sum    / double(scored) : 0.0;
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::stoul(argv[1]) : 2000;
    const std::size_t k = argc > 2 ? std::stoul(argv[2]) : 5;

    auto docs = make_corpus(n, 42);
    std::printf("Contextual Retrieval — %zu documents, subject named once per document, k=%zu\n\n",
                n, k);

    Run off = measure(docs, false, k);
    Run on  = measure(docs, true,  k);

    std::printf("%-22s %14s %14s %10s\n", "", "contextual=off", "contextual=on", "delta");
    std::printf("%-22s %14.1f %14.1f %9.2fx\n", "ingest+build (ms)",
                off.ingest_ms, on.ingest_ms, on.ingest_ms / off.ingest_ms);
    std::printf("%-22s %14zu %14zu %9.2fx\n", "chunks", off.chunks, on.chunks,
                double(on.chunks) / double(off.chunks));
    std::printf("%-22s %14.1f %14.1f %9.2fx\n", "indexed text (KiB)",
                double(off.indexed_bytes) / 1024.0, double(on.indexed_bytes) / 1024.0,
                double(on.indexed_bytes) / double(off.indexed_bytes));
    std::printf("%-22s %14.4f %14.4f %+9.4f\n", "chunk recall@k", off.recall, on.recall,
                on.recall - off.recall);
    std::printf("%-22s %14.4f %14.4f %+9.4f\n", "chunk MRR@k", off.mrr, on.mrr, on.mrr - off.mrr);

    std::printf("\nRead this as a COST/BENEFIT pair, not a win: the ingest column is the\n"
                "price, and it is paid on every document forever. If the recall delta on\n"
                "YOUR corpus is small, the feature is not worth it there — chunks that\n"
                "already repeat their subject have nothing to be situated with.\n");
    return 0;
}
