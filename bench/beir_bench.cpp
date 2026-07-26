// bench/beir_bench.cpp — measure the ENGINE, not a shortcut.
//
// The CLI's `ragcpp eval` (and eval::evaluate_corpus) drives corpus.dense_search
// OR corpus.lexical_search directly — it never runs fusion, the reranker, or any
// pipeline. That makes it a fine BM25 baseline but a poor measurement of what the
// engine actually ships, and useless for A/B-ing a pipeline change.
//
// This bench evaluates the REAL production path — Engine::search, i.e. the full
// pipeline including hybrid fusion, filtering, reranking and top-k — and can run
// several configurations over the same dataset so a change is a diff, not a
// claim. That is the only way to say "this made retrieval better" honestly.
//
// Usage:
//   ragcpp_beir_bench <beir-dir> [--split=test] [--variant=NAME]
// Variants (default: all):
//   lexical    corpus.lexical_search only (the published-BM25 comparison point)
//   standard   Engine + Pipeline::standard()   — hybrid → filter → rerank → topk
//   quality    Engine + Pipeline::quality()    — + MMR diversity
//   context    Engine + Pipeline::context()    — + ParentStitch small-to-big
//
// Every variant uses the SAME corpus, chunking, and embedder so the only thing
// that differs is the ranking policy under test.

#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rag/engine.hpp"
#include "rag/eval/beir.hpp"
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"

namespace {

std::string opt(int argc, char** argv, const std::string& key, std::string dflt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(key + "=", 0) == 0) return a.substr(key.size() + 1);
    }
    return dflt;
}

// Index the BEIR corpus once into `corpus`, returning DocId → beir id.
std::unordered_map<std::uint32_t, std::string>
index_dataset(const rag::eval::BeirDataset& ds, rag::index::Corpus& corpus) {
    std::unordered_map<std::uint32_t, std::string> doc_uri;
    for (const auto& d : ds.corpus) {
        std::string body = d.title.empty() ? d.text : (d.title + ". " + d.text);
        auto did = corpus.add_document(d.id, std::move(body), {}, d.title);
        if (did) doc_uri[did->get()] = d.id;
    }
    (void)corpus.build();
    return doc_uri;
}

// Map a hit list to BEIR doc ids, de-duplicating chunks of the same document
// (BEIR judges documents, so two chunks of one doc must not take two slots).
rag::eval::Ranking to_ranking(const std::vector<rag::Hit>& hits,
                              const rag::index::Corpus& corpus,
                              const std::unordered_map<std::uint32_t, std::string>& doc_uri) {
    rag::eval::Ranking run;
    std::unordered_set<std::string> seen;
    for (const auto& h : hits) {
        const rag::Chunk* ch = corpus.chunk(h.chunk);
        if (!ch) continue;
        auto it = doc_uri.find(ch->doc.get());
        if (it == doc_uri.end()) continue;
        if (seen.insert(it->second).second) run.push_back(it->second);
    }
    return run;
}

void run_variant(const char* name, const rag::eval::BeirDataset& ds,
                 const rag::eval::EvalConfig& cfg,
                 const rag::eval::RetrieveFn& fn) {
    auto m = rag::eval::evaluate(ds, fn, cfg);
    auto ndcg10 = m.ndcg.count(10) ? m.ndcg.at(10) : 0.0;
    auto ndcg1  = m.ndcg.count(1)  ? m.ndcg.at(1)  : 0.0;
    auto r10    = m.recall.count(10) ? m.recall.at(10) : 0.0;
    auto r100   = m.recall.count(100) ? m.recall.at(100) : 0.0;
    std::printf("  %-10s nDCG@1 %.4f  nDCG@10 %.4f  R@10 %.4f  R@100 %.4f  MAP %.4f  MRR %.4f\n",
                name, ndcg1, ndcg10, r10, r100, m.map, m.mrr);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: ragcpp_beir_bench <beir-dir> [--split=test] [--variant=NAME]\n");
        return 2;
    }
    const std::string dir     = argv[1];
    const std::string split   = opt(argc, argv, "--split", "test");
    const std::string variant = opt(argc, argv, "--variant", "all");

    auto ds = rag::eval::BeirDataset::load(dir, split);
    if (!ds) { std::printf("load error: %s\n", ds.error().message.c_str()); return 1; }

    rag::eval::EvalConfig cfg;   // cutoffs 1,3,5,10,100; depth 100

    std::printf("BEIR %s (%s): %zu docs, %zu queries\n",
                dir.c_str(), split.c_str(), ds->corpus.size(), ds->queries.size());

    // One corpus, reused across every ranking policy, so the only difference
    // between variants is the policy itself.
    rag::index::Corpus corpus;
    auto doc_uri = index_dataset(*ds, corpus);
    std::printf("indexed %zu chunks\n\n", corpus.chunk_count());

    const bool all = (variant == "all");

    if (all || variant == "lexical") {
        run_variant("lexical", *ds, cfg, [&](const std::string& q, std::size_t k) {
            return to_ranking(corpus.lexical_search(q, k), corpus, doc_uri);
        });
    }

    // The pipeline variants run over the same corpus through Pipeline::run,
    // which is exactly what Engine::search does.
    auto pipeline_variant = [&](const char* name, rag::pipeline::Pipeline p) {
        run_variant(name, *ds, cfg, [&, p = std::move(p)](const std::string& q, std::size_t k) {
            auto hits = p.run(corpus, q, k);
            if (!hits) return rag::eval::Ranking{};
            return to_ranking(*hits, corpus, doc_uri);
        });
    };

    if (all || variant == "standard") pipeline_variant("standard", rag::pipeline::Pipeline::standard());
    if (all || variant == "quality")  pipeline_variant("quality",  rag::pipeline::Pipeline::quality());
    if (all || variant == "context")  pipeline_variant("context",  rag::pipeline::Pipeline::context());

    std::printf("\nRead nDCG@10 as the headline. R@100 is the ceiling any reranking\n"
                "policy could reach: when R@100 >> nDCG@10, the right documents are\n"
                "being retrieved but not ordered into the top-k, and the lever is\n"
                "ranking, not recall.\n");
    return 0;
}
