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
//                      [--model=model.onnx --tokenizer=tokenizer.json]
//
// WITHOUT --model the corpus is lexical-only, so the `standard` pipeline fuses a
// single retriever and the dense half never runs — useful as a pure-BM25
// reference, but it is NOT what the engine does in production. WITH a model the
// dense retriever is live and `standard` performs real hybrid fusion, which is
// where the engine's actual quality lives (SciFact nDCG@10 0.6809 -> 0.7343).
// Requires -DRAGCPP_WITH_ONNX=ON.
// Variants (default: all):
//   lexical    corpus.lexical_search only (the published-BM25 comparison point)
//   standard   Engine + Pipeline::standard()   — hybrid → filter → rerank → topk
//   quality    Engine + Pipeline::quality()    — + MMR diversity
//   context    Engine + Pipeline::context()    — + ParentStitch small-to-big
//
// Every variant uses the SAME corpus, chunking, and embedder so the only thing
// that differs is the ranking policy under test.

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rag/dense/local_embedder.hpp"
#include "rag/engine.hpp"
#include "rag/eval/beir.hpp"
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"
#include "rag/rerank/reranker.hpp"

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

    // Attach a real embedding model when one is given. This is the difference
    // between measuring BM25 and measuring the ENGINE: with no embedder the
    // hybrid stage has only one retriever to fuse.
    const std::string model = opt(argc, argv, "--model", "");
    const std::string tok   = opt(argc, argv, "--tokenizer", "");
    if (!model.empty()) {
        rag::dense::LocalEmbedderConfig ec;
        ec.model_path     = model;
        ec.tokenizer_path = tok;
        ec.max_tokens     = 256;
        auto emb = rag::dense::OnnxEmbedder::load(ec);
        if (!emb) {
            std::printf("embedder error: %s\n", emb.error().message.c_str());
            return 1;
        }
        std::printf("embedder: %s (dim %zu)\n", model.c_str(), emb->dimension());
        corpus.set_embedder(rag::dense::AnyEmbedder{std::move(*emb)});
    } else {
        std::printf("embedder: none (LEXICAL ONLY — pass --model for hybrid)\n");
    }

    auto doc_uri = index_dataset(*ds, corpus);
    std::printf("indexed %zu chunks\n\n", corpus.chunk_count());

    // Optionally attach an in-process cross-encoder. Reranking is the accuracy
    // ceiling of the funnel: retrieval decides WHICH ~100 chunks are candidates,
    // the cross-encoder re-scores each (query,passage) jointly and reorders the
    // top-k. It only ever sees what retrieval already narrowed to, so the
    // measured lift is purely "better ordering of the same candidates".
    const std::string rr_model = opt(argc, argv, "--reranker", "");
    const std::string rr_tok   = opt(argc, argv, "--reranker-tokenizer", "");
    const std::size_t rr_topn  = std::strtoul(opt(argc, argv, "--rerank-topn", "100").c_str(), nullptr, 10);
    const float rr_blend = std::strtof(opt(argc, argv, "--rerank-blend", "1.0").c_str(), nullptr);
    std::optional<rag::rerank::AnyReranker> reranker;
    if (!rr_model.empty()) {
        rag::dense::LocalEmbedderConfig rc;
        rc.model_path = rr_model;
        rc.tokenizer_path = rr_tok;
        rc.max_tokens = 512;   // cross-encoders read query+passage; give room
        auto rr = rag::rerank::OnnxReranker::load(rc);
        if (!rr) { std::printf("reranker error: %s\n", rr.error().message.c_str()); return 1; }
        std::printf("reranker: %s (in-process cross-encoder, top-%zu, blend=%.2f)\n", rr_model.c_str(), rr_topn, rr_blend);
        reranker.emplace(rag::rerank::AnyReranker{std::move(*rr)});
    }

    // With a model attached, the pure-dense arm is worth reporting on its own:
    // hybrid should beat BOTH halves, and showing only the winner hides whether
    // fusion is actually contributing.
    if (corpus.has_embedder()) {
        run_variant("dense", *ds, cfg, [&](const std::string& q, std::size_t k) {
            auto h = corpus.dense_search(q, k);
            return h ? to_ranking(*h, corpus, doc_uri) : rag::eval::Ranking{};
        });
    }

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

    // The rerank arm: same retrieval, then an in-process cross-encoder reorders
    // the top-N. This is where hybrid retrieval crosses into SOTA — if the
    // cross-encoder cannot lift nDCG@10 above the retrieval-only number, it is
    // not earning its latency, and this measures exactly that. AnyReranker holds
    // a shared_ptr to the loaded model, so the copy into the stage is cheap.
    if (reranker) {
        auto std_rr = rag::pipeline::Pipeline::standard().add(
            rag::rerank::make_rerank_stage(*reranker, rr_topn, /*blend=*/rr_blend, "cross_encoder"));
        pipeline_variant("standard+rerank", std::move(std_rr));
    }

    std::printf("\nRead nDCG@10 as the headline. R@100 is the ceiling any reranking\n"
                "policy could reach: when R@100 >> nDCG@10, the right documents are\n"
                "being retrieved but not ordered into the top-k, and the lever is\n"
                "ranking, not recall.\n");
    return 0;
}
