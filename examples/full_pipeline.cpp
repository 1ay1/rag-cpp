// examples/full_pipeline.cpp — assemble a maximal retrieval pipeline by hand.
//
// Demonstrates: custom embedder, PRF query expansion, hybrid retrieval with
// weighted RRF, metadata pre-filtering, a local cross-encoder reranker,
// parent-document stitching, persistence, and reload.

#include <cstdio>
#include <rag/rag.hpp>

int main() {
    using namespace rag;

    // Corpus config: force HNSW so the ANN path is exercised, tune fusion later.
    index::CorpusConfig cfg;
    cfg.hnsw_threshold = 1;   // build HNSW even for a tiny corpus (demo)
    Engine engine(cfg);
    engine.with_embedder(dense::AnyEmbedder{dense::HashEmbedder{256}});

    engine.add("ann.md",
        "# Approximate Nearest Neighbours\n\nHNSW builds a layered proximity graph "
        "for logarithmic-time vector search. Binary quantization speeds the walk.",
        {{"section","index"}});
    engine.add("bm25.md",
        "# Lexical Retrieval\n\nBM25 scores documents by term frequency and inverse "
        "document frequency, catching exact keyword matches embeddings miss.",
        {{"section","lexical"}});
    engine.add("fusion.md",
        "# Fusion\n\nReciprocal rank fusion combines lexical and dense rankings "
        "without score normalization, the most robust hybrid retrieval method.",
        {{"section","fusion"}});
    engine.add("risotto.md",
        "# Risotto\n\nToast arborio rice, then add warm stock one ladle at a time.",
        {{"section","cooking"}});
    engine.build();

    // A local cross-encoder stand-in: reward passages sharing query terms.
    rerank::ScoreFnReranker ce([](std::string_view q, std::string_view p) {
        int hits = 0;
        std::string qs(q);
        for (auto& w : {std::string("nearest"), std::string("neighbour"), std::string("search")})
            if (p.find(w) != std::string_view::npos) ++hits;
        return static_cast<float>(hits);
    });

    // Assemble the full funnel.
    pipeline::HybridRetrieveConfig hcfg;
    hcfg.bm25_weight = 1.0f;
    hcfg.dense_weight = 1.2f;              // trust the dense side slightly more
    pipeline::Pipeline p;
    p.add(std::make_shared<pipeline::PrfExpandStage>())
     .add(std::make_shared<pipeline::HybridRetrieveStage>(hcfg))
     .add(std::make_shared<pipeline::FilterStage>())
     .add(rerank::make_rerank_stage(rerank::AnyReranker{ce}, /*top_n*/10, /*blend*/0.5f))
     .add(std::make_shared<pipeline::ParentStitchStage>(2))
     .add(std::make_shared<pipeline::TopKStage>());
    engine.with_pipeline(std::move(p));

    // Query with a metadata pre-filter that excludes the cooking doc.
    index::MetaFilter only_technical = [](const Metadata& m) {
        auto it = m.find("section");
        return it != m.end() && it->second != "cooking";
    };

    std::vector<std::string> trace;
    auto results = engine.search("how does nearest neighbour search work", 3, only_technical, &trace);
    if (!results) { std::printf("search failed: %s\n", results.error().message.c_str()); return 1; }

    std::printf("=== Results ===\n");
    for (const auto& r : *results)
        std::printf("  [%.3f] %-12s %.70s...\n", r.score.get(), r.uri.c_str(), r.text.c_str());

    std::printf("\n=== Pipeline trace ===\n");
    for (const auto& t : trace) std::printf("  %s\n", t.c_str());

    // Persist + reload.
    const char* path = "/tmp/ragcpp_full.ragdb";
    if (auto s = engine.save(path); !s) { std::printf("save failed\n"); return 1; }
    auto reloaded = index::Corpus::load(path);
    std::printf("\nReloaded corpus: %zu docs, %zu chunks\n",
                reloaded ? reloaded->document_count() : 0,
                reloaded ? reloaded->chunk_count() : 0);
    std::remove(path);
    return 0;
}
