// examples/advanced.cpp — the cutting-edge retrieval stack, end to end.
//
// Demonstrates, on a small corpus: learned-sparse (SPLADE-style) retrieval,
// ColBERT late-interaction reranking, a RAPTOR summary tree, HyDE query
// transformation, and Corrective-RAG grading. Runs with NO model and NO network
// (everything degrades gracefully to deterministic defaults).
//
//   build/examples/ragcpp_advanced

#include <cstdio>
#include <rag/rag.hpp>

int main() {
    rag::Engine engine;
    const char* docs[] = {
        "HNSW is a graph-based approximate nearest neighbour index: a navigable "
        "small-world graph over vectors giving logarithmic search.",
        "BM25 is the Okapi lexical ranker: term-frequency saturation times "
        "inverse document frequency over an inverted index.",
        "Dense retrieval embeds passages with a transformer and ranks by cosine "
        "similarity; it captures semantics that keyword search misses.",
        "Reciprocal rank fusion combines lexical and dense rankings scale-free by "
        "reading only ranks, not raw scores.",
        "Cross-encoder reranking jointly encodes the query and passage for the "
        "highest accuracy at the cost of a forward pass per candidate.",
        "Late interaction (ColBERT) embeds tokens independently and scores by "
        "MaxSim, a middle ground between bi-encoders and cross-encoders.",
        "Risotto: toast arborio rice, then add warm stock gradually while "
        "stirring until creamy; finish with butter and parmesan.",
    };
    for (int i = 0; i < 7; ++i) engine.add("d" + std::to_string(i), docs[i]);
    engine.build();

    const char* Q = "how does approximate nearest neighbour vector search work";
    std::printf("Query: %s\n\n", Q);

    // ── Learned sparse (SPLADE-style) ─────────────────────────────────────────
    if (auto idx = rag::sparse::SpladeIndex::build(engine.corpus())) {
        std::printf("SPLADE learned-sparse (vocab %zu):\n", idx->vocab_size());
        for (const auto& h : idx->search(Q, 3)) {
            auto r = engine.corpus().resolve(h);
            std::printf("  [%.3f] %-4s %.55s\n", h.score.get(), r.uri.c_str(), r.text.c_str());
        }
    }

    // ── ColBERT late-interaction rerank of the BM25 pool ──────────────────────
    std::printf("\nColBERT late-interaction rerank:\n");
    auto pool = engine.corpus().lexical_search(Q, 5);
    rag::late::ColbertReranker cb(rag::late::hashed_token_embedder(96));
    (void)cb.rerank_hits(Q, pool, [&](const rag::Hit& h) {
        auto r = engine.corpus().resolve(h); return r.text;
    });
    for (const auto& h : pool) {
        auto r = engine.corpus().resolve(h);
        std::printf("  [%.3f] %-4s %.55s\n", h.score.get(), r.uri.c_str(), r.text.c_str());
    }

    // ── RAPTOR summary tree ───────────────────────────────────────────────────
    rag::raptor::RaptorConfig rc; rc.cluster_size = 3; rc.max_levels = 3; rc.sim_threshold = 0.0f;
    if (auto tree = rag::raptor::RaptorTree::build(engine.corpus(), rc)) {
        std::printf("\nRAPTOR tree: %zu nodes across %zu levels\n",
                    tree->node_count(), tree->level_count());
        if (auto res = tree->retrieve(engine.corpus(), Q, 3))
            for (const auto& r : *res)
                std::printf("  L%u [%.3f] %.55s\n", r.level, r.score.get(),
                            tree->node_text(r.node).data());
    }

    // ── HyDE (with a canned hypothetical, no LLM required) ────────────────────
    std::printf("\nHyDE query transformation:\n");
    auto gen = [](std::string_view) -> rag::Result<std::vector<std::string>> {
        return std::vector<std::string>{
            "Approximate nearest neighbour search uses HNSW, a navigable "
            "small-world graph, to find close vectors in logarithmic time."};
    };
    if (auto h = rag::query::hyde_search(engine.corpus(), Q, 3, gen))
        for (const auto& hit : *h) {
            auto r = engine.corpus().resolve(hit);
            std::printf("  [%.3f] %-4s %.55s\n", hit.score.get(), r.uri.c_str(), r.text.c_str());
        }

    // ── Corrective RAG grading ────────────────────────────────────────────────
    std::printf("\nCorrective-RAG grade:\n");
    auto c = rag::crag::correct(engine.corpus(), Q, pool);
    std::printf("  action=%s confidence=%.3f kept=%zu strips\n",
                std::string(rag::crag::to_string(c.action)).c_str(),
                c.confidence, c.knowledge.size());
    return 0;
}
