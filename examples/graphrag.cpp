// examples/graphrag.cpp — GraphRAG + RALM assembly end to end.
//
// Builds a tiny linked corpus, constructs the document graph, runs BOTH graph
// local search (PPR expansion) and global search (community summaries), then
// shows the RALM assemblies: REPLUG ensemble weights, RETRO chunked neighbours,
// and a grounded, source-attributed prompt.
//
//   build/examples/ragcpp_graphrag

#include <cstdio>
#include <rag/rag.hpp>

int main() {
    rag::Engine engine;

    // A small corpus where documents LINK to one another (markdown) and cluster
    // by topic. No embedder → similarity edges fall back to lexical Jaccard and
    // retrieval degrades gracefully to BM25. Attach an embedder for dense edges.
    engine.add("intro.md",
        "Retrieval-augmented generation grounds a language model on external "
        "documents. See [dense](dense.md) and [graph](graph.md) for the engine internals.");
    engine.add("dense.md",
        "The dense retriever embeds passages and ranks them by cosine similarity "
        "over an HNSW index. It pairs with [lexical](lexical.md) BM25 scoring.");
    engine.add("lexical.md",
        "BM25 is the Okapi lexical ranker: term frequency saturation plus inverse "
        "document frequency. It fuses with dense results via reciprocal rank fusion.");
    engine.add("graph.md",
        "GraphRAG builds a document graph — link edges and similarity edges — then "
        "detects communities and summarizes each cluster for corpus-level questions.");
    engine.add("cooking.md",
        "Risotto: toast arborio rice, then ladle warm stock gradually while stirring "
        "until creamy. Finish with butter and parmesan off the heat.");
    engine.build();

    // ── Document graph ─────────────────────────────────────────────────────────
    auto g = engine.graph();
    if (!g) { std::printf("graph error: %s\n", (*g ? "" : g.error().message.c_str())); return 1; }
    std::printf("Graph: %zu nodes, %zu edges, %zu communities\n",
                (*g)->node_count(), (*g)->edge_count(), (*g)->communities().size());
    for (const auto& c : (*g)->communities()) {
        std::printf("  community %u (%zu docs, density %.2f): %.90s%s\n",
                    c.id, c.docs.size(), c.density,
                    c.summary.c_str(), c.summary.size() > 90 ? "…" : "");
    }

    // ── Local search: PPR expansion over the graph ────────────────────────────
    std::printf("\nLocal search  'BM25 lexical ranking':\n");
    if (auto hits = engine.graph_local("BM25 lexical ranking", 4))
        for (const auto& h : *hits)
            std::printf("  [%.3f] %-12s %.60s\n", h.score.get(), h.uri.c_str(), h.text.c_str());

    // ── Global search: rank community summaries ───────────────────────────────
    std::printf("\nGlobal search 'how does the retrieval engine fit together':\n");
    if (auto hits = engine.graph_global("how does the retrieval engine fit together", 3))
        for (const auto& h : *hits)
            std::printf("  [%.3f] %-12s %.60s\n", h.score.get(), h.uri.c_str(), h.text.c_str());

    // ── RALM: REPLUG / RAG ensemble weights ───────────────────────────────────
    std::printf("\nREPLUG ensemble weights for 'dense vector search':\n");
    auto pool = engine.corpus().lexical_search("dense vector search", 4);
    auto weighted = rag::ralm::ensemble_weights(engine.corpus(), pool, /*temp=*/0.5f);
    for (const auto& w : weighted)
        std::printf("  p(z|x)=%.3f  %.55s\n", w.weight, w.text.c_str());

    // ── RALM: RETRO chunked-neighbour retrieval ───────────────────────────────
    std::printf("\nRETRO neighbours for a two-stride input:\n");
    rag::ralm::RetroConfig rc; rc.stride = 6; rc.neighbours = 1;
    if (auto rows = rag::ralm::retro_retrieve(engine.corpus(),
            "dense retriever cosine similarity fuses with lexical bm25 ranking", rc))
        for (const auto& row : *rows)
            for (const auto& nb : row.neighbours)
                std::printf("  stride='%.24s…' → nbr[%.3f] %.40s\n",
                            row.stride_text.c_str(), nb.score.get(), nb.neighbour_text.c_str());

    // ── RALM: grounded, source-attributed prompt (In-Context RALM / IBM RAG) ──
    std::printf("\nAssembled grounded prompt:\n");
    auto prompt = rag::ralm::assemble_prompt(
        "How does dense retrieval combine with lexical?", weighted,
        "Answer the question using only the numbered context. Cite sources.");
    std::printf("%s\n", prompt.c_str());
    return 0;
}
