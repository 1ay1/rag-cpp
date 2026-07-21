// examples/quickstart.cpp — the smallest useful rag-cpp program.

#include <cstdio>
#include <rag/rag.hpp>

int main() {
    rag::Engine engine;
    engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{256}});

    engine.add("intro.md",
        "# rag-cpp\n\nrag-cpp is a type-theoretic retrieval engine. It fuses BM25 "
        "lexical scoring with dense vector search using reciprocal rank fusion.");
    engine.add("hnsw.md",
        "# ANN\n\nThe HNSW index gives logarithmic-time nearest-neighbour search "
        "over embeddings, with binary quantization and Matryoshka truncation.");
    engine.add("unrelated.md",
        "# Cooking\n\nTo make a good risotto, toast the rice before adding stock.");

    if (auto b = engine.build(); !b) {
        std::printf("build failed: %s\n", b.error().message.c_str());
        return 1;
    }

    std::vector<std::string> trace;
    auto results = engine.search("how does nearest neighbour search work", 2, {}, &trace);
    if (!results) { std::printf("search failed: %s\n", results.error().message.c_str()); return 1; }

    std::printf("Query: how does nearest neighbour search work\n\n");
    for (const auto& r : *results) {
        std::printf("  [%.3f] %s  (lines %u-%u)\n", r.score.get(), r.uri.c_str(),
                    r.start_line, r.end_line);
        std::printf("        %.80s...\n", r.text.c_str());
    }
    std::printf("\nPipeline trace:\n");
    for (const auto& t : trace) std::printf("  %s\n", t.c_str());
    return 0;
}
