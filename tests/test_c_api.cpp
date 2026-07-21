// tests/test_c_api.cpp — exercise the stable C ABI end-to-end.

#include <cstdio>
#include <cstring>

#include "rag/c/rag.h"

static int failures = 0;
static int checks = 0;
#define C_CHECK(cond) do { ++checks; if(!(cond)){ ++failures; std::printf("  C FAIL: %s (line %d)\n", #cond, __LINE__);} } while(0)

int main() {
    std::printf("C API smoke test (rag_version=%s)\n", rag_version());

    rag_engine* eng = rag_engine_new();
    C_CHECK(eng != nullptr);

    rag_embedder* emb = rag_embedder_hash(128);
    C_CHECK(emb != nullptr);
    C_CHECK(rag_engine_set_embedder(eng, emb) == RAG_OK);

    uint32_t id = 0;
    const char* keys[] = {"topic"};
    const char* vals[] = {"biology"};
    C_CHECK(rag_engine_add(eng, "cell.md",
        "The mitochondria is the powerhouse of the cell producing energy.",
        "Cell", keys, vals, 1, &id) == RAG_OK);
    C_CHECK(rag_engine_add(eng, "rust.md",
        "Rust is a systems programming language focused on memory safety.",
        "Rust", nullptr, nullptr, 0, nullptr) == RAG_OK);
    C_CHECK(rag_engine_build(eng) == RAG_OK);

    rag_results* res = nullptr;
    C_CHECK(rag_engine_search(eng, "cell energy", 2, nullptr, nullptr, 0, &res) == RAG_OK);
    C_CHECK(res != nullptr);
    C_CHECK(rag_results_count(res) > 0);
    if (rag_results_count(res) > 0) {
        std::printf("  top uri: %s  score=%.3f\n", rag_results_uri(res, 0), rag_results_score(res, 0));
        C_CHECK(std::strcmp(rag_results_uri(res, 0), "rust.md") != 0); // biology should win
    }
    rag_results_free(res);

    // Metadata filter through the C API.
    rag_results* fres = nullptr;
    const char* fk[] = {"topic"};
    const char* fv[] = {"biology"};
    C_CHECK(rag_engine_search(eng, "language", 5, fk, fv, 1, &fres) == RAG_OK);
    for (size_t i = 0; i < rag_results_count(fres); ++i)
        C_CHECK(std::strcmp(rag_results_uri(fres, i), "cell.md") == 0);
    rag_results_free(fres);

    // Save + load round-trip.
    const char* path = "/tmp/ragcpp_c_api.ragdb";
    C_CHECK(rag_engine_save(eng, path) == RAG_OK);
    rag_status st = RAG_ERR_UNKNOWN;
    rag_engine* loaded = rag_engine_load(path, &st);
    C_CHECK(st == RAG_OK);
    C_CHECK(loaded != nullptr);
    if (loaded) {
        rag_results* lr = nullptr;
        // Loaded corpus has no live embedder; lexical still works.
        C_CHECK(rag_engine_search(loaded, "memory safety", 2, nullptr, nullptr, 0, &lr) == RAG_OK);
        C_CHECK(rag_results_count(lr) > 0);
        rag_results_free(lr);
        rag_engine_free(loaded);
    }
    std::remove(path);

    rag_engine_free(eng);
    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
