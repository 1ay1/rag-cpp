#pragma once
// rag/query/hyde.hpp — HyDE + multi-query query transformation.
//
// HyDE (Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance
// Labels"): instead of embedding the QUERY (which lives in a different
// distribution than documents), ask an LLM to HALLUCINATE a hypothetical answer
// document, then embed THAT and retrieve its neighbours. The fake document,
// though possibly wrong in details, lands near real relevant documents in
// embedding space — closing the query-document asymmetry gap with zero training.
//
// We also provide multi-query expansion (retrieve for several LLM-generated
// paraphrases and fuse) — the RAG-Fusion pattern.
//
// The LLM is an injected seam (`Generator`), so the library never bundles a
// model. Without a generator, HyDE degrades to embedding the raw query (i.e. a
// normal dense search) — the graceful-degradation contract.

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/fusion/fuse.hpp"
#include "rag/index/corpus.hpp"

namespace rag::query {

// The generator seam: a prompt → one or more text completions.
using Generator = std::function<Result<std::vector<std::string>>(std::string_view prompt)>;

struct HydeConfig {
    std::size_t hypotheticals = 1;    // number of fake docs to generate + embed
    bool        include_query = true; // also embed the raw query and fuse
    std::string instruction =
        "Write a short passage that directly answers the question.";
    fusion::RrfParams rrf{};          // fusion of the per-hypothetical rankings
};

// Run HyDE: generate hypothetical documents for `query`, embed each, retrieve
// their neighbours from the corpus, and RRF-fuse into one ranked list of `k`.
// Requires the corpus to have an embedder (else Errc::unavailable).
[[nodiscard]] Result<std::vector<Hit>>
hyde_search(const index::Corpus& corpus, std::string_view query, std::size_t k,
            const Generator& generate, HydeConfig cfg = {});

// Multi-query / RAG-Fusion: generate `n` paraphrases of the query, retrieve for
// each (hybrid), and RRF-fuse. Improves recall on under-specified queries.
[[nodiscard]] Result<std::vector<Hit>>
multi_query_search(const index::Corpus& corpus, std::string_view query, std::size_t k,
                   const Generator& generate, std::size_t n = 3,
                   fusion::RrfParams rrf = {});

} // namespace rag::query
