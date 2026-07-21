#pragma once
// rag/rerank/mmr.hpp — Maximal Marginal Relevance diversity reranking.
//
// Top-k by pure relevance often returns k near-duplicates: the best passage and
// four paraphrases of it. MMR (Carbonell & Goldstein 1998) greedily builds the
// result set to trade relevance against NOVELTY:
//
//     MMR = argmax_{d ∈ R∖S} [ λ·rel(d,q) − (1−λ)·max_{s ∈ S} sim(d,s) ]
//
// At each step it picks the candidate that is relevant to the query yet
// dissimilar to what's already chosen. λ=1 is pure relevance (no diversity);
// λ=0 is pure diversity. The default λ=0.5 balances both.
//
// Similarity between candidates uses their chunk embeddings (cosine) when the
// corpus is dense, else a lexical Jaccard over chunk terms — the same graceful
// degradation as the rest of the library.

#include <cstddef>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"

namespace rag::rerank {

struct MmrConfig {
    float       lambda = 0.5f;   // relevance↔diversity trade-off in [0,1]
    std::size_t k      = 10;     // final result size
};

// Re-order `candidates` (assumed relevance-sorted, carrying relevance in
// .score) by MMR against each other, returning the top-k diversified hits.
[[nodiscard]] std::vector<Hit>
mmr(const index::Corpus& corpus, std::span<const Hit> candidates, MmrConfig cfg = {});

// A pipeline stage form: diversify the running candidate set in place.
[[nodiscard]] pipeline::StagePtr
make_mmr_stage(float lambda = 0.5f, std::string label = "mmr");

} // namespace rag::rerank
