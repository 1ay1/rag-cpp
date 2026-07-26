#pragma once
// rag/rerank/dartboard.hpp — Dartboard retrieval: optimise the RESULT SET, not
// each result.
//
// MMR (see mmr.hpp) subtracts a penalty: pick whatever is relevant and unlike
// what you already have. That is a heuristic, and it has a known failure mode —
// the penalty is a max over the chosen set, so a candidate is punished for
// resembling ONE selected item even when it would cover a part of the query
// nothing else covers.
//
// Dartboard (Pickett et al. 2024, "Better RAG using Relevant Information Gain")
// reframes the goal: a result set is good if, for each thing the user might have
// meant, SOMETHING in the set is close to it. Formally it greedily maximises
// relevant information gain — the expected best-match quality over a
// relevance-weighted distribution of intents:
//
//     gain(S) = Σ_i  w_i · max_{d ∈ S} sim(d, t_i)
//
// where t_i are the candidate "targets" (here: the retrieved candidates
// themselves, standing in for plausible intents) and w_i their relevance
// weights, softmax-normalised so a much better candidate dominates a marginal
// one. Adding a document is worth exactly the improvement it makes to the
// covered maximum, so a candidate that is the ONLY good cover for some intent is
// selected even if it looks like something already chosen — precisely the case
// MMR gets wrong.
//
// The practical differences from MMR:
//   * MMR asks "is this different from what I picked?"  (pairwise penalty)
//     Dartboard asks "does this cover something nothing else covers?" (marginal
//     gain over the whole set)
//   * MMR's lambda trades relevance against diversity. Dartboard's `sigma`
//     controls how sharply relevance concentrates: small sigma -> near-pure
//     relevance, large sigma -> intents spread out and coverage dominates.
//
// MEASURED on BEIR/SciFact with an otherwise identical pipeline (nDCG@10 / R@10):
//
//   standard, no diversification     0.6809   0.8212
//   quality  (MMR, lambda=0.5)       0.6744   0.8069
//   dartboard (relevance_weight 0.7) 0.6798   0.8212
//
// Any diversification costs something on a benchmark that rewards pure
// relevance, but Dartboard costs ~6x less than MMR (-0.0011 vs -0.0065) and
// does not give up Recall@10 at all. Prefer it when you need diversity.
//
// Same graceful degradation as MMR: cosine over chunk embeddings when the corpus
// is dense, lexical Jaccard otherwise.

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"

namespace rag::rerank {

struct DartboardConfig {
    // Softmax temperature over candidate relevance, forming the intent weights.
    // Smaller = the top candidates dominate the weighting (behaves closer to
    // pure relevance); larger = weight spreads across candidates and coverage
    // matters more. 0.5 is a reasonable middle on normalised scores.
    float       sigma = 0.5f;
    // How much a document's own relevance counts beside its coverage
    // contribution, in [0,1]. 1.0 ignores coverage entirely (pure relevance);
    // 0.0 selects purely for coverage. The default keeps relevance dominant,
    // because on general IR benchmarks diversity is usually a net cost — see
    // BENCHMARKS.md.
    float       relevance_weight = 0.7f;
    std::size_t k = 10;
};

// Re-order `candidates` (relevance-sorted, relevance in .score) by greedy
// relevant-information-gain, returning the top-k.
//
// O(k·n) similarity evaluations, like the cached MMR: adding a document can only
// raise each intent's covered maximum, so the running maxima are updated
// incrementally rather than recomputed.
[[nodiscard]] std::vector<Hit>
dartboard(const index::Corpus& corpus, std::span<const Hit> candidates,
          DartboardConfig cfg = {});

// Pipeline stage form. Like MMR's stage, this REORDERS rather than rescores, so
// it must not be wrapped in a RerankStage (which re-sorts by score afterwards
// and would undo the selection).
[[nodiscard]] pipeline::StagePtr
make_dartboard_stage(DartboardConfig cfg = {}, std::string label = "dartboard");

} // namespace rag::rerank
