#pragma once
// rag/fusion/fuse.hpp — rank/score fusion of multiple retrievers.
//
// Hybrid retrieval's payoff step: combine a lexical (BM25) ranked list and a
// dense (cosine) ranked list into one. Three families, in increasing order of
// how much of the retrievers' output they actually use:
//
//   • Reciprocal Rank Fusion (RRF): score = Σ_lists 1/(k + rank). Scale-free —
//     it only reads RANK, so BM25's unbounded scores and cosine's [-1,1] fuse
//     without normalization. The classic robust default (Cormack et al. 2009).
//     We also support per-list WEIGHTS for weighted RRF.
//
//   • Relative Score Fusion (RSF): min-max normalize each list's scores to
//     [0,1] using the OBSERVED range, then take a weighted sum. Keeps score
//     MAGNITUDE information RRF discards.
//
//   • Convex combination (CC / TM2C2): the same weighted sum, but normalized
//     against THEORETICAL score bounds rather than observed ones. This is the
//     recommended default — see below.
//
// WHY CONVEX COMBINATION IS THE DEFAULT
//
// Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval"
// (ACM TOIS 42(1), 2023; arXiv:2210.11934) re-examined the widely-cited claim
// that RRF is the safer choice, and found the opposite:
//
//   • "TM2C2 significantly outperforms RRF on all datasets in terms of NDCG,
//     and does generally better in terms of Recall" — in BOTH in-domain and
//     out-of-domain (zero-shot) settings.
//   • RRF is NOT the parameter-free method it is usually sold as: they "find
//     RRF to be sensitive to its parameters", and taking a parametric view of
//     it costs one free parameter PER FUSED LIST — always more than the single
//     α a convex combination needs.
//   • RRF discards the score distribution entirely. Two documents ranked 3rd
//     and 4th contribute the same fused difference whether their true scores
//     were 0.9/0.89 or 0.9/0.02. That information is what a convex combination
//     keeps, and it is exactly the information a hybrid system has that a
//     single retriever does not.
//   • A convex combination is sample-efficient: α transfers across domains, and
//     they report α = 0.8 working well across their in-domain datasets.
//
// WHY *THEORETICAL* MIN-MAX (the "TM" in TM2C2)
//
// The obvious normalization is per-query min-max over the retrieved set (what
// `rsf` does). Its flaw is that both endpoints are statistics of the candidate
// set, so the mapping from raw score to normalized score changes from query to
// query: the worst document in EVERY result set is pinned to exactly 0 and the
// best to exactly 1, no matter how good or bad either actually was. A query
// where everything is irrelevant looks identical to one where everything is
// excellent.
//
// TM2C2 removes ONE of the two data-dependent endpoints — the minimum, and
// ONLY the minimum. In the authors' words, φ_TMM "has one fewer data-dependent
// statistic in the transformation (i.e., minimum score in the retrieved set is
// replaced with minimum feasible value regardless of the candidate set)",
// which is why TM2C2-normalized scores are more stable across datasets.
//
//     φ_TMM(s) = (s - theoretical_min) / (observed_max - theoretical_min)
//
// The asymmetry is deliberate and it matters. Substituting a theoretical
// MAXIMUM as well is tempting — cosine is bounded above by 1 — but it is
// actively harmful, because a theoretical bound that is never approached
// compresses that retriever's scores into a narrow sub-interval of [0,1] and
// silently strips it of influence. Measured on this codebase: over a realistic
// top-100 candidate set, cosine spans ~0.30 of raw score. Normalized against
// the theoretical [-1, 1] that becomes 0.15 of influence per unit weight,
// while BM25 (normalized against its observed max) spans the full 1.0 — so the
// dense list needs α ≈ 0.87 merely to TIE the lexical one, and α stops meaning
// what it should. Against the observed max, both lists span [0,1] and α is
// interpretable again.
//
// So for the two retrievers here:
//
//   BM25 — theoretical_min = 0. A document matching no query term scores 0 and
//     no per-term contribution is negative. Ceiling: observed.
//
//   cosine — theoretical_min = -1 by Cauchy-Schwarz on unit vectors. Ceiling:
//     observed, for the reason above.
//
// Where a theoretical minimum genuinely does not exist, the paper falls back
// to empirical min-max and calls it M2C2. `convex_combination` does the same
// automatically: leave `theoretical_min` unset and it degrades to M2C2 for
// that list rather than inventing a bound.

#include <optional>
#include <span>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::fusion {

struct RankedList {
    std::vector<Hit> hits;   // assumed sorted best-first
    float            weight = 1.0f;

    // Theoretical score bounds for the retriever that produced this list, used
    // by `convex_combination`. Unset => that endpoint is taken from the
    // observed candidate set instead (the M2C2 fallback).
    //
    // These are properties of the SCORING FUNCTION, not of the results: a
    // retriever knows them without looking at what it returned, which is the
    // entire point.
    std::optional<float> theoretical_min{};
    std::optional<float> theoretical_max{};
};

// Bounds for the retrievers rag-cpp ships, so callers do not have to restate
// them (and cannot get them subtly wrong).
//
// BM25: floor of 0 — a document sharing no term with the query contributes
// nothing, and no per-term contribution is negative. No principled ceiling.
[[nodiscard]] inline RankedList bm25_list(std::vector<Hit> hits, float weight = 1.0f) {
    return RankedList{.hits = std::move(hits), .weight = weight,
                      .theoretical_min = 0.0f, .theoretical_max = std::nullopt};
}

// Cosine over unit-normalized vectors: bounded below by -1 (Cauchy-Schwarz).
// The upper bound of 1 is deliberately NOT declared — see the header comment:
// a ceiling that real scores never approach would compress this list's
// influence and distort the meaning of α.
[[nodiscard]] inline RankedList cosine_list(std::vector<Hit> hits, float weight = 1.0f) {
    return RankedList{.hits = std::move(hits), .weight = weight,
                      .theoretical_min = -1.0f, .theoretical_max = std::nullopt};
}

struct RrfParams {
    float k = 60.0f;         // rank damping constant (Cormack default 60)
};

struct ConvexParams {
    // Weight on the FIRST list, with (1-alpha) going to the second: the
    // f_Convex = α·f_Sem + (1-α)·f_Lex of the paper. Only consulted when
    // exactly two lists are fused; with any other count each list's own
    // `weight` is used instead, so the same function generalizes beyond the
    // lexical/dense pair.
    //
    // 0.8 is the value Bruch et al. tuned on their in-domain validation splits
    // and then held FIXED across every out-of-domain dataset, where it still
    // beat RRF. It is a starting point, not a constant of nature: α is cheap to
    // tune (the paper's headline is that it is sample-efficient), and `ragcpp
    // eval` can sweep it against your own qrels.
    //
    // What the choice is actually sensitive to is the RELATIVE STRENGTH of the
    // two retrievers on your corpus. Sweeping α on a synthetic benchmark where
    // the lexical side was made artificially strong (queries containing an
    // exact, unique gold term) put the optimum near 0.1-0.3 and made α ≥ 0.8
    // clearly worse — the mirror image of the paper's setting, where the
    // semantic retriever is a trained encoder and carries more of the signal.
    // Neither number is universal. If retrieval quality matters to you, sweep
    // α once against your own labelled queries; it is one parameter and a
    // handful of examples suffice.
    float alpha = 0.8f;

    // Which list α applies to. The pipeline pushes lexical first, so the dense
    // list is index 1; the paper's α weights the SEMANTIC side, hence the
    // default. Ignored unless exactly two lists are fused.
    std::size_t alpha_list = 1;
};

// Reciprocal Rank Fusion (optionally weighted). Returns fused hits, best-first,
// truncated to `top_k` (0 = keep all).
[[nodiscard]] std::vector<Hit>
rrf(std::span<const RankedList> lists, RrfParams params = {}, std::size_t top_k = 0);

// Relative Score Fusion: per-list EMPIRICAL min-max normalization + weighted
// sum. Kept for comparison and for retrievers with no known bounds.
[[nodiscard]] std::vector<Hit>
rsf(std::span<const RankedList> lists, std::size_t top_k = 0);

// Convex combination with theoretical min-max normalization (TM2C2).
// Falls back to the observed value for any endpoint a list does not declare,
// which is the paper's M2C2 variant.
//
// A document missing from one list is scored 0 on that list — i.e. treated as
// "that retriever assigned it the minimum feasible score". This is the right
// reading of a truncated candidate list under theoretical normalization, and
// it is the reason the fusion is meaningful even when the two retrievers
// return largely disjoint sets.
[[nodiscard]] std::vector<Hit>
convex_combination(std::span<const RankedList> lists, ConvexParams params = {},
                   std::size_t top_k = 0);

} // namespace rag::fusion
