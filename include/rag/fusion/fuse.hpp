#pragma once
// rag/fusion/fuse.hpp — rank/score fusion of multiple retrievers.
//
// Hybrid retrieval's payoff step: combine a lexical (BM25) ranked list and a
// dense (cosine) ranked list into one. Two families:
//
//   • Reciprocal Rank Fusion (RRF): score = Σ_lists 1/(k + rank). Scale-free —
//     it only reads RANK, so BM25's unbounded scores and cosine's [-1,1] fuse
//     without normalization. The robust default (Cormack et al. 2009). We also
//     support per-list WEIGHTS for weighted RRF.
//
//   • Relative Score Fusion (RSF): min-max normalize each list's scores to
//     [0,1], then take a weighted sum. Keeps score MAGNITUDE information RRF
//     discards — better when the two retrievers are well-calibrated.

#include <span>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::fusion {

struct RankedList {
    std::vector<Hit> hits;   // assumed sorted best-first
    float            weight = 1.0f;
};

struct RrfParams {
    float k = 60.0f;         // rank damping constant (Cormack default 60)
};

// Reciprocal Rank Fusion (optionally weighted). Returns fused hits, best-first,
// truncated to `top_k` (0 = keep all).
[[nodiscard]] std::vector<Hit>
rrf(std::span<const RankedList> lists, RrfParams params = {}, std::size_t top_k = 0);

// Relative Score Fusion: per-list min-max normalization + weighted sum.
[[nodiscard]] std::vector<Hit>
rsf(std::span<const RankedList> lists, std::size_t top_k = 0);

} // namespace rag::fusion
