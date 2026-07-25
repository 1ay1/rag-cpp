// rag/fusion/fuse.cpp — RRF and RSF fusion implementations.

#include "rag/fusion/fuse.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace rag::fusion {

std::vector<Hit> rrf(std::span<const RankedList> lists, RrfParams params, std::size_t top_k) {
    std::unordered_map<std::uint32_t, float> acc;
    for (const auto& list : lists) {
        for (std::size_t rank = 0; rank < list.hits.size(); ++rank) {
            std::uint32_t id = list.hits[rank].chunk.get();
            acc[id] += list.weight * (1.0f / (params.k + static_cast<float>(rank) + 1.0f));
        }
    }
    std::vector<Hit> out;
    out.reserve(acc.size());
    for (auto& [id, s] : acc) out.push_back(Hit{ChunkId{id}, Score{s}});
    std::sort(out.begin(), out.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    if (top_k && out.size() > top_k) out.resize(top_k);
    return out;
}

std::vector<Hit> rsf(std::span<const RankedList> lists, std::size_t top_k) {
    std::unordered_map<std::uint32_t, float> acc;
    for (const auto& list : lists) {
        if (list.hits.empty()) continue;
        float lo = std::numeric_limits<float>::max();
        float hi = std::numeric_limits<float>::lowest();
        for (const auto& h : list.hits) { lo = std::min(lo, h.score.get()); hi = std::max(hi, h.score.get()); }
        float range = hi - lo;
        for (const auto& h : list.hits) {
            float norm = range > 1e-9f ? (h.score.get() - lo) / range : 1.0f;
            acc[h.chunk.get()] += list.weight * norm;
        }
    }
    std::vector<Hit> out;
    out.reserve(acc.size());
    for (auto& [id, s] : acc) out.push_back(Hit{ChunkId{id}, Score{s}});
    std::sort(out.begin(), out.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    if (top_k && out.size() > top_k) out.resize(top_k);
    return out;
}

std::vector<Hit> convex_combination(std::span<const RankedList> lists, ConvexParams params,
                                    std::size_t top_k) {
    // α applies only to the canonical two-list (lexical, dense) case. With any
    // other arity there is no single α to speak of, so each list's own weight
    // governs — which keeps this usable for 3+ retrievers without pretending
    // the paper's parameterization still applies.
    const bool use_alpha = lists.size() == 2;

    std::unordered_map<std::uint32_t, float> acc;
    for (std::size_t li = 0; li < lists.size(); ++li) {
        const auto& list = lists[li];
        if (list.hits.empty()) continue;

        // Observed range, needed only for endpoints the retriever could not
        // declare a priori.
        float obs_lo = std::numeric_limits<float>::max();
        float obs_hi = std::numeric_limits<float>::lowest();
        for (const auto& h : list.hits) {
            obs_lo = std::min(obs_lo, h.score.get());
            obs_hi = std::max(obs_hi, h.score.get());
        }

        // THE TM2C2 step: prefer the theoretical bound, fall back to the
        // observed one. Using the declared minimum is what stops the mapping
        // from being re-derived per query — a document scoring 0.2 normalizes
        // to the same value whether or not something better happened to be
        // retrieved alongside it.
        const float lo = list.theoretical_min.value_or(obs_lo);
        float       hi = list.theoretical_max.value_or(obs_hi);
        // A theoretical floor with an observed ceiling can invert if every
        // retrieved score sits below the floor (or all scores are equal).
        if (hi <= lo) hi = lo + 1.0f;
        const float inv_range = 1.0f / (hi - lo);

        float w = list.weight;
        if (use_alpha)
            w = (li == params.alpha_list) ? params.alpha : (1.0f - params.alpha);

        for (const auto& h : list.hits) {
            // Clamp: a score may legitimately exceed an observed max used as a
            // stand-in, and normalized contributions outside [0,1] would let
            // one retriever silently outvote its weight.
            const float norm = std::clamp((h.score.get() - lo) * inv_range, 0.0f, 1.0f);
            acc[h.chunk.get()] += w * norm;
        }
    }

    std::vector<Hit> out;
    out.reserve(acc.size());
    for (auto& [id, s] : acc) out.push_back(Hit{ChunkId{id}, Score{s}});
    // Ties broken by id so fusion is deterministic regardless of hash order.
    std::sort(out.begin(), out.end(), [](const Hit& a, const Hit& b) {
        if (a.score.get() != b.score.get()) return a.score.get() > b.score.get();
        return a.chunk.get() < b.chunk.get();
    });
    if (top_k && out.size() > top_k) out.resize(top_k);
    return out;
}

} // namespace rag::fusion
