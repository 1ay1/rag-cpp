// rag/fusion/fuse.cpp — RRF and RSF fusion implementations.

#include "rag/fusion/fuse.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

namespace rag::fusion {

namespace {

// Robust upper endpoint for min-max normalization.
//
// The naive ceiling is the raw observed maximum, but that is fragile: a SINGLE
// anomalous score (e.g. a rare query term whose IDF spikes BM25 far above the
// rest of the list) inflates `hi`, which compresses every other document's
// normalized score toward 0 and silently strips that retriever of nearly all
// its voting weight — the exact failure mode theoretical-min normalization was
// meant to avoid, reintroduced through the ceiling.
//
// The fix is a winsorized ceiling: sort the scores and take the value at the
// `p`-th percentile (default p99). Outliers above it are clamped to 1.0 by the
// caller (they are still the best documents, they just no longer distort the
// scale for everyone below). For short lists we fall back to the raw max, since
// there is no tail to trim. This is a small, self-contained change that removes
// a real tail-quality failure without touching the theoretical-min design.
float robust_ceiling(const std::vector<float>& sorted_desc, float raw_hi, float p) {
    const std::size_t n = sorted_desc.size();
    // Need enough samples for a percentile to mean anything; below this the raw
    // max IS the robust estimate.
    if (n < 8) return raw_hi;
    // sorted_desc is best-first. The p-th percentile from the top is at index
    // floor((1-p) * n), clamped into range. p=0.99 -> ~1% from the top.
    const float frac = std::clamp(1.0f - p, 0.0f, 1.0f);
    std::size_t idx = static_cast<std::size_t>(frac * static_cast<float>(n));
    if (idx >= n) idx = n - 1;
    return sorted_desc[idx];
}

} // namespace

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
    std::vector<float> scores;
    for (const auto& list : lists) {
        if (list.hits.empty()) continue;
        float lo = std::numeric_limits<float>::max();
        float raw_hi = std::numeric_limits<float>::lowest();
        scores.clear();
        scores.reserve(list.hits.size());
        for (const auto& h : list.hits) {
            const float s = h.score.get();
            lo = std::min(lo, s);
            raw_hi = std::max(raw_hi, s);
            scores.push_back(s);
        }
        // Winsorized ceiling: a single outlier no longer collapses the scale.
        // Sort descending so robust_ceiling can index from the top.
        std::sort(scores.begin(), scores.end(), std::greater<float>{});
        const float hi = robust_ceiling(scores, raw_hi, 0.99f);
        const float range = hi - lo;
        for (const auto& h : list.hits) {
            const float norm = range > 1e-9f
                ? std::clamp((h.score.get() - lo) / range, 0.0f, 1.0f)
                : 1.0f;
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
    std::vector<float> scores;
    for (std::size_t li = 0; li < lists.size(); ++li) {
        const auto& list = lists[li];
        if (list.hits.empty()) continue;

        // Observed range, needed only for endpoints the retriever could not
        // declare a priori.
        float obs_lo = std::numeric_limits<float>::max();
        float obs_hi_raw = std::numeric_limits<float>::lowest();
        scores.clear();
        scores.reserve(list.hits.size());
        for (const auto& h : list.hits) {
            const float s = h.score.get();
            obs_lo = std::min(obs_lo, s);
            obs_hi_raw = std::max(obs_hi_raw, s);
            scores.push_back(s);
        }
        // When the ceiling has to come from the candidate set (no declared
        // theoretical_max, which is the shipped case for both BM25 and cosine),
        // use a WINSORIZED max instead of the raw one. A single spiking score
        // (rare-term IDF blow-up) would otherwise inflate the ceiling and crush
        // every other document's contribution to near-zero, silently negating
        // this retriever's weight. Trimming the top ~1% fixes that; the spiking
        // docs are still clamped to 1.0 below, so they remain top-ranked.
        std::sort(scores.begin(), scores.end(), std::greater<float>{});
        const float obs_hi = robust_ceiling(scores, obs_hi_raw, 0.99f);

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
