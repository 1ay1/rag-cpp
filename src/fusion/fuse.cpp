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

} // namespace rag::fusion
