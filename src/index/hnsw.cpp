// rag/index/hnsw.cpp — HNSW build + search + serialization.

#include "rag/index/hnsw.hpp"
#include "rag/dense/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <unordered_set>

namespace rag::index {

HnswIndex::HnswIndex(HnswConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
    if (cfg_.ml <= 0.0f) cfg_.ml = 1.0f / std::log(static_cast<float>(std::max<std::size_t>(cfg_.M, 2)));
}

int HnswIndex::random_level() {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng_);
    if (r <= 0.0f) r = 1e-9f;
    return static_cast<int>(-std::log(r) * cfg_.ml);
}

float HnswIndex::sim(std::size_t node, std::span<const float> q,
                     std::span<const std::uint64_t> q_bits) const {
    const Node& nd = nodes_[node];
    if (cfg_.binary && !q_bits.empty() && !nd.bits.empty()) {
        // Approximate similarity from Hamming distance over sign codes:
        // sim ≈ 1 - 2*hamming/dim. Monotone in the true cosine for the walk.
        std::uint32_t h = dense::hamming(nd.bits, q_bits);
        return 1.0f - 2.0f * static_cast<float>(h) / static_cast<float>(dim_);
    }
    std::size_t d = cfg_.matryoshka_dim > 0 ? std::min(cfg_.matryoshka_dim, dim_) : dim_;
    return dense::dot(std::span(nd.vec.data(), d), std::span(q.data(), d));
}

std::vector<std::uint32_t>
HnswIndex::search_layer(std::span<const float> q, std::span<const std::uint64_t> q_bits,
                        std::uint32_t entry, int layer, std::size_t ef) const {
    // Min-heap of candidates by distance (we use similarity, so invert with -).
    // top = furthest in the result set; we pop it when a closer one arrives.
    using PQ = std::pair<float, std::uint32_t>; // (similarity, node)
    auto cmp_far  = [](const PQ& a, const PQ& b) { return a.first > b.first; }; // min-sim on top
    auto cmp_near = [](const PQ& a, const PQ& b) { return a.first < b.first; }; // max-sim on top

    std::priority_queue<PQ, std::vector<PQ>, decltype(cmp_near)> candidates(cmp_near);
    std::priority_queue<PQ, std::vector<PQ>, decltype(cmp_far)>  result(cmp_far);
    std::unordered_set<std::uint32_t> visited;

    float s0 = sim(entry, q, q_bits);
    candidates.push({s0, entry});
    result.push({s0, entry});
    visited.insert(entry);

    while (!candidates.empty()) {
        auto [csim, cnode] = candidates.top();
        candidates.pop();
        if (!result.empty() && csim < result.top().first && result.size() >= ef) break;

        for (std::uint32_t nb : nodes_[cnode].links[static_cast<std::size_t>(layer)]) {
            if (!visited.insert(nb).second) continue;
            float s = sim(nb, q, q_bits);
            if (result.size() < ef || s > result.top().first) {
                candidates.push({s, nb});
                result.push({s, nb});
                if (result.size() > ef) result.pop();
            }
        }
    }

    std::vector<std::uint32_t> out;
    out.reserve(result.size());
    while (!result.empty()) { out.push_back(result.top().second); result.pop(); }
    std::reverse(out.begin(), out.end()); // best first
    return out;
}

void HnswIndex::connect(std::uint32_t node, int layer, std::vector<std::uint32_t> neighbours) {
    const std::size_t maxM = (layer == 0) ? cfg_.M * 2 : cfg_.M;
    auto& L = nodes_[node].links[static_cast<std::size_t>(layer)];
    // Keep the M closest to `node`.
    std::sort(neighbours.begin(), neighbours.end(),
        [&](std::uint32_t a, std::uint32_t b) {
            return dense::dot(nodes_[node].vec, nodes_[a].vec) >
                   dense::dot(nodes_[node].vec, nodes_[b].vec);
        });
    if (neighbours.size() > maxM) neighbours.resize(maxM);
    L = neighbours;

    // Add back-links, pruning each neighbour to its own maxM.
    for (std::uint32_t nb : L) {
        auto& NL = nodes_[nb].links[static_cast<std::size_t>(layer)];
        if (std::find(NL.begin(), NL.end(), node) == NL.end()) NL.push_back(node);
        if (NL.size() > maxM) {
            std::sort(NL.begin(), NL.end(),
                [&](std::uint32_t a, std::uint32_t b) {
                    return dense::dot(nodes_[nb].vec, nodes_[a].vec) >
                           dense::dot(nodes_[nb].vec, nodes_[b].vec);
                });
            NL.resize(maxM);
        }
    }
}

void HnswIndex::add(std::uint32_t id, std::span<const float> vec) {
    if (dim_ == 0) dim_ = vec.size();
    if (vec.size() != dim_ || dim_ == 0) return; // dimension mismatch: ignore

    // Re-adding a tombstoned id resurrects it (incremental upsert).
    deleted_.erase(id);

    Node nd;
    nd.id  = id;
    nd.vec.assign(vec.begin(), vec.end());
    dense::normalize(nd.vec);
    if (cfg_.binary) nd.bits = dense::pack_signs(nd.vec);

    int level = random_level();
    nd.links.resize(static_cast<std::size_t>(level) + 1);

    std::uint32_t ordinal = static_cast<std::uint32_t>(nodes_.size());
    nodes_.push_back(std::move(nd));

    if (max_layer_ < 0) { max_layer_ = level; entry_ = ordinal; return; }

    std::span<const float> q = nodes_[ordinal].vec;
    std::span<const std::uint64_t> qb = nodes_[ordinal].bits;

    std::uint32_t cur = entry_;
    // Descend from top down to level+1 greedily (ef=1).
    for (int lc = max_layer_; lc > level; --lc) {
        auto r = search_layer(q, qb, cur, lc, 1);
        if (!r.empty()) cur = r.front();
    }
    // Insert into every layer from min(level, max_layer_) down to 0.
    for (int lc = std::min(level, max_layer_); lc >= 0; --lc) {
        auto neighbours = search_layer(q, qb, cur, lc, cfg_.ef_construction);
        if (!neighbours.empty()) cur = neighbours.front();
        connect(ordinal, lc, neighbours);
    }
    if (level > max_layer_) { max_layer_ = level; entry_ = ordinal; }
}

std::vector<Hit> HnswIndex::search(std::span<const float> query, std::size_t k) const {
    if (nodes_.empty() || dim_ == 0) return {};
    std::vector<float> q(query.begin(), query.end());
    if (q.size() != dim_) q.resize(dim_, 0.0f);
    dense::normalize(q);
    std::vector<std::uint64_t> qb;
    if (cfg_.binary) qb = dense::pack_signs(q);

    std::uint32_t cur = entry_;
    for (int lc = max_layer_; lc > 0; --lc) {
        auto r = search_layer(q, qb, cur, lc, 1);
        if (!r.empty()) cur = r.front();
    }
    std::size_t ef = std::max(cfg_.ef_search, k);
    auto cand = search_layer(q, qb, cur, 0, ef);

    // Rescore candidates on the FULL float vector (exact cosine) — corrects any
    // approximation introduced by matryoshka/binary during the walk.
    std::vector<Hit> hits;
    hits.reserve(cand.size());
    for (std::uint32_t node : cand) {
        if (!deleted_.empty() && deleted_.count(nodes_[node].id)) continue;
        float s = dense::dot(nodes_[node].vec, q);
        hits.push_back(Hit{ChunkId{nodes_[node].id}, Score{s}});
    }
    std::sort(hits.begin(), hits.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    if (hits.size() > k) hits.resize(k);
    return hits;
}

void HnswIndex::remove(std::uint32_t id) { deleted_.insert(id); }
bool HnswIndex::is_deleted(std::uint32_t id) const noexcept { return deleted_.count(id) != 0; }

void HnswIndex::compact() {
    if (deleted_.empty()) return;
    // Rebuild the graph from the surviving nodes' vectors (their ids preserved).
    std::vector<std::pair<std::uint32_t, std::vector<float>>> survivors;
    survivors.reserve(nodes_.size());
    for (const auto& nd : nodes_)
        if (!deleted_.count(nd.id)) survivors.emplace_back(nd.id, nd.vec);
    HnswConfig cfg = cfg_;
    *this = HnswIndex(cfg);
    for (auto& [id, v] : survivors) add(id, v);
}

std::vector<Hit> HnswIndex::search_filtered(std::span<const float> query, std::size_t k,
                                            const AllowFn& allow, float ef_boost) const {
    if (!allow) return search(query, k);
    if (nodes_.empty() || dim_ == 0) return {};
    std::vector<float> q(query.begin(), query.end());
    if (q.size() != dim_) q.resize(dim_, 0.0f);
    dense::normalize(q);
    std::vector<std::uint64_t> qb;
    if (cfg_.binary) qb = dense::pack_signs(q);

    // Descend the upper layers greedily (unfiltered — pure navigation).
    std::uint32_t cur = entry_;
    for (int lc = max_layer_; lc > 0; --lc) {
        auto r = search_layer(q, qb, cur, lc, 1);
        if (!r.empty()) cur = r.front();
    }
    // Widen the base-layer beam so a selective filter still yields k results.
    std::size_t ef = static_cast<std::size_t>(
        std::max<float>(static_cast<float>(std::max(cfg_.ef_search, k)) * std::max(1.0f, ef_boost),
                        static_cast<float>(k)));
    ef = std::min(ef, nodes_.size());
    auto cand = search_layer(q, qb, cur, 0, ef);

    // Rescore on the full vector, keeping only ALLOWED candidates.
    std::vector<Hit> hits;
    hits.reserve(cand.size());
    for (std::uint32_t node : cand) {
        std::uint32_t id = nodes_[node].id;
        if (!allow(id)) continue;
        if (!deleted_.empty() && deleted_.count(id)) continue;
        float s = dense::dot(nodes_[node].vec, q);
        hits.push_back(Hit{ChunkId{id}, Score{s}});
    }
    std::sort(hits.begin(), hits.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    if (hits.size() > k) hits.resize(k);
    return hits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────────────────────────────────────
namespace {
constexpr std::uint32_t kMagic   = 0x31574E48; // "HNW1"
constexpr std::uint32_t kVersion = 1;
template <class T> void put(std::string& o, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const char* p = reinterpret_cast<const char*>(&v); o.append(p, p + sizeof(T));
}
template <class T> bool get(std::string_view& in, T& v) {
    if (in.size() < sizeof(T)) return false;
    std::memcpy(&v, in.data(), sizeof(T)); in.remove_prefix(sizeof(T)); return true;
}
} // namespace

std::string HnswIndex::serialize() const {
    std::string o;
    put(o, kMagic); put(o, kVersion);
    put(o, cfg_);
    put<std::uint64_t>(o, dim_);
    put<std::int32_t>(o, max_layer_);
    put(o, entry_);
    put<std::uint32_t>(o, static_cast<std::uint32_t>(nodes_.size()));
    for (const auto& nd : nodes_) {
        put(o, nd.id);
        put<std::uint32_t>(o, static_cast<std::uint32_t>(nd.vec.size()));
        o.append(reinterpret_cast<const char*>(nd.vec.data()), nd.vec.size() * sizeof(float));
        put<std::uint32_t>(o, static_cast<std::uint32_t>(nd.links.size()));
        for (const auto& layer : nd.links) {
            put<std::uint32_t>(o, static_cast<std::uint32_t>(layer.size()));
            o.append(reinterpret_cast<const char*>(layer.data()), layer.size() * sizeof(std::uint32_t));
        }
    }
    return o;
}

Result<HnswIndex> HnswIndex::deserialize(std::string_view in) {
    HnswIndex idx;
    std::uint32_t magic, version;
    if (!get(in, magic) || magic != kMagic)       return fail<HnswIndex>(Errc::corrupt_index, "hnsw magic");
    if (!get(in, version) || version != kVersion)  return fail<HnswIndex>(Errc::corrupt_index, "hnsw version");
    if (!get(in, idx.cfg_)) return fail<HnswIndex>(Errc::corrupt_index, "cfg");
    std::uint64_t dim; std::int32_t maxl; std::uint32_t entry, ncount;
    if (!get(in, dim) || !get(in, maxl) || !get(in, entry) || !get(in, ncount))
        return fail<HnswIndex>(Errc::corrupt_index, "header");
    idx.dim_ = dim; idx.max_layer_ = maxl; idx.entry_ = entry;
    idx.nodes_.resize(ncount);
    for (auto& nd : idx.nodes_) {
        std::uint32_t vlen;
        if (!get(in, nd.id) || !get(in, vlen)) return fail<HnswIndex>(Errc::corrupt_index, "node");
        nd.vec.resize(vlen);
        if (in.size() < vlen * sizeof(float)) return fail<HnswIndex>(Errc::corrupt_index, "vec");
        std::memcpy(nd.vec.data(), in.data(), vlen * sizeof(float));
        in.remove_prefix(vlen * sizeof(float));
        if (idx.cfg_.binary) nd.bits = dense::pack_signs(nd.vec);
        std::uint32_t nlayers;
        if (!get(in, nlayers)) return fail<HnswIndex>(Errc::corrupt_index, "nlayers");
        nd.links.resize(nlayers);
        for (auto& layer : nd.links) {
            std::uint32_t llen;
            if (!get(in, llen)) return fail<HnswIndex>(Errc::corrupt_index, "llen");
            layer.resize(llen);
            if (in.size() < llen * sizeof(std::uint32_t)) return fail<HnswIndex>(Errc::corrupt_index, "links");
            std::memcpy(layer.data(), in.data(), llen * sizeof(std::uint32_t));
            in.remove_prefix(llen * sizeof(std::uint32_t));
        }
    }
    return idx;
}

} // namespace rag::index
