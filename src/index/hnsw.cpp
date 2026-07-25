// rag/index/hnsw.cpp — HNSW build + search + serialization.

#include "rag/index/hnsw.hpp"
#include "rag/dense/simd.hpp"
#include "rag/util/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
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
    return dense::dot(std::span(store_.data() + node * dim_, d), std::span(q.data(), d));
}

std::vector<std::uint32_t>
HnswIndex::search_layer(std::span<const float> q, std::span<const std::uint64_t> q_bits,
                        std::uint32_t entry, int layer, std::size_t ef) const {
    // Two heaps over (similarity, node):
    //   cand   — max-heap by sim: the frontier, best-first expansion order.
    //   result — min-heap by sim: the running top-ef; its root is the WEAKEST
    //            member, so admitting a better node is a pop+push.
    // Both live in thread-local scratch and keep their capacity across calls,
    // and `visited` is an epoch-stamped array rather than a hash set, so this
    // function performs no allocation in steady state.
    using PQ = std::pair<float, std::uint32_t>;
    auto near_less = [](const PQ& a, const PQ& b) { return a.first <  b.first; }; // max-heap
    auto far_less  = [](const PQ& a, const PQ& b) { return a.first >  b.first; }; // min-heap

    Scratch& sc = scratch();
    sc.reset(nodes_.size());
    auto& cand   = sc.cand;
    auto& result = sc.result;

    const std::size_t L = static_cast<std::size_t>(layer);
    // A node only participates in `layer` if it was assigned that many levels.
    // During a CONCURRENT build another thread may publish a taller entry point
    // before its adjacency is filled in, so every links[] access is guarded.
    auto links_at = [this, L](std::uint32_t n) -> const std::vector<std::uint32_t>* {
        const auto& lk = nodes_[n].links;
        return L < lk.size() ? &lk[L] : nullptr;
    };

    float s0 = sim(entry, q, q_bits);
    (void)sc.mark(entry);
    cand.push_back({s0, entry});
    result.push_back({s0, entry});

    while (!cand.empty()) {
        std::pop_heap(cand.begin(), cand.end(), near_less);
        auto [csim, cnode] = cand.back();
        cand.pop_back();
        if (!result.empty() && csim < result.front().first && result.size() >= ef) break;

        const std::vector<std::uint32_t>* links = links_at(cnode);
        if (!links) continue;
        // Prefetch the neighbour payloads we are about to score: the graph walk
        // is pointer-chasing and this hides most of the miss latency.
        for (std::uint32_t nb : *links)
            if (nb < nodes_.size()) __builtin_prefetch(store_.data() + nb * dim_, 0, 1);

        for (std::uint32_t nb : *links) {
            if (nb >= nodes_.size()) continue;
            if (!sc.mark(nb)) continue;
            float s = sim(nb, q, q_bits);
            if (result.size() < ef) {
                result.push_back({s, nb});
                std::push_heap(result.begin(), result.end(), far_less);
                cand.push_back({s, nb});
                std::push_heap(cand.begin(), cand.end(), near_less);
            } else if (s > result.front().first) {
                std::pop_heap(result.begin(), result.end(), far_less);
                result.back() = {s, nb};
                std::push_heap(result.begin(), result.end(), far_less);
                cand.push_back({s, nb});
                std::push_heap(cand.begin(), cand.end(), near_less);
            }
        }
    }

    // Drain the min-heap: sort_heap leaves it descending by similarity, which
    // is exactly the best-first order callers expect.
    std::sort_heap(result.begin(), result.end(), far_less);
    std::vector<std::uint32_t> out;
    out.reserve(result.size());
    for (const auto& [s, n] : result) out.push_back(n);
    return out;
}

namespace {

// Malkov & Yashunin Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC.
//
// Plain "keep the M nearest" produces a clustered graph: all of a node's edges
// point into the same dense neighbourhood, so the greedy walk has no long-range
// escape route and gets trapped in local minima (recall collapses). The
// heuristic instead keeps a candidate `c` only if it is closer to the query
// node than to every neighbour already selected — i.e. it occupies a NEW
// direction. That yields the sparse, well-spread, navigable graph HNSW needs.
//
// `scored` must be pre-sorted best-first as (sim_to_node, candidate) pairs. The
// similarity to the node is passed IN rather than recomputed: it is the same
// value the caller already needed for the sort, and at ef_construction=200 a
// re-computation would be 200 extra dot products per link operation.
// `sim_between(a,b)` is the similarity between two candidates.
template <class SimAB>
std::vector<std::uint32_t>
select_neighbours_heuristic(const std::vector<std::pair<float, std::uint32_t>>& scored,
                            std::size_t M, SimAB&& sim_between) {
    std::vector<std::uint32_t> picked;
    picked.reserve(M);
    for (const auto& [c_q, c] : scored) {
        if (picked.size() >= M) break;
        bool keep = true;
        for (std::uint32_t p : picked) {
            // If c is nearer to an already-picked neighbour than to the query
            // node, p already "covers" that direction — drop c.
            if (sim_between(c, p) > c_q) { keep = false; break; }
        }
        if (keep) picked.push_back(c);
    }
    // Backfill with the best remaining candidates if the heuristic was too
    // strict to reach M (keeps degree up on small/uniform datasets).
    if (picked.size() < M) {
        for (const auto& [c_q, c] : scored) {
            if (picked.size() >= M) break;
            if (std::find(picked.begin(), picked.end(), c) == picked.end())
                picked.push_back(c);
        }
    }
    return picked;
}

} // namespace

// Score `cands` against `nv` ONCE and sort best-first. The obvious spelling —
// std::sort with a comparator that calls dot() — recomputes each vector's
// similarity O(log n) times: at ef_construction=200 that is ~3000 dot products
// where 200 suffice, and it dominated build time before this was hoisted.
void HnswIndex::score_and_sort(std::span<const float> nv,
                               const std::vector<std::uint32_t>& cands,
                               std::vector<std::pair<float, std::uint32_t>>& out) const {
    out.clear();
    out.reserve(cands.size());
    for (std::uint32_t c : cands) out.emplace_back(dense::dot(nv, vec_at(c)), c);
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
}

void HnswIndex::connect(std::uint32_t node, int layer, std::vector<std::uint32_t> neighbours) {
    const std::size_t maxM = (layer == 0) ? cfg_.M * 2 : cfg_.M;
    const std::span<const float> nv = vec_at(node);
    std::vector<std::pair<float, std::uint32_t>> scored;
    score_and_sort(nv, neighbours, scored);
    auto& L = nodes_[node].links[static_cast<std::size_t>(layer)];
    L = select_neighbours_heuristic(scored, maxM,
            [&](std::uint32_t a, std::uint32_t b) { return dense::dot(vec_at(a), vec_at(b)); });

    // Add back-links, pruning each neighbour to its own maxM with the same
    // heuristic so the reverse edges stay diverse too.
    for (std::uint32_t nb : L) {
        auto& NL = nodes_[nb].links[static_cast<std::size_t>(layer)];
        if (std::find(NL.begin(), NL.end(), node) == NL.end()) NL.push_back(node);
        if (NL.size() > maxM) {
            score_and_sort(vec_at(nb), NL, scored);
            NL = select_neighbours_heuristic(scored, maxM,
                    [&](std::uint32_t a, std::uint32_t b) { return dense::dot(vec_at(a), vec_at(b)); });
        }
    }
}

void HnswIndex::connect_locked(std::uint32_t node, int layer,
                               std::vector<std::uint32_t> neighbours,
                               std::vector<NodeLock>& locks) {
    const std::size_t maxM = (layer == 0) ? cfg_.M * 2 : cfg_.M;
    const std::size_t L    = static_cast<std::size_t>(layer);
    const std::span<const float> nv = vec_at(node);

    // Rank candidates by similarity to `node`, then apply the diversity
    // heuristic. This read-only scoring needs no locks: vectors are immutable
    // after staging and the arena never reallocates during phase 2.
    std::vector<std::pair<float, std::uint32_t>> scored;
    score_and_sort(nv, neighbours, scored);
    auto keep = select_neighbours_heuristic(scored, maxM,
            [&](std::uint32_t a, std::uint32_t b) { return dense::dot(vec_at(a), vec_at(b)); });

    {   // Publish this node's own adjacency. Written in place (capacity was
        // reserved to maxM+1 at staging) so the buffer address never changes
        // and a concurrent reader can never chase a freed pointer.
        locks[node].lock();
        nodes_[node].links[L].assign(keep.begin(), keep.end());
        locks[node].unlock();
    }

    // Add back-links, pruning each neighbour to its own maxM. Each neighbour is
    // locked individually and only for the duration of its own edit.
    std::vector<std::uint32_t> snapshot;
    for (std::uint32_t nb : keep) {
        locks[nb].lock();
        auto& NL = nodes_[nb].links[L];
        if (std::find(NL.begin(), NL.end(), node) == NL.end()) NL.push_back(node);
        const bool over = NL.size() > maxM;
        if (over) snapshot.assign(NL.begin(), NL.end());
        locks[nb].unlock();
        if (!over) continue;

        // Prune outside the lock — scoring maxM+1 candidates against each other
        // is O(M²) dot products, far too long to hold a spinlock that readers
        // and other writers are contending for.
        score_and_sort(vec_at(nb), snapshot, scored);
        auto pruned = select_neighbours_heuristic(scored, maxM,
                [&](std::uint32_t a, std::uint32_t b) { return dense::dot(vec_at(a), vec_at(b)); });
        locks[nb].lock();
        auto& NL2 = nodes_[nb].links[L];
        // Re-check under the lock: another thread may have pruned already, and
        // anything it appended meanwhile must not be silently dropped, so only
        // overwrite when our pruned set still covers the current size.
        if (NL2.size() > maxM) NL2.assign(pruned.begin(), pruned.end());  // in place: capacity is stable
        locks[nb].unlock();
    }
}

void HnswIndex::build_batch(std::size_t n,
                            const std::function<std::span<const float>(std::size_t)>& vec_at_fn,
                            const std::function<std::uint32_t(std::size_t)>& id_at) {
    if (n == 0) return;

    // ── Phase 1 (serial): stage every node. ─────────────────────────────
    // Vectors, sign codes and levels are fixed here so that phase 2 never
    // reallocates `nodes_` or `store_` — which is what makes concurrent
    // linking safe.
    const std::size_t base = nodes_.size();
    if (dim_ == 0 && n > 0) dim_ = vec_at_fn(0).size();
    if (dim_ == 0) return;
    nodes_.reserve(base + n);
    store_.reserve((base + n) * dim_);
    for (std::size_t i = 0; i < n; ++i) {
        std::span<const float> v = vec_at_fn(i);
        if (v.size() != dim_) continue;                // dimension mismatch: skip

        Node nd;
        nd.id = id_at(i);
        const std::size_t off = store_.size();
        store_.insert(store_.end(), v.begin(), v.end());
        dense::normalize(std::span<float>(store_.data() + off, dim_));
        if (cfg_.binary) nd.bits = dense::pack_signs(std::span<const float>(store_.data() + off, dim_));
        const std::size_t levels = static_cast<std::size_t>(random_level()) + 1;
        nd.links.resize(levels);
        // Reserve each layer to its hard maximum NOW. During phase 2 readers
        // walk `links` without a lock while writers push_back into it; if a
        // push_back could reallocate, a reader would follow a dangling pointer.
        // Reserving to maxM up front makes every write in-place, so the buffer
        // address is stable for the whole build. (The only mutation that can
        // exceed maxM is the transient push before the prune, hence maxM+1.)
        for (std::size_t l = 0; l < levels; ++l)
            nd.links[l].reserve((l == 0 ? cfg_.M * 2 : cfg_.M) + 1);
        deleted_.erase(nd.id);
        nodes_.push_back(std::move(nd));
    }
    const std::size_t total = nodes_.size();
    if (total == base) return;

    // Seed the graph with the first node if the index was empty.
    std::size_t start = base;
    if (max_layer_ < 0) {
        max_layer_ = static_cast<int>(nodes_[base].links.size()) - 1;
        entry_     = static_cast<std::uint32_t>(base);
        ++start;
    }

    // ── Phase 2 (parallel): search + link. ──────────────────────────────
    // The entry point and max_layer_ are promoted under a mutex; every other
    // mutation is per-node and spinlock-guarded. Nodes link against whatever
    // portion of the graph is already visible, exactly as in serial insertion.
    std::vector<NodeLock> locks(total);
    std::mutex entry_mu;

    util::parallel_for(total - start, [&](std::size_t idx) {
        const std::uint32_t ordinal = static_cast<std::uint32_t>(start + idx);
        const int level = static_cast<int>(nodes_[ordinal].links.size()) - 1;

        std::span<const float>         q  = vec_at(ordinal);
        std::span<const std::uint64_t> qb = nodes_[ordinal].bits;

        int top;
        std::uint32_t cur;
        { std::lock_guard lk(entry_mu); top = max_layer_; cur = entry_; }

        for (int lc = top; lc > level; --lc) {
            auto r = search_layer(q, qb, cur, lc, 1);
            if (!r.empty()) cur = r.front();
        }
        for (int lc = std::min(level, top); lc >= 0; --lc) {
            auto neighbours = search_layer(q, qb, cur, lc, cfg_.ef_construction);
            if (!neighbours.empty()) cur = neighbours.front();
            connect_locked(ordinal, lc, std::move(neighbours), locks);
        }

        if (level > top) {
            std::lock_guard lk(entry_mu);
            if (level > max_layer_) { max_layer_ = level; entry_ = ordinal; }
        }
    });
}

void HnswIndex::add(std::uint32_t id, std::span<const float> vec) {
    if (dim_ == 0) dim_ = vec.size();
    if (vec.size() != dim_ || dim_ == 0) return; // dimension mismatch: ignore

    // Re-adding a tombstoned id resurrects it (incremental upsert).
    deleted_.erase(id);

    Node nd;
    nd.id  = id;
    const std::size_t off = store_.size();
    store_.insert(store_.end(), vec.begin(), vec.end());
    dense::normalize(std::span<float>(store_.data() + off, dim_));
    if (cfg_.binary) nd.bits = dense::pack_signs(std::span<const float>(store_.data() + off, dim_));

    int level = random_level();
    nd.links.resize(static_cast<std::size_t>(level) + 1);

    std::uint32_t ordinal = static_cast<std::uint32_t>(nodes_.size());
    nodes_.push_back(std::move(nd));

    if (max_layer_ < 0) { max_layer_ = level; entry_ = ordinal; return; }

    std::span<const float> q = vec_at(ordinal);
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
        float s = dense::dot(vec_at(node), q);
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
    std::vector<std::uint32_t> ids;
    std::vector<float>         vecs;
    ids.reserve(nodes_.size());
    vecs.reserve(nodes_.size() * dim_);
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (deleted_.count(nodes_[i].id)) continue;
        ids.push_back(nodes_[i].id);
        auto v = vec_at(i);
        vecs.insert(vecs.end(), v.begin(), v.end());
    }
    const std::size_t d = dim_;
    HnswConfig cfg = cfg_;
    *this = HnswIndex(cfg);
    // Bulk rebuild rather than a serial add loop: same graph quality, and the
    // linking runs across every core.
    build_batch(ids.size(),
                [&](std::size_t i) { return std::span<const float>(vecs.data() + i * d, d); },
                [&](std::size_t i) { return ids[i]; });
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
        float s = dense::dot(vec_at(node), q);
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
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& nd = nodes_[i];
        put(o, nd.id);
        // Wire format is unchanged (per-node length + payload) even though the
        // in-memory layout is now one arena: the blob stays readable by any
        // build, and dim_ is already in the header for the fast path.
        put<std::uint32_t>(o, static_cast<std::uint32_t>(dim_));
        o.append(reinterpret_cast<const char*>(store_.data() + i * dim_), dim_ * sizeof(float));
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
    idx.store_.assign(static_cast<std::size_t>(ncount) * idx.dim_, 0.0f);
    for (std::uint32_t i = 0; i < ncount; ++i) {
        auto& nd = idx.nodes_[i];
        std::uint32_t vlen;
        if (!get(in, nd.id) || !get(in, vlen)) return fail<HnswIndex>(Errc::corrupt_index, "node");
        if (vlen != idx.dim_) return fail<HnswIndex>(Errc::corrupt_index, "vec dim");
        if (in.size() < vlen * sizeof(float)) return fail<HnswIndex>(Errc::corrupt_index, "vec");
        std::memcpy(idx.store_.data() + static_cast<std::size_t>(i) * idx.dim_,
                    in.data(), vlen * sizeof(float));
        in.remove_prefix(vlen * sizeof(float));
        if (idx.cfg_.binary)
            nd.bits = dense::pack_signs(
                std::span<const float>(idx.store_.data() + static_cast<std::size_t>(i) * idx.dim_, idx.dim_));
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
