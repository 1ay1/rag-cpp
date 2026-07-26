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
                     std::span<const std::uint64_t> q_bits,
                     std::span<const std::int8_t> q8,
                     std::span<const float> adc) const {
    const Node& nd = nodes_[node];
    if (cfg_.binary && !q_bits.empty() && !nd.bits.empty()) {
        // Approximate similarity from Hamming distance over sign codes:
        // sim ≈ 1 - 2*hamming/dim. Monotone in the true cosine for the walk.
        std::uint32_t h = dense::hamming(nd.bits, q_bits);
        return 1.0f - 2.0f * static_cast<float>(h) / static_cast<float>(dim_);
    }
    // PQ fast path: m bytes per candidate (m ≪ dim), scored by m table lookups.
    // Checked before SQ8 because it is the coarser, cheaper rung — if the
    // caller asked for PQ they asked for the smaller footprint.
    if (!adc.empty() && !pq_codes_.empty())
        return pq_.adc_score(std::span<const std::uint8_t>(pq_at(node), pq_m_), adc);
    // SQ8 fast path: 4× fewer bytes per candidate, which on a memory-bound walk
    // is close to 4× less waiting. Enabled only when the caller supplies a
    // quantized query AND the index is sealed (q8_ is built there) — passed
    // explicitly rather than stashed in scratch so a build-time walk can never
    // pick up a stale query left behind by a previous search on this thread.
    // The result only ORDERS candidates; search() rescores on exact floats.
    if (!q8.empty() && !q8_.empty())
        return static_cast<float>(dense::dot_sq8(q8_at(node), q8.data(), dim_))
             * dense::kSq8Scale;

    std::size_t d = cfg_.matryoshka_dim > 0 ? std::min(cfg_.matryoshka_dim, dim_) : dim_;
    return dense::dot(std::span(store_.data() + node * dim_, d), std::span(q.data(), d));
}

void HnswIndex::seal() const {
    if (sealed_ || nodes_.empty()) return;
    std::size_t layers = 0;
    for (const auto& nd : nodes_) layers = std::max(layers, nd.links.size());
    if (layers == 0) return;

    const std::size_t n = nodes_.size();
    csr_off_.assign(layers, {});
    csr_nbr_.assign(layers, {});
    for (std::size_t L = 0; L < layers; ++L) {
        auto& off = csr_off_[L];
        off.assign(n + 1, 0);
        // Pass 1: degree per node -> exclusive prefix sum.
        for (std::size_t i = 0; i < n; ++i)
            off[i + 1] = static_cast<std::uint32_t>(
                L < nodes_[i].links.size() ? nodes_[i].links[L].size() : 0);
        for (std::size_t i = 0; i < n; ++i) off[i + 1] += off[i];
        // Pass 2: copy the ids into the flat block.
        auto& nbr = csr_nbr_[L];
        nbr.resize(off[n]);
        for (std::size_t i = 0; i < n; ++i) {
            if (L >= nodes_[i].links.size()) continue;
            const auto& src = nodes_[i].links[L];
            std::copy(src.begin(), src.end(), nbr.begin() + off[i]);
        }
    }

    // PQ codec, when requested. Trained here rather than in build_batch so it
    // also covers incrementally-added vectors, and because it needs the whole
    // (finished) arena to train on.
    const bool want_pq = cfg_.pq_codes > 0 && !cfg_.binary && cfg_.matryoshka_dim == 0
                      && dim_ > 0 && cfg_.pq_codes <= dim_ && dim_ % cfg_.pq_codes == 0;
    if (want_pq && pq_codes_.size() != n * cfg_.pq_codes && !floats_dropped_) {
        // Train on a bounded sample: k-means cost is linear in the sample size
        // and the codebook converges long before the whole corpus is needed,
        // so a 100k-vector cap keeps sealing a large index from becoming the
        // dominant cost of building it.
        constexpr std::size_t kMaxTrain = 100'000;
        const std::size_t ntrain = std::min(n, kMaxTrain);
        auto pq = ProductQuantizer::train_flat(
            std::span<const float>(store_.data(), ntrain * dim_), ntrain, dim_,
            PqConfig{.m = cfg_.pq_codes, .ksub = 256, .iters = 15, .seed = cfg_.seed});
        if (pq) {
            pq_   = std::move(*pq);
            pq_m_ = cfg_.pq_codes;
            pq_codes_.resize(n * pq_m_);
            pq_.encode_flat(std::span<const float>(store_.data(), n * dim_), n, pq_codes_);
        }
        // On training failure pq_codes_ stays empty and the walk simply falls
        // back to SQ8 — a worse footprint, never a wrong answer.
    }

    // Quantize every vector once, here, so the walk never pays for it. Skipped
    // when build_batch already did it. Binary mode has its own (coarser) code
    // and matryoshka wants a float prefix, so SQ8 is skipped for both rather
    // than layering approximations.
    //
    // With PQ driving the walk, SQ8 changes role: it is no longer the walk
    // codec but the RESCORE codec. Those two jobs want opposite things — the
    // walk wants the fewest bytes it can order candidates with, the rescore
    // wants the most precision it can afford — and serving both from one
    // representation is what made pure-PQ recall collapse. So keep SQ8 only
    // when it is actually needed for rescoring (i.e. the floats are going
    // away); otherwise it would be dead weight the walk never reads.
    const bool pq_active   = !pq_codes_.empty();
    const bool need_sq8    = !cfg_.binary && cfg_.matryoshka_dim == 0 && dim_ > 0
                          && (!pq_active || cfg_.drop_floats);
    if (need_sq8 && q8_.size() != n * dim_ && !store_.empty()) {
        q8_.resize(n * dim_);
        for (std::size_t i = 0; i < n; ++i)
            dense::quantize_sq8(vec_at(i),
                                std::span<std::int8_t>(q8_.data() + i * dim_, dim_));
    } else if (!need_sq8 && !q8_.empty()) {
        q8_.clear();
        q8_.shrink_to_fit();
    }

    // Release the exact vectors if the caller opted in. Requires SOMETHING to
    // have replaced them — dropping the floats with no codec would empty the
    // index — so this is a no-op under binary/matryoshka, where neither mirror
    // is built.
    if (cfg_.drop_floats && !floats_dropped_ && (!q8_.empty() || !pq_codes_.empty())) {
        store_.clear();
        store_.shrink_to_fit();
        floats_dropped_ = true;
    }

    // Release the per-node adjacency: the CSR built above is a complete,
    // authoritative copy of it, so keeping both doubles the cost of the graph
    // for nothing. That is worth more than it sounds — once vectors are
    // compressed the adjacency IS the index (at pq_codes=32 the codes are 32
    // B/vector against ~310 B of adjacency), so this halves the dominant term.
    // unseal() restores it from the CSR if the graph is ever mutated again.
    for (auto& nd : nodes_) {
        nd.links.clear();
        nd.links.shrink_to_fit();
    }
    sealed_ = true;
}

void HnswIndex::unpack_links() const {
    if (!sealed_ || csr_off_.empty()) return;
    const std::size_t n = nodes_.size();
    for (std::size_t i = 0; i < n; ++i) {
        // Highest layer this node actually reaches, so the restored shape
        // matches what the build produced rather than padding every node to
        // the tallest one.
        std::size_t top = 0;
        for (std::size_t L = 0; L < csr_off_.size(); ++L) {
            const auto& off = csr_off_[L];
            if (i + 1 < off.size() && off[i + 1] > off[i]) top = L + 1;
        }
        nodes_[i].links.assign(top, {});
        for (std::size_t L = 0; L < top; ++L) {
            const auto& off = csr_off_[L];
            if (i + 1 >= off.size()) continue;
            nodes_[i].links[L].assign(csr_nbr_[L].begin() + off[i],
                                      csr_nbr_[L].begin() + off[i + 1]);
        }
    }
}

// The final-ranking score for one candidate. Walks a precision ladder and
// takes the most accurate rung still available:
//
//   floats  — exact cosine; always present unless drop_floats was set
//   SQ8     — ~1/127 per component; enough to reproduce the exact top-k order
//             almost always (measured: 1.000 of the true top-10 land in the
//             SQ8 top-32 on 50k vectors)
//   PQ      — reconstruction from m centroids; the walk codec, far too coarse
//             to decide the FINAL order (it puts only ~0.53 of the true top-10
//             in its own top-10), and used only if nothing better survives
//
// Keeping this ladder distinct from the WALK codec is the whole trick: the
// walk wants minimum bytes, the rescore wants maximum precision, and using one
// representation for both is what collapses recall.
float HnswIndex::rescore(std::uint32_t node, const std::vector<float>& q,
                         std::span<const std::int8_t> q8, Scratch& sc) const {
    if (!store_.empty()) return dense::dot(vec_at(node), q);
    if (!q8_.empty() && !q8.empty())
        return static_cast<float>(dense::dot_sq8(q8_at(node), q8.data(), dim_))
             * dense::kSq8Scale;
    if (!pq_codes_.empty()) {
        pq_.decode_into(std::span<const std::uint8_t>(pq_at(node), pq_m_), sc.recon);
        return dense::dot(sc.recon, q);
    }
    return 0.0f;
}

std::vector<std::uint32_t>
HnswIndex::search_layer(std::span<const float> q, std::span<const std::uint64_t> q_bits,
                        std::uint32_t entry, int layer, std::size_t ef) const {
    std::vector<std::uint32_t> out;
    search_layer_into(q, q_bits, {}, {}, entry, layer, ef, out);
    return out;
}

void HnswIndex::search_layer_into(std::span<const float> q, std::span<const std::uint64_t> q_bits,
                                  std::span<const std::int8_t> q8,
                                  std::span<const float> adc,
                                  std::uint32_t entry, int layer, std::size_t ef,
                                  std::vector<std::uint32_t>& out) const {
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

    float s0 = sim(entry, q, q_bits, q8, adc);
    (void)sc.mark(entry);
    cand.push_back({s0, entry});
    result.push_back({s0, entry});

    while (!cand.empty()) {
        std::pop_heap(cand.begin(), cand.end(), near_less);
        auto [csim, cnode] = cand.back();
        cand.pop_back();
        if (!result.empty() && csim < result.front().first && result.size() >= ef) break;

        // One span, one bounds check, contiguous ids — see the CSR note in the
        // header. During a concurrent build another thread may publish a taller
        // entry point before its adjacency is filled, so an empty span here is
        // normal rather than exceptional.
        const std::span<const std::uint32_t> links = neighbours(cnode, L);
        if (links.empty()) continue;

        // Prefetch the neighbour payloads we are about to score: the graph walk
        // is pointer-chasing and this hides most of the miss latency. Prefetch
        // whichever representation sim() will actually read, in the same order
        // it checks them.
        if (!pq_codes_.empty() && !adc.empty())
            for (std::uint32_t nb : links)
                if (nb < nodes_.size()) __builtin_prefetch(pq_codes_.data() + nb * pq_m_, 0, 1);
        else if (!q8_.empty())
            for (std::uint32_t nb : links)
                if (nb < nodes_.size()) __builtin_prefetch(q8_.data() + nb * dim_, 0, 1);
        else
            for (std::uint32_t nb : links)
                if (nb < nodes_.size()) __builtin_prefetch(store_.data() + nb * dim_, 0, 1);

        for (std::uint32_t nb : links) {
            if (nb >= nodes_.size()) continue;
            if (!sc.mark(nb)) continue;
            float s = sim(nb, q, q_bits, q8, adc);
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
    out.clear();
    out.reserve(result.size());
    for (const auto& [s, n] : result) out.push_back(n);
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
//
// `node` names the vector being linked so the SQ8 rows can be used when
// available: neighbour selection only ORDERS candidates, and the graph it
// produces is rebuilt-equivalent under an error of ~1/127 (verified by the
// recall gates), so it gets the same 4× bandwidth saving as the walk.
void HnswIndex::score_and_sort(std::uint32_t node,
                               const std::vector<std::uint32_t>& cands,
                               std::vector<std::pair<float, std::uint32_t>>& out) const {
    out.clear();
    out.reserve(cands.size());
    if (!q8_.empty()) {
        const std::int8_t* a = q8_at(node);
        for (std::uint32_t c : cands)
            out.emplace_back(static_cast<float>(dense::dot_sq8(a, q8_at(c), dim_)), c);
    } else {
        const std::span<const float> nv = vec_at(node);
        for (std::uint32_t c : cands) out.emplace_back(dense::dot(nv, vec_at(c)), c);
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
}

// Similarity between two INDEXED nodes, used by the diversity heuristic.
// Matches whatever scale score_and_sort produced, so the two are comparable.
float HnswIndex::sim_nodes(std::uint32_t a, std::uint32_t b) const {
    if (!q8_.empty())
        return static_cast<float>(dense::dot_sq8(q8_at(a), q8_at(b), dim_));
    return dense::dot(vec_at(a), vec_at(b));
}

void HnswIndex::connect(std::uint32_t node, int layer, std::vector<std::uint32_t> neighbours) {
    const std::size_t maxM = (layer == 0) ? cfg_.M * 2 : cfg_.M;
    std::vector<std::pair<float, std::uint32_t>> scored;
    score_and_sort(node, neighbours, scored);
    auto& L = nodes_[node].links[static_cast<std::size_t>(layer)];
    L = select_neighbours_heuristic(scored, maxM,
            [&](std::uint32_t a, std::uint32_t b) { return sim_nodes(a, b); });

    // Add back-links, pruning each neighbour to its own maxM with the same
    // heuristic so the reverse edges stay diverse too.
    for (std::uint32_t nb : L) {
        auto& NL = nodes_[nb].links[static_cast<std::size_t>(layer)];
        if (std::find(NL.begin(), NL.end(), node) == NL.end()) NL.push_back(node);
        if (NL.size() > maxM) {
            score_and_sort(nb, NL, scored);
            NL = select_neighbours_heuristic(scored, maxM,
                    [&](std::uint32_t a, std::uint32_t b) { return sim_nodes(a, b); });
        }
    }
}

void HnswIndex::connect_locked(std::uint32_t node, int layer,
                               std::vector<std::uint32_t> neighbours,
                               std::vector<NodeLock>& locks) {
    const std::size_t maxM = (layer == 0) ? cfg_.M * 2 : cfg_.M;
    const std::size_t L    = static_cast<std::size_t>(layer);

    // Rank candidates by similarity to `node`, then apply the diversity
    // heuristic. This read-only scoring needs no locks: vectors are immutable
    // after staging and neither arena reallocates during phase 2.
    std::vector<std::pair<float, std::uint32_t>> scored;
    score_and_sort(node, neighbours, scored);
    auto keep = select_neighbours_heuristic(scored, maxM,
            [&](std::uint32_t a, std::uint32_t b) { return sim_nodes(a, b); });

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
        score_and_sort(nb, snapshot, scored);
        auto pruned = select_neighbours_heuristic(scored, maxM,
                [&](std::uint32_t a, std::uint32_t b) { return sim_nodes(a, b); });
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
    unseal();   // the graph is about to change

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
    // build_batch appends nodes behind the id index's back, so drop it: the next
    // add() rebuilds it from the full node array. Cheap (a bool + a clear) and
    // it keeps the two insert paths from disagreeing about which node owns an
    // id — which would silently reintroduce duplicate hits.
    id_index_built_ = false;
    id_to_ord_.clear();
    const std::size_t total = nodes_.size();
    if (total == base) return;

    // Quantize every staged vector NOW, before any linking runs. The build is
    // the same memory-bound graph walk as a query, only ~n×ef times over, so it
    // benefits from SQ8 exactly as search does — and doing it here (rather than
    // in seal(), after the fact) means the ef_construction-wide walks that
    // dominate build time read int8 instead of float32.
    //
    // Safe to publish before phase 2 for the same reason `store_` is: the
    // buffer is sized once and never reallocated while linking threads hold
    // pointers into it.
    if (!cfg_.binary && cfg_.matryoshka_dim == 0 && dim_ > 0) {
        q8_.resize(total * dim_);
        util::parallel_for(total - base, [&](std::size_t i) {
            const std::size_t node = base + i;
            dense::quantize_sq8(vec_at(node),
                                std::span<std::int8_t>(q8_.data() + node * dim_, dim_));
        });
    }

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
        // This node's own SQ8 row doubles as the quantized query for its walk.
        std::span<const std::int8_t>   q8 =
            q8_.empty() ? std::span<const std::int8_t>{}
                        : std::span<const std::int8_t>(q8_at(ordinal), dim_);

        int top;
        std::uint32_t cur;
        { std::lock_guard lk(entry_mu); top = max_layer_; cur = entry_; }

        std::vector<std::uint32_t> hop;
        for (int lc = top; lc > level; --lc) {
            // No ADC during construction: PQ is trained in seal(), after the
            // graph exists, so the build always walks on SQ8/floats.
            search_layer_into(q, qb, q8, {}, cur, lc, 1, hop);
            if (!hop.empty()) cur = hop.front();
        }
        for (int lc = std::min(level, top); lc >= 0; --lc) {
            search_layer_into(q, qb, q8, {}, cur, lc, cfg_.ef_construction, hop);
            if (!hop.empty()) cur = hop.front();
            connect_locked(ordinal, lc, hop, locks);
        }

        if (level > top) {
            std::lock_guard lk(entry_mu);
            if (level > max_layer_) { max_layer_ = level; entry_ = ordinal; }
        }
    });

    // The graph is final: freeze the adjacency into CSR so every subsequent
    // query walks contiguous memory. Done once here rather than lazily on
    // first search, so query latency has no cold-start cliff.
    seal();
}

void HnswIndex::add(std::uint32_t id, std::span<const float> vec) {
    if (dim_ == 0) dim_ = vec.size();
    if (vec.size() != dim_ || dim_ == 0) return; // dimension mismatch: ignore
    unseal();   // incremental insert invalidates the frozen adjacency

    // Re-adding a tombstoned id resurrects it (incremental upsert).
    deleted_.erase(id);

    // ...and re-adding a LIVE id supersedes its old node, so the id cannot be
    // returned twice (once per node) with the stale vector still competing on
    // its own score. The old node is left in place as a navigation waypoint
    // rather than excised: removing it would mean re-linking every inbound
    // edge, turning an O(log n) insert into an O(n) one. compact() reclaims it.
    ensure_id_index();
    if (auto it = id_to_ord_.find(id); it != id_to_ord_.end())
        nodes_[it->second].superseded = true;

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
    id_to_ord_[id] = ordinal;          // this node is now the live one for `id`

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

std::vector<Hit> HnswIndex::search(std::span<const float> query, std::size_t k,
                                   std::size_t ef_override) const {
    if (nodes_.empty() || dim_ == 0) return {};
    seal();     // memoized: no-op on every query after the first

    // The query vector is the ONE unavoidable allocation per search (it must be
    // normalized, and the caller's span is const), so it comes from thread-local
    // scratch too rather than a fresh heap block per query.
    Scratch& sc = scratch();
    sc.query.assign(query.begin(), query.end());
    if (sc.query.size() != dim_) sc.query.resize(dim_, 0.0f);
    dense::normalize(sc.query);
    const std::vector<float>& q = sc.query;
    // Quantize the query once per search; passed explicitly into the walk.
    std::span<const std::int8_t> q8;
    if (!q8_.empty() && !cfg_.binary) {
        sc.query_q8.resize(dim_);
        dense::quantize_sq8(q, sc.query_q8);
        q8 = sc.query_q8;
    }
    // Likewise the PQ lookup table: built once per query (m·256 dot products of
    // dsub floats), then every candidate costs m lookups instead of a dot
    // product over dim floats.
    std::span<const float> adc;
    if (!pq_codes_.empty() && !cfg_.binary) {
        sc.adc = pq_.adc_table(q);
        adc = sc.adc;
    }
    std::vector<std::uint64_t> qb;
    if (cfg_.binary) qb = dense::pack_signs(q);

    auto& hop = sc.hop;
    std::uint32_t cur = entry_;
    for (int lc = max_layer_; lc > 0; --lc) {
        search_layer_into(q, qb, q8, adc, cur, lc, 1, hop);
        if (!hop.empty()) cur = hop.front();
    }
    std::size_t ef = std::max(ef_override ? ef_override : cfg_.ef_search, k);
    search_layer_into(q, qb, q8, adc, cur, 0, ef, hop);

    // Rescore candidates on the FULL float vector (exact cosine) — corrects any
    // approximation introduced by matryoshka/binary/SQ8/PQ during the walk.
    // With the floats dropped, fall back to the most precise representation
    // still present: SQ8 (~1/127 per component) before a PQ reconstruction,
    // which is far coarser and would otherwise decide the final ranking.
    std::vector<Hit> hits;
    hits.reserve(hop.size());
    for (std::uint32_t node : hop) {
        if (!deleted_.empty() && deleted_.count(nodes_[node].id)) continue;
        if (nodes_[node].superseded) continue;   // stale node of a re-added id
        hits.push_back(Hit{ChunkId{nodes_[node].id}, Score{rescore(node, q, q8, sc)}});
    }
    // Only the top k are needed, and k ≪ ef: partial_sort touches O(n log k)
    // rather than sorting the whole candidate pool.
    if (hits.size() > k) {
        std::partial_sort(hits.begin(), hits.begin() + k, hits.end(),
            [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
        hits.resize(k);
    } else {
        std::sort(hits.begin(), hits.end(),
            [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    }
    return hits;
}

void HnswIndex::ensure_id_index() {
    if (id_index_built_) return;
    id_index_built_ = true;
    id_to_ord_.reserve(nodes_.size());
    // Later ordinals win: if a duplicate id already exists from before this
    // guard was introduced, the newest node is the live one.
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) id_to_ord_[nodes_[i].id] = i;
}

void HnswIndex::remove(std::uint32_t id) { deleted_.insert(id); }
bool HnswIndex::is_deleted(std::uint32_t id) const noexcept { return deleted_.count(id) != 0; }

void HnswIndex::compact() {
    // Superseded nodes are garbage too — stale vectors of ids that were re-added
    // — so compaction must be able to run for them even with no tombstones.
    bool any_superseded = false;
    for (const auto& nd : nodes_) if (nd.superseded) { any_superseded = true; break; }
    if (deleted_.empty() && !any_superseded) return;
    // Rebuild the graph from the surviving nodes' vectors (their ids preserved).
    std::vector<std::uint32_t> ids;
    std::vector<float>         vecs;
    ids.reserve(nodes_.size());
    vecs.reserve(nodes_.size() * dim_);
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (deleted_.count(nodes_[i].id)) continue;
        if (nodes_[i].superseded) continue;
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
    seal();

    // Same scratch/quantization protocol as search(): the filtered path was
    // walking on raw floats with a freshly allocated query vector, so a
    // metadata-filtered query paid several times the latency of an unfiltered
    // one for no reason — the filter applies at rescore, not during the walk.
    Scratch& sc = scratch();
    sc.query.assign(query.begin(), query.end());
    if (sc.query.size() != dim_) sc.query.resize(dim_, 0.0f);
    dense::normalize(sc.query);
    const std::vector<float>& q = sc.query;

    std::span<const std::int8_t> q8;
    if (!q8_.empty() && !cfg_.binary) {
        sc.query_q8.resize(dim_);
        dense::quantize_sq8(q, sc.query_q8);
        q8 = sc.query_q8;
    }
    std::span<const float> adc;
    if (!pq_codes_.empty() && !cfg_.binary) {
        sc.adc = pq_.adc_table(q);
        adc = sc.adc;
    }
    std::vector<std::uint64_t> qb;
    if (cfg_.binary) qb = dense::pack_signs(q);

    // Descend the upper layers greedily (unfiltered — pure navigation).
    auto& hop = sc.hop;
    std::uint32_t cur = entry_;
    for (int lc = max_layer_; lc > 0; --lc) {
        search_layer_into(q, qb, q8, adc, cur, lc, 1, hop);
        if (!hop.empty()) cur = hop.front();
    }
    // Widen the base-layer beam so a selective filter still yields k results.
    std::size_t ef = static_cast<std::size_t>(
        std::max<float>(static_cast<float>(std::max(cfg_.ef_search, k)) * std::max(1.0f, ef_boost),
                        static_cast<float>(k)));
    ef = std::min(ef, nodes_.size());
    search_layer_into(q, qb, q8, adc, cur, 0, ef, hop);

    // Rescore on the full vector, keeping only ALLOWED candidates.
    std::vector<Hit> hits;
    hits.reserve(hop.size());
    for (std::uint32_t node : hop) {
        std::uint32_t id = nodes_[node].id;
        if (!allow(id)) continue;
        if (!deleted_.empty() && deleted_.count(id)) continue;
        if (nodes_[node].superseded) continue;   // stale node of a re-added id
        hits.push_back(Hit{ChunkId{id}, Score{rescore(node, q, q8, sc)}});
    }
    if (hits.size() > k) {
        std::partial_sort(hits.begin(), hits.begin() + k, hits.end(),
            [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
        hits.resize(k);
    } else {
        std::sort(hits.begin(), hits.end(),
            [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    }
    return hits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────────────────────────────────────
namespace {
constexpr std::uint32_t kMagic   = 0x31574E48; // "HNW1"
// v2: the config is written FIELD BY FIELD instead of memcpy'ing the struct,
// and PQ state is persisted. Blindly memcpy'ing HnswConfig meant that adding
// any tuning knob silently changed the blob layout and invalidated every index
// on disk with nothing but a length mismatch to show for it.
constexpr std::uint32_t kVersion = 2;
template <class T> void put(std::string& o, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const char* p = reinterpret_cast<const char*>(&v); o.append(p, p + sizeof(T));
}
template <class T> bool get(std::string_view& in, T& v) {
    if (in.size() < sizeof(T)) return false;
    std::memcpy(&v, in.data(), sizeof(T)); in.remove_prefix(sizeof(T)); return true;
}

void put_cfg(std::string& o, const HnswConfig& c) {
    put<std::uint64_t>(o, c.M);
    put<std::uint64_t>(o, c.ef_construction);
    put<std::uint64_t>(o, c.ef_search);
    put<float>(o, c.ml);
    put<std::uint64_t>(o, c.matryoshka_dim);
    put<std::uint8_t>(o, c.binary ? 1 : 0);
    put<std::uint64_t>(o, c.pq_codes);
    put<std::uint8_t>(o, c.drop_floats ? 1 : 0);
    put<std::uint64_t>(o, c.seed);
}

bool get_cfg(std::string_view& in, HnswConfig& c) {
    std::uint64_t m, efc, efs, mat, pqc, seed;
    float ml;
    std::uint8_t bin, drop;
    if (!get(in, m) || !get(in, efc) || !get(in, efs) || !get(in, ml) || !get(in, mat)
        || !get(in, bin) || !get(in, pqc) || !get(in, drop) || !get(in, seed))
        return false;
    c.M = m; c.ef_construction = efc; c.ef_search = efs; c.ml = ml;
    c.matryoshka_dim = mat; c.binary = bin != 0;
    c.pq_codes = pqc; c.drop_floats = drop != 0; c.seed = seed;
    return true;
}
} // namespace

std::string HnswIndex::serialize() const {
    // The wire format is per-node link lists, but a sealed index keeps its
    // adjacency only in CSR form, so materialize it first. Cheap relative to
    // writing the blob, and it keeps the format independent of which internal
    // representation happens to be live.
    if (sealed_) unpack_links();

    std::string o;
    put(o, kMagic); put(o, kVersion);
    put_cfg(o, cfg_);
    put<std::uint64_t>(o, dim_);
    put<std::int32_t>(o, max_layer_);
    put(o, entry_);
    put<std::uint32_t>(o, static_cast<std::uint32_t>(nodes_.size()));

    // When the floats were dropped the PQ codes ARE the vectors, so they must
    // be persisted or the index would reload empty. A flag rather than an
    // implied state, so deserialize() never has to guess.
    const bool store_pq = floats_dropped_ && !pq_codes_.empty();
    put<std::uint8_t>(o, store_pq ? 1 : 0);
    if (store_pq) {
        std::string pqblob = pq_.serialize();
        put<std::uint64_t>(o, pqblob.size());
        o.append(pqblob);
        put<std::uint64_t>(o, pq_m_);
        put<std::uint64_t>(o, pq_codes_.size());
        o.append(reinterpret_cast<const char*>(pq_codes_.data()), pq_codes_.size());
    }

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& nd = nodes_[i];
        put(o, nd.id);
        // Wire format keeps the per-node length + payload shape even though the
        // in-memory layout is one arena: the blob stays self-describing, and
        // dim_ is already in the header for the fast path. With the floats
        // dropped the payload is zero-length and the codes above carry the
        // vectors instead.
        const std::uint32_t vlen = store_pq ? 0u : static_cast<std::uint32_t>(dim_);
        put<std::uint32_t>(o, vlen);
        if (vlen)
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
    if (!get_cfg(in, idx.cfg_)) return fail<HnswIndex>(Errc::corrupt_index, "cfg");
    std::uint64_t dim; std::int32_t maxl; std::uint32_t entry, ncount;
    if (!get(in, dim) || !get(in, maxl) || !get(in, entry) || !get(in, ncount))
        return fail<HnswIndex>(Errc::corrupt_index, "header");
    idx.dim_ = dim; idx.max_layer_ = maxl; idx.entry_ = entry;
    idx.nodes_.resize(ncount);

    std::uint8_t store_pq = 0;
    if (!get(in, store_pq)) return fail<HnswIndex>(Errc::corrupt_index, "pq flag");
    if (store_pq) {
        std::uint64_t blen;
        if (!get(in, blen) || in.size() < blen) return fail<HnswIndex>(Errc::corrupt_index, "pq blob");
        auto pq = ProductQuantizer::deserialize(in.substr(0, blen));
        if (!pq) return std::unexpected(pq.error());
        idx.pq_ = std::move(*pq);
        in.remove_prefix(blen);
        std::uint64_t pm, clen;
        if (!get(in, pm) || !get(in, clen) || in.size() < clen)
            return fail<HnswIndex>(Errc::corrupt_index, "pq codes");
        idx.pq_m_ = pm;
        idx.pq_codes_.assign(in.begin(), in.begin() + clen);
        in.remove_prefix(clen);
        // The floats are genuinely absent from the blob; reconstruct-on-rescore
        // is the contract this index was saved with.
        idx.floats_dropped_ = true;
    } else {
        idx.store_.assign(static_cast<std::size_t>(ncount) * idx.dim_, 0.0f);
    }

    for (std::uint32_t i = 0; i < ncount; ++i) {
        auto& nd = idx.nodes_[i];
        std::uint32_t vlen;
        if (!get(in, nd.id) || !get(in, vlen)) return fail<HnswIndex>(Errc::corrupt_index, "node");
        if (vlen) {
            if (vlen != idx.dim_) return fail<HnswIndex>(Errc::corrupt_index, "vec dim");
            if (in.size() < vlen * sizeof(float)) return fail<HnswIndex>(Errc::corrupt_index, "vec");
            std::memcpy(idx.store_.data() + static_cast<std::size_t>(i) * idx.dim_,
                        in.data(), vlen * sizeof(float));
            in.remove_prefix(vlen * sizeof(float));
            if (idx.cfg_.binary)
                nd.bits = dense::pack_signs(
                    std::span<const float>(idx.store_.data() + static_cast<std::size_t>(i) * idx.dim_,
                                           idx.dim_));
        }
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
