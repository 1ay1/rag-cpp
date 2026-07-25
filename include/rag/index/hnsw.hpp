#pragma once
// rag/index/hnsw.hpp — Hierarchical Navigable Small World approximate NN index.
//
// Malkov & Yashunin (2016). O(log n) vector search via a layered proximity
// graph: greedy descent through sparse upper layers to land near the target,
// then a beam (ef) search on the dense base layer. This is the algorithm
// behind FAISS-HNSW / hnswlib / every modern vector DB, in pure C++/STL.
//
// Extras that keep it SOTA:
//   • Matryoshka truncation — score on a truncated prefix of each vector
//     (MRL embeddings keep the most information in the leading dims), then
//     rescore survivors on the full vector. Big speedup, negligible recall loss.
//   • Binary quantization — a 1-bit sign code per dim; the graph WALK compares
//     packed codes with Hamming/popcount (32× cheaper hop), and only the final
//     candidates are rescored with the float dot product.
//   • Serialization — adjacency lists + normalized vectors persist to a blob;
//     re-opening a corpus does not rebuild the graph.

#include <cstdint>
#include <atomic>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::index {

struct HnswConfig {
    std::size_t M               = 16;    // max neighbours per node (base layer 2M)
    std::size_t ef_construction = 200;   // beam width during insert
    // Beam width during query. This is THE recall/latency dial: on realistic
    // embedding geometry (topically clustered vectors) ef=32 already reaches
    // ~0.999 recall@10 and ef=64 is comfortably saturated. Raise it only for
    // near-uniform / very high-dimensional data, where every ANN index needs a
    // wider beam because distances concentrate. Cost is roughly linear in ef.
    std::size_t ef_search       = 64;
    float       ml              = 0.0f;  // level multiplier; 0 => 1/ln(M)
    std::size_t matryoshka_dim  = 0;     // >0: walk on this leading-dim prefix
    bool        binary          = false; // 1-bit sign codes for the walk
    std::uint64_t seed          = 0x9E3779B97F4A7C15ull;
};

class HnswIndex {
public:
    HnswIndex() = default;
    explicit HnswIndex(HnswConfig cfg);

    // Insert one embedding (referenced later by `id`). `vec` is copied and
    // unit-normalized internally. The dimension is fixed by the first insert.
    void add(std::uint32_t id, std::span<const float> vec);

    // Bulk-construct the graph from `n` vectors in PARALLEL.
    //
    // Concurrent HNSW insertion is safe when the node array is fixed up front:
    // we materialize every node (vector, sign code, level) serially — cheap and
    // allocation-bound — then run the expensive part, the neighbour search and
    // linking, across all cores. Each node's adjacency lists are guarded by its
    // own spinlock, so writers only contend when they touch the same node; the
    // graph walk itself reads links optimistically, which is what hnswlib and
    // FAISS do. Recall is statistically identical to serial insertion (the
    // link-selection heuristic is order-sensitive either way).
    //
    // `vec_at(i)` must return a span for the i-th vector; `id_at(i)` its id.
    void build_batch(std::size_t n,
                     const std::function<std::span<const float>(std::size_t)>& vec_at,
                     const std::function<std::uint32_t(std::size_t)>& id_at);

    // k-NN search: returns (id, cosine-similarity) pairs, descending.
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query, std::size_t k) const;

    // Soft-delete a node by id (tombstone): it stays in the graph for
    // connectivity but is never returned by search. O(1). Re-adding the same id
    // resurrects it. `compact()` physically removes tombstones by rebuilding.
    void remove(std::uint32_t id);
    [[nodiscard]] bool is_deleted(std::uint32_t id) const noexcept;
    [[nodiscard]] std::size_t deleted_count() const noexcept { return deleted_.size(); }
    // Rebuild the graph without tombstoned nodes (amortizes delete churn).
    void compact();

    // Filtered k-NN (FILTERED-HNSW / pre-filter): `allow(id)` decides whether a
    // chunk id may appear in the result. The predicate is evaluated DURING the
    // graph walk — disallowed nodes are still traversed (to preserve graph
    // connectivity, ACORN-style) but never enter the result set. This beats
    // post-filtering, which can return k=0 when the top-k are all filtered out.
    // `ef_boost` widens the beam (× multiplier) to compensate for selective
    // filters; pass a larger value when `allow` accepts a small fraction.
    using AllowFn = std::function<bool(std::uint32_t id)>;
    [[nodiscard]] std::vector<Hit>
    search_filtered(std::span<const float> query, std::size_t k,
                    const AllowFn& allow, float ef_boost = 4.0f) const;

    [[nodiscard]] std::size_t size()      const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] const HnswConfig& config() const noexcept { return cfg_; }

    [[nodiscard]] std::string serialize() const;
    [[nodiscard]] static Result<HnswIndex> deserialize(std::string_view blob);

private:
    // Node payload WITHOUT the vector. Vectors live in a flat arena (`store_`)
    // rather than one std::vector<float> per node, because the graph walk is
    // the hot loop and per-node heap blocks cost it twice: an extra pointer
    // chase before every distance computation (the header is in cache, the
    // payload is a fresh miss) and no spatial locality between neighbours
    // scored back-to-back. One arena makes `vec_at(n)` pure address arithmetic
    // and makes prefetching the next neighbour actually pay.
    struct Node {
        std::uint32_t              id = 0;
        std::vector<std::uint64_t> bits;    // sign code (binary mode)
        std::vector<std::vector<std::uint32_t>> links;  // links[layer]
    };

    // One spinlock per node, guarding that node's `links` during a concurrent
    // build. A spinlock (not a mutex) because the critical section is a few
    // dozen nanoseconds — sorting at most 2M neighbour ids — and contention is
    // rare: two threads must pick the same neighbour in the same layer.
    // Held in a separate array so Node stays trivially movable/serializable.
    struct alignas(64) NodeLock {                 // cache-line padded: no false sharing
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
        void lock()   noexcept { while (flag.test_and_set(std::memory_order_acquire)) ; }
        void unlock() noexcept { flag.clear(std::memory_order_release); }
    };

    HnswConfig    cfg_{};
    std::size_t   dim_       = 0;
    int           max_layer_ = -1;
    std::uint32_t entry_     = 0;
    std::vector<Node> nodes_;                 // index == internal node ordinal
    // Flat vector arena: node k's unit-normalized vector is
    // store_[k*dim_ .. k*dim_+dim_). Sized in lockstep with nodes_.
    //
    // CONCURRENCY: build_batch stages ALL vectors (and reserves the arena to
    // its final size) before any linking thread starts, so `store_` never
    // reallocates while readers hold spans into it. Do not push into store_
    // from a parallel phase.
    std::vector<float> store_;

    // ── SQ8 mirror of `store_` ──────────────────────────────────────
    // The walk is memory-bound: it touches ~1100 random vectors per query, and
    // at 256 dims each is 1 KiB, so past cache it is a DRAM-latency benchmark
    // (measured 46 ns/distance at dim=32 vs 236 ns at dim=512 on the same graph
    // shape). q8_ holds the same vectors as int8, a 4× smaller footprint that
    // keeps 4× more of the corpus resident and moves 4× fewer bytes per hop.
    //
    // It is used ONLY to order candidates during traversal. The final top-k is
    // always rescored against the exact float vectors in `store_`, so SQ8
    // affects which candidates are considered, never the scores reported — and
    // the graph is robust to that: neighbour ordering barely changes under an
    // error of ~1/127, and the ef-sized pool absorbs what does.
    //
    // Built by seal() alongside the CSR, dropped by unseal().
    mutable std::vector<std::int8_t> q8_;

    // ── Sealed adjacency (CSR) ────────────────────────────────────
    // `nodes_[n].links[L]` is a vector-of-vectors: reaching one neighbour list
    // costs two dependent loads (node header, then the layer's heap block) and
    // scatters the graph across the allocator. That is fine while BUILDING,
    // where lists are mutated constantly — but a finished index is read-only,
    // and the walk is the hottest loop in the library.
    //
    // So once the graph stops changing we seal it into compressed-sparse-row
    // form: all of layer L's neighbour ids for every node, contiguous, indexed
    // by an offset table. One load gets the offsets, one gets the ids, and
    // consecutive neighbours share cache lines. `seal()` builds it; any
    // mutation (add/remove/build_batch) drops it via unseal().
    //
    // Layout: csr_off_[L] has size()+1 entries; node n's layer-L neighbours are
    // csr_nbr_[L][csr_off_[L][n] .. csr_off_[L][n+1]).
    //
    // Mutable because sealing is pure memoization of `nodes_[].links`, not
    // observable state: a sealed and an unsealed index answer every query
    // identically. That lets the const search path seal on demand, so a caller
    // who inserts incrementally pays for it ONCE on the next query rather than
    // once per insert (which would make incremental add quadratic).
    mutable std::vector<std::vector<std::uint32_t>> csr_off_;
    mutable std::vector<std::vector<std::uint32_t>> csr_nbr_;
    mutable bool sealed_ = false;

    std::unordered_set<std::uint32_t> deleted_;  // tombstoned ids (soft-delete)
    mutable std::mt19937_64 rng_{cfg_.seed};

    // Build the CSR mirror of `nodes_[].links`. Idempotent; cheap relative to
    // the build that produced the graph (one pass to count, one to fill).
    void seal() const;
    // Drop the CSR mirror because the graph is about to change.
    void unseal() noexcept { sealed_ = false; csr_off_.clear(); csr_nbr_.clear(); q8_.clear(); }

    [[nodiscard]] std::span<const float> vec_at(std::size_t n) const noexcept {
        return {store_.data() + n * dim_, dim_};
    }
    [[nodiscard]] std::span<float> vec_at(std::size_t n) noexcept {
        return {store_.data() + n * dim_, dim_};
    }
    [[nodiscard]] const std::int8_t* q8_at(std::size_t n) const noexcept {
        return q8_.data() + n * dim_;
    }

    [[nodiscard]] int  random_level();
    // `q8` is the SQ8-quantized query, or empty to score on exact floats.
    [[nodiscard]] float sim(std::size_t node_a, std::span<const float> q,
                            std::span<const std::uint64_t> q_bits,
                            std::span<const std::int8_t> q8 = {}) const;

    // Neighbours of `n` in layer `L`, or {} if it does not reach that layer.
    // Reads the sealed CSR when available and falls back to the mutable
    // vector-of-vectors during a build.
    [[nodiscard]] std::span<const std::uint32_t>
    neighbours(std::uint32_t n, std::size_t L) const noexcept {
        if (sealed_) {
            if (L >= csr_off_.size()) return {};
            const auto& off = csr_off_[L];
            if (n + 1 >= off.size()) return {};
            return {csr_nbr_[L].data() + off[n], off[n + 1] - off[n]};
        }
        const auto& lk = nodes_[n].links;
        if (L >= lk.size()) return {};
        return {lk[L].data(), lk[L].size()};
    }

    // search_layer writes its result into caller-provided scratch instead of
    // returning a fresh vector: at ef_search=64 the return allocation was a
    // per-query malloc/free pair on the hottest path in the library.
    void search_layer_into(std::span<const float> q, std::span<const std::uint64_t> q_bits,
                           std::span<const std::int8_t> q8,
                           std::uint32_t entry, int layer, std::size_t ef,
                           std::vector<std::uint32_t>& out) const;
    [[nodiscard]] std::vector<std::uint32_t>
    search_layer(std::span<const float> q, std::span<const std::uint64_t> q_bits,
                 std::uint32_t entry, int layer, std::size_t ef) const;
    void connect(std::uint32_t node, int layer, std::vector<std::uint32_t> neighbours);

    // Score `cands` against node `n` and sort best-first into `out`, computing
    // each similarity exactly once. Shared by connect() and connect_locked().
    void score_and_sort(std::uint32_t node,
                        const std::vector<std::uint32_t>& cands,
                        std::vector<std::pair<float, std::uint32_t>>& out) const;

    // Similarity between two indexed nodes, on the same scale score_and_sort
    // produced (SQ8 integer or float dot, whichever is in use).
    [[nodiscard]] float sim_nodes(std::uint32_t a, std::uint32_t b) const;

    // connect() variant used during a concurrent build: takes the per-node
    // spinlocks before mutating adjacency.
    void connect_locked(std::uint32_t node, int layer,
                        std::vector<std::uint32_t> neighbours,
                        std::vector<NodeLock>& locks);

    // ── Search scratch ───────────────────────────────────────────────────────
    // search_layer is the hottest function in the index: it runs on every hop
    // of every insert and every query. Allocating a visited-set and two heaps
    // per call dominated its cost. Instead each thread keeps a persistent
    // scratch block: an epoch-stamped visited array (O(1) clear — bump the
    // epoch instead of rewriting n bytes) and vector-backed heaps that keep
    // their capacity across calls.
    struct Scratch {
        std::vector<std::uint32_t> visit_epoch;   // per-node last-seen epoch
        std::uint32_t             epoch = 0;
        std::vector<std::pair<float, std::uint32_t>> cand;    // max-heap by sim
        std::vector<std::pair<float, std::uint32_t>> result;  // min-heap by sim
        std::vector<std::uint32_t> hop;                       // search_layer output reuse
        std::vector<float>         query;                     // normalized query vector
        std::vector<std::int8_t>   query_q8;                  // SQ8 of the same

        void reset(std::size_t n) {
            if (visit_epoch.size() < n) visit_epoch.assign(n, 0);
            if (++epoch == 0) {                      // wrapped: clear for real
                std::fill(visit_epoch.begin(), visit_epoch.end(), 0);
                epoch = 1;
            }
            cand.clear();
            result.clear();
        }
        [[nodiscard]] bool mark(std::uint32_t id) {  // true if newly visited
            if (visit_epoch[id] == epoch) return false;
            visit_epoch[id] = epoch;
            return true;
        }
    };
    // thread_local so concurrent queries and parallel inserts each get their
    // own scratch without locking or per-call allocation.
    static Scratch& scratch() {
        static thread_local Scratch s;
        return s;
    }
};

} // namespace rag::index
