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
    std::size_t ef_search       = 64;    // beam width during query
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
    struct Node {
        std::uint32_t              id = 0;
        std::vector<float>         vec;     // unit-normalized
        std::vector<std::uint64_t> bits;    // sign code (binary mode)
        std::vector<std::vector<std::uint32_t>> links;  // links[layer]
    };

    HnswConfig    cfg_{};
    std::size_t   dim_       = 0;
    int           max_layer_ = -1;
    std::uint32_t entry_     = 0;
    std::vector<Node> nodes_;                 // index == internal node ordinal
    std::unordered_set<std::uint32_t> deleted_;  // tombstoned ids (soft-delete)
    mutable std::mt19937_64 rng_{cfg_.seed};

    [[nodiscard]] int  random_level();
    [[nodiscard]] float sim(std::size_t node_a, std::span<const float> q,
                            std::span<const std::uint64_t> q_bits) const;
    [[nodiscard]] std::vector<std::uint32_t>
    search_layer(std::span<const float> q, std::span<const std::uint64_t> q_bits,
                 std::uint32_t entry, int layer, std::size_t ef) const;
    void connect(std::uint32_t node, int layer, std::vector<std::uint32_t> neighbours);
};

} // namespace rag::index
