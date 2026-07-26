#pragma once
// rag/index/vector_store.hpp — the easy front door for using rag-cpp as a plain
// vector database.
//
// Corpus is the text-RAG engine: you hand it documents and it chunks, embeds,
// and indexes them. But sometimes you already HAVE vectors — from your own
// model, a batch job, another service — and you just want a fast, persistent
// approximate-nearest-neighbour store keyed by your own ids. That is this.
//
// It is a thin façade over the same HnswIndex the Corpus uses, so it inherits
// everything that engine already does well: parallel graph build, SQ8/PQ
// compression, SIMD distance, per-query ef, filtered search, a versioned
// on-disk format. What it adds is a five-method surface with no chunks, no
// documents, no embedder:
//
//   VectorStore store{dim};                 // or {dim, HnswConfig::accurate()}
//   store.add(id, vec);                      // your id, your vector
//   store.build();                           // one call after a batch
//   for (auto& [id, score] : store.search(query, 10)) { ... }
//   store.save("vectors.ragvec");            // and VectorStore::load(...)
//
// Vectors are unit-normalized on insert, so search returns cosine similarity in
// descending order. Below `brute_force_below` vectors it scans exactly (faster
// and exact at small n); above it, it builds and walks the HNSW graph.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/hnsw.hpp"

namespace rag::index {

// (id, similarity) — what a vector search returns. `id` is YOUR id, echoed back.
struct VectorHit {
    std::uint32_t id;
    float         score;   // cosine similarity in [-1, 1], descending
};

class VectorStore {
public:
    // A store of `dim`-dimensional vectors. The dimension is fixed for the
    // store's life; add() rejects a mismatch rather than corrupting the index.
    explicit VectorStore(std::size_t dim, HnswConfig cfg = {})
        : dim_(dim), cfg_(cfg), hnsw_(cfg) {}

    // Add (or replace) the vector for `id`. Re-adding an id upserts: the new
    // vector wins and the old one never returns, so an id appears at most once
    // in any result. The vector is copied and unit-normalized internally.
    // Fails with Errc::invalid_argument on a dimension mismatch.
    Result<void> add(std::uint32_t id, std::span<const float> vec);

    // Remove `id` (soft-delete/tombstone). Idempotent-ish: removing an unknown
    // id is a no-op success. Space is reclaimed on the next build()/compact().
    void remove(std::uint32_t id);

    // Build the ANN graph over everything added so far. Idempotent; call once
    // after a batch of add()s. Below `brute_force_below` this is a cheap no-op —
    // the store scans exactly instead. Safe to call again after more adds.
    Result<void> build();

    // k nearest neighbours of `query`, descending by cosine similarity.
    // `ef` overrides the configured search beam for THIS query only (0 = use
    // the default); it is the recall/latency dial and costs nothing to change.
    [[nodiscard]] std::vector<VectorHit>
    search(std::span<const float> query, std::size_t k, std::size_t ef = 0) const;

    // k-NN restricted to ids the predicate accepts, evaluated DURING the walk
    // (pre-filter), so a selective filter still returns k results rather than
    // running out. Uses brute force below the threshold.
    [[nodiscard]] std::vector<VectorHit>
    search(std::span<const float> query, std::size_t k,
           const std::function<bool(std::uint32_t)>& allow, std::size_t ef = 0) const;

    // ── Introspection ────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::size_t size()      const noexcept { return count_; }
    [[nodiscard]] bool        is_graph_built() const noexcept { return graph_built_; }
    // Heap bytes held by the index, so you can see what a compression setting
    // bought without guessing from process RSS. 0 before a graph is built.
    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        return graph_built_ ? hnsw_.memory_bytes()
                            : raw_.capacity() * sizeof(float) + ids_.capacity() * sizeof(std::uint32_t);
    }
    [[nodiscard]] const HnswConfig& config() const noexcept { return cfg_; }

    // Below this many vectors, search scans exactly (faster and exact); at or
    // above it, build() constructs the HNSW graph. Set before adding, or leave
    // the default. Mirrors CorpusConfig::hnsw_threshold.
    std::size_t brute_force_below = 2000;

    // ── Persistence ──────────────────────────────────────────────────────────
    // A self-describing, versioned, CRC-checked blob — reopening never rebuilds.
    [[nodiscard]] Result<void> save(const std::string& path) const;
    [[nodiscard]] static Result<VectorStore> load(const std::string& path);

    VectorStore(VectorStore&&) noexcept = default;
    VectorStore& operator=(VectorStore&&) noexcept = default;
    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;

private:
    std::size_t  dim_;
    HnswConfig   cfg_;
    // `mutable` so the const save() can compact away tombstones before
    // serializing — a lazy repair, idempotent and invisible to readers.
    mutable HnswIndex hnsw_;
    bool         graph_built_ = false;
    std::size_t  count_       = 0;

    // Small-corpus arena: parallel arrays of ids and unit-normalized vectors,
    // used for the exact brute-force path below the threshold. Once the graph
    // is built these still back save() so the store round-trips before/after a
    // build identically.
    std::vector<std::uint32_t> ids_;
    std::vector<float>         raw_;   // count_ * dim_, row-major, normalized
    std::vector<std::uint8_t>  live_;  // 1 = present, 0 = removed (brute path)

    [[nodiscard]] std::vector<VectorHit>
    brute_search(std::span<const float> query, std::size_t k,
                 const std::function<bool(std::uint32_t)>* allow) const;
};

} // namespace rag::index
