// rag/index/vector_store.cpp — the plain vector-DB façade over HnswIndex.

#include "rag/index/vector_store.hpp"

#include "rag/dense/simd.hpp"
#include "rag/store/container.hpp"
#include "rag/store/format.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace rag::index {

namespace {
// Normalize a copy of `vec` into `out` (length dim). Returns false on a zero
// vector, which has no direction and cannot be scored by cosine.
void normalize_into(std::span<const float> vec, std::span<float> out) {
    std::copy(vec.begin(), vec.end(), out.begin());
    dense::normalize(out);
}
} // namespace

Result<void> VectorStore::add(std::uint32_t id, std::span<const float> vec) {
    if (vec.size() != dim_)
        return fail<void>(Errc::invalid_argument,
                          "vector dimension " + std::to_string(vec.size()) +
                          " != store dimension " + std::to_string(dim_));

    // Upsert: if the id is already present in the brute arena, overwrite in
    // place; the graph (if built) tombstones the old and adds the new below.
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        if (live_[i] && ids_[i] == id) {
            normalize_into(vec, std::span<float>(raw_.data() + i * dim_, dim_));
            if (graph_built_) { hnsw_.remove(id); hnsw_.add(id, {raw_.data() + i * dim_, dim_}); }
            return {};
        }
    }

    ids_.push_back(id);
    live_.push_back(1);
    const std::size_t row = (ids_.size() - 1) * dim_;
    raw_.resize(ids_.size() * dim_);
    normalize_into(vec, std::span<float>(raw_.data() + row, dim_));
    ++count_;

    // If the graph is already built we keep it live by inserting incrementally.
    if (graph_built_) hnsw_.add(id, {raw_.data() + row, dim_});
    return {};
}

void VectorStore::remove(std::uint32_t id) {
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        if (live_[i] && ids_[i] == id) {
            live_[i] = 0;
            if (count_) --count_;
            if (graph_built_) hnsw_.remove(id);
            return;
        }
    }
}

Result<void> VectorStore::build() {
    // Below the threshold the brute path is exact and faster; nothing to build.
    if (count_ < brute_force_below) { graph_built_ = false; return {}; }
    if (graph_built_) return {};

    // Collect the live rows and bulk-build in parallel (the fast path).
    std::vector<std::size_t> live_rows;
    live_rows.reserve(count_);
    for (std::size_t i = 0; i < ids_.size(); ++i)
        if (live_[i]) live_rows.push_back(i);

    HnswIndex idx(cfg_);
    idx.build_batch(
        live_rows.size(),
        [&](std::size_t k) -> std::span<const float> {
            return {raw_.data() + live_rows[k] * dim_, dim_};
        },
        [&](std::size_t k) -> std::uint32_t { return ids_[live_rows[k]]; });
    hnsw_ = std::move(idx);
    graph_built_ = true;
    return {};
}

std::vector<VectorHit>
VectorStore::brute_search(std::span<const float> query, std::size_t k,
                          const std::function<bool(std::uint32_t)>* allow) const {
    // Query is normalized to a copy so callers need not pre-normalize.
    std::vector<float> q(query.begin(), query.end());
    dense::normalize(q);

    std::vector<VectorHit> all;
    all.reserve(count_);
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        if (!live_[i]) continue;
        if (allow && !(*allow)(ids_[i])) continue;
        float s = dense::dot(q, {raw_.data() + i * dim_, dim_});
        all.push_back({ids_[i], s});
    }
    const std::size_t take = std::min(k, all.size());
    std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(take), all.end(),
                      [](const VectorHit& a, const VectorHit& b) {
                          // Total order: score desc, then id asc, so ties are
                          // deterministic (same reason the dense path does it).
                          if (a.score != b.score) return a.score > b.score;
                          return a.id < b.id;
                      });
    all.resize(take);
    return all;
}

std::vector<VectorHit>
VectorStore::search(std::span<const float> query, std::size_t k, std::size_t ef) const {
    if (!graph_built_) return brute_search(query, k, nullptr);
    std::vector<float> q(query.begin(), query.end());
    dense::normalize(q);
    auto hits = hnsw_.search(q, k, ef);
    std::vector<VectorHit> out;
    out.reserve(hits.size());
    for (const auto& h : hits) out.push_back({h.chunk.get(), h.score.get()});
    return out;
}

std::vector<VectorHit>
VectorStore::search(std::span<const float> query, std::size_t k,
                    const std::function<bool(std::uint32_t)>& allow, std::size_t ef) const {
    if (!graph_built_) return brute_search(query, k, &allow);
    std::vector<float> q(query.begin(), query.end());
    dense::normalize(q);
    // search_filtered widens the beam by a multiplier (ef_boost) rather than an
    // absolute ef, to compensate for a selective predicate. A larger `ef`
    // request maps to a larger boost; 0 keeps the default 4x.
    float boost = ef ? std::max(1.0f, float(ef) / float(cfg_.ef_search)) : 4.0f;
    auto hits = hnsw_.search_filtered(q, k, allow, boost);
    std::vector<VectorHit> out;
    out.reserve(hits.size());
    for (const auto& h : hits) out.push_back({h.chunk.get(), h.score.get()});
    return out;
}

// ── Persistence ──────────────────────────────────────────────────────────────
// A .ragvec is a store::Container: a `meta` section carrying dim + config + the
// normalized brute arena (ids + live + vectors), and, when built, an `hnsw`
// section with the serialized graph. Reusing the sectioned container gets us the
// magic/version/CRC framing for free.

Result<void> VectorStore::save(const std::string& path) const {
    store::Writer w;
    w.u<std::uint32_t>(1);                                   // ragvec format version
    w.u<std::uint64_t>(dim_);
    w.u<std::uint64_t>(count_);
    w.u<std::uint8_t>(graph_built_ ? 1 : 0);
    // The HnswConfig fields we need to reconstruct behaviour.
    w.u<std::uint64_t>(cfg_.M);
    w.u<std::uint64_t>(cfg_.ef_construction);
    w.u<std::uint64_t>(cfg_.ef_search);
    w.u<std::uint8_t>(cfg_.drop_floats ? 1 : 0);
    w.u<std::uint64_t>(brute_force_below);
    // The arena (live rows only, so a save after many removes is compact).
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < ids_.size(); ++i) if (live_[i]) rows.push_back(i);
    w.u<std::uint64_t>(rows.size());
    for (std::size_t r : rows) {
        w.u<std::uint32_t>(ids_[r]);
        w.bytes(std::string_view(reinterpret_cast<const char*>(raw_.data() + r * dim_),
                                 dim_ * sizeof(float)));
    }

    store::Container c;
    c.put(store::Tag::meta, std::move(w.data()));
    if (graph_built_) {
        // Tombstones live only in memory: HnswIndex::serialize() does NOT persist
        // the deleted-id set, so a removed vector would come back on reload.
        // compact() physically rebuilds the graph without tombstoned or
        // superseded nodes, which makes the blob match what search() returns.
        // It is idempotent and a no-op when nothing was removed.
        hnsw_.compact();
        c.put(store::Tag::hnsw, hnsw_.serialize());
    }

    std::string blob = c.serialize();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return fail<void>(Errc::io_error, "cannot open '" + path + "' for writing");
    f.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    if (!f) return fail<void>(Errc::io_error, "write failed for '" + path + "'");
    return {};
}

Result<VectorStore> VectorStore::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail<VectorStore>(Errc::io_error, "cannot open '" + path + "'");
    std::ostringstream ss; ss << f.rdbuf();
    std::string blob = ss.str();

    auto cont = store::Container::parse(blob);   // verifies magic + version + CRC
    if (!cont) return std::unexpected(cont.error());
    const std::string* meta = cont->get(store::Tag::meta);
    if (!meta) return fail<VectorStore>(Errc::corrupt_index, "ragvec: missing meta section");

    store::Reader r(*meta);
    std::uint32_t ver = 0; std::uint64_t dim = 0, count = 0;
    std::uint8_t built = 0;
    std::uint64_t M = 16, efc = 200, efs = 64, bfb = 2000; std::uint8_t drop = 0;
    // Read the fixed header fields; r.ok() below catches any truncation. The
    // (void) casts consume the [[nodiscard]] since we validate via ok() once.
    (void)r.u(ver); (void)r.u(dim); (void)r.u(count); (void)r.u(built);
    (void)r.u(M); (void)r.u(efc); (void)r.u(efs); (void)r.u(drop); (void)r.u(bfb);
    if (!r.ok()) return fail<VectorStore>(Errc::corrupt_index, "ragvec: truncated header");
    if (ver != 1) return fail<VectorStore>(Errc::corrupt_index, "ragvec: unsupported version");

    HnswConfig cfg;
    cfg.M = M; cfg.ef_construction = efc; cfg.ef_search = efs; cfg.drop_floats = (drop != 0);
    VectorStore store(static_cast<std::size_t>(dim), cfg);
    store.brute_force_below = static_cast<std::size_t>(bfb);

    std::uint64_t rows = 0;
    if (!r.u(rows)) return fail<VectorStore>(Errc::corrupt_index, "ragvec: truncated row count");
    store.ids_.reserve(rows);
    store.raw_.reserve(rows * dim);
    store.live_.reserve(rows);
    for (std::uint64_t i = 0; i < rows; ++i) {
        std::uint32_t id = 0;
        if (!r.u(id)) return fail<VectorStore>(Errc::corrupt_index, "ragvec: truncated arena");
        std::string_view vec;
        if (!r.bytes(dim * sizeof(float), vec))
            return fail<VectorStore>(Errc::corrupt_index, "ragvec: truncated vector");
        store.ids_.push_back(id);
        store.live_.push_back(1);
        const float* fp = reinterpret_cast<const float*>(vec.data());
        store.raw_.insert(store.raw_.end(), fp, fp + dim);
    }
    store.count_ = static_cast<std::size_t>(count);

    if (built) {
        const std::string* h = cont->get(store::Tag::hnsw);
        if (!h) return fail<VectorStore>(Errc::corrupt_index, "ragvec: built flag set but no graph");
        auto idx = HnswIndex::deserialize(*h);
        if (!idx) return std::unexpected(idx.error());
        store.hnsw_ = std::move(*idx);
        store.graph_built_ = true;
    }
    return store;
}

} // namespace rag::index
