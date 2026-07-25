// rag/index/pq.cpp — Product Quantization codec + ADC search.

#include "rag/index/pq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "rag/store/format.hpp"
#include "rag/util/parallel.hpp"

namespace rag::index {
namespace {

float sub_dot(const float* a, const float* b, std::size_t n) {
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
float sub_l2(const float* a, const float* b, std::size_t n) {
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { float d = a[i] - b[i]; s += d * d; }
    return s;
}

} // namespace

Result<ProductQuantizer>
ProductQuantizer::train(std::span<const Vector> data, PqConfig cfg) {
    if (data.empty()) return std::unexpected(Error{Errc::invalid_argument, "pq: no training data"});
    const std::size_t dim = data[0].size();
    // Flatten once and delegate: one training implementation, not two.
    std::vector<float> flat(data.size() * dim);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != dim)
            return std::unexpected(Error{Errc::invalid_argument, "pq: ragged training data"});
        std::copy(data[i].begin(), data[i].end(), flat.begin() + i * dim);
    }
    return train_flat(flat, data.size(), dim, cfg);
}

Result<ProductQuantizer>
ProductQuantizer::train_flat(std::span<const float> data, std::size_t n, std::size_t dim,
                            PqConfig cfg) {
    if (n == 0) return std::unexpected(Error{Errc::invalid_argument, "pq: no training data"});
    if (dim == 0 || cfg.m == 0 || dim % cfg.m != 0)
        return std::unexpected(Error{Errc::invalid_argument, "pq: dim must be divisible by m"});
    if (cfg.ksub == 0 || cfg.ksub > 256)
        return std::unexpected(Error{Errc::invalid_argument, "pq: ksub must be 1..256"});
    if (data.size() < n * dim)
        return std::unexpected(Error{Errc::invalid_argument, "pq: short training arena"});

    ProductQuantizer pq;
    pq.cfg_  = cfg;
    pq.dim_  = dim;
    pq.dsub_ = dim / cfg.m;
    pq.centroids_.assign(cfg.m * cfg.ksub * pq.dsub_, 0.0f);

    const std::size_t dsub = pq.dsub_;
    const std::size_t ksub = std::min(cfg.ksub, n);

    // The m subspaces are fully independent k-means problems over disjoint
    // slices of the data and disjoint slices of centroids_, so they run
    // concurrently with no synchronization at all.
    //
    // Dynamic scheduling with an explicit worker count, not parallel_for:
    // there are only m (8-64) items, far below kParallelMin, so parallel_for
    // would run them serially — and each item is a full k-means costing
    // hundreds of milliseconds, which is exactly the coarse/uneven shape
    // parallel_for_dynamic exists for. Using transient threads also keeps this
    // callable from inside a pool worker.
    util::parallel_for_dynamic(cfg.m, util::max_workers(), [&](std::size_t s) {
        // Gather this subspace's slice contiguously: the k-means inner loop
        // then walks it with unit stride instead of striding by `dim`.
        std::vector<float> sub(n * dsub);
        for (std::size_t i = 0; i < n; ++i)
            std::copy_n(data.data() + i * dim + s * dsub, dsub, sub.data() + i * dsub);

        float* cents = pq.centroids_.data() + s * cfg.ksub * dsub;
        // Per-subspace RNG stream so the result is deterministic regardless of
        // how the subspaces are scheduled across threads.
        std::mt19937_64 rng(cfg.seed ^ (0x9E3779B97F4A7C15ull * (s + 1)));

        // k-means++ seeding. The previous code drew k centroids uniformly WITH
        // REPLACEMENT, so duplicate seeds were common; a duplicate can never
        // win the strict-less-than assignment test, so those centroids stayed
        // dead for the whole run and the effective codebook was smaller than
        // requested. k-means++ picks each seed proportional to its squared
        // distance from the nearest existing seed, which both removes the
        // duplicate problem and gives a materially better local optimum.
        std::vector<float> d2(n, std::numeric_limits<float>::infinity());
        {
            std::uniform_int_distribution<std::size_t> pick0(0, n - 1);
            std::copy_n(sub.data() + pick0(rng) * dsub, dsub, cents);
            for (std::size_t c = 1; c < ksub; ++c) {
                double total = 0.0;
                const float* prev = cents + (c - 1) * dsub;
                for (std::size_t i = 0; i < n; ++i) {
                    float d = sub_l2(sub.data() + i * dsub, prev, dsub);
                    if (d < d2[i]) d2[i] = d;
                    total += d2[i];
                }
                std::size_t chosen = 0;
                if (total <= 0.0) {
                    // All remaining points coincide with a chosen centroid;
                    // fall back to a uniform draw so the slot is still filled.
                    std::uniform_int_distribution<std::size_t> u(0, n - 1);
                    chosen = u(rng);
                } else {
                    std::uniform_real_distribution<double> u(0.0, total);
                    double target = u(rng), acc = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        acc += d2[i];
                        if (acc >= target) { chosen = i; break; }
                        chosen = i;
                    }
                }
                std::copy_n(sub.data() + chosen * dsub, dsub, cents + c * dsub);
            }
        }

        std::vector<std::uint32_t> assign(n, 0);
        std::vector<float>         acc(ksub * dsub);
        std::vector<std::uint32_t> cnt(ksub);
        for (std::size_t it = 0; it < cfg.iters; ++it) {
            bool changed = false;
            for (std::size_t i = 0; i < n; ++i) {
                const float* v = sub.data() + i * dsub;
                float bestd = std::numeric_limits<float>::infinity();
                std::uint32_t bestc = 0;
                for (std::size_t c = 0; c < ksub; ++c) {
                    float d = sub_l2(v, cents + c * dsub, dsub);
                    if (d < bestd) { bestd = d; bestc = static_cast<std::uint32_t>(c); }
                }
                if (assign[i] != bestc) { assign[i] = bestc; changed = true; }
            }

            std::fill(acc.begin(), acc.end(), 0.0f);
            std::fill(cnt.begin(), cnt.end(), 0u);
            for (std::size_t i = 0; i < n; ++i) {
                const float* v = sub.data() + i * dsub;
                float* a = acc.data() + assign[i] * dsub;
                for (std::size_t d = 0; d < dsub; ++d) a[d] += v[d];
                ++cnt[assign[i]];
            }
            for (std::size_t c = 0; c < ksub; ++c) {
                if (cnt[c]) {
                    for (std::size_t d = 0; d < dsub; ++d)
                        cents[c * dsub + d] = acc[c * dsub + d] / static_cast<float>(cnt[c]);
                } else {
                    // Empty cluster: a wasted code point costs recall directly,
                    // so respawn it on the point furthest from its own centroid
                    // (the standard Lloyd repair) instead of leaving it dead.
                    float worst = -1.0f;
                    std::size_t worst_i = 0;
                    for (std::size_t i = 0; i < n; ++i) {
                        float d = sub_l2(sub.data() + i * dsub, cents + assign[i] * dsub, dsub);
                        if (d > worst) { worst = d; worst_i = i; }
                    }
                    std::copy_n(sub.data() + worst_i * dsub, dsub, cents + c * dsub);
                    assign[worst_i] = static_cast<std::uint32_t>(c);
                    changed = true;
                }
            }
            if (!changed && it > 0) break;
        }
    });
    return pq;
}

std::vector<std::uint8_t> ProductQuantizer::encode(std::span<const float> v) const {
    std::vector<std::uint8_t> code(cfg_.m, 0);
    encode_into(v, code);
    return code;
}

void ProductQuantizer::encode_into(std::span<const float> v,
                                   std::span<std::uint8_t> out) const noexcept {
    for (std::size_t s = 0; s < cfg_.m && s < out.size(); ++s) {
        const float* vs = v.data() + s * dsub_;
        float bestd = std::numeric_limits<float>::infinity();
        std::size_t bestc = 0;
        for (std::size_t c = 0; c < cfg_.ksub; ++c) {
            float d = sub_l2(vs, centroid(s, c), dsub_);
            if (d < bestd) { bestd = d; bestc = c; }
        }
        out[s] = static_cast<std::uint8_t>(bestc);
    }
}

void ProductQuantizer::encode_flat(std::span<const float> data, std::size_t n,
                                   std::span<std::uint8_t> out) const {
    if (dim_ == 0 || out.size() < n * cfg_.m) return;
    // Encoding is n independent nearest-centroid searches (m*ksub*dsub flops
    // each) writing to disjoint output rows — embarrassingly parallel.
    util::parallel_for(n, [&](std::size_t i) {
        encode_into(std::span<const float>(data.data() + i * dim_, dim_),
                    out.subspan(i * cfg_.m, cfg_.m));
    });
}

Vector ProductQuantizer::decode(std::span<const std::uint8_t> code) const {
    Vector v;
    decode_into(code, v);
    return v;
}

void ProductQuantizer::decode_into(std::span<const std::uint8_t> code,
                                   std::vector<float>& out) const {
    out.assign(dim_, 0.0f);
    for (std::size_t s = 0; s < cfg_.m && s < code.size(); ++s)
        std::copy_n(centroid(s, code[s]), dsub_, out.data() + s * dsub_);
}

std::vector<float> ProductQuantizer::adc_table(std::span<const float> query) const {
    // table[s*ksub + c] = query_sub_s · centroid(s,c)
    std::vector<float> table(cfg_.m * cfg_.ksub, 0.0f);
    for (std::size_t s = 0; s < cfg_.m; ++s) {
        const float* qs = query.data() + s * dsub_;
        for (std::size_t c = 0; c < cfg_.ksub; ++c)
            table[s * cfg_.ksub + c] = sub_dot(qs, centroid(s, c), dsub_);
    }
    return table;
}

float ProductQuantizer::adc_score(std::span<const std::uint8_t> code,
                                  std::span<const float> table) const noexcept {
    float s = 0.0f;
    for (std::size_t sub = 0; sub < cfg_.m && sub < code.size(); ++sub)
        s += table[sub * cfg_.ksub + code[sub]];
    return s;
}

void ProductQuantizer::add(std::uint32_t id, std::span<const float> v) {
    auto code = encode(v);
    ids_.push_back(id);
    codes_.insert(codes_.end(), code.begin(), code.end());
}

std::vector<Hit> ProductQuantizer::search(std::span<const float> query, std::size_t k) const {
    auto table = adc_table(query);
    std::vector<Hit> hits;
    hits.reserve(ids_.size());
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        std::span<const std::uint8_t> code(codes_.data() + i * cfg_.m, cfg_.m);
        hits.push_back({ChunkId{ids_[i]}, Score{adc_score(code, table)}});
    }
    std::size_t keep = std::min(hits.size(), k);
    std::partial_sort(hits.begin(), hits.begin() + keep, hits.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    hits.resize(keep);
    return hits;
}

std::string ProductQuantizer::serialize() const {
    store::Writer w;
    w.bytes("2PQ1");
    w.u<std::uint32_t>((std::uint32_t)cfg_.m);
    w.u<std::uint32_t>((std::uint32_t)cfg_.ksub);
    w.u<std::uint32_t>((std::uint32_t)dim_);
    w.u<std::uint32_t>((std::uint32_t)dsub_);
    w.u<std::uint32_t>((std::uint32_t)centroids_.size());
    for (float f : centroids_) w.u<float>(f);
    w.u<std::uint32_t>((std::uint32_t)ids_.size());
    for (auto id : ids_) w.u<std::uint32_t>(id);
    w.u<std::uint32_t>((std::uint32_t)codes_.size());
    w.bytes(std::string_view(reinterpret_cast<const char*>(codes_.data()), codes_.size()));
    return std::move(w.data());
}

Result<ProductQuantizer> ProductQuantizer::deserialize(std::string_view blob) {
    store::Reader r(blob);
    std::string_view magic;
    if (!r.bytes(4, magic) || magic != "2PQ1")
        return std::unexpected(Error{Errc::corrupt_index, "pq: bad magic"});
    ProductQuantizer pq;
    std::uint32_t m, ksub, dim, dsub, ncent;
    if (!r.u(m) || !r.u(ksub) || !r.u(dim) || !r.u(dsub) || !r.u(ncent))
        return std::unexpected(Error{Errc::corrupt_index, "pq: header"});
    pq.cfg_.m = m; pq.cfg_.ksub = ksub; pq.dim_ = dim; pq.dsub_ = dsub;
    pq.centroids_.resize(ncent);
    for (auto& f : pq.centroids_) if (!r.u(f)) return std::unexpected(Error{Errc::corrupt_index, "pq: centroids"});
    std::uint32_t nid;
    if (!r.u(nid)) return std::unexpected(Error{Errc::corrupt_index, "pq: ids"});
    pq.ids_.resize(nid);
    for (auto& id : pq.ids_) if (!r.u(id)) return std::unexpected(Error{Errc::corrupt_index, "pq: id"});
    std::uint32_t nc;
    if (!r.u(nc)) return std::unexpected(Error{Errc::corrupt_index, "pq: codes"});
    std::string_view cv;
    if (!r.bytes(nc, cv)) return std::unexpected(Error{Errc::corrupt_index, "pq: code bytes"});
    pq.codes_.assign(cv.begin(), cv.end());
    return pq;
}

} // namespace rag::index
