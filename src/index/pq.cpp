// rag/index/pq.cpp — Product Quantization codec + ADC search.

#include "rag/index/pq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "rag/store/format.hpp"

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
    std::size_t dim = data[0].size();
    if (dim == 0 || dim % cfg.m != 0)
        return std::unexpected(Error{Errc::invalid_argument, "pq: dim must be divisible by m"});
    if (cfg.ksub == 0 || cfg.ksub > 256)
        return std::unexpected(Error{Errc::invalid_argument, "pq: ksub must be 1..256"});

    ProductQuantizer pq;
    pq.cfg_ = cfg;
    pq.dim_ = dim;
    pq.dsub_ = dim / cfg.m;
    pq.centroids_.assign(cfg.m * cfg.ksub * pq.dsub_, 0.0f);
    std::mt19937_64 rng(cfg.seed);

    // Per-subspace k-means (Lloyd) with k-means++-lite (random distinct seeds).
    std::vector<float> sub(data.size() * pq.dsub_);
    for (std::size_t s = 0; s < cfg.m; ++s) {
        // gather this subspace's slice of every training vector
        for (std::size_t i = 0; i < data.size(); ++i)
            std::copy_n(data[i].data() + s * pq.dsub_, pq.dsub_, sub.data() + i * pq.dsub_);
        std::size_t ksub = std::min(cfg.ksub, data.size());
        // init centroids from random distinct samples
        std::uniform_int_distribution<std::size_t> pick(0, data.size() - 1);
        for (std::size_t c = 0; c < ksub; ++c) {
            std::size_t r = pick(rng);
            std::copy_n(sub.data() + r * pq.dsub_, pq.dsub_,
                        pq.centroids_.data() + (s * cfg.ksub + c) * pq.dsub_);
        }
        std::vector<std::size_t> assign(data.size(), 0);
        for (std::size_t it = 0; it < cfg.iters; ++it) {
            // assignment
            bool changed = false;
            for (std::size_t i = 0; i < data.size(); ++i) {
                const float* v = sub.data() + i * pq.dsub_;
                float bestd = std::numeric_limits<float>::infinity();
                std::size_t bestc = 0;
                for (std::size_t c = 0; c < ksub; ++c) {
                    float d = sub_l2(v, pq.centroid(s, c), pq.dsub_);
                    if (d < bestd) { bestd = d; bestc = c; }
                }
                if (assign[i] != bestc) { assign[i] = bestc; changed = true; }
            }
            // update
            std::vector<float> acc(ksub * pq.dsub_, 0.0f);
            std::vector<std::size_t> cnt(ksub, 0);
            for (std::size_t i = 0; i < data.size(); ++i) {
                const float* v = sub.data() + i * pq.dsub_;
                float* a = acc.data() + assign[i] * pq.dsub_;
                for (std::size_t d = 0; d < pq.dsub_; ++d) a[d] += v[d];
                ++cnt[assign[i]];
            }
            for (std::size_t c = 0; c < ksub; ++c)
                if (cnt[c])
                    for (std::size_t d = 0; d < pq.dsub_; ++d)
                        pq.centroids_[(s * cfg.ksub + c) * pq.dsub_ + d] = acc[c * pq.dsub_ + d] / (float)cnt[c];
            if (!changed && it > 0) break;
        }
    }
    return pq;
}

std::vector<std::uint8_t> ProductQuantizer::encode(std::span<const float> v) const {
    std::vector<std::uint8_t> code(cfg_.m, 0);
    for (std::size_t s = 0; s < cfg_.m; ++s) {
        const float* vs = v.data() + s * dsub_;
        float bestd = std::numeric_limits<float>::infinity();
        std::size_t bestc = 0;
        for (std::size_t c = 0; c < cfg_.ksub; ++c) {
            float d = sub_l2(vs, centroid(s, c), dsub_);
            if (d < bestd) { bestd = d; bestc = c; }
        }
        code[s] = static_cast<std::uint8_t>(bestc);
    }
    return code;
}

Vector ProductQuantizer::decode(std::span<const std::uint8_t> code) const {
    Vector v(dim_, 0.0f);
    for (std::size_t s = 0; s < cfg_.m && s < code.size(); ++s)
        std::copy_n(centroid(s, code[s]), dsub_, v.data() + s * dsub_);
    return v;
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
