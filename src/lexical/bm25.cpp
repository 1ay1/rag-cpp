// rag/lexical/bm25.cpp — Okapi BM25 implementation + binary serialization.

#include "rag/lexical/bm25.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace rag::lexical {

std::size_t Bm25Index::add(std::uint32_t id, std::string_view text) {
    auto terms = tok_.tokenize(text);
    if (terms.empty()) { doc_len_[id] = 0; return 0; }

    // Count term frequencies within this doc.
    std::unordered_map<std::string, std::uint32_t> tf;
    tf.reserve(terms.size());
    for (auto& t : terms) ++tf[t];

    for (auto& [term, freq] : tf) {
        auto& plist = postings_[term];
        // Keep postings sorted by doc id for stable serialization / merge.
        Posting p{id, freq};
        auto it = std::lower_bound(plist.begin(), plist.end(), p,
            [](const Posting& a, const Posting& b) { return a.doc < b.doc; });
        if (it != plist.end() && it->doc == id) it->tf = freq; // overwrite on re-add
        else plist.insert(it, p);
    }

    doc_len_[id]   = static_cast<std::uint32_t>(terms.size());
    total_len_    += terms.size();
    finalized_     = false;
    return terms.size();
}

void Bm25Index::finalize() {
    avgdl_ = doc_len_.empty() ? 0.0f
           : static_cast<float>(total_len_ / static_cast<double>(doc_len_.size()));

    // Materialize the dense doc-length mirror. Worth it only when ordinals are
    // actually dense (a corpus assigns them contiguously); if the id space is
    // sparse we skip it and scoring keeps using the hash map.
    dense_len_.clear();
    max_doc_ = 0;
    for (const auto& [d, _] : doc_len_) max_doc_ = std::max(max_doc_, d);
    if (!doc_len_.empty() && static_cast<std::size_t>(max_doc_) < doc_len_.size() * 2) {
        dense_len_.assign(static_cast<std::size_t>(max_doc_) + 1, 0);
        for (const auto& [d, l] : doc_len_) dense_len_[d] = l;
    }

    // Evaluate the query-independent half of every posting's score once.
    {
        const float avgdl = avgdl_ > 0 ? avgdl_ : 1.0f;
        const float k1 = params_.k1, b = params_.b;
        const float inv_avgdl = 1.0f / avgdl;

        std::size_t total = 0;
        for (const auto& [term, plist] : postings_) total += plist.size();
        pw_.resize(total);
        pw_span_.clear();
        pw_span_.reserve(postings_.size());

        std::uint32_t cursor = 0;
        for (const auto& [term, plist] : postings_) {
            const std::uint32_t begin = cursor;
            for (const auto& p : plist) {
                float dl = avgdl;
                if (!dense_len_.empty() && p.doc < dense_len_.size()) {
                    if (const std::uint32_t raw = dense_len_[p.doc]) dl = static_cast<float>(raw);
                } else if (auto it = doc_len_.find(p.doc); it != doc_len_.end()) {
                    dl = static_cast<float>(it->second);
                }
                const float f   = static_cast<float>(p.tf);
                const float num = f * (k1 + 1.0f);
                const float den = f + k1 * (1.0f - b + b * dl * inv_avgdl);
                pw_[cursor++] = den > 0.0f ? num / den : 0.0f;
            }
            pw_span_.emplace(term, std::pair{begin, cursor});
        }
    }
    finalized_ = true;
}

float Bm25Index::idf(std::size_t n_t) const {
    const double N = static_cast<double>(doc_len_.size());
    // BM25+ smoothed idf: strictly positive.
    return static_cast<float>(std::log(1.0 + (N - static_cast<double>(n_t) + 0.5) /
                                             (static_cast<double>(n_t) + 0.5)));
}

void Bm25Index::term_coverage(const std::vector<std::string>& q_terms,
                              std::span<const std::uint32_t> docs,
                              std::vector<std::uint32_t>& out) const {
    out.assign(docs.size(), 0);
    if (docs.empty() || q_terms.empty()) return;

    // Map doc id -> its slot in `out`. `docs` is a candidate list (tens of
    // entries), so a small sorted index beats a hash map and needs no
    // allocation beyond one vector.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> slot;
    slot.reserve(docs.size());
    for (std::uint32_t i = 0; i < docs.size(); ++i) slot.emplace_back(docs[i], i);
    std::sort(slot.begin(), slot.end());

    // One pass per DISTINCT query term. Because each term is visited once, a
    // doc can be credited at most once per term — which is exactly the
    // "distinct terms covered" semantics, with no per-candidate dedup needed.
    std::vector<std::string> uniq = q_terms;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    for (const auto& t : uniq) {
        auto pit = postings_.find(t);
        if (pit == postings_.end()) continue;
        const auto& plist = pit->second;
        // Postings are sorted by doc id and so is `slot`: walk both once
        // rather than searching one inside the other.
        std::size_t pi = 0, si = 0;
        while (pi < plist.size() && si < slot.size()) {
            if (plist[pi].doc < slot[si].first)      ++pi;
            else if (slot[si].first < plist[pi].doc) ++si;
            else { ++out[slot[si].second]; ++pi; ++si; }
        }
    }
}

float Bm25Index::score_doc(const std::vector<std::string>& q_terms,
                           std::uint32_t doc_id) const {
    auto dl_it = doc_len_.find(doc_id);
    if (dl_it == doc_len_.end()) return 0.0f;
    const float dl = static_cast<float>(dl_it->second);
    const float avgdl = avgdl_ > 0 ? avgdl_ : 1.0f;
    const float k1 = params_.k1, b = params_.b;

    float score = 0.0f;
    for (const auto& term : q_terms) {
        auto pit = postings_.find(term);
        if (pit == postings_.end()) continue;
        const auto& plist = pit->second;
        // Binary search the posting for this doc.
        auto it = std::lower_bound(plist.begin(), plist.end(), doc_id,
            [](const Posting& p, std::uint32_t d) { return p.doc < d; });
        if (it == plist.end() || it->doc != doc_id) continue;
        const float f = static_cast<float>(it->tf);
        const float num = f * (k1 + 1.0f);
        const float den = f + k1 * (1.0f - b + b * dl / avgdl);
        score += idf(plist.size()) * (num / den);
    }
    return score;
}

std::vector<Hit> Bm25Index::search(std::string_view query, std::size_t k) const {
    if (doc_len_.empty()) return {};
    auto q_terms = tok_.tokenize(query);
    if (q_terms.empty()) return {};

    const float avgdl = avgdl_ > 0 ? avgdl_ : 1.0f;
    const float k1 = params_.k1, b = params_.b;
    const float inv_avgdl = 1.0f / avgdl;

    // Fast path: dense ordinals + precomputed weights. Per posting this is one
    // multiply-add over two sequential streams (the posting's doc id, and its
    // precomputed weight) — no division, no doc-length lookup, no hashing.
    if (!dense_len_.empty() && !pw_.empty()) {
        std::vector<float>         acc(dense_len_.size(), 0.0f);
        std::vector<std::uint32_t> touched;
        touched.reserve(1024);

        for (const auto& term : q_terms) {
            auto pit = postings_.find(term);
            if (pit == postings_.end()) continue;
            auto sit = pw_span_.find(term);
            if (sit == pw_span_.end()) continue;
            const auto& plist = pit->second;
            const float term_idf = idf(plist.size());
            const float* w = pw_.data() + sit->second.first;

            const std::size_t m = plist.size();
            for (std::size_t i = 0; i < m; ++i) {
                const std::uint32_t d = plist[i].doc;
                if (d >= acc.size()) continue;
                if (acc[d] == 0.0f) touched.push_back(d);
                acc[d] += term_idf * w[i];
            }
        }

        std::vector<Hit> hits;
        hits.reserve(touched.size());
        for (std::uint32_t d : touched) hits.push_back(Hit{ChunkId{d}, Score{acc[d]}});

        const std::size_t kk = std::min(k, hits.size());
        std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(kk),
                          hits.end(),
                          [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
        hits.resize(kk);
        return hits;
    }

    // Fast path: dense ordinals. Accumulate into a flat array and track which
    // slots were touched, so we neither hash per posting nor scan the whole
    // corpus afterwards. This is the inner loop of every lexical query.
    if (!dense_len_.empty()) {
        std::vector<float>         acc(dense_len_.size(), 0.0f);
        std::vector<std::uint32_t> touched;
        touched.reserve(1024);

        for (const auto& term : q_terms) {
            auto pit = postings_.find(term);
            if (pit == postings_.end()) continue;
            const auto& plist = pit->second;
            const float term_idf = idf(plist.size());
            for (const auto& p : plist) {
                if (p.doc >= acc.size()) continue;
                const std::uint32_t dl_raw = dense_len_[p.doc];
                const float dl  = dl_raw ? static_cast<float>(dl_raw) : avgdl;
                const float f   = static_cast<float>(p.tf);
                const float num = f * (k1 + 1.0f);
                const float den = f + k1 * (1.0f - b + b * dl * inv_avgdl);
                if (acc[p.doc] == 0.0f) touched.push_back(p.doc);
                acc[p.doc] += term_idf * (num / den);
            }
        }

        std::vector<Hit> hits;
        hits.reserve(touched.size());
        for (std::uint32_t d : touched) hits.push_back(Hit{ChunkId{d}, Score{acc[d]}});

        const std::size_t kk = std::min(k, hits.size());
        std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(kk),
                          hits.end(),
                          [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
        hits.resize(kk);
        return hits;
    }

    // Fallback: sparse ordinal space, accumulate in a hash map.
    std::unordered_map<std::uint32_t, float> acc;
    for (const auto& term : q_terms) {
        auto pit = postings_.find(term);
        if (pit == postings_.end()) continue;
        const auto& plist = pit->second;
        const float term_idf = idf(plist.size());
        for (const auto& p : plist) {
            auto dl_it = doc_len_.find(p.doc);
            const float dl = dl_it == doc_len_.end() ? avgdl
                           : static_cast<float>(dl_it->second);
            const float f = static_cast<float>(p.tf);
            const float num = f * (k1 + 1.0f);
            const float den = f + k1 * (1.0f - b + b * dl * inv_avgdl);
            acc[p.doc] += term_idf * (num / den);
        }
    }

    std::vector<Hit> hits;
    hits.reserve(acc.size());
    for (auto& [doc, sc] : acc) hits.push_back(Hit{ChunkId{doc}, Score{sc}});

    const std::size_t kk = std::min(k, hits.size());
    std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(kk),
                      hits.end(),
                      [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    hits.resize(kk);
    return hits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization — a compact, versioned little-endian blob.
// ─────────────────────────────────────────────────────────────────────────────
namespace {
constexpr std::uint32_t kMagic   = 0x314D4232; // "2BM1"
constexpr std::uint32_t kVersion = 1;

template <class T> void put(std::string& out, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const char* p = reinterpret_cast<const char*>(&v);
    out.append(p, p + sizeof(T));
}
void put_str(std::string& out, std::string_view s) {
    put<std::uint32_t>(out, static_cast<std::uint32_t>(s.size()));
    out.append(s.data(), s.size());
}
template <class T> bool get(std::string_view& in, T& v) {
    if (in.size() < sizeof(T)) return false;
    std::memcpy(&v, in.data(), sizeof(T));
    in.remove_prefix(sizeof(T));
    return true;
}
bool get_str(std::string_view& in, std::string& s) {
    std::uint32_t n;
    if (!get(in, n) || in.size() < n) return false;
    s.assign(in.data(), n);
    in.remove_prefix(n);
    return true;
}
} // namespace

std::string Bm25Index::serialize() const {
    std::string out;
    put(out, kMagic);
    put(out, kVersion);
    put(out, params_.k1);
    put(out, params_.b);
    put<std::uint64_t>(out, static_cast<std::uint64_t>(total_len_));
    // doc lengths
    put<std::uint32_t>(out, static_cast<std::uint32_t>(doc_len_.size()));
    for (auto& [doc, len] : doc_len_) { put(out, doc); put(out, len); }
    // postings
    put<std::uint32_t>(out, static_cast<std::uint32_t>(postings_.size()));
    for (auto& [term, plist] : postings_) {
        put_str(out, term);
        put<std::uint32_t>(out, static_cast<std::uint32_t>(plist.size()));
        for (auto& p : plist) { put(out, p.doc); put(out, p.tf); }
    }
    return out;
}

Result<Bm25Index> Bm25Index::deserialize(std::string_view blob) {
    Bm25Index idx;
    std::uint32_t magic, version;
    if (!get(blob, magic) || magic != kMagic)   return fail<Bm25Index>(Errc::corrupt_index, "bad magic");
    if (!get(blob, version) || version != kVersion) return fail<Bm25Index>(Errc::corrupt_index, "bad version");
    if (!get(blob, idx.params_.k1) || !get(blob, idx.params_.b))
        return fail<Bm25Index>(Errc::corrupt_index, "params");
    std::uint64_t total;
    if (!get(blob, total)) return fail<Bm25Index>(Errc::corrupt_index, "total_len");
    idx.total_len_ = static_cast<double>(total);

    std::uint32_t ndocs;
    if (!get(blob, ndocs)) return fail<Bm25Index>(Errc::corrupt_index, "ndocs");
    idx.doc_len_.reserve(ndocs);
    for (std::uint32_t i = 0; i < ndocs; ++i) {
        std::uint32_t doc, len;
        if (!get(blob, doc) || !get(blob, len)) return fail<Bm25Index>(Errc::corrupt_index, "doc_len");
        idx.doc_len_[doc] = len;
    }
    std::uint32_t nterms;
    if (!get(blob, nterms)) return fail<Bm25Index>(Errc::corrupt_index, "nterms");
    idx.postings_.reserve(nterms);
    for (std::uint32_t i = 0; i < nterms; ++i) {
        std::string term;
        std::uint32_t np;
        if (!get_str(blob, term) || !get(blob, np)) return fail<Bm25Index>(Errc::corrupt_index, "term");
        std::vector<Posting> plist(np);
        for (auto& p : plist)
            if (!get(blob, p.doc) || !get(blob, p.tf)) return fail<Bm25Index>(Errc::corrupt_index, "posting");
        idx.postings_.emplace(std::move(term), std::move(plist));
    }
    idx.finalize();
    return idx;
}

} // namespace rag::lexical
