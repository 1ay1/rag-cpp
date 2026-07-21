// rag/sparse/splade.cpp — learned-sparse (SPLADE-style) retrieval.

#include "rag/sparse/splade.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace rag::sparse {

std::uint32_t SpladeIndex::intern(std::string_view term) {
    auto it = term_id_.find(std::string(term));
    if (it != term_id_.end()) return it->second;
    auto id = static_cast<std::uint32_t>(term_id_.size());
    term_id_.emplace(std::string(term), id);
    return id;
}

const std::uint32_t* SpladeIndex::lookup(std::string_view term) const {
    auto it = term_id_.find(std::string(term));
    return it == term_id_.end() ? nullptr : &it->second;
}

Result<SpladeIndex>
SpladeIndex::build(const index::Corpus& corpus, SpladeConfig cfg) {
    if (corpus.chunk_count() == 0)
        return std::unexpected(Error{Errc::empty_corpus, "splade: empty corpus"});
    SpladeIndex s;
    s.cfg_ = cfg;
    s.tok_ = corpus.tokenizer();

    // Pass 1: intern terms, document frequency, and co-occurrence counts.
    std::vector<std::vector<std::uint32_t>> chunk_terms;
    chunk_terms.reserve(corpus.chunk_count());
    std::unordered_map<std::uint32_t, std::uint32_t> df;
    // Co-occurrence: (min<<32|max) → count, within a chunk.
    std::unordered_map<std::uint64_t, std::uint32_t> cooc;

    for (const auto& ch : corpus.chunks()) {
        auto toks = s.tok_.tokenize(ch.indexed_text());
        std::vector<std::uint32_t> ids;
        ids.reserve(toks.size());
        for (auto& t : toks) ids.push_back(s.intern(t));
        // document frequency (unique terms)
        std::unordered_set<std::uint32_t> uniq(ids.begin(), ids.end());
        for (auto t : uniq) ++df[t];
        // co-occurrence over unique terms (bounded to keep the graph sane)
        std::vector<std::uint32_t> u(uniq.begin(), uniq.end());
        if (u.size() <= 64) {  // skip pathological chunks
            for (std::size_t i = 0; i < u.size(); ++i)
                for (std::size_t j = i + 1; j < u.size(); ++j) {
                    std::uint32_t a = u[i], b = u[j];
                    if (a > b) std::swap(a, b);
                    ++cooc[(static_cast<std::uint64_t>(a) << 32) | b];
                }
        }
        chunk_terms.push_back(std::move(ids));
    }

    const std::size_t V = s.term_id_.size();
    const auto N = static_cast<float>(corpus.chunk_count());
    s.idf_.assign(V, 0.0f);
    for (auto& [t, n] : df)
        s.idf_[t] = std::log(1.0f + (N - n + 0.5f) / (n + 0.5f));

    // Build expansion graph: per term, its top co-occurring neighbours by PMI-ish
    // weight = cooc / sqrt(df_a·df_b), normalized to the source term's idf.
    s.expand_.assign(V, {});
    if (V <= cfg.max_cooc_vocab) {
        std::vector<std::vector<std::pair<std::uint32_t, float>>> tmp(V);
        for (auto& [key, c] : cooc) {
            auto a = static_cast<std::uint32_t>(key >> 32);
            auto b = static_cast<std::uint32_t>(key & 0xffffffffu);
            float dfa = static_cast<float>(df[a]), dfb = static_cast<float>(df[b]);
            float w = static_cast<float>(c) / std::sqrt(dfa * dfb + 1.0f);
            tmp[a].emplace_back(b, w);
            tmp[b].emplace_back(a, w);
        }
        for (std::uint32_t t = 0; t < V; ++t) {
            auto& v = tmp[t];
            std::size_t keep = std::min(v.size(), cfg.expansion_terms);
            std::partial_sort(v.begin(), v.begin() + keep, v.end(),
                [](auto& x, auto& y) { return x.second > y.second; });
            v.resize(keep);
            s.expand_[t] = std::move(v);
        }
    }

    // Pass 2: encode each chunk into a sparse impact vector, build inverted idx.
    s.doc_vecs_.reserve(corpus.chunk_count());
    s.doc_ids_.reserve(corpus.chunk_count());
    s.inverted_.assign(V, {});
    std::size_t di = 0;
    for (const auto& ch : corpus.chunks()) {
        // term frequency in this chunk
        std::unordered_map<std::uint32_t, std::uint32_t> tf;
        for (auto t : chunk_terms[di]) ++tf[t];
        SparseVec vec;
        for (auto& [t, c] : tf) {
            // saturated weight: log(1+tf)·idf  (SPLADE log-saturation shape)
            float w = std::log(1.0f + static_cast<float>(c)) * s.idf_[t];
            if (w > cfg.min_weight) vec[t] = std::max(vec[t], w);
        }
        // NOTE: documents are stored WITHOUT expansion; expansion is applied to
        // the QUERY at search time (asymmetric SPLADE-doc, the efficient variant).
        for (auto& [t, w] : vec)
            s.inverted_[t].emplace_back(static_cast<std::uint32_t>(di), w);
        s.doc_vecs_.push_back(std::move(vec));
        s.doc_ids_.push_back(ch.id);
        ++di;
    }
    return s;
}

SparseVec SpladeIndex::encode(std::string_view text, bool expand) const {
    std::unordered_map<std::uint32_t, std::uint32_t> tf;
    for (auto& t : tok_.tokenize(text)) {
        if (auto* id = lookup(t)) ++tf[*id];
    }
    SparseVec vec;
    for (auto& [t, c] : tf) {
        float w = std::log(1.0f + static_cast<float>(c)) * idf_[t];
        if (w > cfg_.min_weight) vec[t] = std::max(vec[t], w);
    }
    if (expand) {
        // add damped weights for each source term's co-occurrence neighbours.
        SparseVec added;
        for (auto& [t, w] : vec)
            if (t < expand_.size())
                for (auto& [nbr, ew] : expand_[t]) {
                    float contrib = cfg_.expansion_decay * w * ew;
                    if (contrib > cfg_.min_weight)
                        added[nbr] = std::max(added[nbr], contrib);
                }
        for (auto& [t, w] : added)
            vec[t] = std::max(vec[t], w);
    }
    return vec;
}

std::vector<Hit> SpladeIndex::search(std::string_view query, std::size_t k) const {
    SparseVec q = encode(query, /*expand=*/true);
    if (q.empty()) return {};
    std::vector<float> acc(doc_vecs_.size(), 0.0f);
    for (auto& [t, qw] : q) {
        if (t >= inverted_.size()) continue;
        for (auto& [d, dw] : inverted_[t]) acc[d] += qw * dw;
    }
    std::vector<Hit> hits;
    for (std::size_t d = 0; d < acc.size(); ++d)
        if (acc[d] > 0.0f) hits.push_back({doc_ids_[d], Score{acc[d]}});
    std::size_t keep = std::min(hits.size(), k);
    std::partial_sort(hits.begin(), hits.begin() + keep, hits.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    hits.resize(keep);
    return hits;
}

} // namespace rag::sparse
