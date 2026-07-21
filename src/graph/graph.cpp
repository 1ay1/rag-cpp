// rag/graph/graph.cpp — GraphRAG implementation (model-free by default).

#include "rag/graph/graph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "rag/dense/simd.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::graph {
namespace {

// ── Link mining ────────────────────────────────────────────────────────────
// Extract candidate reference tokens from a document body: markdown link
// targets [txt](target), bare http(s) URLs, and [[wikilinks]]. We return the
// raw targets; resolution against corpus URIs/titles happens in build().
std::vector<std::string> mine_links(std::string_view s) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        // [[wikilink]]
        if (s[i] == '[' && i + 1 < s.size() && s[i + 1] == '[') {
            auto end = s.find("]]", i + 2);
            if (end != std::string_view::npos) {
                out.emplace_back(s.substr(i + 2, end - (i + 2)));
                i = end + 1;
                continue;
            }
        }
        // markdown [text](target)
        if (s[i] == ']' && i + 1 < s.size() && s[i + 1] == '(') {
            auto end = s.find(')', i + 2);
            if (end != std::string_view::npos) {
                out.emplace_back(s.substr(i + 2, end - (i + 2)));
                i = end;
                continue;
            }
        }
        // bare URL
        if (s.compare(i, 4, "http") == 0) {
            std::size_t j = i;
            while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j])) &&
                   s[j] != ')' && s[j] != ']' && s[j] != '>' && s[j] != '"')
                ++j;
            out.emplace_back(s.substr(i, j - i));
            i = j;
        }
    }
    return out;
}

// A loose canonical key for URI/title matching: basename without extension,
// lowercased, non-alnum stripped.
std::string canon(std::string_view s) {
    // strip fragment / query
    if (auto h = s.find('#'); h != std::string_view::npos) s = s.substr(0, h);
    if (auto q = s.find('?'); q != std::string_view::npos) s = s.substr(0, q);
    // basename
    if (auto sl = s.find_last_of("/\\"); sl != std::string_view::npos) s = s.substr(sl + 1);
    // drop extension
    if (auto dot = s.find_last_of('.'); dot != std::string_view::npos && dot > 0) s = s.substr(0, dot);
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (std::isalnum(static_cast<unsigned char>(c)))
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

// Split into naive sentences (for extractive summaries).
std::vector<std::string_view> sentences(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '.' || s[i] == '!' || s[i] == '?' || s[i] == '\n') {
            auto piece = s.substr(start, i - start + 1);
            // trim
            while (!piece.empty() && std::isspace(static_cast<unsigned char>(piece.front())))
                piece.remove_prefix(1);
            while (!piece.empty() && std::isspace(static_cast<unsigned char>(piece.back())))
                piece.remove_suffix(1);
            if (piece.size() > 12) out.push_back(piece);
            start = i + 1;
        }
    }
    if (start < s.size()) {
        auto piece = s.substr(start);
        while (!piece.empty() && std::isspace(static_cast<unsigned char>(piece.front())))
            piece.remove_prefix(1);
        while (!piece.empty() && std::isspace(static_cast<unsigned char>(piece.back())))
            piece.remove_suffix(1);
        if (piece.size() > 12) out.push_back(piece);
    }
    return out;
}

// Mean of a document's chunk embeddings (unit-normalized), if any exist.
std::optional<std::vector<float>>
doc_centroid(const index::Corpus& corpus, DocId doc) {
    std::vector<float> acc;
    std::size_t n = 0;
    for (const auto& ch : corpus.chunks()) {
        if (ch.doc.get() != doc.get()) continue;
        if (ch.embedding.empty()) continue;
        if (acc.empty()) acc.assign(ch.embedding.size(), 0.0f);
        if (acc.size() != ch.embedding.size()) continue;
        for (std::size_t i = 0; i < acc.size(); ++i) acc[i] += ch.embedding[i];
        ++n;
    }
    if (n == 0) return std::nullopt;
    dense::normalize(acc);
    return acc;
}

// Bag of stemmed terms for a document (union over its chunks' indexed text).
std::unordered_set<std::string>
doc_terms(const index::Corpus& corpus, DocId doc) {
    std::unordered_set<std::string> bag;
    const auto& tok = corpus.tokenizer();
    if (const Document* d = corpus.document(doc)) {
        for (auto& t : tok.tokenize(d->title)) bag.insert(t);
    }
    for (const auto& ch : corpus.chunks()) {
        if (ch.doc.get() != doc.get()) continue;
        for (auto& t : tok.tokenize(ch.text)) bag.insert(t);
    }
    return bag;
}

float jaccard(const std::unordered_set<std::string>& a,
              const std::unordered_set<std::string>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    const auto& small = a.size() < b.size() ? a : b;
    const auto& big   = a.size() < b.size() ? b : a;
    std::size_t inter = 0;
    for (const auto& t : small) if (big.count(t)) ++inter;
    std::size_t uni = a.size() + b.size() - inter;
    return uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

} // namespace

std::span<const std::pair<std::uint32_t, float>>
DocGraph::neighbours(std::uint32_t doc) const {
    if (doc >= adj_.size()) return {};
    return adj_[doc];
}

std::string_view DocGraph::community_summary(std::uint32_t id) const {
    for (const auto& c : communities_)
        if (c.id == id) return c.summary;
    return {};
}

// ── Personalized PageRank ──────────────────────────────────────────────────
std::vector<float>
DocGraph::personalized_pagerank(
    const std::vector<std::pair<std::uint32_t, float>>& restart) const {
    const std::size_t n = adj_.size();
    std::vector<float> teleport(n, 0.0f);
    float tsum = 0.0f;
    for (auto [d, m] : restart) if (d < n) { teleport[d] += m; tsum += m; }
    if (tsum <= 0.0f) {                       // uniform fallback
        for (auto& x : teleport) x = 1.0f / static_cast<float>(std::max<std::size_t>(n, 1));
    } else {
        for (auto& x : teleport) x /= tsum;
    }
    // Row-normalized weighted adjacency, applied via power iteration.
    std::vector<float> deg(n, 0.0f);
    for (std::size_t u = 0; u < n; ++u)
        for (auto [v, w] : adj_[u]) deg[u] += w;

    std::vector<float> rank = teleport, next(n, 0.0f);
    const float a = cfg_.ppr_restart;
    for (std::size_t it = 0; it < cfg_.ppr_iters; ++it) {
        std::fill(next.begin(), next.end(), 0.0f);
        float dangling = 0.0f;
        for (std::size_t u = 0; u < n; ++u) {
            if (deg[u] <= 0.0f) { dangling += rank[u]; continue; }
            float share = rank[u] / deg[u];
            for (auto [v, w] : adj_[u]) next[v] += share * w;
        }
        for (std::size_t v = 0; v < n; ++v)
            next[v] = a * teleport[v] + (1.0f - a) * (next[v] + dangling * teleport[v]);
        rank.swap(next);
    }
    return rank;
}

// ── Build ──────────────────────────────────────────────────────────────────
Result<DocGraph>
DocGraph::build(const index::Corpus& corpus, GraphConfig cfg, Summarizer summarize) {
    DocGraph g;
    g.cfg_ = cfg;
    const std::size_t n = corpus.document_count();
    g.adj_.assign(n, {});
    if (n == 0) return g;

    // Accumulate undirected edge weights, deduplicated by (min,max) key.
    std::unordered_map<std::uint64_t, Edge> acc;
    auto key = [](std::uint32_t x, std::uint32_t y) {
        if (x > y) std::swap(x, y);
        return (static_cast<std::uint64_t>(x) << 32) | y;
    };
    auto add_edge = [&](std::uint32_t x, std::uint32_t y, float w, Edge::Kind k) {
        if (x == y) return;
        auto kk = key(x, y);
        auto it = acc.find(kk);
        if (it == acc.end()) {
            std::uint32_t lo = std::min(x, y), hi = std::max(x, y);
            acc.emplace(kk, Edge{lo, hi, w, k});
        } else {
            it->second.weight += w;
            if (k == Edge::Kind::link) it->second.kind = Edge::Kind::link;
        }
    };

    // 1) Link edges — resolve mined targets against document uri/title canon.
    if (cfg.use_links) {
        std::unordered_map<std::string, std::uint32_t> by_canon;
        for (std::uint32_t i = 0; i < n; ++i) {
            const Document* d = corpus.document(DocId{i});
            if (!d) continue;
            if (auto c = canon(d->uri); !c.empty()) by_canon.emplace(c, i);
            if (auto c = canon(d->title); !c.empty()) by_canon.emplace(c, i);
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            const Document* d = corpus.document(DocId{i});
            if (!d) continue;
            for (auto& tgt : mine_links(d->text)) {
                auto c = canon(tgt);
                if (c.empty()) continue;
                if (auto it = by_canon.find(c); it != by_canon.end())
                    add_edge(i, it->second, 2.0f, Edge::Kind::link);  // links weigh 2×
            }
        }
    }

    // 2) Similarity edges — dense centroid cosine if embedded, else lexical
    //    Jaccard. k-NN sparsify: keep each node's top-`knn` above threshold.
    if (cfg.use_similarity) {
        std::vector<std::optional<std::vector<float>>> cents(n);
        std::vector<std::unordered_set<std::string>>   terms(n);
        bool dense = corpus.has_embedder();
        for (std::uint32_t i = 0; i < n; ++i) {
            if (dense) cents[i] = doc_centroid(corpus, DocId{i});
            if (!dense || !cents[i]) terms[i] = doc_terms(corpus, DocId{i});
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            std::vector<std::pair<float, std::uint32_t>> sims;
            for (std::uint32_t j = 0; j < n; ++j) {
                if (i == j) continue;
                float s;
                if (dense && cents[i] && cents[j])
                    s = dense::dot(*cents[i], *cents[j]);
                else
                    s = jaccard(terms[i], terms[j]);
                if (s >= cfg.sim_threshold) sims.emplace_back(s, j);
            }
            std::partial_sort(sims.begin(),
                sims.begin() + std::min(sims.size(), cfg.knn), sims.end(),
                [](auto& a, auto& b) { return a.first > b.first; });
            std::size_t keep = std::min(sims.size(), cfg.knn);
            for (std::size_t t = 0; t < keep; ++t)
                add_edge(i, sims[t].second, sims[t].first, Edge::Kind::similarity);
        }
    }

    // Materialize edge list + adjacency.
    g.edges_.reserve(acc.size());
    for (auto& [_, e] : acc) {
        g.edges_.push_back(e);
        g.adj_[e.a].emplace_back(e.b, e.weight);
        g.adj_[e.b].emplace_back(e.a, e.weight);
    }

    // 3) Community detection — synchronous label propagation, deterministic.
    std::vector<std::uint32_t> label(n);
    std::iota(label.begin(), label.end(), 0u);
    for (std::size_t round = 0; round < cfg.lp_rounds; ++round) {
        bool changed = false;
        for (std::uint32_t u = 0; u < n; ++u) {
            if (g.adj_[u].empty()) continue;
            std::unordered_map<std::uint32_t, float> votes;
            for (auto [v, w] : g.adj_[u]) votes[label[v]] += w;
            // pick max weight; tie-break to the smallest label id (determinism).
            std::uint32_t best = label[u];
            float bestw = -1.0f;
            for (auto [lab, w] : votes)
                if (w > bestw || (w == bestw && lab < best)) { bestw = w; best = lab; }
            if (best != label[u]) { label[u] = best; changed = true; }
        }
        if (!changed) break;
    }

    // Compact labels → 0..C-1, gather members.
    std::unordered_map<std::uint32_t, std::uint32_t> remap;
    g.doc_community_.assign(n, 0);
    for (std::uint32_t u = 0; u < n; ++u) {
        auto [it, ins] = remap.emplace(label[u], static_cast<std::uint32_t>(remap.size()));
        g.doc_community_[u] = it->second;
    }
    g.communities_.resize(remap.size());
    for (std::uint32_t cid = 0; cid < g.communities_.size(); ++cid) g.communities_[cid].id = cid;
    for (std::uint32_t u = 0; u < n; ++u)
        g.communities_[g.doc_community_[u]].docs.push_back(DocId{u});

    // 4) Community reports — extractive summary + lead chunk + density.
    for (auto& c : g.communities_) {
        // density = intra-edges / (intra + inter) over member docs.
        std::size_t intra = 0, inter = 0;
        std::unordered_set<std::uint32_t> members;
        for (auto d : c.docs) members.insert(d.get());
        for (auto d : c.docs)
            for (auto [v, w] : g.adj_[d.get()]) {
                (void)w;
                if (members.count(v)) ++intra; else ++inter;
            }
        c.density = (intra + inter) ? static_cast<float>(intra) / static_cast<float>(intra + inter) : 0.0f;

        // Score each community sentence by term-frequency centrality within the
        // community; take the top `summary_sentences`. Also pick the lead chunk
        // (the chunk whose text overlaps the most community terms).
        const auto& tok = corpus.tokenizer();
        std::unordered_map<std::string, int> df;   // community document freq
        for (auto d : c.docs)
            for (auto& t : doc_terms(corpus, d)) ++df[t];

        struct Cand { float score; std::string text; };
        std::vector<Cand> cands;
        for (auto d : c.docs) {
            const Document* doc = corpus.document(d);
            if (!doc) continue;
            for (auto sv : sentences(doc->text)) {
                float sc = 0.0f;
                for (auto& t : tok.tokenize(sv)) {
                    auto it = df.find(t);
                    if (it != df.end()) sc += static_cast<float>(it->second);
                }
                // length-normalize so we don't just pick the longest sentence.
                auto len = static_cast<float>(std::max<std::size_t>(sv.size(), 1));
                cands.push_back({sc / std::sqrt(len), std::string(sv)});
            }
        }
        std::partial_sort(cands.begin(),
            cands.begin() + std::min(cands.size(), cfg.summary_sentences), cands.end(),
            [](const Cand& a, const Cand& b) { return a.score > b.score; });
        std::string report;
        for (std::size_t i = 0; i < std::min(cands.size(), cfg.summary_sentences); ++i) {
            if (!report.empty()) report += ' ';
            report += cands[i].text;
        }
        c.summary = report;

        // lead chunk: highest community-term overlap.
        float bestlead = -1.0f;
        for (const auto& ch : corpus.chunks()) {
            if (!members.count(ch.doc.get())) continue;
            float sc = 0.0f;
            for (auto& t : tok.tokenize(ch.text))
                if (df.count(t)) sc += static_cast<float>(df[t]);
            if (sc > bestlead) { bestlead = sc; c.lead = ch.id; }
        }

        // Optional abstractive upgrade.
        if (summarize && !c.summary.empty()) {
            if (auto r = summarize(c.summary); r) c.summary = std::move(*r);
            // On summarizer failure we silently keep the extractive report.
        }
    }

    return g;
}

// ── Local search ────────────────────────────────────────────────────────────
Result<std::vector<Hit>>
DocGraph::local_search(const index::Corpus& corpus, std::string_view query,
                       std::size_t k, std::size_t seed_k) const {
    // Seed with a hybrid pool: dense if available, else lexical.
    std::vector<Hit> seed;
    if (corpus.has_embedder()) {
        auto d = corpus.dense_search(query, seed_k);
        if (d) seed = std::move(*d);
    }
    if (seed.empty()) seed = corpus.lexical_search(query, seed_k);
    if (seed.empty()) return std::vector<Hit>{};

    // Build the PPR restart distribution: seed chunk scores → their documents.
    std::unordered_map<std::uint32_t, float> doc_mass;
    for (const auto& h : seed) {
        const Chunk* ch = corpus.chunk(h.chunk);
        if (!ch) continue;
        doc_mass[ch->doc.get()] += std::max(h.score.get(), 0.0f) + 1e-3f;
    }
    std::vector<std::pair<std::uint32_t, float>> restart(doc_mass.begin(), doc_mass.end());
    std::vector<float> ppr = personalized_pagerank(restart);

    // Re-score every chunk: base seed score (if present) blended with the doc's
    // PPR centrality. Chunks in linked/similar docs surface even if not seeded.
    std::unordered_map<std::uint32_t, float> seed_score;
    for (const auto& h : seed) seed_score[h.chunk.get()] = h.score.get();

    std::vector<Hit> out;
    float maxppr = 0.0f;
    for (float x : ppr) maxppr = std::max(maxppr, x);
    if (maxppr <= 0.0f) maxppr = 1.0f;

    for (const auto& ch : corpus.chunks()) {
        std::uint32_t d = ch.doc.get();
        if (d >= ppr.size()) continue;
        float central = ppr[d] / maxppr;
        if (central <= 1e-6f) continue;
        auto it = seed_score.find(ch.id.get());
        float base = it != seed_score.end() ? it->second : 0.0f;
        // Blend: retrieval evidence + graph evidence. Both matter.
        float score = 0.6f * base + 0.4f * central;
        if (score > 0.0f) out.push_back({ch.id, Score{score}});
    }
    std::partial_sort(out.begin(), out.begin() + std::min(out.size(), k), out.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    if (out.size() > k) out.resize(k);
    return out;
}

// ── Global search ────────────────────────────────────────────────────────────
Result<std::vector<Hit>>
DocGraph::global_search(const index::Corpus& corpus, std::string_view query,
                        std::size_t k) const {
    if (communities_.empty()) return std::vector<Hit>{};
    const auto& tok = corpus.tokenizer();
    auto qterms = tok.tokenize(query);
    std::unordered_set<std::string> qset(qterms.begin(), qterms.end());

    // Rank communities by their summary's overlap with the query, weighted by
    // community density (cohesive communities make better global answers).
    std::vector<std::pair<float, std::uint32_t>> scored;
    for (const auto& c : communities_) {
        if (c.summary.empty() && !c.lead.valid()) continue;
        std::unordered_map<std::string, int> tf;
        for (auto& t : tok.tokenize(c.summary)) ++tf[t];
        float sc = 0.0f;
        for (auto& q : qset) {
            auto it = tf.find(q);
            if (it != tf.end()) sc += 1.0f + std::log(1.0f + static_cast<float>(it->second));
        }
        // small density prior; +1 so a zero-overlap cohesive hub can still rank.
        sc = sc * (0.5f + c.density) + 1e-4f * static_cast<float>(c.docs.size());
        if (sc > 0.0f) scored.emplace_back(sc, c.id);
    }
    std::sort(scored.begin(), scored.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<Hit> out;
    float top = scored.empty() ? 1.0f : std::max(scored.front().first, 1e-6f);
    for (std::size_t i = 0; i < scored.size() && out.size() < k; ++i) {
        const Community& c = communities_[scored[i].second];
        if (!c.lead.valid()) continue;
        out.push_back({c.lead, Score{scored[i].first / top}});
    }
    return out;
}

} // namespace rag::graph
