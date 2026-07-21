// rag/raptor/raptor.cpp — RAPTOR recursive tree construction + collapsed retrieval.

#include "rag/raptor/raptor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "rag/dense/simd.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::raptor {
namespace {

std::vector<std::string_view> sentences(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '.' || s[i] == '!' || s[i] == '?' || s[i] == '\n') {
            auto p = s.substr(start, i - start + 1);
            while (!p.empty() && std::isspace((unsigned char)p.front())) p.remove_prefix(1);
            while (!p.empty() && std::isspace((unsigned char)p.back()))  p.remove_suffix(1);
            if (p.size() > 12) out.push_back(p);
            start = i + 1;
        }
    }
    if (start < s.size()) {
        auto p = s.substr(start);
        while (!p.empty() && std::isspace((unsigned char)p.front())) p.remove_prefix(1);
        while (!p.empty() && std::isspace((unsigned char)p.back()))  p.remove_suffix(1);
        if (p.size() > 12) out.push_back(p);
    }
    return out;
}

// Extractive summary of concatenated child text: pick the sentences with the
// highest corpus-term coverage (a cheap, deterministic "abstract").
std::string extractive_summary(const text::Tokenizer& tok, std::string_view joined,
                               std::size_t n_sentences) {
    std::unordered_map<std::string, int> df;
    for (auto& t : tok.tokenize(joined)) ++df[t];
    struct C { float score; std::string_view text; };
    std::vector<C> cands;
    for (auto sv : sentences(joined)) {
        float sc = 0.0f;
        for (auto& t : tok.tokenize(sv)) if (auto it = df.find(t); it != df.end()) sc += it->second;
        cands.push_back({sc / std::sqrt((float)std::max<std::size_t>(sv.size(), 1)), sv});
    }
    std::size_t keep = std::min(cands.size(), n_sentences);
    std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(),
        [](const C& a, const C& b) { return a.score > b.score; });
    std::string out;
    for (std::size_t i = 0; i < keep; ++i) { if (!out.empty()) out += ' '; out += std::string(cands[i].text); }
    return out;
}

float lex_jaccard(const std::unordered_set<std::string>& a,
                  const std::unordered_set<std::string>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    const auto& sm = a.size() < b.size() ? a : b;
    const auto& bg = a.size() < b.size() ? b : a;
    std::size_t inter = 0;
    for (auto& t : sm) if (bg.count(t)) ++inter;
    return (float)inter / (float)(a.size() + b.size() - inter);
}

// Greedy agglomerative clustering: repeatedly seed a cluster with an unused node
// and pull in its most-similar neighbours until the target size or the
// similarity floor. Deterministic (index order). Returns clusters of node idx.
std::vector<std::vector<std::size_t>>
cluster(const std::vector<Vector>& embs,
        const std::vector<std::unordered_set<std::string>>& bags,
        bool dense, std::size_t target, float floor) {
    const std::size_t n = dense ? embs.size() : bags.size();
    std::vector<char> used(n, 0);
    std::vector<std::vector<std::size_t>> clusters;
    auto sim = [&](std::size_t i, std::size_t j) {
        if (dense) return dense::dot(embs[i], embs[j]);
        return lex_jaccard(bags[i], bags[j]);
    };
    for (std::size_t seed = 0; seed < n; ++seed) {
        if (used[seed]) continue;
        std::vector<std::size_t> cl{seed};
        used[seed] = 1;
        // rank the rest by similarity to the seed, pull best ones in.
        std::vector<std::pair<float, std::size_t>> nbr;
        for (std::size_t j = 0; j < n; ++j)
            if (!used[j] && sim(seed, j) >= floor) nbr.emplace_back(sim(seed, j), j);
        std::sort(nbr.begin(), nbr.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (auto& [s, j] : nbr) {
            if (cl.size() >= target) break;
            used[j] = 1; cl.push_back(j);
        }
        clusters.push_back(std::move(cl));
    }
    return clusters;
}

} // namespace

std::string_view RaptorTree::node_text(std::uint32_t id) const {
    return id < nodes_.size() ? std::string_view(nodes_[id].text) : std::string_view{};
}

Result<RaptorTree>
RaptorTree::build(const index::Corpus& corpus, RaptorConfig cfg, Summarizer summarize) {
    if (corpus.chunk_count() == 0)
        return std::unexpected(Error{Errc::empty_corpus, "raptor: empty corpus"});
    RaptorTree tree;
    tree.dense_ = corpus.has_embedder();
    text::Tokenizer tok = corpus.tokenizer();

    // Level 0 — leaves from the corpus chunks.
    std::vector<std::uint32_t> current;   // node ids at the level being clustered
    for (const auto& ch : corpus.chunks()) {
        Node n;
        n.id = (std::uint32_t)tree.nodes_.size();
        n.level = 0;
        n.chunk = ch.id;
        n.text = ch.indexed_text();
        n.embedding = ch.embedding;   // may be empty if no embedder
        current.push_back(n.id);
        tree.nodes_.push_back(std::move(n));
    }

    std::size_t level = 0;
    while (level + 1 < cfg.max_levels && current.size() > cfg.min_cluster) {
        // Gather embeddings / term bags for the current level's nodes.
        std::vector<Vector> embs;
        std::vector<std::unordered_set<std::string>> bags;
        bool dense = tree.dense_;
        for (auto id : current) {
            const Node& nd = tree.nodes_[id];
            if (dense && !nd.embedding.empty()) embs.push_back(nd.embedding);
            else dense = false;   // fall back to lexical if any node lacks a vec
        }
        if (!dense) {
            embs.clear();
            for (auto id : current) {
                auto toks = tok.tokenize(tree.nodes_[id].text);
                bags.emplace_back(toks.begin(), toks.end());
            }
        }
        auto clusters = cluster(embs, bags, dense, cfg.cluster_size, cfg.sim_threshold);

        // A level that can't reduce (every cluster is a singleton) → stop.
        bool reduced = std::any_of(clusters.begin(), clusters.end(),
            [&](auto& c) { return c.size() >= cfg.min_cluster; });
        if (!reduced) break;

        std::vector<std::uint32_t> next;
        for (auto& cl : clusters) {
            if (cl.size() < cfg.min_cluster) {
                // carry singletons up unchanged so they remain retrievable.
                next.push_back(current[cl[0]]);
                continue;
            }
            // Summarize the cluster's child text.
            std::string joined;
            std::vector<std::uint32_t> child_ids;
            for (auto local : cl) {
                std::uint32_t cid = current[local];
                child_ids.push_back(cid);
                if (!joined.empty()) joined += "\n";
                joined += tree.nodes_[cid].text;
            }
            std::string summary = extractive_summary(tok, joined, cfg.summary_sentences);
            if (summarize && !joined.empty())
                if (auto r = summarize(joined)) summary = std::move(*r);

            Node parent;
            parent.id = (std::uint32_t)tree.nodes_.size();
            parent.level = (std::uint32_t)(level + 1);
            parent.text = std::move(summary);
            parent.children = std::move(child_ids);
            // Parent embedding = mean of child embeddings, or embed the summary.
            if (tree.dense_) {
                Vector acc;
                std::size_t cnt = 0;
                for (auto cid : parent.children) {
                    const auto& e = tree.nodes_[cid].embedding;
                    if (e.empty()) continue;
                    if (acc.empty()) acc.assign(e.size(), 0.0f);
                    if (acc.size() != e.size()) continue;
                    for (std::size_t d = 0; d < acc.size(); ++d) acc[d] += e[d];
                    ++cnt;
                }
                if (cnt) { dense::normalize(acc); parent.embedding = std::move(acc); }
                else if (auto ev = corpus.embed_text(parent.text)) parent.embedding = std::move(*ev);
            }
            next.push_back(parent.id);
            tree.nodes_.push_back(std::move(parent));
        }
        if (next.size() >= current.size()) break;   // no progress guard
        current = std::move(next);
        ++level;
    }
    tree.levels_ = level + 1;
    return tree;
}

Result<std::vector<RaptorTree::Retrieved>>
RaptorTree::retrieve(const index::Corpus& corpus, std::string_view query,
                     std::size_t k) const {
    // Collapsed tree: score ALL nodes (all levels) against the query.
    std::vector<std::pair<float, std::uint32_t>> scored;
    scored.reserve(nodes_.size());

    if (dense_) {
        auto qv = corpus.embed_text(std::string(query));
        if (qv) {
            for (const auto& n : nodes_)
                if (!n.embedding.empty() && n.embedding.size() == qv->size())
                    scored.emplace_back(dense::dot(n.embedding, *qv), n.id);
        }
    }
    if (scored.empty()) {
        // Lexical fallback: term-overlap score.
        text::Tokenizer tok = corpus.tokenizer();
        auto qt = tok.tokenize(query);
        std::unordered_set<std::string> qs(qt.begin(), qt.end());
        for (const auto& n : nodes_) {
            float sc = 0.0f;
            for (auto& t : tok.tokenize(n.text)) if (qs.count(t)) sc += 1.0f;
            if (sc > 0.0f) scored.emplace_back(sc, n.id);
        }
    }
    std::size_t keep = std::min(scored.size(), k);
    std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
        [](auto& a, auto& b) { return a.first > b.first; });
    scored.resize(keep);

    std::vector<Retrieved> out;
    out.reserve(keep);
    for (auto& [s, id] : scored) {
        const Node& n = nodes_[id];
        out.push_back({id, n.chunk, Score{s}, n.level});
    }
    return out;
}

} // namespace rag::raptor
