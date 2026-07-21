// rag/crag/crag.cpp — Corrective RAG + Self-RAG reflection (model-free default).

#include "rag/crag/crag.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>

#include "rag/text/tokenizer.hpp"

namespace rag::crag {
namespace {

std::vector<std::string_view> sentences(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
        if (s[i] == '.' || s[i] == '!' || s[i] == '?' || s[i] == '\n') {
            auto p = s.substr(start, i - start + 1);
            while (!p.empty() && std::isspace((unsigned char)p.front())) p.remove_prefix(1);
            while (!p.empty() && std::isspace((unsigned char)p.back()))  p.remove_suffix(1);
            if (p.size() > 6) out.push_back(p);
            start = i + 1;
        }
    if (start < s.size()) {
        auto p = s.substr(start);
        while (!p.empty() && std::isspace((unsigned char)p.front())) p.remove_prefix(1);
        while (!p.empty() && std::isspace((unsigned char)p.back()))  p.remove_suffix(1);
        if (p.size() > 6) out.push_back(p);
    }
    return out;
}

// Model-free per-passage relevance: query-term coverage (recall of query terms
// in the passage), a deterministic proxy for a learned IsRelevant grader.
float lexical_relevance(const text::Tokenizer& tok, std::string_view query, std::string_view passage) {
    auto q = tok.tokenize(query);
    if (q.empty()) return 0.0f;
    std::unordered_set<std::string> qs(q.begin(), q.end());
    std::unordered_set<std::string> ps;
    for (auto& t : tok.tokenize(passage)) ps.insert(t);
    std::size_t hit = 0;
    for (auto& t : qs) if (ps.count(t)) ++hit;
    return (float)hit / (float)qs.size();
}

} // namespace

Correction
correct(const index::Corpus& corpus, std::string_view query, std::span<const Hit> hits,
        CragConfig cfg, Evaluator eval, ExternalSource external) {
    Correction c;
    text::Tokenizer tok = corpus.tokenizer();

    // Score each passage's relevance (learned evaluator or lexical default).
    struct Graded { Hit hit; float rel; std::string text; };
    std::vector<Graded> graded;
    graded.reserve(hits.size());
    for (const auto& h : hits) {
        const Chunk* ch = corpus.chunk(h.chunk);
        std::string text = ch ? ch->indexed_text() : std::string{};
        float rel = eval ? eval(query, text) : lexical_relevance(tok, query, text);
        graded.push_back({h, std::clamp(rel, 0.0f, 1.0f), std::move(text)});
    }
    std::sort(graded.begin(), graded.end(), [](const Graded& a, const Graded& b) { return a.rel > b.rel; });

    // Overall confidence = the best passage's relevance (CRAG grades the SET by
    // its strongest evidence; a single strong hit is enough to be "correct").
    c.confidence = graded.empty() ? 0.0f : graded.front().rel;

    // Decide the action from the confidence thresholds.
    if (c.confidence >= cfg.upper)      c.action = Action::correct;
    else if (c.confidence <= cfg.lower) c.action = Action::incorrect;
    else                                c.action = Action::ambiguous;

    // Decompose-recompose: keep the top-N relevant "knowledge strips", optionally
    // dropping the irrelevant ones (Self-RAG IsRelevant filter).
    for (const auto& g : graded) {
        bool relevant = g.rel > cfg.lower;
        if (cfg.drop_irrelevant && !relevant) continue;
        if (c.kept.size() >= cfg.strips) break;
        c.kept.push_back(g.hit);
        c.knowledge.push_back(g.text);
    }

    // External fallback for Ambiguous / Incorrect (CRAG's web-search branch).
    if (external && c.action != Action::correct) {
        if (auto ext = external(query)) {
            c.external = std::move(*ext);
            // Ambiguous = combine; Incorrect = external replaces internal.
            if (c.action == Action::incorrect) c.knowledge.clear();
            for (const auto& e : c.external) c.knowledge.push_back(e);
        }
    }
    return c;
}

float support_score(const index::Corpus& corpus, std::string_view answer,
                    std::span<const std::string> knowledge) {
    text::Tokenizer tok = corpus.tokenizer();
    // Build the knowledge term set.
    std::unordered_set<std::string> kbag;
    for (const auto& k : knowledge)
        for (auto& t : tok.tokenize(k)) kbag.insert(t);
    if (kbag.empty()) return 0.0f;

    // A sentence is "supported" if a strong majority of its content terms appear
    // in the knowledge (deterministic IsSupported proxy). Score = supported frac.
    auto sents = sentences(answer);
    if (sents.empty()) return 0.0f;
    std::size_t supported = 0;
    for (auto s : sents) {
        auto terms = tok.tokenize(s);
        if (terms.empty()) continue;
        std::size_t in = 0;
        for (auto& t : terms) if (kbag.count(t)) ++in;
        if ((float)in / (float)terms.size() >= 0.5f) ++supported;
    }
    return (float)supported / (float)sents.size();
}

} // namespace rag::crag
