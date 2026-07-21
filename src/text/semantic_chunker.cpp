// rag/text/semantic_chunker.cpp — embedding-boundary + proposition chunking.

#include "rag/text/semantic_chunker.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>

#include "rag/dense/simd.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::text {
namespace {

// Sentence segmentation preserving offsets (so we can set line spans loosely).
struct Sent { std::string text; };

std::vector<Sent> split_sentences(const std::string& body) {
    std::vector<Sent> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i < body.size(); ++i) {
        char c = body[i];
        if (c == '.' || c == '!' || c == '?' || c == '\n') {
            // treat "\n\n" and terminal punctuation as boundaries
            std::string s = body.substr(start, i - start + 1);
            // trim
            std::size_t a = 0, b = s.size();
            while (a < b && std::isspace((unsigned char)s[a])) ++a;
            while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
            s = s.substr(a, b - a);
            if (s.size() > 2) out.push_back({std::move(s)});
            start = i + 1;
        }
    }
    if (start < body.size()) {
        std::string s = body.substr(start);
        std::size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        s = s.substr(a, b - a);
        if (s.size() > 2) out.push_back({std::move(s)});
    }
    return out;
}

float percentile(std::vector<float> v, float pct) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    std::size_t idx = (std::size_t)std::clamp(pct / 100.0f * (v.size() - 1), 0.0f, (float)(v.size() - 1));
    return v[idx];
}

Chunk make_chunk(DocId doc, std::string text) {
    Chunk ch;
    ch.doc = doc;
    ch.text = std::move(text);
    return ch;
}

// Assemble chunks from sentences given the breakpoint indices (start-of-chunk).
std::vector<Chunk> assemble(DocId doc, const std::vector<Sent>& sents,
                            const std::vector<char>& breaks, std::size_t max_chars) {
    std::vector<Chunk> out;
    std::string cur;
    for (std::size_t i = 0; i < sents.size(); ++i) {
        if ((breaks[i] && !cur.empty()) || cur.size() + sents[i].text.size() > max_chars) {
            if (!cur.empty()) out.push_back(make_chunk(doc, std::move(cur)));
            cur.clear();
        }
        if (!cur.empty()) cur += ' ';
        cur += sents[i].text;
    }
    if (!cur.empty()) out.push_back(make_chunk(doc, std::move(cur)));
    return out;
}

} // namespace

Result<std::vector<Chunk>>
semantic_chunk(DocId doc_id, const std::string& body,
               const dense::AnyEmbedder& embedder, SemanticChunkOptions opts) {
    auto sents = split_sentences(body);
    if (sents.size() <= 1) {
        std::vector<Chunk> one;
        if (!sents.empty()) one.push_back(make_chunk(doc_id, sents[0].text));
        return one;
    }
    // Embed every sentence.
    std::vector<std::string> texts;
    texts.reserve(sents.size());
    for (auto& s : sents) texts.push_back(s.text);
    auto embs = embedder.embed(texts);
    if (!embs) return std::unexpected(embs.error());
    for (auto& e : *embs) dense::normalize(e);

    // Distance between consecutive sentences = 1 - cosine.
    std::vector<float> dist(sents.size(), 0.0f);
    for (std::size_t i = 1; i < sents.size(); ++i) {
        float cos = (*embs)[i].size() == (*embs)[i - 1].size()
                  ? dense::dot((*embs)[i], (*embs)[i - 1]) : 0.0f;
        dist[i] = 1.0f - cos;
    }
    float thresh = percentile(std::vector<float>(dist.begin() + 1, dist.end()),
                              opts.breakpoint_percentile);
    std::vector<char> breaks(sents.size(), 0);
    for (std::size_t i = 1; i < sents.size(); ++i)
        if (dist[i] > thresh) breaks[i] = 1;
    return assemble(doc_id, sents, breaks, opts.max_chars);
}

std::vector<Chunk>
semantic_chunk_lexical(DocId doc_id, const std::string& body, SemanticChunkOptions opts) {
    auto sents = split_sentences(body);
    if (sents.size() <= 1) {
        std::vector<Chunk> one;
        if (!sents.empty()) one.push_back(make_chunk(doc_id, sents[0].text));
        return one;
    }
    Tokenizer tok;
    std::vector<std::unordered_set<std::string>> bags(sents.size());
    for (std::size_t i = 0; i < sents.size(); ++i) {
        auto t = tok.tokenize(sents[i].text);
        bags[i].insert(t.begin(), t.end());
    }
    std::vector<float> dist(sents.size(), 0.0f);
    for (std::size_t i = 1; i < sents.size(); ++i) {
        const auto& a = bags[i]; const auto& b = bags[i - 1];
        float j = 0.0f;
        if (!a.empty() && !b.empty()) {
            const auto& sm = a.size() < b.size() ? a : b;
            const auto& bg = a.size() < b.size() ? b : a;
            std::size_t inter = 0;
            for (auto& t : sm) if (bg.count(t)) ++inter;
            j = (float)inter / (float)(a.size() + b.size() - inter);
        }
        dist[i] = 1.0f - j;
    }
    float thresh = percentile(std::vector<float>(dist.begin() + 1, dist.end()),
                              opts.breakpoint_percentile);
    std::vector<char> breaks(sents.size(), 0);
    for (std::size_t i = 1; i < sents.size(); ++i)
        if (dist[i] > thresh) breaks[i] = 1;
    return assemble(doc_id, sents, breaks, opts.max_chars);
}

std::vector<Chunk>
proposition_chunk(DocId doc_id, const std::string& body, PropositionFn extract) {
    std::vector<Chunk> out;
    if (extract) {
        if (auto props = extract(body)) {
            for (auto& p : *props) if (!p.empty()) out.push_back(make_chunk(doc_id, std::move(p)));
            if (!out.empty()) return out;
        }
    }
    // Deterministic default: one sentence per proposition (atomic statements).
    for (auto& s : split_sentences(body)) out.push_back(make_chunk(doc_id, s.text));
    return out;
}

} // namespace rag::text
