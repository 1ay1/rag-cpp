// rag/text/contextual.cpp — Anthropic Contextual Retrieval situating context.

#include "rag/text/contextual.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <unordered_set>

#include "rag/text/tokenizer.hpp"

namespace rag::text {
namespace {

std::vector<std::string_view> doc_sentences(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
        if (s[i] == '.' || s[i] == '!' || s[i] == '?' || s[i] == '\n') {
            auto p = s.substr(start, i - start + 1);
            while (!p.empty() && std::isspace((unsigned char)p.front())) p.remove_prefix(1);
            while (!p.empty() && std::isspace((unsigned char)p.back()))  p.remove_suffix(1);
            if (p.size() > 8) out.push_back(p);
            start = i + 1;
        }
    if (start < s.size()) {
        auto p = s.substr(start);
        while (!p.empty() && std::isspace((unsigned char)p.front())) p.remove_prefix(1);
        while (!p.empty() && std::isspace((unsigned char)p.back()))  p.remove_suffix(1);
        if (p.size() > 8) out.push_back(p);
    }
    return out;
}

std::string_view first_line(std::string_view s) {
    auto nl = s.find('\n');
    auto line = nl == std::string_view::npos ? s : s.substr(0, nl);
    // strip markdown heading marks
    while (!line.empty() && (line.front() == '#' || line.front() == ' ')) line.remove_prefix(1);
    return line;
}

} // namespace

namespace {

// A document prepared ONCE for contextualizing all of its chunks: its title,
// its sentences, and each sentence's token set.
//
// This exists because the obvious implementation — call extractive_context()
// per chunk — re-splits and re-tokenizes the ENTIRE document for every chunk of
// it, which is O(chunks x document) per document and made contextual ingest
// 5.3x the baseline on bench/contextual_bench.cpp. The document only has to be
// read once; only the chunk side varies.
struct PreparedDoc {
    std::string                                    title;
    std::vector<std::string_view>                  sentences;
    std::vector<std::vector<std::string>>          sent_tokens;   // parallel to sentences
};

PreparedDoc prepare(const Tokenizer& tok, std::string_view document) {
    PreparedDoc p;
    p.title     = std::string(first_line(document));
    p.sentences = doc_sentences(document);
    p.sent_tokens.reserve(p.sentences.size());
    for (auto s : p.sentences) p.sent_tokens.push_back(tok.tokenize(s));
    return p;
}

// The chunk-varying half of extractive_context().
std::string situate(const Tokenizer& tok, const PreparedDoc& p, std::string_view chunk) {
    auto ctoks = tok.tokenize(chunk);
    std::unordered_set<std::string> cset(ctoks.begin(), ctoks.end());
    std::string_view best;
    std::size_t best_overlap = 0;
    for (std::size_t i = 0; i < p.sentences.size(); ++i) {
        // Skip a sentence the chunk already contains: repeating the chunk back
        // to itself situates nothing.
        if (chunk.find(p.sentences[i]) != std::string_view::npos) continue;
        std::size_t ov = 0;
        for (const auto& t : p.sent_tokens[i]) if (cset.count(t)) ++ov;
        if (ov > best_overlap) { best_overlap = ov; best = p.sentences[i]; }
    }
    std::string out = p.title;
    if (!best.empty()) { if (!out.empty()) out += " — "; out += std::string(best); }
    return out;
}

} // namespace

std::string extractive_context(std::string_view document, std::string_view chunk) {
    Tokenizer tok;
    return situate(tok, prepare(tok, document), chunk);
}

void contextualize(std::vector<Chunk>& chunks, std::string_view document,
                   const Contextualizer& ctx) {
    Tokenizer tok;
    // Prepared lazily: with an LLM contextualizer that never fails, the
    // extractive path is dead code and the document is never tokenized at all.
    std::optional<PreparedDoc> prepared;

    for (auto& ch : chunks) {
        std::string situating;
        if (ctx) {
            // A contextualizer that errors on one chunk falls back for THAT
            // chunk. A flaky model must not be able to reject a document.
            if (auto r = ctx(document, ch.text)) situating = std::move(*r);
        }
        if (situating.empty()) {
            if (!prepared) prepared = prepare(tok, document);
            situating = situate(tok, *prepared, ch.text);
        }
        if (situating.empty()) continue;
        // Preserve any existing heading breadcrumb; prepend the situating blurb.
        if (ch.context.empty()) ch.context = std::move(situating);
        else ch.context = situating + "\n" + ch.context;
    }
}

} // namespace rag::text
