// rag/text/contextual.cpp — Anthropic Contextual Retrieval situating context.

#include "rag/text/contextual.hpp"

#include <algorithm>
#include <cctype>
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

std::string extractive_context(std::string_view document, std::string_view chunk) {
    Tokenizer tok;
    auto ctoks = tok.tokenize(chunk);
    std::unordered_set<std::string> cset(ctoks.begin(), ctoks.end());
    // best-overlap document sentence (excluding the chunk's own text).
    std::string_view best;
    std::size_t best_overlap = 0;
    for (auto s : doc_sentences(document)) {
        if (chunk.find(s) != std::string_view::npos) continue;   // skip self
        std::size_t ov = 0;
        for (auto& t : tok.tokenize(s)) if (cset.count(t)) ++ov;
        if (ov > best_overlap) { best_overlap = ov; best = s; }
    }
    std::string title(first_line(document));
    std::string out;
    if (!title.empty()) out = title;
    if (!best.empty()) { if (!out.empty()) out += " — "; out += std::string(best); }
    return out;
}

void contextualize(std::vector<Chunk>& chunks, std::string_view document,
                   const Contextualizer& ctx) {
    for (auto& ch : chunks) {
        std::string situating;
        if (ctx) {
            if (auto r = ctx(document, ch.text)) situating = std::move(*r);
        }
        if (situating.empty()) situating = extractive_context(document, ch.text);
        if (situating.empty()) continue;
        // Preserve any existing heading breadcrumb; prepend the situating blurb.
        if (ch.context.empty()) ch.context = std::move(situating);
        else ch.context = situating + "\n" + ch.context;
    }
}

} // namespace rag::text
