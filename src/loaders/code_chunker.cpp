// rag/loaders/code_chunker.cpp — definition-aligned chunking for source code.

#include "rag/loaders/code_chunker.hpp"

#include <array>
#include <cctype>

namespace rag::loaders {

Language detect_language(std::string_view ext) {
    if (ext==".c"||ext==".h"||ext==".cpp"||ext==".hpp"||ext==".cc"||ext==".cxx"||
        ext==".java"||ext==".cs"||ext==".js"||ext==".ts"||ext==".tsx"||ext==".jsx"||
        ext==".swift"||ext==".kt"||ext==".scala"||ext==".php") return Language::c_like;
    if (ext==".py") return Language::python;
    if (ext==".rb") return Language::ruby;
    if (ext==".go") return Language::go;
    if (ext==".rs") return Language::rust;
    if (ext==".md"||ext==".markdown"||ext==".html"||ext==".htm"||ext==".rst") return Language::markup;
    return Language::unknown;
}

std::string_view language_name(Language l) {
    switch (l) {
        case Language::c_like:     return "c_like";
        case Language::python:     return "python";
        case Language::ruby:       return "ruby";
        case Language::go:         return "go";
        case Language::rust:       return "rust";
        case Language::javascript: return "javascript";
        case Language::markup:     return "markup";
        default:                   return "unknown";
    }
}

namespace {

std::vector<std::string> split_lines(const std::string& body) {
    std::vector<std::string> lines; std::string cur;
    for (char c : body) { if (c=='\n') { lines.push_back(std::move(cur)); cur.clear(); } else cur.push_back(c); }
    lines.push_back(std::move(cur));
    return lines;
}

std::size_t indent_of(const std::string& s) {
    std::size_t i = 0; while (i < s.size() && (s[i]==' '||s[i]=='\t')) ++i; return i;
}

// Is this line a top-level definition boundary for the language? Heuristic:
// a def/class/func keyword at low indentation.
bool is_boundary(Language lang, const std::string& line) {
    std::size_t ind = indent_of(line);
    std::string_view s(line);
    s.remove_prefix(ind);
    auto starts = [&](std::string_view kw) { return s.substr(0, kw.size()) == kw; };

    switch (lang) {
        case Language::python:
            return ind == 0 && (starts("def ") || starts("class ") || starts("async def "));
        case Language::ruby:
            return ind <= 2 && (starts("def ") || starts("class ") || starts("module "));
        case Language::go:
            return ind == 0 && (starts("func ") || starts("type "));
        case Language::rust:
            return ind == 0 && (starts("fn ") || starts("pub fn ") || starts("impl ") ||
                                starts("struct ") || starts("enum ") || starts("trait ") ||
                                starts("pub struct ") || starts("pub enum ") || starts("mod "));
        case Language::c_like: {
            // A line at column 0 that looks like a signature and ends with '{' or ')'.
            if (ind != 0) return false;
            if (s.empty() || s[0]=='/' || s[0]=='*' || s[0]=='#' || s[0]=='}') return false;
            bool has_paren = s.find('(') != std::string_view::npos;
            bool sig_end = !s.empty() && (s.back()=='{' || s.back()==')');
            if (starts("class ")||starts("struct ")||starts("enum ")||starts("namespace ")||
                starts("interface ")||starts("public ")||starts("private ")||starts("export ")||
                starts("function ")) return true;
            return has_paren && sig_end;
        }
        default: return false;
    }
}

} // namespace

std::vector<Chunk> chunk_code(DocId doc, std::string_view ext, const std::string& body,
                              const CodeChunkOptions& opts) {
    Language lang = detect_language(ext);
    auto lines = split_lines(body);
    std::vector<Chunk> chunks;

    auto flush = [&](std::size_t start, std::size_t stop, const std::string& ctx) {
        if (stop <= start) return;
        std::string text;
        for (std::size_t i = start; i < stop && i < lines.size(); ++i) {
            text += lines[i]; text += '\n';
        }
        while (!text.empty() && (text.back()=='\n'||text.back()==' ')) text.pop_back();
        if (text.empty()) return;
        Chunk ch;
        ch.doc = doc; ch.text = std::move(text); ch.context = ctx;
        ch.start_line = static_cast<std::uint32_t>(start);
        ch.end_line = static_cast<std::uint32_t>(stop > 0 ? stop - 1 : 0);
        chunks.push_back(std::move(ch));
    };

    if (lang == Language::unknown || lang == Language::markup) {
        // Fall back to fixed windows.
        for (std::size_t i = 0; i < lines.size(); i += opts.max_lines)
            flush(i, std::min(i + opts.max_lines, lines.size()), std::string(language_name(lang)));
        return chunks;
    }

    std::size_t seg_start = 0;
    std::size_t chars = 0;
    std::string last_def;   // running symbol context

    for (std::size_t i = 0; i < lines.size(); ++i) {
        bool boundary = is_boundary(lang, lines[i]);
        std::size_t seg_len = i - seg_start;
        bool too_big = seg_len >= opts.max_lines || chars >= opts.max_chars;

        if ((boundary && seg_len >= opts.min_lines) || too_big) {
            flush(seg_start, i, last_def);
            seg_start = i;
            chars = 0;
        }
        if (boundary) {
            // Capture the definition line (trimmed) as context for the next chunk.
            std::string d = lines[i];
            std::size_t ind = indent_of(d);
            last_def = d.substr(ind, std::min<std::size_t>(d.size()-ind, 120));
        }
        chars += lines[i].size() + 1;
    }
    flush(seg_start, lines.size(), last_def);
    return chunks;
}

} // namespace rag::loaders
