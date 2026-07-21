// rag/text/chunker.cpp — semantic line-aligned chunking with heading context.

#include "rag/text/chunker.hpp"

#include <algorithm>
#include <array>
#include <sstream>

namespace rag::text {

namespace {

// Return markdown heading level (1-6) if `line` is an ATX heading, else 0.
int heading_level(std::string_view line) {
    std::size_t i = 0;
    while (i < line.size() && line[i] == '#') ++i;
    if (i >= 1 && i <= 6 && i < line.size() && line[i] == ' ') return static_cast<int>(i);
    return 0;
}

std::string heading_text(std::string_view line) {
    std::size_t i = 0;
    while (i < line.size() && (line[i] == '#' || line[i] == ' ')) ++i;
    std::string t(line.substr(i));
    while (!t.empty() && (t.back() == ' ' || t.back() == '#')) t.pop_back();
    return t;
}

} // namespace

std::vector<Chunk> chunk_document(DocId doc_id, const std::string& body,
                                  const ChunkOptions& opts) {
    // Split into lines, preserving line numbers.
    std::vector<std::string> lines;
    {
        std::string cur;
        for (char c : body) {
            if (c == '\n') { lines.push_back(std::move(cur)); cur.clear(); }
            else cur.push_back(c);
        }
        lines.push_back(std::move(cur));
    }

    std::vector<Chunk> chunks;
    // Running heading breadcrumb: heading text per level (1..6).
    std::array<std::string, 7> crumb{};

    auto build_context = [&]() -> std::string {
        std::string ctx;
        for (int lvl = 1; lvl <= 6; ++lvl) {
            if (crumb[lvl].empty()) continue;
            if (!ctx.empty()) ctx += " › ";
            ctx += crumb[lvl];
        }
        return ctx;
    };

    std::size_t i = 0;
    const std::size_t n = lines.size();
    while (i < n) {
        // Update breadcrumb from any heading at the window start.
        std::size_t start = i;
        std::string chunk_text;
        std::size_t line_count = 0;
        std::size_t char_count = 0;
        std::string ctx_at_start;
        bool ctx_captured = false;

        while (i < n && line_count < opts.max_lines &&
               char_count < opts.max_chars) {
            const std::string& ln = lines[i];
            if (opts.heading_context) {
                if (int lvl = heading_level(ln); lvl > 0) {
                    crumb[lvl] = heading_text(ln);
                    for (int deeper = lvl + 1; deeper <= 6; ++deeper) crumb[deeper].clear();
                }
            }
            if (!ctx_captured) { ctx_at_start = build_context(); ctx_captured = true; }

            if (!chunk_text.empty()) chunk_text += '\n';
            chunk_text += ln;
            char_count += ln.size() + 1;
            ++line_count;
            ++i;

            // Prefer to end the chunk on a blank line once we're past half budget.
            if (ln.empty() && line_count >= opts.max_lines / 2) break;
        }

        // Trim trailing whitespace-only chunk.
        std::string trimmed = chunk_text;
        while (!trimmed.empty() && (trimmed.back() == '\n' || trimmed.back() == ' '))
            trimmed.pop_back();
        if (!trimmed.empty()) {
            Chunk ch;
            ch.doc        = doc_id;
            ch.text       = std::move(trimmed);
            ch.context    = ctx_at_start;
            ch.start_line = static_cast<std::uint32_t>(start);
            ch.end_line   = static_cast<std::uint32_t>(i > 0 ? i - 1 : 0);
            chunks.push_back(std::move(ch));
        }

        // Overlap: step back a few lines so the next chunk repeats the tail.
        if (opts.overlap_lines > 0 && i < n && i > start + opts.overlap_lines)
            i -= opts.overlap_lines;
    }

    return chunks;
}

} // namespace rag::text
