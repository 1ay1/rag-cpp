// rag/loaders/structure.cpp — inferring a file's own conventions.
//
// The scoring model, and why each term is there. A candidate leading token is
// a definition keyword if it looks like this in the file:
//
//   REPEATED   it appears several times. A token seen once is a coincidence;
//              a token seen twenty times is a convention. But frequency alone
//              elects "the" in prose and "if" in code, so it is necessary and
//              nowhere near sufficient.
//
//   ALIGNED    its occurrences sit at ONE indentation column, over and over.
//              This is the strongest single signal, and the reason inference
//              works at all: `if` and `return` scatter across every depth of a
//              file, while `def`/`func`/`SECTION` are pinned to one. Column
//              consistency separates structure from statements without knowing
//              what either word means.
//
//   ANCHORING  the line after it tends to be MORE indented, or the line before
//              it tends to be blank or a comment. Definitions open a body and
//              are announced; statements do neither.
//
//   SPACED     its occurrences are spread through the file rather than bunched.
//              Import blocks and constant tables are repeated and aligned, so
//              without this term a Python file chunks on `import`. Definitions
//              are what you find every few dozen lines all the way down.
//
// Multiplying these gives a score that is near zero unless a token is doing all
// four things, which in practice only definition keywords do.

#include "rag/loaders/structure.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

namespace rag::loaders {

bool StructureProfile::usable() const noexcept {
    return confidence >= kMinConfidence && !definition_tokens.empty();
}

namespace {

struct Line {
    std::string_view text;      // full line
    std::string_view body;      // leading whitespace removed
    std::size_t indent = 0;     // in columns (tab = 4)
    bool blank = true;
};

std::vector<Line> scan_lines(std::string_view src) {
    std::vector<Line> out;
    std::size_t pos = 0;
    while (pos <= src.size()) {
        std::size_t nl = src.find('\n', pos);
        if (nl == std::string_view::npos) nl = src.size();
        Line l;
        l.text = src.substr(pos, nl - pos);
        std::size_t i = 0, col = 0;
        while (i < l.text.size() && (l.text[i] == ' ' || l.text[i] == '\t')) {
            col += (l.text[i] == '\t') ? 4 : 1;
            ++i;
        }
        l.indent = col;
        l.body = l.text.substr(i);
        // Trailing whitespace should not make a line non-blank.
        while (!l.body.empty() && (l.body.back() == ' ' || l.body.back() == '\t' ||
                                   l.body.back() == '\r'))
            l.body.remove_suffix(1);
        l.blank = l.body.empty();
        out.push_back(l);
        if (nl == src.size()) break;
        pos = nl + 1;
    }
    return out;
}

// The leading token of a line: its first run of "word" characters, or, if the
// line starts with punctuation, that punctuation run. Both matter — plenty of
// DSLs mark definitions with sigils (`@route`, `.macro`, `[section]`, `#+TITLE`)
// rather than words, and a word-only tokenizer is blind to exactly the formats
// least likely to have a parser already.
std::string_view leading_token(std::string_view body) {
    if (body.empty()) return {};
    auto is_word = [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '-';
    };
    std::size_t i = 0;
    if (is_word(static_cast<unsigned char>(body[0]))) {
        while (i < body.size() && is_word(static_cast<unsigned char>(body[i]))) ++i;
    } else {
        // A punctuation sigil, plus the word that follows it if there is one:
        // "@route" is a far more useful token than "@".
        while (i < body.size() && !is_word(static_cast<unsigned char>(body[i])) &&
               body[i] != ' ' && body[i] != '\t' && i < 3)
            ++i;
        std::size_t j = i;
        while (j < body.size() && is_word(static_cast<unsigned char>(body[j]))) ++j;
        i = j;
    }
    if (i > 24) return {};   // not a keyword; probably a long identifier or URL
    return body.substr(0, i);
}

// Comment prefixes we can recognize without knowing the language. We do not
// need to be exhaustive: this only affects whether a leading comment is glued
// to the definition below it, never whether the definition is found.
const char* const kCommentCandidates[] = {"#", "//", "--", ";;", ";", "%", "*", "'"};

std::string detect_comment_prefix(const std::vector<Line>& lines) {
    std::unordered_map<std::string, std::size_t> hits;
    std::size_t nonblank = 0;
    for (const Line& l : lines) {
        if (l.blank) continue;
        ++nonblank;
        for (const char* cand : kCommentCandidates) {
            std::string_view c(cand);
            if (l.body.size() >= c.size() && l.body.substr(0, c.size()) == c) {
                hits[std::string(c)]++;
                break;     // longest-first ordering matters: "//" before "/"
            }
        }
    }
    if (nonblank == 0) return {};
    std::string best;
    std::size_t best_n = 0;
    for (const auto& [pfx, n] : hits) {
        // A real comment prefix accounts for a visible slice of the file but
        // not nearly all of it (that would be prose with bullets, or a banner).
        double frac = static_cast<double>(n) / static_cast<double>(nonblank);
        if (n >= 3 && frac > 0.02 && frac < 0.7 && n > best_n) { best = pfx; best_n = n; }
    }
    return best;
}

struct TokenStats {
    std::size_t count = 0;
    std::unordered_map<std::size_t, std::size_t> indents;  // column -> hits
    std::size_t opens_body = 0;      // next non-blank line is deeper
    std::size_t announced = 0;       // previous line blank or a comment
    std::size_t first_line = 0;
    std::size_t last_line = 0;
    bool first_seen = false;
};

// Tokens that are common enough across languages to be worth suppressing by
// name. This list is a shortcut, not the mechanism: with it removed, inference
// still works (the alignment and spacing terms already push these down), it
// just wastes a candidate slot. Keeping it small on purpose — the whole point
// of this file is to NOT need a per-language word list.
bool is_control_flow(std::string_view t) {
    static const std::string_view kNoise[] = {
        "if", "else", "elif", "for", "while", "return", "break", "continue",
        "case", "switch", "try", "catch", "except", "finally", "with", "do",
        "then", "end", "endif", "fi", "esac", "done", "and", "or", "not",
        "the", "a", "an", "this", "that", "it", "is", "was", "in", "of", "to",
    };
    std::string lower;
    lower.reserve(t.size());
    for (char c : t) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    for (std::string_view n : kNoise) if (lower == n) return true;
    return false;
}

} // namespace

StructureProfile infer_structure(std::string_view body) {
    StructureProfile prof;
    auto lines = scan_lines(body);
    if (lines.size() < 8) return prof;         // too small to infer anything

    std::size_t blanks = 0, nonblank = 0, bracket_lines = 0;
    for (const Line& l : lines) {
        if (l.blank) { ++blanks; continue; }
        ++nonblank;
        char b = l.body.back();
        if (b == '{' || b == '(' || b == '[') ++bracket_lines;
    }
    if (nonblank < 6) return prof;
    prof.blank_ratio = static_cast<double>(blanks) / static_cast<double>(lines.size());
    prof.bracket_nesting =
        static_cast<double>(bracket_lines) / static_cast<double>(nonblank) > 0.10;
    prof.comment_prefix = detect_comment_prefix(lines);

    // Does this file NEST at all? A definition, whatever the language, opens a
    // body: the lines under it are indented, or bracketed. Text that never
    // indents anything has no definitions to find, whatever its vocabulary
    // looks like.
    //
    // This gate is load-bearing, and the bench's prose corpus is why. Blank-line
    // separated paragraphs beat every other term in the model: each paragraph is
    // one long line at column 0, so a word that opens several of them ("Revenue",
    // "However") is repeated, perfectly aligned, spread across the whole file,
    // and "announced" by the blank line above it. It scored 0.75 confidence and
    // would have chunked an annual report on the word "Revenue". What it could
    // never do is open a body, because prose has no bodies — which is the one
    // property that actually distinguishes a definition from a sentence.
    std::size_t deeper = 0, comparable = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
        if (lines[i].blank) continue;
        std::size_t j = i;
        while (j > 0 && lines[j - 1].blank) --j;
        if (j == 0) continue;
        ++comparable;
        if (lines[i].indent > lines[j - 1].indent) ++deeper;
    }
    const double nesting_ratio =
        comparable ? static_cast<double>(deeper) / static_cast<double>(comparable) : 0.0;

    auto is_comment = [&](const Line& l) {
        return !prof.comment_prefix.empty() && !l.blank &&
               l.body.size() >= prof.comment_prefix.size() &&
               l.body.substr(0, prof.comment_prefix.size()) == prof.comment_prefix;
    };

    // ─── Gather per-token evidence ────────────────────────────────────────────
    std::unordered_map<std::string, TokenStats> stats;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const Line& l = lines[i];
        if (l.blank || is_comment(l)) continue;
        std::string_view tok = leading_token(l.body);
        if (tok.size() < 2 || is_control_flow(tok)) continue;

        TokenStats& s = stats[std::string(tok)];
        ++s.count;
        s.indents[l.indent]++;
        if (!s.first_seen) { s.first_line = i; s.first_seen = true; }
        s.last_line = i;

        // Does it open a body? Look at the next non-blank line.
        for (std::size_t j = i + 1; j < lines.size(); ++j) {
            if (lines[j].blank) continue;
            if (lines[j].indent > l.indent) ++s.opens_body;
            break;
        }
        // Is it announced? A blank line or a comment above it.
        if (i > 0 && (lines[i - 1].blank || is_comment(lines[i - 1]))) ++s.announced;
    }

    // ─── Score ────────────────────────────────────────────────────────────────
    struct Scored { std::string token; double score; std::size_t indent; };
    std::vector<Scored> scored;
    const double total_lines = static_cast<double>(lines.size());

    for (const auto& [tok, s] : stats) {
        if (s.count < 3) continue;

        // Dominant indentation column, and how dominant it is.
        std::size_t best_col = 0, best_hits = 0;
        for (const auto& [col, n] : s.indents)
            if (n > best_hits) { best_hits = n; best_col = col; }
        const double alignment = static_cast<double>(best_hits) / static_cast<double>(s.count);
        if (alignment < 0.6) continue;         // scattered: a statement, not a definition

        const double count_d = static_cast<double>(s.count);

        // Repetition, saturating: 20 occurrences is as convincing as 200.
        const double repetition = std::min(1.0, std::log1p(count_d) / std::log1p(20.0));

        // Anchoring: opens a body and/or is announced.
        const double anchoring =
            0.5 * (static_cast<double>(s.opens_body) / count_d) +
            0.5 * (static_cast<double>(s.announced) / count_d);

        // Spread: how much of the file the token's occurrences span, relative
        // to what an evenly-distributed token would span. This is the term that
        // demotes `import`/`use`/`#include` blocks, which are perfectly aligned
        // and highly repeated but confined to the top of the file.
        const double span = static_cast<double>(s.last_line - s.first_line + 1) / total_lines;
        const double spread = std::min(1.0, span / 0.5);

        // Shallow definitions are likelier than deep ones. Not decisive — some
        // formats indent everything under a root — just a tiebreak.
        const double depth_bonus = 1.0 / (1.0 + static_cast<double>(best_col) / 8.0);

        const double score = repetition * (0.15 + anchoring) * (0.15 + spread) *
                             alignment * depth_bonus;
        if (score > 0.0) scored.push_back({tok, score, best_col});
    }

    if (scored.empty()) return prof;
    std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.token < b.token;              // deterministic ties
    });

    // Keep tokens within a factor of the best, up to 6. A language usually has
    // a handful of definition forms (func/type/var, def/class, rule/macro), and
    // taking only the top one chunks a mixed file on one of its shapes.
    const double cutoff = scored.front().score * 0.35;
    for (const Scored& s : scored) {
        if (s.score < cutoff || prof.definition_tokens.size() >= 6) break;
        prof.definition_tokens.push_back(s.token);
    }
    prof.definition_indent = scored.front().indent;

    // Confidence: the best token's score, lifted when the winner stands clearly
    // apart from the runner-up (a decisive convention) and damped when the file
    // has too few candidate boundaries to be worth structural chunking at all.
    double margin = 1.0;
    if (scored.size() > 1 && scored.front().score > 0.0)
        margin = 1.0 + 0.5 * (1.0 - scored[1].score / scored.front().score);

    std::size_t boundaries = 0;
    for (const auto& t : prof.definition_tokens) boundaries += stats[t].count;
    const double density = std::min(1.0, static_cast<double>(boundaries) / (total_lines / 25.0));

    // Structural evidence, per the gate described above: either the file nests,
    // or the winning token is itself seen opening bodies. Absent both, there is
    // no structure here and any boundary we report is invented.
    const TokenStats& best = stats[scored.front().token];
    const double opens_frac =
        best.count ? static_cast<double>(best.opens_body) / static_cast<double>(best.count) : 0.0;
    const double structural = std::max({nesting_ratio / 0.15, opens_frac / 0.5,
                                        prof.bracket_nesting ? 1.0 : 0.0});
    if (structural < 1.0) {
        prof.definition_tokens.clear();
        prof.confidence = 0.0;
        return prof;
    }

    prof.confidence = std::min(1.0, scored.front().score * 2.2 * margin * (0.4 + 0.6 * density));
    return prof;
}

// ─── Chunking on an inferred profile ──────────────────────────────────────────

std::vector<Chunk> chunk_by_structure(DocId doc, const std::string& body,
                                      const StructureProfile& profile,
                                      const CodeChunkOptions& opts) {
    std::vector<Chunk> chunks;
    if (body.empty()) return chunks;
    auto lines = scan_lines(body);

    auto is_comment = [&](const Line& l) {
        return !profile.comment_prefix.empty() && !l.blank &&
               l.body.size() >= profile.comment_prefix.size() &&
               l.body.substr(0, profile.comment_prefix.size()) == profile.comment_prefix;
    };
    auto is_boundary = [&](const Line& l) {
        if (l.blank || is_comment(l)) return false;
        std::string_view tok = leading_token(l.body);
        if (tok.empty()) return false;
        for (const auto& d : profile.definition_tokens)
            if (tok == d) return l.indent <= profile.definition_indent;
        return false;
    };

    // A definition's leading comment block belongs to it, not to the previous
    // definition. Walk back over comments (and one blank line) from a boundary.
    auto claim_leading_comments = [&](std::size_t at) {
        std::size_t start = at;
        while (start > 0) {
            const Line& prev = lines[start - 1];
            if (is_comment(prev)) { --start; continue; }
            if (prev.blank && start >= 2 && is_comment(lines[start - 2])) { --start; continue; }
            break;
        }
        return start;
    };

    std::string pending_context;
    auto flush = [&](std::size_t from, std::size_t to, const std::string& ctx) {
        if (from >= to || from >= lines.size()) return;
        std::string text;
        for (std::size_t i = from; i < to && i < lines.size(); ++i) {
            text.append(lines[i].text);
            text.push_back('\n');
        }
        while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
        if (text.empty()) return;
        Chunk ch;
        ch.doc = doc;
        ch.text = std::move(text);
        ch.context = ctx;
        ch.start_line = static_cast<std::uint32_t>(from);
        ch.end_line = static_cast<std::uint32_t>(to > 0 ? to - 1 : 0);
        chunks.push_back(std::move(ch));
    };

    std::size_t seg_start = 0, chars = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const bool boundary = is_boundary(lines[i]);
        const std::size_t cut = boundary ? claim_leading_comments(i) : i;
        const bool too_big = (i - seg_start) >= opts.max_lines || chars >= opts.max_chars;

        if (((boundary && cut > seg_start && (cut - seg_start) >= opts.min_lines)) || too_big) {
            flush(seg_start, cut, pending_context);
            seg_start = cut;
            chars = 0;
        }
        if (boundary) {
            std::string_view d = lines[i].body;
            pending_context = std::string(d.substr(0, std::min<std::size_t>(d.size(), 120)));
        }
        chars += lines[i].text.size() + 1;
    }
    flush(seg_start, lines.size(), pending_context);
    return chunks;
}

// ─── Strategy selection ───────────────────────────────────────────────────────

ChunkStrategy strategy_for(std::string_view ext, const std::string& body) {
    if (detect_language(ext) != Language::unknown) return ChunkStrategy::known_language;
    return infer_structure(body).usable() ? ChunkStrategy::inferred : ChunkStrategy::windows;
}

std::string_view strategy_name(ChunkStrategy s) {
    switch (s) {
        case ChunkStrategy::known_language: return "known_language";
        case ChunkStrategy::inferred:       return "inferred";
        default:                            return "windows";
    }
}

std::vector<Chunk> chunk_source(DocId doc, std::string_view ext, const std::string& body,
                                const CodeChunkOptions& opts) {
    if (detect_language(ext) != Language::unknown)
        return chunk_code(doc, ext, body, opts);

    StructureProfile prof = infer_structure(body);
    if (prof.usable()) return chunk_by_structure(doc, body, prof, opts);

    // No structure worth trusting: size-bounded windows, which is what
    // chunk_code's unknown-language path already does.
    return chunk_code(doc, ext, body, opts);
}

} // namespace rag::loaders
