// rag/loaders/loaders.cpp — HTML/PDF/file/directory loaders.

#include "rag/loaders/loaders.hpp"

#include "rag/loaders/extract.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace rag::loaders {

namespace fs = std::filesystem;

// ─── HTML → text ──────────────────────────────────────────────────────────────
std::string html_to_text(std::string_view html) {
    static const std::unordered_map<std::string, std::string> kEntities = {
        {"amp","&"},{"lt","<"},{"gt",">"},{"quot","\""},{"apos","'"},{"nbsp"," "},
        {"copy","(c)"},{"mdash","—"},{"ndash","–"},{"hellip","…"},{"rsquo","'"},{"lsquo","'"},
        {"ldquo","\""},{"rdquo","\""},
    };
    static const std::unordered_set<std::string> kSkip = {"script","style","head","noscript"};

    std::string out;
    out.reserve(html.size() / 2);
    std::size_t i = 0;
    const std::size_t n = html.size();
    std::string skip_until; // when inside script/style, skip to its close tag

    auto append_space = [&] { if (!out.empty() && out.back() != ' ' && out.back() != '\n') out.push_back(' '); };

    while (i < n) {
        if (html[i] == '<') {
            std::size_t end = html.find('>', i);
            if (end == std::string_view::npos) break;
            std::string_view tag = html.substr(i + 1, end - i - 1);
            // tag name
            std::size_t p = 0; bool closing = false;
            if (!tag.empty() && tag[0] == '/') { closing = true; p = 1; }
            std::string name;
            while (p < tag.size() && (std::isalnum(static_cast<unsigned char>(tag[p])))) name.push_back(static_cast<char>(std::tolower(tag[p++])));

            if (!skip_until.empty()) {
                if (closing && name == skip_until) skip_until.clear();
            } else if (kSkip.contains(name) && !closing) {
                skip_until = name;
            } else {
                // Block-level tags become line breaks; inline become spaces.
                static const std::unordered_set<std::string> block = {
                    "p","div","br","li","tr","h1","h2","h3","h4","h5","h6","section","article","header","footer","pre","blockquote"};
                if (block.contains(name)) { if (!out.empty() && out.back() != '\n') out.push_back('\n'); }
                else append_space();
            }
            i = end + 1;
            continue;
        }
        if (!skip_until.empty()) { ++i; continue; }
        if (html[i] == '&') {
            std::size_t semi = html.find(';', i);
            if (semi != std::string_view::npos && semi - i <= 8) {
                std::string ent(html.substr(i + 1, semi - i - 1));
                if (auto it = kEntities.find(ent); it != kEntities.end()) { out += it->second; i = semi + 1; continue; }
                if (!ent.empty() && ent[0] == '#') { out.push_back(' '); i = semi + 1; continue; }
            }
        }
        char c = html[i];
        if (c == '\r') { ++i; continue; }
        out.push_back(c);
        ++i;
    }
    // Collapse 3+ newlines to 2.
    std::string clean; clean.reserve(out.size());
    int nl = 0;
    for (char c : out) {
        if (c == '\n') { if (++nl <= 2) clean.push_back(c); }
        else { nl = 0; clean.push_back(c); }
    }
    return clean;
}

// ─── RTF → text (in-process) ──────────────────────────────────────────────────
std::string rtf_to_text(std::string_view rtf) {
    // RTF is a stream of {groups}, \control words, and literal text. Text
    // retrieval wants the literal text plus the few control words that are
    // structure (\par, \line, \tab, \cell). Everything else — font tables,
    // colour tables, stylesheets, embedded pictures, revision metadata — lives
    // in "destination" groups that carry no reading content, so those are
    // skipped wholesale.
    static const std::unordered_set<std::string> kSkipDest = {
        "fonttbl","colortbl","stylesheet","info","pict","object","header",
        "footer","footnote","annotation","themedata","colorschememapping",
        "latentstyles","datastore","generator","filetbl","listtable",
        "listoverridetable","revtbl","rsidtbl","mmath","xmlnstbl"};

    std::string out;
    out.reserve(rtf.size() / 2);
    const std::size_t n = rtf.size();

    struct Group { bool skip; };
    std::vector<Group> stack;
    stack.push_back({false});
    auto skipping = [&] { return stack.back().skip; };
    // A \* before a control word marks the whole group as an ignorable
    // destination unless the reader understands it; treat the next control word
    // as a destination name.
    bool pending_optional_dest = false;

    auto emit = [&](char c) { if (!skipping()) out.push_back(c); };

    std::size_t i = 0;
    while (i < n) {
        char c = rtf[i];
        if (c == '{') { stack.push_back({skipping()}); ++i; continue; }
        if (c == '}') { if (stack.size() > 1) stack.pop_back(); ++i; continue; }
        if (c == '\\') {
            // Escape or control word.
            if (i + 1 < n) {
                char d = rtf[i + 1];
                if (d == '\\' || d == '{' || d == '}') { emit(d); i += 2; continue; }
                if (d == '\'' ) {   // \'hh : one byte in the document codepage
                    if (i + 3 < n) {
                        auto hex = [](char x)->int{ if(x>='0'&&x<='9')return x-'0'; x|=0x20; if(x>='a'&&x<='f')return x-'a'+10; return -1; };
                        int hi = hex(rtf[i+2]), lo = hex(rtf[i+3]);
                        if (hi >= 0 && lo >= 0) {
                            unsigned byte = static_cast<unsigned>((hi<<4)|lo);
                            // RTF \'hh is a codepage byte, not UTF-8. Map the
                            // 0x80..0xFF range through CP1252 (a superset of
                            // Latin-1 for the printable high range) to a Unicode
                            // code point, then emit UTF-8 -- so caf\'e9 comes out
                            // as UTF-8 "cafe-acute" rather than a lone 0xE9 byte
                            // that corrupts the string.
                            unsigned long cp = byte;
                            if (byte >= 0x80) {
                                static const unsigned short cp1252[32] = {
                                    0x20AC,0x0081,0x201A,0x0192,0x201E,0x2026,0x2020,0x2021,
                                    0x02C6,0x2030,0x0160,0x2039,0x0152,0x008D,0x017D,0x008F,
                                    0x0090,0x2018,0x2019,0x201C,0x201D,0x2022,0x2013,0x2014,
                                    0x02DC,0x2122,0x0161,0x203A,0x0153,0x009D,0x017E,0x0178};
                                cp = (byte < 0xA0) ? cp1252[byte - 0x80] : byte;
                            }
                            if (!skipping()) {
                                if (cp < 0x80) out.push_back(static_cast<char>(cp));
                                else if (cp < 0x800) { out.push_back(char(0xC0|(cp>>6))); out.push_back(char(0x80|(cp&0x3F))); }
                                else { out.push_back(char(0xE0|(cp>>12))); out.push_back(char(0x80|((cp>>6)&0x3F))); out.push_back(char(0x80|(cp&0x3F))); }
                            }
                        }
                        i += 4; continue;
                    }
                    i += 2; continue;
                }
                if (d == '*') { pending_optional_dest = true; i += 2; continue; }
                if (!std::isalpha(static_cast<unsigned char>(d))) {
                    // A control SYMBOL like \~ (nbsp) or \- (opt hyphen): skip.
                    i += 2; continue;
                }
                // Control WORD: letters then an optional signed number.
                std::size_t j = i + 1;
                while (j < n && std::isalpha(static_cast<unsigned char>(rtf[j]))) ++j;
                std::string word(rtf.substr(i + 1, j - (i + 1)));
                bool neg = false; long param = 0; bool has_param = false;
                if (j < n && rtf[j] == '-') { neg = true; ++j; }
                std::size_t ps = j;
                while (j < n && std::isdigit(static_cast<unsigned char>(rtf[j]))) { param = param*10 + (rtf[j]-'0'); ++j; }
                has_param = (j > ps);
                if (neg) param = -param;
                // A control word's numeric parameter is delimited by exactly one
                // space, which is consumed as syntax. But for \uN the fallback
                // char (if the generator emitted one, e.g. \u9731?) sits BEFORE
                // that space, so decide the fallback skip from the char at `j`
                // before eating the delimiter.
                bool u_had_inline_fallback = (word == "u" && has_param && j < n &&
                                              rtf[j] != ' ' && rtf[j] != '\\' &&
                                              rtf[j] != '{' && rtf[j] != '}');
                if (word == "u" && has_param && u_had_inline_fallback) ++j;   // drop the fallback char
                if (j < n && rtf[j] == ' ') ++j;

                if (pending_optional_dest) {
                    // {\*\word ...} : the whole group is an unknown destination.
                    stack.back().skip = true;
                    pending_optional_dest = false;
                } else if (kSkipDest.contains(word)) {
                    stack.back().skip = true;
                } else if (word == "par" || word == "line" || word == "sect" || word == "page") {
                    if (!skipping() && !out.empty() && out.back() != '\n') out.push_back('\n');
                } else if (word == "tab" || word == "cell" || word == "row") {
                    emit('\t');
                } else if (word == "u" && has_param) {
                    // \uN : a Unicode code point (may be negative = +65536).
                    long cp = param < 0 ? param + 65536 : param;
                    if (!skipping() && cp > 0 && cp <= 0x10FFFF) {
                        if (cp < 0x80) out.push_back(static_cast<char>(cp));
                        else if (cp < 0x800) { out.push_back(char(0xC0|(cp>>6))); out.push_back(char(0x80|(cp&0x3F))); }
                        else { out.push_back(char(0xE0|(cp>>12))); out.push_back(char(0x80|((cp>>6)&0x3F))); out.push_back(char(0x80|(cp&0x3F))); }
                    }
                    // The fallback char, when present, was already skipped above.
                }
                // else: an ignored formatting control word (\b, \fs20, ...).
                i = j; continue;
            }
            ++i; continue;
        }
        if (c == '\r' || c == '\n') { ++i; continue; }   // RTF newlines are not text
        emit(c);
        ++i;
    }

    // Collapse runs of blank lines, trim trailing whitespace per line.
    std::string clean; clean.reserve(out.size());
    int nl = 0;
    for (char ch : out) {
        if (ch == '\n') { if (++nl <= 2) clean.push_back(ch); }
        else { nl = 0; clean.push_back(ch); }
    }
    return clean;
}

// ─── PDF → text (via pdftotext) ───────────────────────────────────────────────
Result<std::string> pdf_to_text(const fs::path& path) {
    // pdftotext <in> - : write extracted text to stdout.
    std::string cmd = "pdftotext -q " + std::string("\"") + path.string() + "\" - 2>/dev/null";
    std::FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) return fail<std::string>(Errc::unavailable, "pdftotext not available");
    std::string out;
    std::array<char, 4096> buf;
    std::size_t got;
    while ((got = std::fread(buf.data(), 1, buf.size(), pipe)) > 0) out.append(buf.data(), got);
    int rc = ::pclose(pipe);
    if (rc != 0 && out.empty())
        return fail<std::string>(Errc::unavailable, "pdftotext failed (install poppler-utils)");
    return out;
}

// ─── Single file ──────────────────────────────────────────────────────────────
namespace {
std::string lower_ext(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return e;
}
} // namespace

Result<LoadedDoc> load_file(const fs::path& path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return fail<LoadedDoc>(Errc::not_found, path.string());
    std::string ext = lower_ext(path);

    LoadedDoc d;
    d.uri = path.string();
    d.title = path.stem().string();
    d.meta["ext"] = ext;

    // One dispatcher for every format — in-process for the ones we own
    // (text, HTML, and the whole OOXML family), an external converter for the
    // ones a library owns, a named error for the ones that need OCR. Going
    // through extract_file here rather than special-casing extensions is what
    // makes a .docx work identically in load_file, load_directory, and the CLI,
    // and what lets register_extractor() add a company's internal format to all
    // three at once.
    auto r = extract_file(path);
    if (!r) return std::unexpected(r.error());
    d.text = std::move(r->text);
    if (r->kind == ExtractKind::external && !r->tool.empty()) d.meta["extractor"] = r->tool;

    if (d.text.empty()) return fail<LoadedDoc>(Errc::parse_error, "empty extraction: " + path.string());
    return d;
}

// ─── Directory walk ───────────────────────────────────────────────────────────
Result<std::vector<LoadedDoc>> load_directory(const fs::path& root, const DirOptions& opts) {
    return load_directory(root, opts, nullptr);
}

Result<std::vector<LoadedDoc>>
load_directory(const fs::path& root, const DirOptions& opts, const ProgressFn& on_file) {
    std::error_code ec;
    if (!fs::exists(root, ec)) return fail<std::vector<LoadedDoc>>(Errc::not_found, root.string());
    std::unordered_set<std::string> want(opts.include_ext.begin(), opts.include_ext.end());
    std::unordered_set<std::string> skip_dirs(opts.exclude_dirs.begin(), opts.exclude_dirs.end());

    std::vector<LoadedDoc> docs;
    auto it_opts = opts.follow_symlinks ? fs::directory_options::follow_directory_symlink
                                        : fs::directory_options::none;
    fs::recursive_directory_iterator it(root, it_opts, ec), end;
    for (; it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        const auto& entry = *it;
        if (entry.is_directory(ec)) {
            if (skip_dirs.contains(entry.path().filename().string())) it.disable_recursion_pending();
            continue;
        }
        if (!entry.is_regular_file(ec)) continue;
        std::string ext = lower_ext(entry.path());
        if (!want.contains(ext)) continue;
        if (entry.file_size(ec) > opts.max_file_bytes) { if (on_file) on_file(entry.path(), false); continue; }

        auto d = load_file(entry.path());
        bool ok = d.has_value();
        if (ok) {
            d->meta["rel"] = fs::relative(entry.path(), root, ec).string();
            docs.push_back(std::move(*d));
        }
        if (on_file) on_file(entry.path(), ok);
    }
    return docs;
}

} // namespace rag::loaders
