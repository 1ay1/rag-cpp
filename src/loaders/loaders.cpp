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
