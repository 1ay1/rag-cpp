// rag/loaders/extract.cpp — the one place a file becomes text.

#include "rag/loaders/extract.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>

#include "rag/loaders/loaders.hpp"
#include "rag/loaders/ooxml.hpp"

namespace rag::loaders {

namespace fs = std::filesystem;

namespace {

std::shared_mutex& registry_mutex() {
    static std::shared_mutex m;
    return m;
}
struct Custom { ExtractFn fn; std::string desc; };
std::unordered_map<std::string, Custom>& registry() {
    static std::unordered_map<std::string, Custom> r;
    return r;
}

std::string lower(std::string_view s) {
    std::string o(s);
    std::transform(o.begin(), o.end(), o.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return o;
}

std::string ext_of(std::string_view hint) {
    if (hint.empty()) return {};
    if (hint.front() == '.' && hint.find('/') == std::string_view::npos &&
        hint.find('.', 1) == std::string_view::npos)
        return lower(hint);
    auto dot = hint.rfind('.');
    if (dot == std::string_view::npos) return {};
    return lower(hint.substr(dot));
}

std::string run_capture(const std::string& cmd) {
    std::string out;
    std::FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) return out;
    std::array<char, 4096> buf;
    std::size_t got;
    while ((got = std::fread(buf.data(), 1, buf.size(), pipe)) > 0) out.append(buf.data(), got);
    ::pclose(pipe);
    return out;
}

std::string shell_quote(const std::string& s) {
    std::string q = "'";
    for (char c : s) {
        if (c == '\'') q += "'\\''";
        else q += c;
    }
    q += "'";
    return q;
}

// External converters, in preference order per extension. The first one
// present on the machine wins.
struct Converter { const char* tool; const char* argfmt; };

const std::unordered_map<std::string, std::vector<Converter>>& converters() {
    static const std::unordered_map<std::string, std::vector<Converter>> k = {
        {".pdf",  {{"pdftotext", "pdftotext -q %s -"},
                   {"mutool",    "mutool draw -F txt %s"}}},
        {".doc",  {{"antiword",  "antiword %s"},
                   {"catdoc",    "catdoc %s"},
                   {"textutil",  "textutil -stdout -convert txt %s"}}},
        {".rtf",  {{"textutil",  "textutil -stdout -convert txt %s"},
                   {"unrtf",     "unrtf --text %s"}}},
        {".xls",  {{"xls2csv",   "xls2csv %s"}}},
        {".ppt",  {{"catppt",    "catppt %s"}}},
        {".epub", {{"pandoc",    "pandoc -t plain %s"}}},
        {".odt",  {{"pandoc",    "pandoc -t plain %s"}}},
    };
    return k;
}

// A PDF whose pages carry images but no text layer. pdftotext returns almost
// nothing for these, and the difference between "extraction failed" and "this
// document is a photograph" is the difference between a bug report and an
// actionable instruction to the operator.
bool looks_like_scanned_pdf(const std::string& extracted, std::uintmax_t file_bytes) {
    if (file_bytes < 4096) return false;
    // Under ~1 character of text per KB of file, there is no text layer.
    const double per_kb = static_cast<double>(extracted.size()) /
                          (static_cast<double>(file_bytes) / 1024.0);
    return per_kb < 1.0;
}

} // namespace

bool tool_available(std::string_view tool) {
    static std::mutex m;
    static std::unordered_map<std::string, bool> cache;
    std::string t(tool);
    {
        std::lock_guard lk(m);
        if (auto it = cache.find(t); it != cache.end()) return it->second;
    }
    const bool ok = !run_capture("command -v " + shell_quote(t) + " 2>/dev/null").empty();
    std::lock_guard lk(m);
    cache[t] = ok;
    return ok;
}

void register_extractor(std::string extension, ExtractFn fn, std::string description) {
    std::string e = lower(extension);
    if (!e.empty() && e.front() != '.') e.insert(e.begin(), '.');
    std::unique_lock lk(registry_mutex());
    registry()[std::move(e)] = Custom{std::move(fn), std::move(description)};
}

std::vector<std::string> custom_extractors() {
    std::shared_lock lk(registry_mutex());
    std::vector<std::string> out;
    out.reserve(registry().size());
    for (const auto& [k, v] : registry()) out.push_back(k);
    std::sort(out.begin(), out.end());
    return out;
}

Result<ExtractResult> extract_bytes(std::string_view bytes, std::string_view hint) {
    const std::string ext = ext_of(hint);

    // A custom extractor is the caller's explicit instruction and outranks
    // everything, including content sniffing: if someone registered a handler
    // for ".rpt" they know more about their format than we do.
    {
        std::shared_lock lk(registry_mutex());
        if (auto it = registry().find(ext); it != registry().end()) {
            ExtractFn fn = it->second.fn;
            lk.unlock();
            auto t = fn(bytes);
            if (!t) return std::unexpected(t.error());
            return ExtractResult{std::move(*t), ExtractKind::native, "custom"};
        }
    }

    // Content sniffing beats the extension. A .docx renamed to .txt is common
    // (mail gateways do it), and reading a zip container as UTF-8 produces a
    // chunk of binary noise that indexes without complaint.
    if (looks_like_zip(bytes)) {
        if (auto t = ooxml_to_text(bytes)) return ExtractResult{std::move(*t), ExtractKind::native, {}};
        // A zip that is not OOXML (a .jar, a plain archive) has no text of its
        // own; saying so beats indexing the compressed bytes.
        return fail<ExtractResult>(Errc::parse_error, "zip archive is not an office document");
    }

    if (ext == ".html" || ext == ".htm" || ext == ".xhtml")
        return ExtractResult{html_to_text(bytes), ExtractKind::native, {}};

    // The legacy binary Office formats begin with the OLE compound-document
    // magic. They cannot be read from memory here, so this is a clear error
    // rather than a silent pass-through of binary rubbish.
    static const unsigned char kOle[8] = {0xD0,0xCF,0x11,0xE0,0xA1,0xB1,0x1A,0xE1};
    if (bytes.size() >= 8 &&
        std::memcmp(bytes.data(), kOle, 8) == 0)
        return fail<ExtractResult>(Errc::unavailable,
            "legacy binary Office format (.doc/.xls/.ppt): needs an external converter, "
            "use extract_file() or install antiword/catdoc/textutil");

    if (bytes.size() >= 5 && bytes.substr(0, 5) == "%PDF-")
        return fail<ExtractResult>(Errc::unavailable,
            "PDF: needs an external converter, use extract_file() or install poppler-utils");

    // Reject binary content rather than indexing it. A NUL byte in the first
    // kilobyte is the cheapest reliable signal.
    const std::size_t probe = std::min<std::size_t>(bytes.size(), 1024);
    if (bytes.substr(0, probe).find('\0') != std::string_view::npos)
        return fail<ExtractResult>(Errc::parse_error, "binary content, no text extractor registered");

    return ExtractResult{std::string(bytes), ExtractKind::native, {}};
}

Result<ExtractResult> extract_file(const fs::path& path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return fail<ExtractResult>(Errc::not_found, path.string());
    const std::string ext = lower(path.extension().string());
    const std::uintmax_t size = fs::file_size(path, ec);

    // Custom extractors and in-process formats work from bytes.
    const auto& conv = converters();
    auto cit = conv.find(ext);
    bool has_custom = false;
    {
        std::shared_lock lk(registry_mutex());
        has_custom = registry().contains(ext);
    }

    if (has_custom || cit == conv.end()) {
        std::ifstream in(path, std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        std::string bytes = ss.str();
        auto r = extract_bytes(bytes, path.filename().string());
        if (r) return r;
        // Fall through to a converter when one exists for the sniffed type
        // (a .pdf named .bin, say); otherwise report the original error.
        if (cit == conv.end()) return r;
    }

    // An external converter is needed. Pick the first installed one.
    for (const Converter& c : cit->second) {
        if (!tool_available(c.tool)) continue;
        std::string cmd(c.argfmt);
        const std::string q = shell_quote(path.string());
        if (auto at = cmd.find("%s"); at != std::string::npos) cmd.replace(at, 2, q);
        cmd += " 2>/dev/null";
        std::string text = run_capture(cmd);

        if (ext == ".pdf" && looks_like_scanned_pdf(text, size)) {
            // Be explicit. This is the case people spend an afternoon on,
            // convinced the loader is broken, when the document is a photo.
            return fail<ExtractResult>(Errc::unavailable,
                "PDF appears to be scanned images with no text layer (" +
                std::to_string(text.size()) + " chars from " + std::to_string(size / 1024) +
                " KB): OCR required, e.g. `ocrmypdf in.pdf out.pdf` then re-ingest");
        }
        if (text.empty()) continue;
        return ExtractResult{std::move(text), ExtractKind::external, c.tool};
    }

    // Nothing installed. Name the tools, so the message is actionable.
    std::string tools;
    for (const Converter& c : cit->second) {
        if (!tools.empty()) tools += " or ";
        tools += c.tool;
    }
    return fail<ExtractResult>(Errc::unavailable,
        "no extractor for " + ext + ": install " + tools);
}

std::vector<FormatSupport> capabilities() {
    std::vector<FormatSupport> out;
    auto native = [&](const char* e, const char* note) {
        out.push_back({e, ExtractKind::native, true, {}, note});
    };
    native(".txt",  "plain text");
    native(".md",   "markdown");
    native(".html", "tag stripping + entity decode");
    native(".csv",  "one document per row, columns as filterable metadata");
    native(".docx", "in-process ZIP + DEFLATE + OOXML");
    native(".xlsx", "in-process, shared-string table resolved");
    native(".pptx", "in-process, slides ordered, speaker notes included");
    native("(source code)", "definition-aligned chunking; unknown languages inferred");

    for (const auto& [ext, list] : converters()) {
        FormatSupport f;
        f.extension = ext;
        f.kind = ExtractKind::external;
        for (const Converter& c : list) {
            if (tool_available(c.tool)) { f.available = true; f.requires_tool = c.tool; break; }
        }
        if (!f.available) {
            for (const Converter& c : list) {
                if (!f.requires_tool.empty()) f.requires_tool += " or ";
                f.requires_tool += c.tool;
            }
            f.note = "not installed";
        } else {
            f.note = "via " + f.requires_tool;
        }
        out.push_back(std::move(f));
    }

    out.push_back({"(scanned pdf)", ExtractKind::ocr, false, "ocrmypdf / tesseract",
                   "no OCR engine bundled; detected and reported, never guessed"});

    std::shared_lock lk(registry_mutex());
    for (const auto& [ext, c] : registry())
        out.push_back({ext, ExtractKind::native, true, {}, c.desc.empty() ? "custom" : c.desc});

    std::sort(out.begin(), out.end(), [](const FormatSupport& a, const FormatSupport& b) {
        return a.extension < b.extension;
    });
    return out;
}

} // namespace rag::loaders
