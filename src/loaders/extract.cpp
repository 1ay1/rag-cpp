// rag/loaders/extract.cpp — the one place a file becomes text.

#include "rag/loaders/extract.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

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

std::FILE* open_pipe(const std::string& command) {
#ifdef _WIN32
    return ::_popen(command.c_str(), "r");
#else
    return ::popen(command.c_str(), "r");
#endif
}

int close_pipe(std::FILE* pipe) {
#ifdef _WIN32
    return ::_pclose(pipe);
#else
    return ::pclose(pipe);
#endif
}

std::string run_capture(const std::string& cmd) {
    std::string out;
    std::FILE* pipe = open_pipe(cmd);
    if (!pipe) return out;
    std::array<char, 4096> buf;
    std::size_t got;
    while ((got = std::fread(buf.data(), 1, buf.size(), pipe)) > 0) out.append(buf.data(), got);
    close_pipe(pipe);
    return out;
}

std::string shell_quote(const std::string& s) {
#ifdef _WIN32
    // Windows filenames cannot contain a double quote. Quoting the complete
    // path keeps spaces and cmd metacharacters such as '&' literal.
    return "\"" + s + "\"";
#else
    std::string q = "'";
    for (char c : s) {
        if (c == '\'') q += "'\\''";
        else q += c;
    }
    q += "'";
    return q;
#endif
}

std::string null_redirect() {
#ifdef _WIN32
    return " 2>NUL";
#else
    return " 2>/dev/null";
#endif
}

// Append one Unicode code point to `out` as UTF-8.
void append_utf8(std::string& out, unsigned long cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// If `bytes` starts with a Unicode Byte-Order-Mark, decode the whole buffer to
// UTF-8 and return it. A UTF-8 BOM is stripped; UTF-16 LE/BE is transcoded
// (with surrogate-pair handling). Returns nullopt when there is no BOM, so the
// caller falls through to its plain-text / binary logic unchanged.
std::optional<std::string> decode_bom_text(std::string_view b) {
    auto u = [&](std::size_t i) { return static_cast<unsigned char>(b[i]); };

    // UTF-8 BOM: EF BB BF. Strip it; the rest is already UTF-8.
    if (b.size() >= 3 && u(0) == 0xEF && u(1) == 0xBB && u(2) == 0xBF)
        return std::string(b.substr(3));

    const bool le = b.size() >= 2 && u(0) == 0xFF && u(1) == 0xFE;
    const bool be = b.size() >= 2 && u(0) == 0xFE && u(1) == 0xFF;
    if (!le && !be) return std::nullopt;

    std::string out;
    out.reserve(b.size() / 2);
    for (std::size_t i = 2; i + 1 < b.size(); i += 2) {
        unsigned unit = le ? (u(i) | (u(i + 1) << 8))
                           : ((u(i) << 8) | u(i + 1));
        // High surrogate: combine with the following low surrogate.
        if (unit >= 0xD800 && unit <= 0xDBFF && i + 3 < b.size()) {
            unsigned lo = le ? (u(i + 2) | (u(i + 3) << 8))
                             : ((u(i + 2) << 8) | u(i + 3));
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                unsigned long cp = 0x10000 + (((unit - 0xD800) << 10) | (lo - 0xDC00));
                append_utf8(out, cp);
                i += 2;
                continue;
            }
        }
        if (unit == 0) continue;   // stray padding NUL, drop it
        append_utf8(out, unit);
    }
    return out;
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
        // .rtf is handled in-process (rtf_to_text); no converter needed.
        {".xls",  {{"xls2csv",   "xls2csv %s"}}},
        {".ppt",  {{"catppt",    "catppt %s"}}},
        // .epub / .odt / .ods / .odp are handled in-process (ZIP of XML), so no
        // external converter is listed for them — see zip_document_to_text.
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
#ifdef _WIN32
    const bool ok = !run_capture("where " + shell_quote(t) + null_redirect()).empty();
#else
    const bool ok = !run_capture("command -v " + shell_quote(t) + null_redirect()).empty();
#endif
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
    // chunk of binary noise that indexes without complaint. One door handles
    // OOXML, OpenDocument (.odt/.ods/.odp) and EPUB — all ZIP-of-XML — so all of
    // them extract in-process regardless of what the file was named.
    if (looks_like_zip(bytes)) {
        if (auto t = zip_document_to_text(bytes)) return ExtractResult{std::move(*t), ExtractKind::native, {}};
        // A zip that is not a known document (a .jar, a plain archive) has no
        // text of its own; saying so beats indexing the compressed bytes.
        return fail<ExtractResult>(Errc::parse_error,
            "zip archive is not a recognised document (OOXML / OpenDocument / EPUB)");
    }

    if (ext == ".html" || ext == ".htm" || ext == ".xhtml")
        return ExtractResult{html_to_text(bytes), ExtractKind::native, {}};

    // RTF is plain ASCII beginning with `{\rtf`. Sniff it so a renamed or
    // extension-less RTF still extracts, and handle it in-process — no textutil.
    if (ext == ".rtf" || bytes.rfind("{\\rtf", 0) == 0)
        return ExtractResult{rtf_to_text(bytes), ExtractKind::native, {}};

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

    // A Unicode Byte-Order-Mark means the text is encoded, not binary. Windows
    // Notepad, Excel CSV export and many editors write UTF-16 (LE or BE) or a
    // UTF-8 BOM. Decoding these to UTF-8 is the difference between indexing a
    // Windows-authored file and rejecting it below as "binary content" the
    // moment its first character sits in the high plane and shows a NUL byte.
    if (auto decoded = decode_bom_text(bytes))
        return ExtractResult{std::move(*decoded), ExtractKind::native, "bom-decoded"};

    // Reject binary content rather than indexing it. A NUL byte in the first
    // kilobyte is the cheapest reliable signal. (BOM-marked UTF-16 is handled
    // above, so a NUL here is genuine binary, not a wide-char text file.)
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

    // ZIP-container documents (OOXML + OpenDocument + EPUB) are extracted
    // in-process; the converter, if any, is only a fallback. So even though
    // .odt/.epub appear in the converters map (pandoc), try the byte path FIRST
    // for them — a self-contained extraction beats shelling out when it works.
    static const std::unordered_set<std::string> kZipNative = {
        ".docx", ".xlsx", ".pptx", ".docm", ".xlsm", ".pptm",
        ".odt", ".ods", ".odp", ".odg", ".epub"};
    // RTF is handled in-process too, but a converter (textutil/unrtf) stays as a
    // fallback for the rare document our stripper cannot make sense of.
    const bool zip_native = kZipNative.contains(ext) || ext == ".rtf";

    if (has_custom || zip_native || cit == conv.end()) {
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
        cmd += null_redirect();
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
    native(".odt",  "in-process OpenDocument (LibreOffice/OpenOffice) text");
    native(".ods",  "in-process OpenDocument spreadsheet, rows preserved");
    native(".odp",  "in-process OpenDocument presentation");
    native(".epub", "in-process ZIP + XHTML spine, reading order");
    native(".rtf",  "in-process control-word stripping, \\'hh + \\uN escapes");
    native("(utf-16)", "BOM-detected UTF-16 LE/BE and UTF-8 decoded to UTF-8");
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
