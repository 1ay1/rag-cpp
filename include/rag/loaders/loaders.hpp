#pragma once
// rag/loaders/loaders.hpp — turn files and folders into ingestable documents.
//
// A Loader reads a source (a path, a byte blob) and yields LoadedDoc records
// (uri + plaintext + metadata) ready for Engine::add / Corpus::add_document.
// Extraction is best-effort and dependency-light:
//
//   • PlainText     — utf-8 text as-is.
//   • Markdown      — text as-is (the chunker already understands headings).
//   • Html          — tag-stripping + entity decode to readable text.
//   • Pdf           — shells out to `pdftotext` if present; else reports
//                     unavailable (no bundled PDF parser — that stays a plug-in).
//   • Directory     — recursive walk with include/exclude globs, dispatching
//                     each file to the loader matching its extension.
//
// Code files get language-aware chunking via loaders/code_chunker.hpp.

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"

namespace rag::loaders {

struct LoadedDoc {
    std::string uri;
    std::string title;
    std::string text;
    Metadata    meta;
};

// ─── Single-source extractors ──────────────────────────────────────
[[nodiscard]] std::string html_to_text(std::string_view html);

// RTF -> text, in-process. Handles the control words that carry text structure
// (\par, \line, \tab, \cell), \'hh hex byte escapes, unicode \uN, and skips
// the destination groups ({\*\...}, fonts, stylesheets, pictures) that hold no
// document text. It is not a full RTF reader — it extracts the words, which is
// all retrieval wants — so a self-contained .rtf loader needs no textutil.
[[nodiscard]] std::string rtf_to_text(std::string_view rtf);

[[nodiscard]] Result<std::string> pdf_to_text(const std::filesystem::path& path); // needs `pdftotext`

// Load one file, choosing the extractor from its extension. Returns unavailable
// for binary/unsupported types.
[[nodiscard]] Result<LoadedDoc> load_file(const std::filesystem::path& path);

// ─── Tabular (CSV/TSV) ───────────────────────────────────────────────
// A CSV is not prose, and chunking it as prose destroys it: rows are
// independent records, and a token window that spans a row boundary produces a
// chunk describing two unrelated things. So each ROW becomes its own document,
// with every column also attached as filterable metadata — which means the
// structure survives into the query layer, where you can filter on
// meta["status"] == "open" instead of hoping the word "open" was retrieved.
struct CsvOptions {
    char        delimiter    = ',';    // ';' or '\t' for European CSV / TSV
    bool        has_header   = true;   // first row names the columns
    // Columns to concatenate into the searchable text. Empty = every column.
    // Naming them matters when the table has one prose column and ten id
    // columns: indexing the ids adds noise and dilutes BM25's term statistics.
    std::vector<std::string> text_columns;
    // Column whose value becomes the document title (empty = none).
    std::string title_column;
    // Column whose value becomes the uri suffix. Empty = the 1-based row number,
    // which is always available and always unique.
    std::string id_column;
    // Attach every column as metadata, not just the unindexed ones.
    bool        meta_all_columns = true;
};

// Parse CSV text into one LoadedDoc per data row.
//
// Handles the parts of RFC 4180 that occur in practice: quoted fields, embedded
// delimiters and newlines inside quotes, and "" as an escaped quote. Returns
// invalid_argument on a row whose field count does not match the header, rather
// than silently misaligning every column after the error.
[[nodiscard]] Result<std::vector<LoadedDoc>>
load_csv_text(std::string_view csv, const std::string& uri_prefix, const CsvOptions& opts = {});

// Same, reading from a file. The extension picks a default delimiter (.tsv -> tab).
[[nodiscard]] Result<std::vector<LoadedDoc>>
load_csv(const std::filesystem::path& path, const CsvOptions& opts = {});

// ─── Directory walk ───────────────────────────────────────────────────────────
struct DirOptions {
    std::vector<std::string> include_ext = {   // lowercase, with dot
        ".md",".markdown",".txt",".rst",".html",".htm",".pdf",
        ".c",".h",".cpp",".hpp",".cc",".cxx",".py",".js",".ts",".tsx",".jsx",
        ".go",".rs",".java",".rb",".php",".cs",".swift",".kt",".scala",".sh",
        ".json",".yaml",".yml",".toml",".sql",
        // Office and e-book formats extracted IN-PROCESS — no dependency.
        // OOXML, OpenDocument (LibreOffice/OpenOffice) and EPUB are all ZIP of
        // XML; RTF is a control-word stream. See ooxml.hpp / rtf_to_text.
        ".docx",".xlsx",".pptx",".odt",".ods",".odp",".epub",".rtf",
        // Legacy binary Office, via an external converter when one is installed.
        // Listed here so they are ATTEMPTED: a corpus that silently skipped
        // every .doc looked, to its owner, like one that had indexed them.
        ".doc",".xls",".ppt",
    };
    std::vector<std::string> exclude_dirs = {
        ".git","node_modules","build","dist","target","__pycache__",".venv","venv",
    };
    std::size_t max_file_bytes = 4 * 1024 * 1024;   // skip huge files
    bool        follow_symlinks = false;
};

// Recursively load a directory. Each returned LoadedDoc has meta["ext"],
// meta["lang"] (for code), and meta["rel"] (path relative to root).
[[nodiscard]] Result<std::vector<LoadedDoc>>
load_directory(const std::filesystem::path& root, const DirOptions& opts = {});

// Progress callback variant (called per file); returns count loaded.
using ProgressFn = std::function<void(const std::filesystem::path&, bool ok)>;
[[nodiscard]] Result<std::vector<LoadedDoc>>
load_directory(const std::filesystem::path& root, const DirOptions& opts, const ProgressFn& on_file);

} // namespace rag::loaders
