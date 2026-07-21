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

// ─── Single-source extractors ─────────────────────────────────────────────────
[[nodiscard]] std::string html_to_text(std::string_view html);
[[nodiscard]] Result<std::string> pdf_to_text(const std::filesystem::path& path); // needs `pdftotext`

// Load one file, choosing the extractor from its extension. Returns unavailable
// for binary/unsupported types.
[[nodiscard]] Result<LoadedDoc> load_file(const std::filesystem::path& path);

// ─── Directory walk ───────────────────────────────────────────────────────────
struct DirOptions {
    std::vector<std::string> include_ext = {   // lowercase, with dot
        ".md",".markdown",".txt",".rst",".html",".htm",".pdf",
        ".c",".h",".cpp",".hpp",".cc",".cxx",".py",".js",".ts",".tsx",".jsx",
        ".go",".rs",".java",".rb",".php",".cs",".swift",".kt",".scala",".sh",
        ".json",".yaml",".yml",".toml",".sql",
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
