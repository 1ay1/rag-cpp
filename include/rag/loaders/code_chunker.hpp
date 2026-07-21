#pragma once
// rag/loaders/code_chunker.hpp — language-aware chunking for source code.
//
// Splitting code on blank lines (the prose chunker) shreds functions. This
// chunker instead breaks on TOP-LEVEL definition boundaries (functions,
// classes, methods) detected by lightweight per-language heuristics, so each
// chunk is a coherent semantic unit. It falls back to size-bounded windows for
// languages it doesn't recognize, and always respects a max size.
//
// This is retrieval-oriented, NOT a parser: it uses brace/indent heuristics
// that are robust to the messiness of real code without a grammar per language.

#include <string>
#include <string_view>
#include <vector>

#include "rag/core/document.hpp"

namespace rag::loaders {

enum class Language {
    unknown, c_like, python, ruby, go, rust, javascript, markup,
};

[[nodiscard]] Language detect_language(std::string_view ext);
[[nodiscard]] std::string_view language_name(Language l);

struct CodeChunkOptions {
    std::size_t max_lines = 80;
    std::size_t max_chars = 3000;
    std::size_t min_lines = 3;    // merge tiny fragments upward
};

// Chunk source `body` into definition-aligned chunks. Each chunk's `context`
// carries the enclosing symbol path (e.g. "class Foo › method bar") when it can
// be inferred.
[[nodiscard]] std::vector<Chunk>
chunk_code(DocId doc, std::string_view ext, const std::string& body,
           const CodeChunkOptions& opts = {});

} // namespace rag::loaders
