#pragma once
// rag/loaders/structure.hpp — chunking for languages nobody has a parser for.
//
// The usual answer to "support every language" is a grammar per language:
// tree-sitter and friends. That answer has a hard ceiling. It cannot help with
// the in-house DSL your config lives in, the fourth-generation language your
// mainframe team maintains, the vendor format with no public spec, or the
// dialect someone invented last quarter — precisely the code that is most
// valuable to search, because none of it is on the public internet and no
// model has memorized it.
//
// This header takes the other route: INFER the structure from the file itself.
// A source file is not arbitrary text. It is written by people following a
// convention, and a convention repeated a few hundred times leaves statistical
// fingerprints that are visible without knowing what the language MEANS:
//
//   * definitions start at a repeated, low indentation level
//   * a definition line begins with one of a small set of leading tokens
//     ("def", "func", "DEFINE", "SECTION", "rule", "@endpoint", "  ---")
//   * nesting is delimited either by brackets or by indentation, consistently
//   * comment lines share a prefix and are not themselves structure
//
// So: scan the file once, score every candidate leading token by how well it
// behaves like a definition keyword, and chunk on the winners. No grammar, no
// dependency, no per-language code — and it works the first time it sees a
// language, which a grammar by construction cannot.
//
// This is deliberately a DIFFERENT tool from code_chunker.hpp's hand-written
// per-language heuristics. Those are better when they apply, because a human
// encoded real knowledge of the language. `chunk_source` below uses them when
// the extension is recognized and falls back to inference when it isn't, so
// known languages never regress and unknown ones stop being shredded.
//
// Measured by bench/structure_bench.cpp — see BENCHMARKS.md.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/loaders/code_chunker.hpp"

namespace rag::loaders {

// What a single pass over a file concluded about how it is written.
struct StructureProfile {
    // Leading tokens that behave like definition keywords, best first.
    std::vector<std::string> definition_tokens;

    // Line prefix that starts a comment ("#", "//", "--", ";", "*"), empty if
    // no consistent one was found. Comment lines immediately above a definition
    // belong to it (they are its docstring), which is why we must detect them.
    std::string comment_prefix;

    // Indentation column at which definitions live. Most languages: 0.
    std::size_t definition_indent = 0;

    // Does nesting use brackets, or indentation? Decides how we find the END
    // of a definition, which is the part that actually matters for chunking.
    bool bracket_nesting = false;

    // Fraction of lines that are blank. High values (>0.25) mean blank lines
    // are used as separators and are a usable structural signal on their own.
    double blank_ratio = 0.0;

    // Confidence in [0,1] that the above describes real structure rather than
    // noise found in prose. Below `kMinConfidence` the caller should not use
    // this profile; chunking prose as if it were code produces worse chunks
    // than the prose chunker does.
    double confidence = 0.0;

    [[nodiscard]] bool usable() const noexcept;
};

// Confidence below which an inferred profile should be ignored.
inline constexpr double kMinConfidence = 0.35;

// Infer how `body` is structured. Single pass, no allocation per line beyond
// the token table. Safe on any input including binary garbage and empty files.
[[nodiscard]] StructureProfile infer_structure(std::string_view body);

// Chunk `body` on the boundaries implied by `profile`. Chunks carry the
// definition line as `context`, same convention as chunk_code.
[[nodiscard]] std::vector<Chunk>
chunk_by_structure(DocId doc, const std::string& body,
                   const StructureProfile& profile,
                   const CodeChunkOptions& opts = {});

// ─── The front door ───────────────────────────────────────────────────────────
//
// Chunk source text the best way available for it, in this order:
//   1. a hand-written per-language chunker, if `ext` names a known language
//   2. inferred structure, if the file has any worth using
//   3. size-bounded windows (what the caller would have gotten anyway)
//
// This is what the loaders and the CLI call. Pass the extension when you have
// it; pass "" when you don't (stdin, a blob store, a language whose files have
// no extension) and inference will carry the file on its own.
[[nodiscard]] std::vector<Chunk>
chunk_source(DocId doc, std::string_view ext, const std::string& body,
             const CodeChunkOptions& opts = {});

// Reports which of the three strategies chunk_source would pick. Exposed for
// diagnostics and for the bench; retrieval does not need it.
enum class ChunkStrategy : std::uint8_t { known_language, inferred, windows };
[[nodiscard]] ChunkStrategy strategy_for(std::string_view ext, const std::string& body);
[[nodiscard]] std::string_view strategy_name(ChunkStrategy s);

} // namespace rag::loaders
