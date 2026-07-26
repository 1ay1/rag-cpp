#pragma once
// rag/loaders/ooxml.hpp — Word, Excel and PowerPoint without a dependency.
//
// A .docx is a ZIP archive of XML parts. So is .xlsx, .pptx, and every other
// OOXML format Microsoft shipped since 2007. That means extracting text from
// them needs exactly two things: a ZIP reader and a DEFLATE decompressor.
// Both are small enough to own outright, which is the entire argument for
// doing it here rather than linking libzip + zlib + a DOM parser:
//
//   * an office document loader that pulls in three transitive dependencies is
//     one nobody enables in a hardened build, and it becomes dead code
//   * the formats' text extraction is genuinely simple; the complexity in
//     OOXML lives in layout and styling, which retrieval discards anyway
//   * a self-contained implementation works identically on every platform,
//     which is the property that makes a corpus portable
//
// So: an in-process ZIP central-directory reader, an in-process INFLATE (the
// raw DEFLATE of RFC 1951, both fixed and dynamic Huffman), and per-format
// text extraction that understands enough of each schema to preserve the
// structure retrieval actually uses -- paragraphs, headings, table rows,
// sheet cells, slide titles.
//
// What this deliberately does NOT do: layout reconstruction, style resolution,
// change tracking, embedded objects, or the legacy pre-2007 binary formats
// (.doc/.xls/.ppt), which share no code with these and are handled by the
// external-tool seam in extract.hpp instead.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::loaders {

// ─── ZIP ──────────────────────────────────────────────────────────────────────

struct ZipEntry {
    std::string name;
    std::size_t offset = 0;          // local header offset
    std::uint32_t comp_size = 0;
    std::uint32_t uncomp_size = 0;
    std::uint16_t method = 0;        // 0 = stored, 8 = deflate
};

// List the archive's central directory. Returns invalid_argument when `data`
// is not a ZIP; this is the check that keeps a mislabelled file from being
// interpreted as an office document.
[[nodiscard]] Result<std::vector<ZipEntry>> zip_entries(std::string_view data);

// Extract one entry's bytes. Handles stored and deflated members.
[[nodiscard]] Result<std::string> zip_read(std::string_view data, const ZipEntry& e);

// Convenience: find an entry by exact name and extract it.
[[nodiscard]] Result<std::string> zip_read_name(std::string_view data, std::string_view name);

// Raw DEFLATE (RFC 1951) decompression. `expected` is a size hint from the ZIP
// header, used to pre-size the output; decoding does not trust it.
[[nodiscard]] Result<std::string> inflate_raw(std::string_view in, std::size_t expected);

// ─── OOXML text extraction ────────────────────────────────────────────────────

// Strip XML tags to text, honouring a small set of OOXML elements that carry
// structure worth keeping: <w:p> and <w:br> become newlines, <w:tab> a tab,
// <a:p> (drawing text) a newline. Everything else is dropped, and entities are
// decoded. Shared by all three formats.
[[nodiscard]] std::string ooxml_xml_to_text(std::string_view xml);

// Extract readable text from a whole .docx / .xlsx / .pptx byte buffer.
// `parse_error` if the archive has none of the expected parts.
[[nodiscard]] Result<std::string> docx_to_text(std::string_view bytes);
[[nodiscard]] Result<std::string> xlsx_to_text(std::string_view bytes);
[[nodiscard]] Result<std::string> pptx_to_text(std::string_view bytes);

// Dispatch on the archive's contents rather than on a file extension, so a
// document that was renamed, served without a name, or handed over as a blob
// still extracts correctly.
[[nodiscard]] Result<std::string> ooxml_to_text(std::string_view bytes);

// ─── OpenDocument (LibreOffice / OpenOffice) ──────────────────────────────────
// .odt / .ods / .odp are also ZIP archives of XML, with the body in
// `content.xml` and structure carried by text:/table: elements. Extraction
// reuses the same ZIP reader and the shared xml_to_text (which recognises the
// ODF element names), so these formats are in-process and dependency-free too.
[[nodiscard]] Result<std::string> odf_to_text(std::string_view bytes);

// ─── EPUB ─────────────────────────────────────────────────────────────────────
// An .epub is a ZIP of XHTML content documents. Extraction reads the spine
// order from the OPF package when present (so chapters read in reading order),
// falls back to sorted XHTML parts otherwise, and strips each to text.
[[nodiscard]] Result<std::string> epub_to_text(std::string_view bytes);

// The one entry point for any ZIP-container document: sniffs OOXML vs ODF vs
// EPUB from the archive contents and dispatches. Returns parse_error if the
// zip is a container we do not understand (a .jar, a plain archive).
[[nodiscard]] Result<std::string> zip_document_to_text(std::string_view bytes);

// Is this buffer a ZIP container? Cheap magic-number check.
[[nodiscard]] bool looks_like_zip(std::string_view bytes) noexcept;

} // namespace rag::loaders
