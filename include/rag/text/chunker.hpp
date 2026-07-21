#pragma once
// rag/text/chunker.hpp — semantic, line-aligned document chunking.
//
// Splits a document into bounded chunks that (a) never break mid-line, (b)
// prefer to break on blank lines / markdown headings (semantic boundaries),
// (c) overlap a few trailing lines so a fact straddling a boundary survives,
// and (d) carry a `context` breadcrumb (the enclosing markdown heading chain)
// for contextual retrieval.

#include <string>
#include <vector>

#include "rag/core/document.hpp"

namespace rag::text {

struct ChunkOptions {
    std::size_t max_lines     = 40;
    std::size_t max_chars     = 1600;
    std::size_t overlap_lines = 4;
    bool        heading_context = true;   // synthesize breadcrumb from headings
};

// Split `body` into chunks. Chunk ids are left invalid (the Corpus assigns
// them on ingest); doc is set to `doc_id`.
[[nodiscard]] std::vector<Chunk>
chunk_document(DocId doc_id, const std::string& body, const ChunkOptions& opts = {});

} // namespace rag::text
