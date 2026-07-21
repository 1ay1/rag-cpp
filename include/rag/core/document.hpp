#pragma once
// rag/core/document.hpp — the corpus record types (product types).

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "rag/core/types.hpp"

namespace rag {

// Arbitrary user metadata attached to a document. Used for filtered retrieval.
using Metadata = std::map<std::string, std::string>;

// A source document as ingested by the user. `id` is assigned by the Corpus.
struct Document {
    DocId       id = DocId::invalid();
    std::string uri;       // stable external identity (path, URL, key)
    std::string title;     // optional display title / heading breadcrumb root
    std::string text;      // full body
    Metadata    meta;      // filterable key/value pairs
};

// A chunk is a bounded, retrievable window into a Document. It carries the
// "contextual retrieval" breadcrumb (`context`) prepended into the indexed
// text so a fragment that lost its heading still ranks (Anthropic 2024).
struct Chunk {
    ChunkId               id     = ChunkId::invalid();
    DocId                 doc    = DocId::invalid();
    std::string           text;          // the chunk body (raw)
    std::string           context;       // heading breadcrumb / synthesized ctx
    std::uint32_t         start_line = 0;
    std::uint32_t         end_line   = 0;
    Vector                embedding;      // dense vector (empty until embedded)
    const Metadata*       meta = nullptr; // borrowed from the owning Document

    // The text actually fed to the lexical + dense indexes: context ⊕ body.
    [[nodiscard]] std::string indexed_text() const {
        if (context.empty()) return text;
        return context + "\n" + text;
    }
};

// A ranked, resolved retrieval result the caller consumes.
struct SearchResult {
    ChunkId       chunk = ChunkId::invalid();
    DocId         doc   = DocId::invalid();
    Score         score{0.0f};
    std::string   text;
    std::string   context;
    std::string   uri;
    std::uint32_t start_line = 0;
    std::uint32_t end_line   = 0;
};

} // namespace rag
