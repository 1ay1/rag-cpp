#pragma once
// rag/text/semantic_chunker.hpp — embedding-boundary + proposition chunking.
//
// The default chunker splits on lines/headings/size. That can cut mid-topic.
// SEMANTIC chunking (Kamradt 2024) instead splits where MEANING shifts: embed
// each sentence, walk the document, and start a new chunk when the cosine
// similarity between consecutive sentences drops below a percentile threshold —
// so a chunk is a run of topically-coherent sentences, however long the prose.
//
// PROPOSITION chunking (Chen et al. 2023, "dense-x") is the finer extreme:
// break text into atomic, self-contained factual statements. We provide a
// deterministic proposition splitter (sentence-level with pronoun-free
// heuristics) and an optional LLM seam for true proposition extraction.
//
// Both require sentence embeddings; without an embedder they degrade to the
// standard line/size chunker via a lexical-shift fallback.

#include <functional>
#include <string>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"

namespace rag::text {

struct SemanticChunkOptions {
    float       breakpoint_percentile = 90.0f;  // split when dist > this pct
    std::size_t min_sentences = 1;              // never emit shorter chunks
    std::size_t max_chars     = 2000;           // hard cap even if coherent
    bool        heading_context = true;
};

// Split `body` into semantically-coherent chunks using `embedder` to measure
// sentence-to-sentence drift. Falls back to lexical drift if embedding fails.
[[nodiscard]] Result<std::vector<Chunk>>
semantic_chunk(DocId doc_id, const std::string& body,
               const dense::AnyEmbedder& embedder, SemanticChunkOptions opts = {});

// Lexical-only variant (no embedder): splits on Jaccard drift between adjacent
// sentence windows. Deterministic, dependency-free.
[[nodiscard]] std::vector<Chunk>
semantic_chunk_lexical(DocId doc_id, const std::string& body, SemanticChunkOptions opts = {});

// Proposition extraction seam: text → atomic statements. Absent → the built-in
// deterministic splitter.
using PropositionFn = std::function<Result<std::vector<std::string>>(std::string_view)>;

// Break `body` into proposition-level chunks (one atomic statement each).
[[nodiscard]] std::vector<Chunk>
proposition_chunk(DocId doc_id, const std::string& body, PropositionFn extract = {});

} // namespace rag::text
