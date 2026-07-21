#pragma once
// rag/text/contextual.hpp — Anthropic Contextual Retrieval (2024).
//
// A chunk ripped from its document loses the context that disambiguates it
// ("the company's revenue grew 3%" — which company? which quarter?). Contextual
// Retrieval prepends to each chunk a short, LLM-generated blurb SITUATING it
// within its source document, BEFORE indexing. Anthropic reports this cuts
// retrieval failures ~35% (49% with reranking) because both the BM25 and dense
// representations now carry the disambiguating context.
//
// The library already threads a `context` breadcrumb (heading chain) into every
// chunk's indexed_text(). This module adds the paper's stronger variant: an
// LLM `Contextualizer` seam that, given (whole document, this chunk), returns a
// 1-2 sentence situating context. Absent a model, we fall back to a
// deterministic extractive context (the document title + its most salient
// sentence overlapping the chunk) — free, and still a real disambiguator.

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"

namespace rag::text {

// The LLM seam: (full document text, chunk text) → a short situating context.
using Contextualizer =
    std::function<Result<std::string>(std::string_view document, std::string_view chunk)>;

// Populate each chunk's `context` field with a situating blurb derived from
// `document`. Uses `ctx` if provided, else the deterministic extractive default.
// Existing (heading-breadcrumb) context is preserved and appended to.
void contextualize(std::vector<Chunk>& chunks, std::string_view document,
                   const Contextualizer& ctx = {});

// The deterministic default: document title (first non-empty line) + the
// document sentence with the highest term overlap with the chunk.
[[nodiscard]] std::string
extractive_context(std::string_view document, std::string_view chunk);

} // namespace rag::text
