#pragma once
// rag/crag/crag.hpp — Corrective RAG (CRAG) + Self-RAG reflection gates.
//
// RAG's blind spot: it prepends whatever retrieval returned, even when retrieval
// FAILED. CRAG (Yan et al. 2024) adds a lightweight RETRIEVAL EVALUATOR that
// grades the retrieved set and triggers one of three actions:
//
//   • Correct   — confidence high → refine (decompose-then-recompose: keep the
//     knowledge "strips" that matter, drop the noise).
//   • Ambiguous — middling → combine refined internal knowledge WITH an external
//     (web) fallback.
//   • Incorrect — confidence low → discard retrieval, go to the external source.
//
// Self-RAG (Asai et al. 2024) adds REFLECTION: per-passage "IsRelevant" and, at
// answer time, "IsSupported" / "IsUseful" critique tokens. We expose those as
// gate hooks so a caller can drop irrelevant passages and score groundedness.
//
// The evaluator here is model-FREE by default: it grades a hit by blending its
// retrieval score's separation from the pack with lexical query-overlap of the
// passage — a real, deterministic confidence signal. A learned evaluator (a
// T5-grader, an LLM) can be injected through `Evaluator` to match the paper.

#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"

namespace rag::crag {

enum class Action { correct, ambiguous, incorrect };

[[nodiscard]] constexpr std::string_view to_string(Action a) noexcept {
    switch (a) {
        case Action::correct:   return "correct";
        case Action::ambiguous: return "ambiguous";
        case Action::incorrect: return "incorrect";
    }
    return "?";
}

// Per-passage relevance evaluator seam (Self-RAG IsRelevant). Given the query
// and a passage, return a relevance score in [0,1]. Absent → model-free default.
using Evaluator = std::function<float(std::string_view query, std::string_view passage)>;

// External-knowledge fallback seam (CRAG's web search). Given the query, return
// replacement/supplementary passages. Absent → no fallback (retrieval is kept).
using ExternalSource = std::function<Result<std::vector<std::string>>(std::string_view)>;

struct CragConfig {
    float upper = 0.60f;   // ≥ upper  → Correct
    float lower = 0.30f;   // ≤ lower  → Incorrect ; between → Ambiguous
    std::size_t strips = 3;  // decompose-recompose: keep top-N knowledge strips
    bool  drop_irrelevant = true;   // Self-RAG: filter passages below `lower`
};

// A graded retrieval result: the action taken, the overall confidence, and the
// corrected / filtered knowledge (refined internal strips ⊕ external fallback).
struct Correction {
    Action                   action = Action::correct;
    float                    confidence = 0.0f;
    std::vector<Hit>         kept;            // surviving/refined internal hits
    std::vector<std::string> external;        // fallback passages (if triggered)
    std::vector<std::string> knowledge;       // final recomposed knowledge strips
};

// Grade a query's retrieval and correct it. `hits` is the raw retrieved set.
[[nodiscard]] Correction
correct(const index::Corpus& corpus, std::string_view query, std::span<const Hit> hits,
        CragConfig cfg = {}, Evaluator eval = {}, ExternalSource external = {});

// Self-RAG groundedness: fraction of the answer's claims (sentence-level)
// supported by the knowledge — a deterministic "IsSupported" signal in [0,1].
[[nodiscard]] float
support_score(const index::Corpus& corpus, std::string_view answer,
              std::span<const std::string> knowledge);

} // namespace rag::crag
