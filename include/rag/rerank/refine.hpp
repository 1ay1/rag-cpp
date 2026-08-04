#pragma once
// rag/rerank/refine.hpp — candidate-set refinement stages: near-duplicate
// dedup and relevance autocut.
//
// Two cheap, composable stages that clean up a ranked candidate set before it
// reaches an expensive downstream consumer (a cross-encoder reranker, or the
// caller's LLM context window). Both model the RetrievalStage interface and
// degrade gracefully — dense cosine when the corpus is embedded, lexical
// Jaccard otherwise — exactly like MMR.
//
//   • DedupStage — folds NEAR-DUPLICATE chunks (paraphrases, boilerplate,
//     re-hosted copies) that a retriever loves to surface together. It keeps
//     the highest-ranked representative of each cluster and drops the rest, so
//     a top-n rerank budget (or an LLM context) is not wasted re-reading the
//     same content five times. This is a QUALITY-and-COST stage: it does not
//     reorder survivors, it removes redundancy.
//
//   • AutocutStage — trims the long low-relevance tail by finding the largest
//     relative DROP in the score curve and cutting there (Weaviate's "autocut"
//     / the classic knee/elbow heuristic). A query with three good answers and
//     forty weak ones returns three, not k; a query with a smooth relevance
//     ramp is left alone. Keeps precision high without a hand-tuned threshold.

#include <cstddef>
#include <string>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"

namespace rag::rerank {

// ─── Near-duplicate dedup ─────────────────────────────────────────────────────

struct DedupConfig {
    // Two candidates are duplicates when their similarity ≥ this. 0.92 is
    // deliberately high: it catches paraphrases and re-hosted copies without
    // folding two genuinely distinct passages that merely share a topic.
    float       threshold = 0.92f;
    // Only consider the top `window` candidates for folding (a duplicate far
    // down the tail will be cut by top-k anyway). 0 = whole set.
    std::size_t window    = 0;
};

// Fold near-duplicates in `candidates` (assumed relevance-sorted), keeping the
// first (highest-ranked) member of each duplicate cluster. Returns the deduped
// list in the same order. O(n²) similarity in the window, but n is the rerank
// pool (tens), and each sim() is one cached dot product on the dense path.
[[nodiscard]] std::vector<Hit>
dedup(const index::Corpus& corpus, std::span<const Hit> candidates, DedupConfig cfg = {});

// Pipeline-stage form. Insert BEFORE an expensive rerank stage (so the model's
// top-n budget is spent on distinct passages) or before top-k.
[[nodiscard]] pipeline::StagePtr
make_dedup_stage(DedupConfig cfg = {}, std::string label = "dedup");

// ─── Relevance autocut ────────────────────────────────────────────────────────

struct AutocutConfig {
    // Cut at the first gap whose size is ≥ mean-gap × `sensitivity`. Higher =
    // more conservative (fewer cuts). 2.0 cuts at a gap twice the average.
    float       sensitivity = 2.0f;
    // Never cut above this many results (protects recall for smooth curves).
    std::size_t min_keep     = 1;
    // Only scan the first `scan` candidates for the knee. 0 = whole set.
    std::size_t scan         = 0;
};

// Trim `candidates` (relevance-sorted) at the largest relative score drop.
[[nodiscard]] std::vector<Hit>
autocut(std::span<const Hit> candidates, AutocutConfig cfg = {});

// Pipeline-stage form. Insert AFTER rerank, BEFORE top-k, so it cuts on the
// final relevance order.
[[nodiscard]] pipeline::StagePtr
make_autocut_stage(AutocutConfig cfg = {}, std::string label = "autocut");

} // namespace rag::rerank
