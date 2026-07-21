#pragma once
// rag/ralm/ralm.hpp — Retrieval-Augmented Language Modeling assembly.
//
// The retrieval library's job stops at the boundary of the generator, but the
// SHAPE of what you hand the generator is itself a research contribution. This
// module implements the retrieval-side machinery of four landmark RALM recipes
// so a caller can wire any black-box LM behind them:
//
//   • RAG (Lewis et al. 2020) — marginalize the generator over the top-k
//     retrieved documents, each weighted by its (softmax-normalized) retrieval
//     probability p(z|x). We expose those normalized weights.
//
//   • RETRO (Borgeaud et al. 2022) — CHUNKED cross-attention: split the input
//     into fixed strides, retrieve nearest neighbours per stride, and attach
//     each neighbour's CONTINUATION (the following chunk) so the LM sees "what
//     came next". We build those (neighbour ⊕ continuation) records.
//
//   • In-Context RALM (Ram et al. 2023) — leave the LM untouched; retrieve at a
//     fixed generation STRIDE and prepend documents, RERANKING the retrieved
//     set to pick the single best in-context document. We provide the stride
//     schedule + a rerank hook.
//
//   • REPLUG (Shi et al. 2024) — treat the LM as a black box; run it once PER
//     retrieved document and ENSEMBLE the output distributions weighted by the
//     retrieval likelihood. We compute the ensemble weights (temperature-scaled
//     softmax over retrieval scores) and expose an ensembling combinator.
//
// Everything here is generation-agnostic: you supply the corpus + query, we
// return typed, weighted retrieval assemblies. The LM call is YOUR seam.

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"

namespace rag::ralm {

// A retrieved document paired with its ensemble weight (weights sum to 1).
struct WeightedDoc {
    Hit         hit;
    float       weight = 0.0f;   // p(z|x): normalized retrieval probability
    std::string text;            // resolved chunk text (context ⊕ body)
};

// ── RAG / REPLUG ensemble weighting ─────────────────────────────────────────
// Turn raw retrieval scores into a probability distribution over the retrieved
// documents via a temperature-scaled softmax. `temperature`→0 sharpens toward
// the top hit; →∞ flattens toward uniform. This is p(z|x) in RAG and the
// ensemble weight in REPLUG.
[[nodiscard]] std::vector<WeightedDoc>
ensemble_weights(const index::Corpus& corpus, std::span<const Hit> hits,
                 float temperature = 1.0f);

// REPLUG combination: given, for each retrieved doc, the LM's next-token
// distribution (a probability row over the vocabulary), return the ensembled
// distribution Σ_z p(z|x)·p_LM(y|x,z). Rows must share length; weights are the
// ensemble_weights above. This is the numeric heart of REPLUG, LM-agnostic.
[[nodiscard]] std::vector<float>
replug_combine(std::span<const WeightedDoc> docs,
               std::span<const std::vector<float>> per_doc_logits);

// The LM seam REPLUG drives: given (query, one retrieved passage) → the LM's
// next-token distribution. You implement it over your model.
using LmScorer = std::function<Result<std::vector<float>>(std::string_view query,
                                                          std::string_view passage)>;

// End-to-end REPLUG step: retrieve k, weight, call `score` per passage, combine.
[[nodiscard]] Result<std::vector<float>>
replug_step(const index::Corpus& corpus, std::string_view query, std::size_t k,
            const LmScorer& score, float temperature = 1.0f);

// ── RETRO chunked-neighbour retrieval ───────────────────────────────────────
// A RETRO neighbour record: the retrieved chunk PLUS its continuation (the next
// chunk in the same document), so the LM attends to "neighbour and what
// followed it" — the key RETRO trick.
struct RetroNeighbour {
    ChunkId     neighbour = ChunkId::invalid();
    ChunkId     continuation = ChunkId::invalid();  // next chunk in same doc
    Score       score{0.0f};
    std::string neighbour_text;
    std::string continuation_text;
};

// One RETRO retrieval "row": the input stride and its neighbours (with conts).
struct RetroRow {
    std::string                 stride_text;
    std::vector<RetroNeighbour> neighbours;
};

struct RetroConfig {
    std::size_t stride     = 64;   // input split granularity (tokens, approx)
    std::size_t neighbours = 2;    // neighbours retrieved per stride
    bool        with_continuation = true;
};

// Split `input` into strides, retrieve per-stride neighbours from the corpus,
// and attach continuations. This is RETRO's retrieval frontend; the chunked
// cross-attention itself lives in the model.
[[nodiscard]] Result<std::vector<RetroRow>>
retro_retrieve(const index::Corpus& corpus, std::string_view input,
               RetroConfig cfg = {});

// ── In-Context RALM stride schedule ─────────────────────────────────────────
// In-Context RALM re-retrieves every `stride` generated tokens using the last
// `query_len` tokens as the query, then reranks the retrieved set and prepends
// the winner. We produce the schedule of (position → chosen document) given a
// rerank hook, so the caller drives generation and just consults the plan.
struct RalmConfig {
    std::size_t stride    = 4;    // retrieval stride (tokens); paper uses small
    std::size_t retrieve_k = 8;   // pool reranked at each retrieval point
    std::size_t query_len = 32;   // trailing tokens used as the query
};

// Rerank hook: given (query, candidate passages) return the index of the best
// passage. Absent → pick the top-retrieved (rank-0) passage.
using RerankPick = std::function<std::size_t(std::string_view,
                                             std::span<const std::string>)>;

// A single retrieval decision at a generation position.
struct RalmDecision {
    std::size_t position = 0;      // token index where retrieval fired
    Hit         chosen;            // the prepended document
    std::string text;
};

// Plan the in-context RALM retrievals for generating `n_tokens`, given the
// evolving context supplied lazily via `context_at(position)` (the trailing
// text so far). Deterministic given the hooks.
[[nodiscard]] Result<std::vector<RalmDecision>>
incontext_plan(const index::Corpus& corpus, std::size_t n_tokens,
               const std::function<std::string(std::size_t)>& context_at,
               RalmConfig cfg = {}, RerankPick pick = {});

// ── Prompt assembly ─────────────────────────────────────────────────────────
// Assemble a grounded prompt from weighted docs: numbered, source-attributed
// passages (In-Context RALM / IBM-RAG style) followed by the query. The
// attribution is what makes RAG answers checkable.
[[nodiscard]] std::string
assemble_prompt(std::string_view query, std::span<const WeightedDoc> docs,
                std::string_view instruction = {});

} // namespace rag::ralm
