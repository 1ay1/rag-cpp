#pragma once
// rag/cascade/cascade.hpp — a multi-stage retrieval cascade with budgets.
//
// The accuracy/latency frontier of retrieval is a CASCADE: a cheap wide first
// stage, then progressively more expensive, more accurate re-scorers applied to
// progressively smaller candidate sets. This is how production search actually
// runs (the "telescoping" or "funnel" pattern):
//
//   stage 0  hybrid retrieve      top-N₀   (BM25+dense, milliseconds)
//   stage 1  ColBERT late-interact top-N₁   (token MaxSim, N₁ ≪ N₀)
//   stage 2  cross-encoder rerank  top-N₂   (joint encode, N₂ ≪ N₁)
//   → return top-k
//
// Each stage has a BUDGET (how many candidates it is allowed to score) so the
// expensive stages never blow up. Stages are optional and pluggable — omit the
// cross-encoder and you have a bi-encoder→ColBERT cascade; the shape is data.
//
// This ties together the retrieval, late-interaction, and rerank modules behind
// one call, and degrades gracefully: any stage that is unavailable is skipped
// and the pipeline continues with the prior stage's ranking.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"
#include "rag/late/colbert.hpp"
#include "rag/rerank/reranker.hpp"

namespace rag::cascade {

struct CascadeConfig {
    std::size_t retrieve_k  = 200;   // stage-0 pool
    std::size_t colbert_k   = 50;    // survivors after late interaction
    std::size_t rerank_k    = 20;    // survivors after cross-encoder
    std::size_t final_k     = 10;    // returned
    bool        use_colbert = true;
    bool        use_rerank  = true;
};

// A trace entry per executed stage (for observability / tuning).
struct StageTrace { std::string stage; std::size_t in; std::size_t out; };

class Cascade {
public:
    explicit Cascade(CascadeConfig cfg = {}) : cfg_(cfg) {}

    // Attach the late-interaction reranker (stage 1). Fluent.
    Cascade& with_colbert(late::ColbertReranker cb) { colbert_ = std::make_shared<late::ColbertReranker>(std::move(cb)); return *this; }
    // Attach the cross-encoder reranker (stage 2). Fluent.
    Cascade& with_reranker(rerank::AnyReranker rr) { reranker_ = std::make_shared<rerank::AnyReranker>(std::move(rr)); return *this; }

    // Run the cascade. `trace`, if non-null, receives per-stage in/out counts.
    [[nodiscard]] Result<std::vector<Hit>>
    run(const index::Corpus& corpus, std::string_view query,
        std::vector<StageTrace>* trace = nullptr) const;

private:
    CascadeConfig cfg_;
    std::shared_ptr<late::ColbertReranker> colbert_;
    std::shared_ptr<rerank::AnyReranker>   reranker_;
};

} // namespace rag::cascade
