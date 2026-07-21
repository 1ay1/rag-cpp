#pragma once
// rag/pipeline/pipeline.hpp — composable retrieval stages + the Engine.
//
// A retrieval query flows through an ordered list of Stages, each transforming
// a Context (the query + the running candidate set + scratch). This is the
// "retrieve → expand → fuse → rerank → compress" funnel as a pipes-and-filters
// architecture: every stage has the same interface, so stages compose in any
// order and new capabilities are added without touching the core.
//
// Stages are runtime-polymorphic (abstract base RetrievalStage) because a
// pipeline is assembled at run time from a config. The hot inner scoring loops
// they call (BM25, cosine, HNSW) remain non-virtual inside Corpus.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/fusion/fuse.hpp"
#include "rag/index/corpus.hpp"

namespace rag::pipeline {

// The mutable state threaded through the pipeline.
struct Context {
    std::string                query;          // possibly rewritten by a stage
    std::string                original_query; // never mutated
    std::vector<Hit>           candidates;     // the running result set
    std::size_t                k = 10;         // desired final count
    const index::Corpus*       corpus = nullptr;
    index::MetaFilter          filter;         // optional metadata predicate
    std::vector<std::string>   trace;          // per-stage diagnostics
};

// A composable transformation. Total: returns the (possibly failed) Context.
class RetrievalStage {
public:
    virtual ~RetrievalStage() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual Result<Context> process(Context ctx) const = 0;
};

using StagePtr = std::shared_ptr<RetrievalStage>;

// ─────────────────────────────────────────────────────────────────────────────
// Concrete stages
// ─────────────────────────────────────────────────────────────────────────────

// Hybrid retrieval: runs BM25 + dense (if available) and fuses with RRF/RSF.
// This is normally the FIRST stage — it populates `candidates`.
struct HybridRetrieveConfig {
    std::size_t candidate_k = 60;    // per-retriever pool before fusion
    float       bm25_weight = 1.0f;
    float       dense_weight = 1.0f;
    enum class Fusion { rrf, rsf } fusion = Fusion::rrf;
    fusion::RrfParams rrf{};
};

class HybridRetrieveStage final : public RetrievalStage {
public:
    explicit HybridRetrieveStage(HybridRetrieveConfig cfg = {}) : cfg_(cfg) {}
    std::string_view name() const noexcept override { return "hybrid_retrieve"; }
    Result<Context> process(Context ctx) const override;
private:
    HybridRetrieveConfig cfg_;
};

// Metadata pre/post filter: drops candidates whose document metadata fails the
// context's filter predicate.
class FilterStage final : public RetrievalStage {
public:
    std::string_view name() const noexcept override { return "filter"; }
    Result<Context> process(Context ctx) const override;
};

// A generic Ranker adapter: lifts anything modelling the Ranker concept (or a
// std::function) into a stage that reorders candidates.
class RerankStage final : public RetrievalStage {
public:
    using RerankFn = std::function<Result<void>(std::string_view, std::vector<Hit>&, const index::Corpus&)>;
    explicit RerankStage(std::string label, RerankFn fn)
        : label_(std::move(label)), fn_(std::move(fn)) {}
    std::string_view name() const noexcept override { return label_; }
    Result<Context> process(Context ctx) const override;
private:
    std::string label_;
    RerankFn    fn_;
};

// Truncate to k. Usually the final stage.
class TopKStage final : public RetrievalStage {
public:
    std::string_view name() const noexcept override { return "top_k"; }
    Result<Context> process(Context ctx) const override;
};

// Pseudo-Relevance Feedback (RM3-lite): run an initial retrieval, harvest the
// top terms from the top-`fb_docs` chunks, append the best `fb_terms` to the
// query, and let the NEXT retrieve stage use the expanded query. Classic recall
// booster for under-specified queries. Insert BEFORE HybridRetrieveStage.
struct ExpandConfig {
    std::size_t fb_docs  = 5;    // pseudo-relevant docs to mine
    std::size_t fb_terms = 8;    // expansion terms to add
    std::size_t probe_k  = 20;   // initial probe depth
};
class PrfExpandStage final : public RetrievalStage {
public:
    explicit PrfExpandStage(ExpandConfig cfg = {}) : cfg_(cfg) {}
    std::string_view name() const noexcept override { return "prf_expand"; }
    Result<Context> process(Context ctx) const override;
private:
    ExpandConfig cfg_;
};

// Parent-document stitch (small-to-big): after ranking on fine chunks, merge
// adjacent chunks from the SAME document that both survived into the result, so
// the caller gets coherent, de-fragmented context windows. Insert AFTER rerank,
// BEFORE top-k.
class ParentStitchStage final : public RetrievalStage {
public:
    explicit ParentStitchStage(std::size_t max_gap = 1) : max_gap_(max_gap) {}
    std::string_view name() const noexcept override { return "parent_stitch"; }
    Result<Context> process(Context ctx) const override;
private:
    std::size_t max_gap_;   // merge chunks whose line ranges are within this gap
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline — an ordered sequence of stages.
// ─────────────────────────────────────────────────────────────────────────────
class Pipeline {
public:
    Pipeline& add(StagePtr stage) { stages_.push_back(std::move(stage)); return *this; }

    [[nodiscard]] Result<std::vector<Hit>>
    run(const index::Corpus& corpus, std::string_view query, std::size_t k,
        index::MetaFilter filter = {}, std::vector<std::string>* trace = nullptr) const;

    // A sensible default: hybrid retrieve → filter → feature rerank → top-k.
    [[nodiscard]] static Pipeline standard();

private:
    std::vector<StagePtr> stages_;
};

} // namespace rag::pipeline
