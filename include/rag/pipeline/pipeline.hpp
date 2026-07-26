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
    // Convex combination (TM2C2) is the default: Bruch et al. 2023 found it
    // "significantly outperforms RRF on all datasets in terms of NDCG", in
    // both in-domain and zero-shot settings, because it preserves the score
    // distribution that RRF throws away. `rrf` remains available and is still
    // the right choice when fusing retrievers whose scores are not
    // commensurable at all. Note bm25_weight/dense_weight are ignored under
    // `convex`, which uses `convex.alpha` instead.
    enum class Fusion { rrf, rsf, convex } fusion = Fusion::convex;
    fusion::RrfParams    rrf{};
    fusion::ConvexParams convex{};
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

    // The standard pipeline with the hybrid retrieval stage configured — the
    // seam for per-request overrides (an RCP client choosing a fusion method)
    // without mutating the server's shared Engine.
    [[nodiscard]] static Pipeline standard_with(HybridRetrieveConfig cfg);

    // OPT-IN: standard() plus MMR diversity reranking.
    //
    // WHY THIS IS NOT THE DEFAULT. MMR does not make ranking more accurate — it
    // makes the top-k COVER MORE of the answer, by refusing to spend several
    // slots on near-duplicates of one passage. Those are different goods, and
    // only the second is measurable with a coverage metric. On a benchmark
    // built for the distinction (12 topics x 10 distinct facets x 20 near
    // duplicate documents, so the top 10 cannot hold every facet):
    //
    //   pipeline                nDCG@10   distinct facets in top 10 (of 10)
    //   relevance only           1.0000        6.25 / 6.00 / 6.17
    //   + MMR lambda=0.3         1.0000        7.50 / 7.75 / 7.58
    //   + MMR lambda=0.5         1.0000        7.08 / 7.42 / 7.75
    //   + MMR lambda=0.7         1.0000        6.67 / 6.67 / 7.50
    //                                          (three seeds: 42 / 7 / 1234)
    //
    // nDCG is IDENTICAL at 1.0 everywhere, which is the honest headline: on
    // this corpus every top-10 hit is already on-topic, so a relevance metric
    // cannot see the difference at all. Coverage rises ~20-25%.
    //
    // The default stays narrow because the trade is real: lambda < 1 will
    // demote a genuinely more relevant passage in favour of a more novel one,
    // which is wrong for a lookup query with exactly one right answer, and
    // right for a broad question whose answer spans several passages. That is a
    // property of the QUERY, not of the corpus, so the caller chooses.
    //
    // lambda=0.5 balances the two; the benchmark above prefers 0.3, but on a
    // corpus with less redundancy that would over-diversify, so the paper's
    // balanced default is kept rather than one tuned to a synthetic benchmark.
    [[nodiscard]] static Pipeline quality(float mmr_lambda = 0.5f);

    // quality() with the retrieval stage configured, for the same reason
    // standard_with exists.
    [[nodiscard]] static Pipeline quality_with(HybridRetrieveConfig cfg, float mmr_lambda = 0.5f);

private:
    std::vector<StagePtr> stages_;
};

} // namespace rag::pipeline
