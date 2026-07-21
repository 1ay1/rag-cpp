// rag/pipeline/pipeline.cpp — stage implementations + the standard pipeline.

#include "rag/pipeline/pipeline.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>

namespace rag::pipeline {

// ── HybridRetrieveStage ──────────────────────────────────────────────────────
Result<Context> HybridRetrieveStage::process(Context ctx) const {
    if (!ctx.corpus) return fail<Context>(Errc::invalid_argument, "no corpus in context");
    const auto& corpus = *ctx.corpus;

    std::vector<fusion::RankedList> lists;

    // Lexical.
    auto lex = corpus.lexical_search(ctx.query, cfg_.candidate_k);
    lists.push_back({std::move(lex), cfg_.bm25_weight});

    // Dense (degrade gracefully if unavailable/offline).
    if (corpus.has_embedder()) {
        auto dense = corpus.dense_search(ctx.query, cfg_.candidate_k);
        if (dense) lists.push_back({std::move(*dense), cfg_.dense_weight});
        else ctx.trace.push_back(std::string("dense unavailable: ") + std::string(to_string(dense.error().code)));
    }

    std::span<const fusion::RankedList> sp(lists);
    ctx.candidates = (cfg_.fusion == HybridRetrieveConfig::Fusion::rrf)
                   ? fusion::rrf(sp, cfg_.rrf, cfg_.candidate_k)
                   : fusion::rsf(sp, cfg_.candidate_k);
    ctx.trace.push_back("hybrid: " + std::to_string(ctx.candidates.size()) + " candidates");
    return ctx;
}

// ── FilterStage ──────────────────────────────────────────────────────────────
Result<Context> FilterStage::process(Context ctx) const {
    if (!ctx.filter || !ctx.corpus) return ctx;
    std::vector<Hit> kept;
    kept.reserve(ctx.candidates.size());
    for (const auto& h : ctx.candidates)
        if (ctx.corpus->passes(h.chunk, ctx.filter)) kept.push_back(h);
    ctx.candidates = std::move(kept);
    return ctx;
}

// ── RerankStage ──────────────────────────────────────────────────────────────
Result<Context> RerankStage::process(Context ctx) const {
    if (!ctx.corpus || ctx.candidates.empty()) return ctx;
    if (auto r = fn_(ctx.query, ctx.candidates, *ctx.corpus); !r)
        return std::unexpected(r.error());
    std::sort(ctx.candidates.begin(), ctx.candidates.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    return ctx;
}

// ── TopKStage ────────────────────────────────────────────────────────────────
Result<Context> TopKStage::process(Context ctx) const {
    if (ctx.candidates.size() > ctx.k) ctx.candidates.resize(ctx.k);
    return ctx;
}

// ── Pipeline ─────────────────────────────────────────────────────────────────
Result<std::vector<Hit>> Pipeline::run(const index::Corpus& corpus, std::string_view query,
                                       std::size_t k, index::MetaFilter filter,
                                       std::vector<std::string>* trace) const {
    Context ctx;
    ctx.query = ctx.original_query = std::string(query);
    ctx.k = k; ctx.corpus = &corpus; ctx.filter = std::move(filter);

    for (const auto& stage : stages_) {
        auto r = stage->process(std::move(ctx));
        if (!r) return std::unexpected(r.error());
        ctx = std::move(*r);
        ctx.trace.push_back(std::string("→ ") + std::string(stage->name()));
    }
    if (trace) *trace = ctx.trace;
    return ctx.candidates;
}

// ── The feature reranker (deterministic, no model) ───────────────────────────
//
// Blends the fused rank score with a lexical-coverage feature: how many of the
// query's content terms actually appear in the candidate's text. This is the
// cheap, calibrated signal that rescues fusion when one retriever dominates.
namespace {
Result<void> feature_rerank(std::string_view query, std::vector<Hit>& cands,
                            const index::Corpus& corpus) {
    auto q_terms = corpus.tokenizer().tokenize(query);
    std::unordered_set<std::string> qset(q_terms.begin(), q_terms.end());
    if (qset.empty()) return {};

    // Normalize fusion scores to [0,1] for a stable blend.
    float lo = 1e30f, hi = -1e30f;
    for (auto& h : cands) { lo = std::min(lo, h.score.get()); hi = std::max(hi, h.score.get()); }
    float range = hi - lo;

    for (auto& h : cands) {
        const Chunk* ch = corpus.chunk(h.chunk);
        if (!ch) continue;
        auto terms = corpus.tokenizer().tokenize(ch->indexed_text());
        std::unordered_set<std::string> tset(terms.begin(), terms.end());
        std::size_t hit = 0;
        for (const auto& q : qset) if (tset.contains(q)) ++hit;
        float coverage = static_cast<float>(hit) / static_cast<float>(qset.size());
        float base = range > 1e-9f ? (h.score.get() - lo) / range : 1.0f;
        // 0.6 fusion + 0.4 coverage — coverage anchors on ABSOLUTE term presence.
        h.score = Score{0.6f * base + 0.4f * coverage};
    }
    return {};
}
} // namespace

Pipeline Pipeline::standard() {
    Pipeline p;
    p.add(std::make_shared<HybridRetrieveStage>())
     .add(std::make_shared<FilterStage>())
     .add(std::make_shared<RerankStage>("feature_rerank", feature_rerank))
     .add(std::make_shared<TopKStage>());
    return p;
}

} // namespace rag::pipeline
