// rag/pipeline/pipeline.cpp — stage implementations + the standard pipeline.

#include "rag/pipeline/pipeline.hpp"
#include "rag/rerank/mmr.hpp"
#include "rag/rerank/refine.hpp"

#include <algorithm>
#include <thread>
#include <array>
#include <unordered_map>
#include <unordered_set>

namespace rag::pipeline {

// ── HybridRetrieveStage ───────────────────────────────────────────
Result<Context> HybridRetrieveStage::process(Context ctx) const {
    if (!ctx.corpus) return fail<Context>(Errc::invalid_argument, "no corpus in context");
    const auto& corpus = *ctx.corpus;

    std::vector<fusion::RankedList> lists;

    // The two retrievers are INDEPENDENT: they read disjoint index structures
    // (inverted lists vs the ANN graph) and neither observes the other's
    // output, so running them one after the other simply adds their latencies.
    // Overlapping them makes hybrid cost max(lexical, dense) instead of the
    // sum — and they have complementary profiles (BM25 is pointer-chasing over
    // postings, the dense walk is bandwidth-bound), so they interleave well.
    //
    // Only worth a thread when there IS a second retriever to run.
    const bool run_dense = corpus.has_embedder();

    // The candidate pool must be at least as deep as what the CALLER asked for,
    // and wider still so the reranker has alternatives to promote. A flat
    // cfg_.candidate_k made the funnel NARROWER THAN THE REQUEST: with the
    // default 60, a k=100 query could never return more than 60 documents, and
    // measured on BEIR/SciFact that capped Recall@100 at 0.9002 against 0.9160
    // for an unbounded lexical scan — the pipeline was throwing away documents
    // it had already found. Scale with k (a 4x rerank funnel is the usual
    // ratio), never below the configured floor.
    const std::size_t pool = std::max(cfg_.candidate_k, ctx.k * 4);

    std::vector<Hit>                 lex;
    Result<std::vector<Hit>>         dense = std::vector<Hit>{};

    auto do_lexical = [&] { lex = corpus.lexical_search(ctx.query, pool); };
    auto do_dense   = [&] {
        // The metadata filter is pushed into the ANN walk as a PRE-filter so a
        // selective predicate still returns a full candidate pool.
        dense = ctx.filter ? corpus.dense_search(ctx.query, pool, ctx.filter)
                           : corpus.dense_search(ctx.query, pool);
    };

    if (run_dense) {
        // Dense on a helper, lexical on this thread. A plain std::thread rather
        // than the shared pool: this may itself be running on a pool worker
        // (a server handling queries in parallel), and blocking a pool thread
        // on work queued to that same pool is the classic way to deadlock.
        std::thread helper(do_dense);
        do_lexical();
        helper.join();
    } else {
        do_lexical();
    }

    // Declare each retriever's THEORETICAL score bounds rather than letting
    // fusion infer them from the candidate set (see fuse.hpp). BM25 has a true
    // floor of 0 and no natural ceiling; cosine over unit vectors is bounded
    // in [-1,1] at both ends.
    lists.push_back(fusion::bm25_list(std::move(lex), cfg_.bm25_weight));

    if (run_dense) {
        // Degrade gracefully if the embedder is unavailable/offline.
        if (dense) lists.push_back(fusion::cosine_list(std::move(*dense), cfg_.dense_weight));
        else ctx.trace.push_back(std::string("dense unavailable: ") + std::string(to_string(dense.error().code)));
    }

    std::span<const fusion::RankedList> sp(lists);
    switch (cfg_.fusion) {
        case HybridRetrieveConfig::Fusion::rrf:
            ctx.candidates = fusion::rrf(sp, cfg_.rrf, pool);
            break;
        case HybridRetrieveConfig::Fusion::rsf:
            ctx.candidates = fusion::rsf(sp, pool);
            break;
        case HybridRetrieveConfig::Fusion::convex:
            ctx.candidates = fusion::convex_combination(sp, cfg_.convex, pool);
            break;
    }
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

// ── PrfExpandStage (RM3-lite pseudo-relevance feedback) ──────────────────────
Result<Context> PrfExpandStage::process(Context ctx) const {
    if (!ctx.corpus) return ctx;
    // Initial probe on the raw query (lexical is enough to seed expansion).
    auto probe = ctx.corpus->lexical_search(ctx.query, cfg_.probe_k);
    if (probe.empty()) return ctx;

    // Mine term frequencies from the top pseudo-relevant chunks, minus the
    // terms already in the query (avoid double-weighting).
    auto q_terms = ctx.corpus->tokenizer().tokenize(ctx.query);
    std::unordered_set<std::string> qset(q_terms.begin(), q_terms.end());
    std::unordered_map<std::string, int> freq;
    std::size_t used = std::min(cfg_.fb_docs, probe.size());
    for (std::size_t i = 0; i < used; ++i) {
        const Chunk* ch = ctx.corpus->chunk(probe[i].chunk);
        if (!ch) continue;
        for (auto& t : ctx.corpus->tokenizer().tokenize(ch->indexed_text()))
            if (!qset.contains(t)) ++freq[t];
    }
    if (freq.empty()) return ctx;

    // Pick the top expansion terms by frequency.
    std::vector<std::pair<std::string,int>> ranked(freq.begin(), freq.end());
    std::partial_sort(ranked.begin(),
        ranked.begin() + static_cast<std::ptrdiff_t>(std::min(cfg_.fb_terms, ranked.size())),
        ranked.end(), [](auto& a, auto& b){ return a.second > b.second; });

    std::string expanded = ctx.query;
    std::size_t added = 0;
    for (auto& [term, f] : ranked) {
        if (added >= cfg_.fb_terms) break;
        expanded += ' '; expanded += term; ++added;
    }
    ctx.query = expanded;
    ctx.trace.push_back("prf: +" + std::to_string(added) + " terms");
    return ctx;
}

// ── ParentStitchStage (small-to-big) ─────────────────────────────────────────
Result<Context> ParentStitchStage::process(Context ctx) const {
    if (!ctx.corpus || ctx.candidates.size() < 2) return ctx;
    // Group surviving candidates by document, keeping best score per group and
    // dropping a chunk that is adjacent-or-overlapping a higher-ranked sibling
    // (its content is already represented by the neighbour we keep).
    std::vector<Hit> kept;
    std::vector<char> dropped(ctx.candidates.size(), 0);

    for (std::size_t i = 0; i < ctx.candidates.size(); ++i) {
        if (dropped[i]) continue;
        const Chunk* ci = ctx.corpus->chunk(ctx.candidates[i].chunk);
        if (!ci) { kept.push_back(ctx.candidates[i]); continue; }
        for (std::size_t j = i + 1; j < ctx.candidates.size(); ++j) {
            if (dropped[j]) continue;
            const Chunk* cj = ctx.corpus->chunk(ctx.candidates[j].chunk);
            if (!cj || cj->doc.get() != ci->doc.get()) continue;
            // Adjacent if line ranges are within max_gap of each other.
            std::uint32_t gap = (cj->start_line > ci->end_line)
                              ? cj->start_line - ci->end_line
                              : (ci->start_line > cj->end_line ? ci->start_line - cj->end_line : 0);
            if (gap <= max_gap_) dropped[j] = 1;   // fold lower-ranked neighbour away
        }
        kept.push_back(ctx.candidates[i]);
    }
    std::size_t merged = ctx.candidates.size() - kept.size();
    ctx.candidates = std::move(kept);
    if (merged) ctx.trace.push_back("stitch: merged " + std::to_string(merged) + " adjacent");
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
    if (q_terms.empty()) return {};
    // Distinct query terms — coverage is over the SET, not the multiset.
    std::sort(q_terms.begin(), q_terms.end());
    q_terms.erase(std::unique(q_terms.begin(), q_terms.end()), q_terms.end());
    const float nq = static_cast<float>(q_terms.size());

    // Normalize fusion scores to [0,1] for a stable blend.
    float lo = 1e30f, hi = -1e30f;
    for (auto& h : cands) { lo = std::min(lo, h.score.get()); hi = std::max(hi, h.score.get()); }
    float range = hi - lo;

    // Coverage from the INVERTED INDEX. The previous implementation tokenized
    // every candidate's full text and built an unordered_set per candidate:
    // with ~60 candidates per query that is thousands of string allocations
    // and hash inserts per query, and it made the hybrid path an order of
    // magnitude slower than either of its halves (2.2ms vs 0.15/0.24ms).
    // The index already records which chunks contain which terms; asking it
    // costs one merge-walk of the query terms' postings, no tokenization.
    std::vector<std::uint32_t> ids;
    ids.reserve(cands.size());
    for (const auto& h : cands) ids.push_back(h.chunk.get());
    std::vector<std::uint32_t> covered;
    corpus.term_coverage(q_terms, ids, covered);

    for (std::size_t i = 0; i < cands.size(); ++i) {
        float coverage = static_cast<float>(covered[i]) / nq;
        float base = range > 1e-9f ? (cands[i].score.get() - lo) / range : 1.0f;
        // Blend fusion score with absolute term coverage.
        //
        // The weight used to be 0.4, which was a GUESS and a bad one. MEASURED
        // on three BEIR datasets (nDCG@10, lexical corpus, this exact pipeline),
        // sweeping w in score = (1-w)*fusion + w*coverage:
        //
        //     w      SciFact   NFCorpus  ArguAna   mean
        //     0.00   0.6800    0.3266    0.3720    0.4595   <- best mean
        //     0.10   0.6809    0.3261    0.3628    0.4566
        //     0.20   0.6836    0.3254    0.3508    0.4533
        //     0.30   0.6798    0.3236    0.3398    0.4477
        //     0.40   0.6751    0.3221    0.3275    0.4416   <- the old default
        //     0.50   0.6681    0.3190    0.3093    0.4321
        //
        // Coverage helps ONLY on SciFact (peak 0.20, +0.0036) and monotonically
        // HURTS on NFCorpus and ArguAna — on ArguAna the old default cost
        // -0.045 nDCG@10, because counting distinct query terms present throws
        // away the IDF weighting BM25 computed so carefully: a common term
        // counts exactly as much as a rare discriminative one.
        //
        // Tuning to SciFact's 0.20 optimum would be overfitting to one dataset.
        // A small non-zero weight is kept because coverage is a genuine signal
        // when fusion scores are degenerate (all-equal, e.g. a single-retriever
        // corpus), and 0.10 costs ~0.003 mean nDCG while retaining that
        // tie-breaking behaviour. Override via RerankStage with your own fn if
        // your corpus measures differently — and measure, do not guess.
        constexpr float kCoverageWeight = 0.10f;
        cands[i].score = Score{(1.0f - kCoverageWeight) * base + kCoverageWeight * coverage};
    }
    return {};
}
} // namespace

StagePtr make_feature_rerank_stage(std::string label) {
    return std::make_shared<RerankStage>(std::move(label), feature_rerank);
}

Pipeline Pipeline::standard() { return standard_with(HybridRetrieveConfig{}); }

Pipeline Pipeline::standard_with(HybridRetrieveConfig cfg) {
    Pipeline p;
    p.add(std::make_shared<HybridRetrieveStage>(std::move(cfg)))
     .add(std::make_shared<FilterStage>())
     .add(std::make_shared<RerankStage>("feature_rerank", feature_rerank))
     .add(std::make_shared<TopKStage>());
    return p;
}

Pipeline Pipeline::quality(float mmr_lambda) {
    return quality_with(HybridRetrieveConfig{}, mmr_lambda);
}

Pipeline Pipeline::quality_with(HybridRetrieveConfig cfg, float mmr_lambda) {
    Pipeline p;
    p.add(std::make_shared<HybridRetrieveStage>(std::move(cfg)))
     .add(std::make_shared<FilterStage>())
     .add(std::make_shared<RerankStage>("feature_rerank", feature_rerank))
     // MMR runs on the RELEVANCE-ORDERED candidate pool, before the trim to k.
     // Order matters: after TopKStage there would be nothing left to diversify
     // — the duplicates it exists to displace would already have been kept and
     // the alternatives already discarded.
     .add(rerank::make_mmr_stage(mmr_lambda))
     .add(std::make_shared<TopKStage>());
    return p;
}

Pipeline Pipeline::context(std::size_t max_gap) {
    return context_with(HybridRetrieveConfig{}, max_gap);
}

Pipeline Pipeline::context_with(HybridRetrieveConfig cfg, std::size_t max_gap) {
    Pipeline p;
    p.add(std::make_shared<HybridRetrieveStage>(std::move(cfg)))
     .add(std::make_shared<FilterStage>())
     .add(std::make_shared<RerankStage>("feature_rerank", feature_rerank))
     // ParentStitch folds adjacent same-document fragments into their
     // higher-ranked sibling — so it must run AFTER the rerank that establishes
     // that order, and BEFORE the top-k that would trim away the pool it
     // promotes distinct locations from. Same slot-ordering argument as MMR.
     .add(std::make_shared<ParentStitchStage>(max_gap))
     .add(std::make_shared<TopKStage>());
    return p;
}

Pipeline Pipeline::best() {
    HybridRetrieveConfig cfg;
    // Per-query adaptive alpha: weight the more confident retriever more on
    // each query. The static prior (cfg.convex.alpha) is the fallback it
    // regresses toward, so this never strays far on a normal query.
    cfg.convex.adaptive = true;
    return best_with(std::move(cfg));
}

Pipeline Pipeline::best_with(HybridRetrieveConfig cfg) {
    Pipeline p;
    p.add(std::make_shared<HybridRetrieveStage>(std::move(cfg)))
     .add(std::make_shared<FilterStage>())
     .add(std::make_shared<RerankStage>("feature_rerank", feature_rerank))
     // Dedup AFTER rerank so it keeps the higher-ranked member of each
     // near-duplicate cluster; BEFORE autocut/top-k so the survivors are
     // distinct and the tail is measured on real content, not paraphrases.
     .add(rerank::make_dedup_stage())
     // Autocut trims the low-relevance tail at the score knee. Last refinement
     // before top-k so it cuts on the final relevance order.
     .add(rerank::make_autocut_stage())
     .add(std::make_shared<TopKStage>());
    return p;
}

} // namespace rag::pipeline
