// rag/pipeline/pipeline.cpp — stage implementations + the standard pipeline.

#include "rag/pipeline/pipeline.hpp"

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

    std::vector<Hit>                 lex;
    Result<std::vector<Hit>>         dense = std::vector<Hit>{};

    auto do_lexical = [&] { lex = corpus.lexical_search(ctx.query, cfg_.candidate_k); };
    auto do_dense   = [&] {
        // The metadata filter is pushed into the ANN walk as a PRE-filter so a
        // selective predicate still returns a full candidate pool.
        dense = ctx.filter ? corpus.dense_search(ctx.query, cfg_.candidate_k, ctx.filter)
                           : corpus.dense_search(ctx.query, cfg_.candidate_k);
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

    lists.push_back({std::move(lex), cfg_.bm25_weight});

    if (run_dense) {
        // Degrade gracefully if the embedder is unavailable/offline.
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
        // 0.6 fusion + 0.4 coverage — coverage anchors on ABSOLUTE term presence.
        cands[i].score = Score{0.6f * base + 0.4f * coverage};
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
