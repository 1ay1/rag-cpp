// rag/cascade/cascade.cpp — multi-stage retrieval cascade with budgets.

#include "rag/cascade/cascade.hpp"

#include <algorithm>

namespace rag::cascade {
namespace {

std::string resolve_text(const index::Corpus& corpus, ChunkId id) {
    const Chunk* ch = corpus.chunk(id);
    return ch ? ch->indexed_text() : std::string{};
}

void truncate(std::vector<Hit>& v, std::size_t k) { if (v.size() > k) v.resize(k); }

} // namespace

Result<std::vector<Hit>>
Cascade::run(const index::Corpus& corpus, std::string_view query,
             std::vector<StageTrace>* trace) const {
    // ── Stage 0: hybrid retrieve ──────────────────────────────────────────────
    std::vector<Hit> cands;
    if (corpus.has_embedder()) {
        if (auto d = corpus.dense_search(query, cfg_.retrieve_k)) cands = std::move(*d);
    }
    if (cands.empty()) cands = corpus.lexical_search(query, cfg_.retrieve_k);
    if (trace) trace->push_back({"hybrid", cfg_.retrieve_k, cands.size()});
    if (cands.empty()) return cands;

    // ── Stage 1: ColBERT late interaction ─────────────────────────────────────
    if (cfg_.use_colbert && colbert_) {
        std::size_t budget = std::min(cands.size(), cfg_.colbert_k * 3);  // score a wider set
        std::vector<Hit> pool(cands.begin(), cands.begin() + budget);
        std::size_t in = pool.size();
        auto r = colbert_->rerank_hits(query, pool,
            [&](const Hit& h) { return resolve_text(corpus, h.chunk); });
        if (r) {                       // on failure, keep stage-0 order
            truncate(pool, cfg_.colbert_k);
            cands = std::move(pool);
        }
        if (trace) trace->push_back({"colbert", in, cands.size()});
    }

    // ── Stage 2: cross-encoder rerank ─────────────────────────────────────────
    if (cfg_.use_rerank && reranker_) {
        std::size_t budget = std::min(cands.size(), cfg_.rerank_k * 2);
        std::vector<Hit> pool(cands.begin(), cands.begin() + budget);
        std::size_t in = pool.size();
        std::vector<std::string> passages;
        passages.reserve(pool.size());
        for (const auto& h : pool) passages.push_back(resolve_text(corpus, h.chunk));
        auto scores = reranker_->rerank(query, passages);
        if (scores && scores->size() == pool.size()) {
            std::vector<std::pair<float, Hit>> scored;
            scored.reserve(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i) scored.emplace_back((*scores)[i], pool[i]);
            std::stable_sort(scored.begin(), scored.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            pool.clear();
            for (auto& [s, h] : scored) { Hit x = h; x.score = Score{s}; pool.push_back(x); }
            truncate(pool, cfg_.rerank_k);
            cands = std::move(pool);
        }
        if (trace) trace->push_back({"cross_encoder", in, cands.size()});
    }

    truncate(cands, cfg_.final_k);
    if (trace) trace->push_back({"final", cands.size(), cands.size()});
    return cands;
}

} // namespace rag::cascade
