// rag/rerank/mmr.cpp — Maximal Marginal Relevance diversity reranking.

#include "rag/rerank/mmr.hpp"

#include <algorithm>
#include <unordered_set>

#include "rag/dense/simd.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::rerank {
namespace {

// Pairwise similarity between two chunks: cosine of embeddings if both present,
// else lexical Jaccard over their tokens.
struct SimCtx {
    const index::Corpus* corpus;
    text::Tokenizer      tok;
    std::vector<std::unordered_set<std::string>> bags;  // lazily filled per hit
};

} // namespace

std::vector<Hit>
mmr(const index::Corpus& corpus, std::span<const Hit> candidates, MmrConfig cfg) {
    const std::size_t n = candidates.size();
    if (n == 0) return {};
    float lambda = std::clamp(cfg.lambda, 0.0f, 1.0f);
    std::size_t k = std::min(cfg.k == 0 ? n : cfg.k, n);

    // Normalize incoming relevance to [0,1] so it's commensurate with cosine.
    float lo = candidates[0].score.get(), hi = lo;
    for (const auto& h : candidates) { lo = std::min(lo, h.score.get()); hi = std::max(hi, h.score.get()); }
    float span = hi - lo > 1e-9f ? hi - lo : 1.0f;
    std::vector<float> rel(n);
    for (std::size_t i = 0; i < n; ++i) rel[i] = (candidates[i].score.get() - lo) / span;

    // Precompute embeddings / term bags for similarity.
    text::Tokenizer tok = corpus.tokenizer();
    std::vector<const Vector*> emb(n, nullptr);
    std::vector<std::unordered_set<std::string>> bags(n);
    bool dense = true;
    for (std::size_t i = 0; i < n; ++i) {
        const Chunk* ch = corpus.chunk(candidates[i].chunk);
        if (ch && !ch->embedding.empty()) emb[i] = &ch->embedding;
        else dense = false;
    }
    if (!dense)
        for (std::size_t i = 0; i < n; ++i) {
            const Chunk* ch = corpus.chunk(candidates[i].chunk);
            if (ch) { auto t = tok.tokenize(ch->text); bags[i].insert(t.begin(), t.end()); }
        }

    auto sim = [&](std::size_t a, std::size_t b) -> float {
        if (dense && emb[a] && emb[b] && emb[a]->size() == emb[b]->size())
            return dense::dot(*emb[a], *emb[b]);
        const auto& x = bags[a]; const auto& y = bags[b];
        if (x.empty() || y.empty()) return 0.0f;
        const auto& sm = x.size() < y.size() ? x : y;
        const auto& bg = x.size() < y.size() ? y : x;
        std::size_t inter = 0;
        for (auto& t : sm) if (bg.count(t)) ++inter;
        return (float)inter / (float)(x.size() + y.size() - inter);
    };

    std::vector<char> chosen(n, 0);
    std::vector<std::size_t> order;
    order.reserve(k);
    // Greedy MMR selection.
    for (std::size_t step = 0; step < k; ++step) {
        std::size_t best = n;
        float best_score = -1e30f;
        for (std::size_t i = 0; i < n; ++i) {
            if (chosen[i]) continue;
            float max_sim = 0.0f;
            for (std::size_t s : order) max_sim = std::max(max_sim, sim(i, s));
            float score = lambda * rel[i] - (1.0f - lambda) * max_sim;
            if (score > best_score) { best_score = score; best = i; }
        }
        if (best == n) break;
        chosen[best] = 1;
        order.push_back(best);
    }

    std::vector<Hit> out;
    out.reserve(order.size());
    for (std::size_t i : order) out.push_back(candidates[i]);
    return out;
}

namespace {

// MMR cannot be a RerankStage, and the reason is subtle enough that it shipped
// broken: RerankStage RE-SORTS the candidates by score immediately after its
// callback returns.
//
//     if (auto r = fn_(...); !r) return ...;
//     std::sort(candidates, by descending score);      // <-- destroys MMR
//
// That is correct for a reranker, whose whole output IS the new scores. But
// MMR does not rescore anything — it reorders, and its result is carried
// entirely by the ORDER of the returned vector. Sorting by score afterwards
// throws that away and restores the pure-relevance ranking, so the stage ran,
// the trace showed it running, and the output was identical to no MMR at all
// (measured: 2 of 4 distinct facets in the top 8, exactly the same as
// standard()).
//
// A first-class stage owns its own Context and is not post-processed, so the
// order it produces is the order that survives. It also gets access to ctx.k,
// which the RerankStage callback signature does not carry — letting MMR select
// the k slots that will actually be returned rather than permuting the whole
// candidate pool.
class MmrStage final : public pipeline::RetrievalStage {
public:
    MmrStage(float lambda, std::string label)
        : lambda_(lambda), label_(std::move(label)) {}

    std::string_view name() const noexcept override { return label_; }

    Result<pipeline::Context> process(pipeline::Context ctx) const override {
        if (!ctx.corpus || ctx.candidates.empty()) return ctx;
        MmrConfig cfg;
        cfg.lambda = lambda_;
        // Select exactly the slots that will survive. A pool wider than k is
        // what gives MMR alternatives to swap in.
        cfg.k = ctx.k ? ctx.k : ctx.candidates.size();
        ctx.candidates = mmr(*ctx.corpus, ctx.candidates, cfg);
        return ctx;
    }

private:
    float       lambda_;
    std::string label_;
};

} // namespace

pipeline::StagePtr
make_mmr_stage(float lambda, std::string label) {
    return std::make_shared<MmrStage>(lambda, std::move(label));
}

} // namespace rag::rerank
