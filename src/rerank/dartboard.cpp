// rag/rerank/dartboard.cpp — greedy relevant-information-gain selection.

#include "rag/rerank/dartboard.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "rag/dense/simd.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::rerank {

std::vector<Hit>
dartboard(const index::Corpus& corpus, std::span<const Hit> candidates, DartboardConfig cfg) {
    const std::size_t n = candidates.size();
    if (n == 0) return {};
    const std::size_t k = std::min(cfg.k == 0 ? n : cfg.k, n);
    const float alpha = std::clamp(cfg.relevance_weight, 0.0f, 1.0f);
    const float sigma = cfg.sigma > 1e-6f ? cfg.sigma : 1e-6f;

    // Normalise relevance to [0,1] so sigma means the same thing regardless of
    // whether scores arrived from BM25, cosine, or a fusion of both.
    float lo = candidates[0].score.get(), hi = lo;
    for (const auto& h : candidates) {
        lo = std::min(lo, h.score.get());
        hi = std::max(hi, h.score.get());
    }
    const float span = (hi - lo) > 1e-9f ? (hi - lo) : 1.0f;
    std::vector<float> rel(n);
    for (std::size_t i = 0; i < n; ++i) rel[i] = (candidates[i].score.get() - lo) / span;

    // Intent weights: softmax over relevance. This is what makes the objective
    // "relevant information gain" rather than plain coverage — covering an
    // intent nobody asked for is worth nothing, so each target is weighted by
    // how plausible it is as the user's actual need.
    std::vector<float> w(n);
    float wsum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { w[i] = std::exp(rel[i] / sigma); wsum += w[i]; }
    if (wsum > 0.0f) for (auto& x : w) x /= wsum;

    // Same similarity basis as MMR: cosine when every candidate is embedded,
    // lexical Jaccard otherwise. Kept identical on purpose so an A/B between the
    // two policies measures the OBJECTIVE, not two different notions of "alike".
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

    auto raw_sim = [&](std::size_t a, std::size_t b) -> float {
        if (a == b) return 1.0f;
        if (dense && emb[a] && emb[b] && emb[a]->size() == emb[b]->size())
            return dense::dot(*emb[a], *emb[b]);
        const auto& x = bags[a]; const auto& y = bags[b];
        if (x.empty() || y.empty()) return 0.0f;
        const auto& sm = x.size() < y.size() ? x : y;
        const auto& bg = x.size() < y.size() ? y : x;
        std::size_t inter = 0;
        for (auto& t : sm) if (bg.count(t)) ++inter;
        return static_cast<float>(inter) / static_cast<float>(x.size() + y.size() - inter);
    };

    // Precompute the full n x n similarity matrix ONCE.
    //
    // The gain of adding d is a sum over every intent i, so the naive greedy
    // form evaluates sim(d,i) for all (d,i) on every one of the k steps:
    // O(k n^2) similarity calls, and sim() is a Jaccard set intersection when
    // the corpus is not dense. MEASURED on BEIR/SciFact before this change:
    //
    //     pool  40 (k=10):    24.68 ms/query
    //     pool 100 (k=25):   369.09 ms/query
    //     pool 200 (k=50):  2803.39 ms/query      <- 2.8 SECONDS
    //
    // The matrix is symmetric and never changes during selection, so computing
    // it once costs n^2/2 sims and reduces every step to array lookups: total
    // O(n^2 + k n). This is the same trap MMR fell into (see mmr.cpp); the
    // shape differs because the objective sums over intents rather than taking
    // a max over the chosen set, so caching a running max is not enough here.
    std::vector<float> S(n * n, 0.0f);
    for (std::size_t a = 0; a < n; ++a) {
        S[a * n + a] = 1.0f;
        for (std::size_t b = a + 1; b < n; ++b) {
            const float s = raw_sim(a, b);
            S[a * n + b] = s;
            S[b * n + a] = s;
        }
    }
    auto sim = [&](std::size_t a, std::size_t b) -> float { return S[a * n + b]; };

    // best[i] = how well the CURRENT selection covers intent i. Adding a
    // document can only raise it, so the gain of adding d is
    //     Σ_i w_i · max(0, sim(d,i) − best[i])
    std::vector<float> best(n, 0.0f);
    std::vector<char>  chosen(n, 0);
    std::vector<std::size_t> order;
    order.reserve(k);

    for (std::size_t step = 0; step < k; ++step) {
        std::size_t pick = n;
        float pick_score = -1e30f;
        for (std::size_t d = 0; d < n; ++d) {
            if (chosen[d]) continue;
            float gain = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                const float s = sim(d, i);
                if (s > best[i]) gain += w[i] * (s - best[i]);
            }
            // Blend the document's own relevance with what it adds to coverage.
            // At alpha=1 this degenerates to relevance order, which is the
            // correct behaviour for a caller who wants no diversification.
            const float score = alpha * rel[d] + (1.0f - alpha) * gain;
            if (score > pick_score) { pick_score = score; pick = d; }
        }
        if (pick == n) break;
        chosen[pick] = 1;
        order.push_back(pick);
        for (std::size_t i = 0; i < n; ++i) best[i] = std::max(best[i], sim(pick, i));
    }

    std::vector<Hit> out;
    out.reserve(order.size());
    for (std::size_t i : order) out.push_back(candidates[i]);
    return out;
}

namespace {

// Reorders rather than rescores, so — exactly like MmrStage — it is a plain
// Stage and NOT a RerankStage: RerankStage re-sorts by score after its callback
// returns, which would discard the selection this computed.
class DartboardStage final : public pipeline::RetrievalStage {
public:
    DartboardStage(DartboardConfig cfg, std::string label)
        : cfg_(cfg), label_(std::move(label)) {}

    std::string_view name() const noexcept override { return label_; }

    Result<pipeline::Context> process(pipeline::Context ctx) const override {
        if (!ctx.corpus || ctx.candidates.empty()) return ctx;
        DartboardConfig cfg = cfg_;
        cfg.k = ctx.k ? ctx.k : ctx.candidates.size();
        ctx.candidates = dartboard(*ctx.corpus, ctx.candidates, cfg);
        return ctx;
    }

private:
    DartboardConfig cfg_;
    std::string     label_;
};

} // namespace

pipeline::StagePtr
make_dartboard_stage(DartboardConfig cfg, std::string label) {
    return std::make_shared<DartboardStage>(cfg, std::move(label));
}

} // namespace rag::rerank
