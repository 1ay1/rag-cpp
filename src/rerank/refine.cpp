// rag/rerank/refine.cpp — near-duplicate dedup + relevance autocut stages.

#include "rag/rerank/refine.hpp"

#include <algorithm>
#include <unordered_set>

#include "rag/dense/simd.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::rerank {

// ─── Near-duplicate dedup ─────────────────────────────────────────────────────

std::vector<Hit>
dedup(const index::Corpus& corpus, std::span<const Hit> candidates, DedupConfig cfg) {
    const std::size_t n = candidates.size();
    if (n <= 1) return std::vector<Hit>(candidates.begin(), candidates.end());

    const std::size_t win = (cfg.window == 0) ? n : std::min(cfg.window, n);
    const float thresh = std::clamp(cfg.threshold, 0.0f, 1.0f);

    // Same graceful similarity as MMR: chunk-embedding cosine when the corpus is
    // dense, else lexical Jaccard over chunk terms.
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
        for (const auto& t : sm) if (bg.count(t)) ++inter;
        return static_cast<float>(inter) / static_cast<float>(x.size() + y.size() - inter);
    };

    // Keep the first (highest-ranked) member of each cluster. A candidate is
    // dropped only if it is a near-duplicate of an already-KEPT one — comparing
    // against kept (not all prior) is what makes the survivors mutually
    // distinct rather than merely non-adjacent.
    std::vector<char> dropped(n, 0);
    std::vector<std::size_t> kept;
    kept.reserve(win);
    for (std::size_t i = 0; i < win; ++i) {
        bool dup = false;
        for (std::size_t k : kept) {
            if (sim(i, k) >= thresh) { dup = true; break; }
        }
        if (dup) dropped[i] = 1;
        else kept.push_back(i);
    }

    std::vector<Hit> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (i >= win || !dropped[i]) out.push_back(candidates[i]);
    return out;
}

// ─── Relevance autocut ────────────────────────────────────────────────────────

std::vector<Hit>
autocut(std::span<const Hit> candidates, AutocutConfig cfg) {
    const std::size_t n = candidates.size();
    const std::size_t min_keep = std::max<std::size_t>(1, cfg.min_keep);
    if (n <= min_keep) return std::vector<Hit>(candidates.begin(), candidates.end());

    const std::size_t scan = (cfg.scan == 0) ? n : std::min(cfg.scan, n);

    // Consecutive score drops along the (relevance-sorted) curve. The first gap
    // that is markedly larger than the average gap is the knee: everything past
    // it is the low-relevance tail. Robust to scale because it compares gaps to
    // the mean gap, not to an absolute threshold.
    if (scan < 3) return std::vector<Hit>(candidates.begin(), candidates.end());

    double sum_gap = 0.0;
    std::size_t gaps = 0;
    for (std::size_t i = 1; i < scan; ++i) {
        const float g = candidates[i - 1].score.get() - candidates[i].score.get();
        if (g > 0.0f) { sum_gap += g; ++gaps; }
    }
    if (gaps == 0) return std::vector<Hit>(candidates.begin(), candidates.end()); // flat: keep all
    const double mean_gap = sum_gap / static_cast<double>(gaps);
    const double trigger  = mean_gap * static_cast<double>(std::max(0.0f, cfg.sensitivity));

    std::size_t cut = n;  // default: no cut
    for (std::size_t i = min_keep; i < scan; ++i) {
        const float g = candidates[i - 1].score.get() - candidates[i].score.get();
        if (static_cast<double>(g) >= trigger) { cut = i; break; }
    }
    if (cut >= n) return std::vector<Hit>(candidates.begin(), candidates.end());
    return std::vector<Hit>(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(cut));
}

// ─── Pipeline stages ──────────────────────────────────────────────────────────
namespace {

class DedupStage final : public pipeline::RetrievalStage {
public:
    DedupStage(DedupConfig cfg, std::string label) : cfg_(cfg), label_(std::move(label)) {}
    std::string_view name() const noexcept override { return label_; }
    Result<pipeline::Context> process(pipeline::Context ctx) const override {
        if (!ctx.corpus || ctx.candidates.size() <= 1) return ctx;
        const std::size_t before = ctx.candidates.size();
        ctx.candidates = dedup(*ctx.corpus, ctx.candidates, cfg_);
        ctx.trace.push_back("dedup " + std::to_string(before) + " -> " +
                            std::to_string(ctx.candidates.size()));
        return ctx;
    }
private:
    DedupConfig cfg_;
    std::string label_;
};

class AutocutStage final : public pipeline::RetrievalStage {
public:
    AutocutStage(AutocutConfig cfg, std::string label) : cfg_(cfg), label_(std::move(label)) {}
    std::string_view name() const noexcept override { return label_; }
    Result<pipeline::Context> process(pipeline::Context ctx) const override {
        if (ctx.candidates.size() <= 1) return ctx;
        const std::size_t before = ctx.candidates.size();
        // Never cut below what the caller actually asked for: autocut trims the
        // low-relevance TAIL past k, it must not starve the requested result
        // set. A noisy/weak retriever can otherwise produce a sharp early knee
        // inside the top-k and collapse recall (seen on a weak-embedder hybrid
        // run). Floor min_keep at ctx.k so the cut only ever removes surplus.
        AutocutConfig cfg = cfg_;
        cfg.min_keep = std::max(cfg.min_keep, ctx.k);
        ctx.candidates = autocut(ctx.candidates, cfg);
        ctx.trace.push_back("autocut " + std::to_string(before) + " -> " +
                            std::to_string(ctx.candidates.size()));
        return ctx;
    }
private:
    AutocutConfig cfg_;
    std::string   label_;
};

} // namespace

pipeline::StagePtr make_dedup_stage(DedupConfig cfg, std::string label) {
    return std::make_shared<DedupStage>(cfg, std::move(label));
}

pipeline::StagePtr make_autocut_stage(AutocutConfig cfg, std::string label) {
    return std::make_shared<AutocutStage>(cfg, std::move(label));
}

} // namespace rag::rerank
