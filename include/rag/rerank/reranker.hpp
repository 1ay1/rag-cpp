#pragma once
// rag/rerank/reranker.hpp — cross-encoder reranking as a first-class stage.
//
// A retriever answers "which chunks are plausibly relevant?" cheaply. A
// cross-encoder answers "how relevant is THIS chunk to THIS query?" precisely,
// by jointly encoding (query, passage) — the accuracy ceiling of the funnel.
// It is expensive, so it only ever sees the top-N candidates the retriever
// already narrowed to.
//
// This module ships:
//   • CrossEncoderReranker — HTTP to a reranker server. Supports the two
//     dominant wire formats: HuggingFace TEI `/rerank` and Cohere-style
//     `/v1/rerank` (also what Jina / Voyage / vLLM-rerank expose). Bearer-auth
//     + TLS via the same injected HttpTransport as embedders.
//   • ScoreFnReranker — lift any local scoring function (an in-process ONNX
//     model, a heuristic) into a reranker with zero network.
//   • make_rerank_stage() — adapt a reranker into a pipeline RerankStage.
//
// All rerankers model the `Reranker` concept (below) and are exposed via the
// type-erased AnyReranker for the runtime pipeline path.

#include <chrono>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/cache/cache.hpp"           // RerankCache for the caching decorator
#include "rag/dense/embedder.hpp"     // HttpTransport seam
#include "rag/dense/local_embedder.hpp" // LocalEmbedderConfig (reused for paths)
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"

namespace rag::rerank {

// A (passage-index, relevance) pair the reranker returns.
struct Scored {
    std::size_t index;   // position in the input passage list
    float       score;   // relevance; higher is better
};

// The Reranker concept: score a query against N passages, return per-passage
// relevance in input order (or as {index,score}).
template <class R>
concept Reranker = requires(const R& r, std::string_view q, std::span<const std::string> ps) {
    { r.rerank(q, ps) } -> std::same_as<Result<std::vector<float>>>;
};

// ─── Cross-encoder over HTTP ──────────────────────────────────────────────────
struct CrossEncoderConfig {
    enum class Wire { tei, cohere };            // /rerank vs /v1/rerank shapes
    std::string   host    = "127.0.0.1";
    std::uint16_t port    = 8080;
    bool          tls      = false;
    std::string   path    = "/rerank";
    std::string   model;                        // required for cohere-style
    std::string   api_key;                      // Bearer (cohere/jina/voyage)
    Wire          wire     = Wire::tei;
    std::chrono::milliseconds timeout{30'000};

    static CrossEncoderConfig tei(std::string host, std::uint16_t port);
    static CrossEncoderConfig cohere(std::string key, std::string model = "rerank-english-v3.0");
    static CrossEncoderConfig jina(std::string key, std::string model = "jina-reranker-v2-base-multilingual");
};

class CrossEncoderReranker {
public:
    explicit CrossEncoderReranker(CrossEncoderConfig cfg,
                                  std::shared_ptr<dense::HttpTransport> tp = dense::default_http_transport())
        : cfg_(std::move(cfg)), tp_(std::move(tp)) {}
    [[nodiscard]] Result<std::vector<float>>
    rerank(std::string_view query, std::span<const std::string> passages) const;
private:
    CrossEncoderConfig cfg_;
    std::shared_ptr<dense::HttpTransport> tp_;
};

// ─── Local scoring-function reranker (no network; wrap an in-process model) ────
class ScoreFnReranker {
public:
    using Fn = std::function<float(std::string_view query, std::string_view passage)>;
    explicit ScoreFnReranker(Fn fn) : fn_(std::move(fn)) {}
    [[nodiscard]] Result<std::vector<float>>
    rerank(std::string_view query, std::span<const std::string> passages) const {
        std::vector<float> out; out.reserve(passages.size());
        for (const auto& p : passages) out.push_back(fn_(query, p));
        return out;
    }
private:
    Fn fn_;
};

// ─── In-process ONNX cross-encoder (no network) ───────────────────────────
// The reranking counterpart to OnnxEmbedder: loads a HuggingFace sequence-
// classification cross-encoder exported to ONNX (bge-reranker, ms-marco
// MiniLM cross-encoders, mono-BERT) and scores (query, passage) pairs IN
// PROCESS — no server, no network — which is what most deployments want. It
// jointly encodes the pair as `[CLS] query [SEP] passage [SEP]` and reads the
// model's single relevance logit, the accuracy ceiling of the retrieval funnel.
//
// Gated behind RAGCPP_WITH_ONNX exactly like OnnxEmbedder: without the flag the
// class is still DECLARED (callers compile) but load() returns Errc::unavailable
// — the same graceful-degradation contract. Reuses LocalEmbedderConfig for the
// model_path / tokenizer_path / max_tokens / threads fields.
class OnnxReranker {
public:
    // Load a cross-encoder. Fails with Errc::unavailable if built without ONNX,
    // or a typed error if the model/tokenizer file is missing or malformed.
    [[nodiscard]] static Result<OnnxReranker> load(dense::LocalEmbedderConfig cfg);

    // Score `query` against each passage; returns one relevance per passage in
    // input order. Higher is more relevant (raw logit; monotonic, un-normalized).
    [[nodiscard]] Result<std::vector<float>>
    rerank(std::string_view query, std::span<const std::string> passages) const;

    [[nodiscard]] std::string_view identity() const;

    // Compile-time capability flag (mirrors OnnxEmbedder::available()).
    [[nodiscard]] static constexpr bool available() noexcept {
#ifdef RAGCPP_WITH_ONNX
        return true;
#else
        return false;
#endif
    }

    OnnxReranker(OnnxReranker&&) noexcept;
    OnnxReranker& operator=(OnnxReranker&&) noexcept;
    ~OnnxReranker();

private:
    OnnxReranker();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ─── Caching decorator ────────────────────────────────────────────────────────
// Wraps any Reranker and memoizes (query, passage) → score in a shared
// RerankCache. A cross-encoder pass is the funnel's most expensive op and it
// repeats constantly (interactive re-queries, overlapping retrieval windows);
// this makes every repeat free. On a call it splits passages into cache hits
// and misses, invokes the wrapped reranker on ONLY the misses (so a partial
// overlap still shrinks the batch), then backfills the cache. Models the
// Reranker concept itself, so it composes with AnyReranker and make_rerank_stage.
//
// The cache is shared (shared_ptr) so multiple stages/threads reuse one table;
// `identity` pins entries to a specific model so a swap never returns a stale
// logit. Pass a distinct identity per underlying reranker.
template <Reranker R>
class CachingReranker {
public:
    CachingReranker(R inner, std::shared_ptr<cache::RerankCache> cache, std::string identity)
        : inner_(std::move(inner)), cache_(std::move(cache)), identity_(std::move(identity)) {
        if (!cache_) cache_ = std::make_shared<cache::RerankCache>();
    }

    [[nodiscard]] Result<std::vector<float>>
    rerank(std::string_view query, std::span<const std::string> passages) const {
        std::vector<float>       out(passages.size(), 0.0f);
        std::vector<std::string> miss;              // passages not in cache
        std::vector<std::size_t> miss_idx;          // their positions in `out`
        miss.reserve(passages.size());
        miss_idx.reserve(passages.size());

        for (std::size_t i = 0; i < passages.size(); ++i) {
            if (auto hit = cache_->get(identity_, query, passages[i])) {
                out[i] = *hit;
            } else {
                miss.push_back(passages[i]);
                miss_idx.push_back(i);
            }
        }

        if (!miss.empty()) {
            auto scored = inner_.rerank(query, std::span<const std::string>(miss));
            if (!scored) return std::unexpected(scored.error());
            if (scored->size() != miss.size())
                return fail<std::vector<float>>(Errc::invalid_argument, "reranker returned wrong count");
            for (std::size_t j = 0; j < miss.size(); ++j) {
                out[miss_idx[j]] = (*scored)[j];
                cache_->put(identity_, query, miss[j], (*scored)[j]);
            }
        }
        return out;
    }

    [[nodiscard]] const cache::RerankCache& cache() const { return *cache_; }

private:
    R                                   inner_;
    std::shared_ptr<cache::RerankCache> cache_;
    std::string                         identity_;
};

// Deduce-and-wrap helper: cached(myReranker, cache, "bge-reranker-v2").
template <Reranker R>
[[nodiscard]] CachingReranker<R>
cached(R inner, std::shared_ptr<cache::RerankCache> cache, std::string identity) {
    return CachingReranker<R>(std::move(inner), std::move(cache), std::move(identity));
}

// ─── Type-erased reranker ─────────────────────────────────────────────────────
class AnyReranker {
public:
    template <Reranker R>
    explicit AnyReranker(R r) : self_(std::make_shared<Model<R>>(std::move(r))) {}
    [[nodiscard]] Result<std::vector<float>>
    rerank(std::string_view q, std::span<const std::string> ps) const { return self_->rerank(q, ps); }
private:
    struct Concept {
        virtual ~Concept() = default;
        virtual Result<std::vector<float>> rerank(std::string_view, std::span<const std::string>) const = 0;
    };
    template <Reranker R> struct Model final : Concept {
        R r; explicit Model(R x) : r(std::move(x)) {}
        Result<std::vector<float>> rerank(std::string_view q, std::span<const std::string> ps) const override {
            return r.rerank(q, ps);
        }
    };
    std::shared_ptr<const Concept> self_;
};

// ─── Adapt a reranker into a pipeline stage ───────────────────────────────────
// Reranks the top `top_n` candidates (the rest keep their fused order below the
// reranked block). `blend` in [0,1] mixes cross-encoder score with the incoming
// fused score (both min-max normalized over the reranked block): 1.0 = pure
// cross-encoder, 0.0 = ignore it. Graceful: if the reranker is unavailable,
// candidates pass through untouched.
//
// The default is 0.5, NOT 1.0, and that is a measured choice: trusting an
// out-of-domain cross-encoder outright (blend=1.0) cost 0.049 nDCG@10 on SciFact
// (0.7355 -> 0.6859), while a 50/50 blend recovered it to 0.7280 and improved
// R@10. A reranker should sharpen a good retriever, not overrule it. Pass 1.0
// explicitly only when the cross-encoder is known to be in-domain and stronger
// than the retriever it is reordering. See BENCHMARKS.md "Reranking, measured".
[[nodiscard]] pipeline::StagePtr
make_rerank_stage(AnyReranker reranker, std::size_t top_n = 50, float blend = 0.5f,
                  std::string label = "cross_encoder");

// Same, but memoizes (query, passage) → score through a shared RerankCache so a
// repeated or overlapping query never re-runs the cross-encoder forward pass.
// `identity` pins cache entries to this specific model (pass the reranker's
// identity() or any stable name); a distinct identity per model prevents a swap
// from returning a stale logit. The cache is shared — hand the same shared_ptr
// to multiple stages/engines to pool it. Everything else matches the overload
// above; the cache is purely additive and transparent.
[[nodiscard]] pipeline::StagePtr
make_cached_rerank_stage(AnyReranker reranker, std::shared_ptr<cache::RerankCache> cache,
                         std::string identity, std::size_t top_n = 50, float blend = 0.5f,
                         std::string label = "cross_encoder");

static_assert(Reranker<CrossEncoderReranker>);
static_assert(Reranker<ScoreFnReranker>);
static_assert(Reranker<OnnxReranker>);

} // namespace rag::rerank
