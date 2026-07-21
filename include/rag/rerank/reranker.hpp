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
#include "rag/dense/embedder.hpp"     // HttpTransport seam
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
// fused score: 1.0 = pure cross-encoder, 0.0 = ignore it. Graceful: if the
// reranker is unavailable, candidates pass through untouched.
[[nodiscard]] pipeline::StagePtr
make_rerank_stage(AnyReranker reranker, std::size_t top_n = 50, float blend = 1.0f,
                  std::string label = "cross_encoder");

static_assert(Reranker<CrossEncoderReranker>);
static_assert(Reranker<ScoreFnReranker>);

} // namespace rag::rerank
