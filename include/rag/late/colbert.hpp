#pragma once
// rag/late/colbert.hpp — ColBERT-style late interaction (MaxSim) reranking.
//
// Dense bi-encoders compress a whole passage into ONE vector — cheap, but they
// lose token-level detail. Cross-encoders score every query-passage pair jointly
// — accurate, but O(candidates) full forward passes. ColBERT (Khattab & Zaharia
// 2020) sits between: it embeds each token INDEPENDENTLY, then scores a pair by
//
//     MaxSim(Q, D) = Σ_{q ∈ Q}  max_{d ∈ D}  (E_q · E_d)
//
// i.e. every query token softly matches its best document token, and the
// matches sum. This "late interaction" keeps token granularity while letting
// document token embeddings be PRECOMPUTED — so reranking is fast dot products,
// not transformer passes.
//
// The token embeddings come from a `TokenEmbedder` seam (any model that maps a
// piece of text to a SEQUENCE of unit vectors — an ONNX ColBERT checkpoint, or
// for a dependency-free deterministic default, the hashed n-gram embedder here).
// The MaxSim scorer itself is exact and model-agnostic.

#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::late {

// A token embedding matrix: rows are unit-normalized per-token vectors.
using TokenMatrix = std::vector<Vector>;

// The token-embedder seam: text → sequence of unit vectors (one per token).
using TokenEmbedder = std::function<Result<TokenMatrix>(std::string_view)>;

// Exact MaxSim late-interaction score between a query and a document matrix.
// Both are sequences of unit vectors of equal dimension.
[[nodiscard]] float
maxsim(std::span<const Vector> query_tokens, std::span<const Vector> doc_tokens) noexcept;

// A deterministic, dependency-free TokenEmbedder: hashes character n-grams of
// each whitespace token into a fixed-dim unit vector. Not a trained ColBERT, but
// a faithful late-interaction substrate (identical tokens ⇒ identical vectors,
// so exact-match query tokens score 1.0) — good for tests, demos, and as a drop-
// in until a real checkpoint is wired through the seam.
[[nodiscard]] TokenEmbedder hashed_token_embedder(std::size_t dim = 64);

// The reranker: given a query and candidate passages, embeds tokens for each and
// returns MaxSim scores aligned to the input order. Precomputes the query matrix
// once. Models the pattern of rerank/reranker.hpp's scorers.
class ColbertReranker {
public:
    explicit ColbertReranker(TokenEmbedder embed) : embed_(std::move(embed)) {}

    // Score every passage; returns a score per passage (higher = better).
    [[nodiscard]] Result<std::vector<float>>
    rerank(std::string_view query, std::span<const std::string> passages) const;

    // Convenience: rerank a Hit list in place by MaxSim, given the passage text
    // resolver. Stable-sorts best-first.
    [[nodiscard]] Result<void>
    rerank_hits(std::string_view query, std::vector<Hit>& hits,
                const std::function<std::string(const Hit&)>& text_of) const;

private:
    TokenEmbedder embed_;
};

} // namespace rag::late
