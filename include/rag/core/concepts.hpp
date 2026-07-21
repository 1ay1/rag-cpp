#pragma once
// rag/core/concepts.hpp — the structural interfaces of the framework.
//
// Where types.hpp gives nominal typing (StrongId), this file gives STRUCTURAL
// typing via C++20 concepts: any type whose shape satisfies `Embedder` IS an
// embedder — no inheritance, no vtable on the hot path. Concepts are the
// framework's extension points; a user plugs in a new backend by writing a
// struct that models the concept, and the templates accept it for free.
//
// For the cases that need runtime polymorphism (a pipeline of heterogeneous
// stages chosen at run time), we ALSO expose type-erased abstract bases in the
// relevant headers. Concepts constrain the generic path; abstract bases carry
// the dynamic path. The two are bridged by thin `AnyX` adapters.

#include <concepts>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"

namespace rag {

// ─────────────────────────────────────────────────────────────────────────────
// Embedder — turns text into a dense vector. May fail (backend offline), so it
// is a total function returning Result. Batched form is the primitive; single
// is derived.
// ─────────────────────────────────────────────────────────────────────────────
template <class E>
concept Embedder = requires(const E& e, std::span<const std::string> texts) {
    { e.dimension() }        -> std::convertible_to<std::size_t>;
    { e.embed(texts) }       -> std::same_as<Result<std::vector<Vector>>>;
    { e.identity() }         -> std::convertible_to<std::string_view>; // model id, for cache keying
};

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer — text → normalized terms. Pure, total, deterministic.
// ─────────────────────────────────────────────────────────────────────────────
template <class T>
concept Tokenizer = requires(const T& t, std::string_view text) {
    { t.tokenize(text) } -> std::convertible_to<std::vector<std::string>>;
};

// ─────────────────────────────────────────────────────────────────────────────
// Retriever — searches SOME index and returns ranked Hits. The unit that
// fusion combines. Total (returns Result) because a dense retriever can fail
// when its embedder is offline.
// ─────────────────────────────────────────────────────────────────────────────
template <class R>
concept Retriever = requires(const R& r, std::string_view query, std::size_t k) {
    { r.retrieve(query, k) } -> std::same_as<Result<std::vector<Hit>>>;
};

// ─────────────────────────────────────────────────────────────────────────────
// Ranker — reorders a candidate set (cross-encoder, feature reranker, learned
// prior). Distinct from Retriever: it refines, it does not fetch.
// ─────────────────────────────────────────────────────────────────────────────
template <class R>
concept Ranker = requires(const R& r, std::string_view query, std::vector<Hit>& cands) {
    { r.rerank(query, cands) } -> std::same_as<Result<void>>;
};

} // namespace rag
