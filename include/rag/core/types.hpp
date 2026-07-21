#pragma once
// rag/core/types.hpp — the type-theoretic foundation.
//
// Design creed: make illegal states unrepresentable, and make every domain
// quantity a DISTINCT type so the compiler enforces the algebra. We lean on
// four constructs:
//
//   1. Strong types (the "newtype" pattern): a DocId is not an int, a Score
//      is not a float. A function that wants a ChunkId cannot be handed a
//      DocId by accident — even though both wrap uint32.
//
//   2. Phantom tags: StrongId<T, Tag> shares one implementation but the Tag
//      makes DocId and ChunkId incompatible at compile time (nominal, not
//      structural, typing — exactly what a type theorist wants).
//
//   3. Sum types (std::variant) for closed alternatives, product types
//      (struct) for records — the algebraic data types of the domain.
//
//   4. Result<T> = std::expected<T, Error>: total functions. Fallible
//      operations return a value OR a typed error, never throw for control
//      flow, and compose monadically via and_then / transform.

#include <cstdint>
#include <compare>
#include <concepts>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rag {

// ─────────────────────────────────────────────────────────────────────────────
// Strong scalar (the newtype pattern).
//
// Wraps a scalar `Rep` under a phantom `Tag`, giving a nominally-distinct type
// with value semantics, defaulted three-way comparison, and explicit access to
// the underlying representation via get(). No implicit conversions in or out —
// that is the whole point.
// ─────────────────────────────────────────────────────────────────────────────
template <class Rep, class Tag>
struct Strong {
    Rep value{};

    constexpr Strong() = default;
    constexpr explicit Strong(Rep v) noexcept : value(v) {}

    [[nodiscard]] constexpr Rep  get()  const noexcept { return value; }
    [[nodiscard]] constexpr explicit operator Rep() const noexcept { return value; }

    friend constexpr auto operator<=>(const Strong&, const Strong&) = default;
    friend constexpr bool operator==(const Strong&, const Strong&) = default;
};

// Phantom-tagged unsigned identifier. Nominal typing: DocId and ChunkId are
// different types the compiler will not silently interconvert.
template <class Tag>
struct StrongId : Strong<std::uint32_t, Tag> {
    using Strong<std::uint32_t, Tag>::Strong;
    static constexpr StrongId invalid() noexcept { return StrongId{UINT32_MAX}; }
    [[nodiscard]] constexpr bool valid() const noexcept { return this->value != UINT32_MAX; }
};

struct DocIdTag {};
struct ChunkIdTag {};
struct TermIdTag {};

using DocId   = StrongId<DocIdTag>;    // identifies a source document
using ChunkId = StrongId<ChunkIdTag>;  // identifies a chunk within the corpus
using TermId  = StrongId<TermIdTag>;   // interned vocabulary term

// ─────────────────────────────────────────────────────────────────────────────
// Domain scalars — distinct float newtypes so a relevance Score can never be
// confused with a raw cosine Similarity or a fused rank contribution.
// ─────────────────────────────────────────────────────────────────────────────
struct ScoreTag {};
struct SimilarityTag {};

using Score      = Strong<float, ScoreTag>;       // an opaque ranking score
using Similarity = Strong<float, SimilarityTag>;  // cosine ∈ [-1, 1]

// ─────────────────────────────────────────────────────────────────────────────
// Error — a closed sum type of failure kinds plus a human message.
// Total functions return Result<T>; they never throw for expected failure.
// ─────────────────────────────────────────────────────────────────────────────
enum class Errc {
    ok = 0,
    not_found,
    invalid_argument,
    dimension_mismatch,
    io_error,
    parse_error,
    transport_error,
    unavailable,        // an optional backend (e.g. embedder) is offline
    empty_corpus,
    already_exists,
    corrupt_index,
};

[[nodiscard]] constexpr std::string_view to_string(Errc e) noexcept {
    switch (e) {
        case Errc::ok:                 return "ok";
        case Errc::not_found:          return "not_found";
        case Errc::invalid_argument:   return "invalid_argument";
        case Errc::dimension_mismatch: return "dimension_mismatch";
        case Errc::io_error:           return "io_error";
        case Errc::parse_error:        return "parse_error";
        case Errc::transport_error:    return "transport_error";
        case Errc::unavailable:        return "unavailable";
        case Errc::empty_corpus:       return "empty_corpus";
        case Errc::already_exists:     return "already_exists";
        case Errc::corrupt_index:      return "corrupt_index";
    }
    return "unknown";
}

struct Error {
    Errc        code = Errc::ok;
    std::string message;

    Error() = default;
    Error(Errc c, std::string msg) : code(c), message(std::move(msg)) {}
    explicit Error(Errc c) : code(c), message(std::string(to_string(c))) {}
};

// Result<T> = value | Error. The library's total-function return type.
template <class T>
using Result = std::expected<T, Error>;

// Convenience: construct a failed Result<T>.
template <class T>
[[nodiscard]] Result<T> fail(Errc c, std::string msg = {}) {
    return std::unexpected(Error(c, msg.empty() ? std::string(to_string(c)) : std::move(msg)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector — a fixed-dimension dense embedding. Value type; unit-normalization
// and dimension are part of its contract, checked at the seams.
// ─────────────────────────────────────────────────────────────────────────────
using Vector = std::vector<float>;

// A read-only view over an embedding for zero-copy scoring.
using VectorView = std::span<const float>;

// ─────────────────────────────────────────────────────────────────────────────
// Hit / ScoredChunk — the product types the retrieval pipeline speaks in.
// ─────────────────────────────────────────────────────────────────────────────
struct Hit {
    ChunkId chunk = ChunkId::invalid();
    Score   score{0.0f};

    friend constexpr bool operator==(const Hit&, const Hit&) = default;
};

} // namespace rag
