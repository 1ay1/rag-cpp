#pragma once
// rag/dense/simd.hpp — vectorized distance kernels with runtime dispatch.
//
// All embeddings the library scores are UNIT-NORMALIZED, so cosine similarity
// is a plain dot product. We provide dot, l2-norm normalization, and Hamming
// distance over packed sign bits (for binary-quantized ANN prefiltering).
//
// Runtime dispatch: on x86 we detect AVX2 once; on ARM we always have NEON.
// The scalar path is always correct and is the fallback.

#include <cstdint>
#include <span>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::dense {

// Dot product of two equal-length float spans.
[[nodiscard]] float dot(std::span<const float> a, std::span<const float> b) noexcept;

// In-place L2 normalization; no-op on a zero vector. The span overload lets
// callers normalize a slice of a larger arena (e.g. one row of a flat vector
// store) without owning a std::vector.
void normalize(std::span<float> v) noexcept;
inline void normalize(std::vector<float>& v) noexcept { normalize(std::span<float>(v)); }

// Cosine similarity of two vectors that need NOT be normalized.
[[nodiscard]] float cosine(std::span<const float> a, std::span<const float> b) noexcept;

// Pack the sign bits of `v` into 64-bit words (bit i set iff v[i] >= 0). Used
// by binary-quantized HNSW: the walk compares packed codes with popcount.
[[nodiscard]] std::vector<std::uint64_t> pack_signs(std::span<const float> v);

// Hamming distance between two equal-length packed sign codes.
[[nodiscard]] std::uint32_t hamming(std::span<const std::uint64_t> a,
                                    std::span<const std::uint64_t> b) noexcept;

// Which SIMD tier is active (for diagnostics/bench reporting).
[[nodiscard]] const char* simd_tier() noexcept;

} // namespace rag::dense
