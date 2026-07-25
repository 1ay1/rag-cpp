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

// ──────────────────────────────────────────────────────────────────────
// SQ8 — symmetric int8 scalar quantization.
//
// An ANN graph walk on float32 vectors is MEMORY-bound, not compute-bound:
// at 256 dims each vector is 1 KiB, the access pattern is random, and once the
// corpus exceeds cache the walk spends its time waiting on DRAM rather than
// doing arithmetic (measured: 46 ns/distance at dim=32 rising to 236 ns/distance
// at dim=512, on identical graph shapes).
//
// Quantizing to int8 cuts the bytes moved by 4×, which cuts the wait by ~4×.
// The vectors here are unit-normalized, so every component is in [-1,1] and a
// single global scale suffices — no per-vector scale to load and no dequantize
// step in the inner loop, because a dot product of two symmetrically-quantized
// vectors is just the integer dot times a constant.
//
// Accuracy is preserved by only ever using SQ8 to ORDER candidates during the
// walk; the final top-k is always rescored against the exact float vectors.

// Quantize unit-normalized `v` to int8 with scale 127 (component c -> c*127).
void quantize_sq8(std::span<const float> v, std::span<std::int8_t> out) noexcept;

// Integer dot product of two SQ8 vectors. Returns the RAW integer sum; it is
// monotone in the true cosine, which is all the walk needs. Multiply by
// kSq8Scale to recover an approximate float cosine.
[[nodiscard]] std::int32_t dot_sq8(const std::int8_t* a, const std::int8_t* b,
                                   std::size_t n) noexcept;

// Recovers an approximate cosine from dot_sq8's integer result.
inline constexpr float kSq8Scale = 1.0f / (127.0f * 127.0f);

// Which SIMD tier is active (for diagnostics/bench reporting).
[[nodiscard]] const char* simd_tier() noexcept;

} // namespace rag::dense
