// rag/dense/simd.cpp — SIMD distance kernels with runtime dispatch.

#include "rag/dense/simd.hpp"

#include <bit>
#include <cmath>
#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64)
#  define RAGCPP_X86 1
#  include <immintrin.h>
#  if defined(__GNUC__)
#    include <cpuid.h>
#  endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
#  define RAGCPP_NEON 1
#  include <arm_neon.h>
#endif

namespace rag::dense {

namespace {

float dot_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

#if defined(RAGCPP_X86)
bool has_avx2() {
    static const bool v = [] {
#  if defined(__GNUC__)
        unsigned eax, ebx, ecx, edx;
        if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
        return (ebx & (1u << 5)) != 0; // AVX2 bit
#  else
        return false;
#  endif
    }();
    return v;
}

__attribute__((target("avx2,fma")))
float dot_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

#if defined(RAGCPP_NEON)
float dot_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    float s = vaddvq_f32(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

} // namespace

float dot(std::span<const float> a, std::span<const float> b) noexcept {
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
#if defined(RAGCPP_X86)
    if (has_avx2()) return dot_avx2(a.data(), b.data(), n);
#elif defined(RAGCPP_NEON)
    return dot_neon(a.data(), b.data(), n);
#endif
    return dot_scalar(a.data(), b.data(), n);
}

void normalize(std::vector<float>& v) noexcept {
    float norm = std::sqrt(dot(v, v));
    if (norm <= 1e-12f) return;
    const float inv = 1.0f / norm;
    for (float& x : v) x *= inv;
}

float cosine(std::span<const float> a, std::span<const float> b) noexcept {
    float na = std::sqrt(dot(a, a));
    float nb = std::sqrt(dot(b, b));
    if (na <= 1e-12f || nb <= 1e-12f) return 0.0f;
    return dot(a, b) / (na * nb);
}

std::vector<std::uint64_t> pack_signs(std::span<const float> v) {
    const std::size_t words = (v.size() + 63) / 64;
    std::vector<std::uint64_t> out(words, 0);
    for (std::size_t i = 0; i < v.size(); ++i)
        if (v[i] >= 0.0f) out[i >> 6] |= (std::uint64_t{1} << (i & 63));
    return out;
}

std::uint32_t hamming(std::span<const std::uint64_t> a,
                      std::span<const std::uint64_t> b) noexcept {
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    std::uint32_t d = 0;
    for (std::size_t i = 0; i < n; ++i)
        d += static_cast<std::uint32_t>(std::popcount(a[i] ^ b[i]));
    return d;
}

const char* simd_tier() noexcept {
#if defined(RAGCPP_X86)
    return has_avx2() ? "avx2+fma" : "scalar-x86";
#elif defined(RAGCPP_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

} // namespace rag::dense
