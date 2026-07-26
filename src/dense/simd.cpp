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

// Four independent accumulators. A single acc chains every FMA through one
// dependency (~4-cycle latency each); four lets the CPU keep its FMA units
// saturated, which is what actually makes this throughput- rather than
// latency-bound. Embedding dims (256/384/768/1024) are all multiples of 32,
// so the 32-wide main loop covers the vast majority of the work.
__attribute__((target("avx2,fma")))
float dot_avx2(const float* a, const float* b, std::size_t n) noexcept {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm_prefetch(reinterpret_cast<const char*>(a + i + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b + i + 128), _MM_HINT_T0);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);

    a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    // Horizontal reduce: 256 -> 128 -> scalar.
    __m128 lo = _mm256_castps256_ps128(a0);
    __m128 hi = _mm256_extractf128_ps(a0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float s = _mm_cvtss_f32(lo);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

#if defined(RAGCPP_NEON)
// Same four-accumulator structure. vfmaq_f32 is a true fused multiply-add
// (single rounding, one instruction); vmlaq_f32 is not guaranteed to fuse.
float dot_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __builtin_prefetch(a + i + 64, 0, 0);
        __builtin_prefetch(b + i + 64, 0, 0);
        a0 = vfmaq_f32(a0, vld1q_f32(a + i +  0), vld1q_f32(b + i +  0));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));

    a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    float s = vaddvq_f32(a0);
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

void normalize(std::span<float> v) noexcept {
    float norm = std::sqrt(dot(v, v));
    if (norm <= 1e-12f) return;
    const float inv = 1.0f / norm;
    for (float& x : v) x *= inv;
}

// ─── SQ8 ───────────────────────────────────────────────────────────────
void quantize_sq8(std::span<const float> v, std::span<std::int8_t> out) noexcept {
    const std::size_t n = v.size() < out.size() ? v.size() : out.size();
    for (std::size_t i = 0; i < n; ++i) {
        // Round-to-nearest, then clamp: a component slightly outside [-1,1]
        // (numerical drift in normalize) must not wrap around.
        float s = v[i] * 127.0f;
        s = s < -127.0f ? -127.0f : (s > 127.0f ? 127.0f : s);
        out[i] = static_cast<std::int8_t>(s < 0 ? s - 0.5f : s + 0.5f);
    }
}

#if defined(RAGCPP_X86)
// Hoisted into its own target-attributed function for the same reason
// dot_avx2 is: AVX2 intrinsics are `always_inline`, and inlining one into a
// function compiled for the BASELINE ISA is a hard error, not a fallback
// ("inlining failed in call to always_inline ... target specific option
// mismatch"). Writing the block inline behind a runtime has_avx2() check
// compiles fine on ARM — where the whole branch is preprocessed away — and
// fails on every x86-64 build. That is exactly how it shipped: this file did
// not compile on x86-64 at all until CI ran on Linux.
//
// The runtime check stays where it is; the attribute only tells the compiler
// it may EMIT AVX2 here, it does not decide whether we CALL it.
__attribute__((target("avx2")))
std::int32_t dot_sq8_avx2(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
    __m256i acc = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        // Sign-extend to int16 and multiply-add pairs into int32.
        const __m256i alo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        const __m256i blo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        const __m256i ahi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        const __m256i bhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(alo, blo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ahi, bhi));
    }
    alignas(32) std::int32_t buf[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf), acc);
    std::int32_t sum = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < n; ++i) sum += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return sum;
}
#endif

std::int32_t dot_sq8(const std::int8_t* a, const std::int8_t* b, std::size_t n) noexcept {
#if defined(RAGCPP_NEON)
    {
        // Widening multiply-accumulate: 16 int8 lanes per iteration, accumulated
        // into int32 so nothing can overflow (worst case 127*127*dim ≪ 2^31).
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        std::size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            const int8x16_t va = vld1q_s8(a + i);
            const int8x16_t vb = vld1q_s8(b + i);
            int16x8_t p0 = vmull_s8(vget_low_s8(va),  vget_low_s8(vb));
            int16x8_t p1 = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            acc0 = vpadalq_s16(acc0, p0);
            acc1 = vpadalq_s16(acc1, p1);
        }
        std::int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
        for (; i < n; ++i)
            sum += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
        return sum;
    }
#elif defined(RAGCPP_X86)
    if (has_avx2()) return dot_sq8_avx2(a, b, n);
#endif
    std::int32_t s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return s;
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
