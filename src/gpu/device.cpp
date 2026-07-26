// rag/gpu/device.cpp — backend-neutral half of the GPU seam.
//
// This file is always compiled. When a GPU backend is built in, it defers to
// that backend through the small internal hook below; otherwise every entry
// point reports "no device" and callers take their CPU path. Keeping the
// dispatch decision here (rather than in each backend) means there is exactly
// one place that decides whether a job goes to the GPU.

#include "rag/gpu/device.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace rag::gpu {

const char* backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::metal:  return "metal";
        case Backend::opencl: return "opencl";
        case Backend::none:   break;
    }
    return "none";
}

std::vector<Backend> compiled_backends() {
    std::vector<Backend> v;
#if defined(RAGCPP_WITH_METAL)
    v.push_back(Backend::metal);
#endif
#if defined(RAGCPP_WITH_OPENCL)
    v.push_back(Backend::opencl);
#endif
    return v;
}

namespace detail {

// Implemented by a backend translation unit when one is compiled in. Declared
// here rather than in the public header so the backend stays invisible to
// users of the library.
//
// THE SIGNATURES ARE PLAIN C ON PURPOSE, and this is not stylistic. The Metal
// backend is Objective-C++, which only Clang can compile, while this library is
// routinely built with Homebrew GCC — so the two object files use DIFFERENT
// STANDARD LIBRARIES (libc++ vs libstdc++). Passing a std::string or any type
// containing one across that boundary is undefined behaviour: the layouts do
// not match. It is also not a theoretical hazard. An earlier version of this
// returned `DeviceInfo&` and the symptoms were an empty device name, a device
// reporting no unified memory on hardware that has nothing else, and an abort
// on exit. Only trivially-copyable C types may cross this line.
#if defined(RAGCPP_WITH_METAL)
extern "C" {
bool metal_init() noexcept;
void metal_info(char* name_out, unsigned long name_cap,
                int* unified_out, unsigned long long* max_buffer_out) noexcept;
bool metal_score_batch(const float*, const float*, unsigned long, unsigned long,
                       unsigned long, float*) noexcept;
}
#endif
#if defined(RAGCPP_WITH_OPENCL)
extern "C" {
bool opencl_init() noexcept;
void opencl_info(char* name_out, unsigned long name_cap,
                 int* unified_out, unsigned long long* max_buffer_out) noexcept;
bool opencl_score_batch(const float*, const float*, unsigned long, unsigned long,
                        unsigned long, float*) noexcept;
}
#endif

// One-way kill switch. Relaxed ordering is right: this is a hint that flips at
// most once, and a query racing the flip may legitimately observe either value.
std::atomic<bool>& disabled() noexcept {
    static std::atomic<bool> d{false};
    return d;
}

DeviceInfo& info() noexcept {
    static DeviceInfo i;
    return i;
}

// Probe the devices exactly once and choose a backend. static-local
// initialization is thread-safe (C++11 [stmt.dcl]/4) and the probe is expensive
// — it compiles kernels — so doing it under the initializer rather than a
// hand-rolled flag also avoids a double-compile race.
//
// ORDER OF PREFERENCE, and why:
//   1. RAGCPP_GPU_BACKEND=metal|opencl|none forces the choice. Without this a
//      user on a machine where one driver misbehaves has no recourse short of
//      rebuilding, and a test cannot exercise the non-default backend at all.
//   2. Metal, when built in. On Apple silicon it wins on the merits: unified
//      memory means no host<->device copy, where the OpenCL path must stage
//      through clCreateBuffer.
//   3. OpenCL. The portable path — NVIDIA, AMD, Intel, and Apple as a fallback.
// A backend that fails to initialize falls through to the next rather than
// disabling the GPU entirely: a broken OpenCL ICD should not cost an Apple user
// their Metal path, and vice versa.
bool initialized() noexcept {
    static const bool ok = [] {
        const char* want = std::getenv("RAGCPP_GPU_BACKEND");
        const bool force_metal  = want && std::strcmp(want, "metal")  == 0;
        const bool force_opencl = want && std::strcmp(want, "opencl") == 0;
        const bool force_none   = want && std::strcmp(want, "none")   == 0;
        if (force_none) return false;

        char name[256] = {0};
        int unified = 0;
        unsigned long long max_buf = 0;

#if defined(RAGCPP_WITH_METAL)
        if (!force_opencl && metal_init()) {
            metal_info(name, sizeof name, &unified, &max_buf);
            info().backend          = Backend::metal;
            info().name             = name[0] ? name : "metal";
            info().unified_memory   = unified != 0;
            info().max_buffer_bytes = static_cast<std::size_t>(max_buf);
            return true;
        }
#endif
#if defined(RAGCPP_WITH_OPENCL)
        if (!force_metal && opencl_init()) {
            opencl_info(name, sizeof name, &unified, &max_buf);
            info().backend          = Backend::opencl;
            info().name             = name[0] ? name : "opencl";
            info().unified_memory   = unified != 0;
            info().max_buffer_bytes = static_cast<std::size_t>(max_buf);
            return true;
        }
#endif
        (void)force_metal; (void)force_opencl;
        return false;
    }();
    return ok;
}

} // namespace detail

bool available() noexcept {
    if (detail::disabled().load(std::memory_order_relaxed)) return false;
    return detail::initialized();
}

const DeviceInfo& device_info() noexcept {
    (void)available();          // ensure the probe has run before reporting
    static const DeviceInfo cpu{};
    return detail::disabled().load(std::memory_order_relaxed) ? cpu : detail::info();
}

void disable() noexcept { detail::disabled().store(true, std::memory_order_relaxed); }

std::size_t min_batch_work() noexcept {
    // Measured crossover, not a guess.
    //
    // The competition is not a naive loop — it is the CPU path that already
    // ships: NEON dot products, 8 threads, iterating candidates outer and
    // queries inner so each candidate row stays in L1 across the whole query
    // batch. That baseline is strong, and the GPU only overtakes it once the
    // batch is big enough to amortize a ~0.2 ms dispatch round trip AND wide
    // enough for the query tile to pay:
    //
    //   n x nq (dim=384)   CPU blocked   GPU     ratio
    //    50k x   8            3.4 ms    11.0 ms   0.3x
    //    50k x  32           10.9 ms    14.2 ms   0.8x
    //   200k x  32           40.8 ms    27.8 ms   1.5x
    //   200k x 128          222.1 ms    71.9 ms   3.1x
    //   500k x 128          527.0 ms   187.2 ms   2.8x
    //
    // 50k x 32 x 384 is 614M MACs and still loses; 200k x 32 x 384 is 2.5G and
    // wins. The threshold sits between them, deliberately nearer the winning
    // end — declining a job the GPU would have won costs a little speed, while
    // accepting one it loses costs correctness of the whole premise.
    return 2ull * 1000ull * 1000ull * 1000ull;
}

bool score_batch(std::span<const float> corpus, std::span<const float> queries,
                 std::size_t dim, std::span<float> out) noexcept {
    if (dim == 0) return false;
    if (corpus.size() % dim || queries.size() % dim) return false;
    const std::size_t n  = corpus.size() / dim;
    const std::size_t nq = queries.size() / dim;
    if (n == 0 || nq == 0) return false;
    if (out.size() < n * nq) return false;
    if (n * nq * dim < min_batch_work()) return false;   // too small to pay for
    if (!available()) return false;

    // A buffer bigger than the device can allocate is a decline, not an error:
    // the caller's CPU path handles it correctly and is what would have run
    // anyway. Checked against whichever backend was selected.
    if (detail::info().max_buffer_bytes &&
        corpus.size() * sizeof(float) > detail::info().max_buffer_bytes) return false;

    switch (detail::info().backend) {
#if defined(RAGCPP_WITH_METAL)
        case Backend::metal:
            return detail::metal_score_batch(corpus.data(), queries.data(),
                                             static_cast<unsigned long>(dim),
                                             static_cast<unsigned long>(n),
                                             static_cast<unsigned long>(nq), out.data());
#endif
#if defined(RAGCPP_WITH_OPENCL)
        case Backend::opencl:
            return detail::opencl_score_batch(corpus.data(), queries.data(),
                                              static_cast<unsigned long>(dim),
                                              static_cast<unsigned long>(n),
                                              static_cast<unsigned long>(nq), out.data());
#endif
        default: break;
    }
    return false;
}

} // namespace rag::gpu
