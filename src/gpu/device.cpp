// rag/gpu/device.cpp — backend-neutral half of the GPU seam.
//
// This file is always compiled. When a GPU backend is built in, it defers to
// that backend through the small internal hook below; otherwise every entry
// point reports "no device" and callers take their CPU path. Keeping the
// dispatch decision here (rather than in each backend) means there is exactly
// one place that decides whether a job goes to the GPU.

#include "rag/gpu/device.hpp"

#include <utility>

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
void* metal_open(char* name_out, unsigned long name_cap,
                 int* unified_out, unsigned long long* max_buffer_out) noexcept;
void metal_close(void* ctx) noexcept;
bool metal_score_batch(void* ctx, const float*, const float*, unsigned long,
                       unsigned long, unsigned long, float*) noexcept;
}
#endif
#if defined(RAGCPP_WITH_OPENCL)
extern "C" {
void* opencl_open(char* name_out, unsigned long name_cap,
                  int* unified_out, unsigned long long* max_buffer_out) noexcept;
void opencl_close(void* ctx) noexcept;
bool opencl_score_batch(void* ctx, const float*, const float*, unsigned long,
                        unsigned long, unsigned long, float*) noexcept;
}
#endif

} // namespace detail

// Metal first when allowed: on Apple silicon unified memory means no copy,
// where OpenCL must stage through clCreateBuffer. A backend that fails to
// initialise falls through to the next, so a broken OpenCL ICD does not cost
// an Apple user their Metal path.
Device Device::open(Prefer prefer) {
    Device d;
    char name[256] = {0};
    int unified = 0;
    unsigned long long max_buf = 0;
    (void)name; (void)unified; (void)max_buf; (void)prefer;

    auto adopt = [&](void* h, Backend b, const char* fallback) {
        d.handle_                = h;
        d.info_.backend          = b;
        d.info_.name             = name[0] ? name : fallback;
        d.info_.unified_memory   = unified != 0;
        d.info_.max_buffer_bytes = static_cast<std::size_t>(max_buf);
    };
#if defined(RAGCPP_WITH_METAL)
    if (prefer != Prefer::opencl)
        if (void* h = detail::metal_open(name, sizeof name, &unified, &max_buf)) {
            adopt(h, Backend::metal, "metal");
            return d;
        }
#endif
#if defined(RAGCPP_WITH_OPENCL)
    if (prefer != Prefer::metal)
        if (void* h = detail::opencl_open(name, sizeof name, &unified, &max_buf)) {
            adopt(h, Backend::opencl, "opencl");
            return d;
        }
#endif
    (void)adopt;
    return d;
}

void Device::reset() noexcept {
    if (!handle_) return;
    switch (info_.backend) {
#if defined(RAGCPP_WITH_METAL)
        case Backend::metal:  detail::metal_close(handle_); break;
#endif
#if defined(RAGCPP_WITH_OPENCL)
        case Backend::opencl: detail::opencl_close(handle_); break;
#endif
        default: break;
    }
    handle_ = nullptr;
    info_ = {};
}

Device::Device(Device&& o) noexcept
    : info_(std::move(o.info_)), handle_(std::exchange(o.handle_, nullptr)) {
    o.info_ = {};
}

Device& Device::operator=(Device&& o) noexcept {
    if (this != &o) {
        reset();
        info_   = std::move(o.info_);
        handle_ = std::exchange(o.handle_, nullptr);
        o.info_ = {};
    }
    return *this;
}

Device::~Device() { reset(); }

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

bool Device::score_batch(std::span<const float> corpus, std::span<const float> queries,
                         std::size_t dim, std::span<float> out) const noexcept {
    if (!handle_ || dim == 0) return false;
    if (corpus.size() % dim || queries.size() % dim) return false;
    const std::size_t n  = corpus.size() / dim;
    const std::size_t nq = queries.size() / dim;
    if (n == 0 || nq == 0) return false;
    if (out.size() < n * nq) return false;
    if (n * nq * dim < min_batch_work()) return false;   // too small to pay for

    // A buffer bigger than the device can allocate is a decline, not an error:
    // the caller's CPU path handles it.
    if (info_.max_buffer_bytes && corpus.size() * sizeof(float) > info_.max_buffer_bytes)
        return false;

    switch (info_.backend) {
#if defined(RAGCPP_WITH_METAL)
        case Backend::metal:
            return detail::metal_score_batch(handle_, corpus.data(), queries.data(),
                                             static_cast<unsigned long>(dim),
                                             static_cast<unsigned long>(n),
                                             static_cast<unsigned long>(nq), out.data());
#endif
#if defined(RAGCPP_WITH_OPENCL)
        case Backend::opencl:
            return detail::opencl_score_batch(handle_, corpus.data(), queries.data(),
                                              static_cast<unsigned long>(dim),
                                              static_cast<unsigned long>(n),
                                              static_cast<unsigned long>(nq), out.data());
#endif
        default: break;
    }
    return false;
}

} // namespace rag::gpu
