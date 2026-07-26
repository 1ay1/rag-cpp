// src/gpu/opencl_backend.cpp — the cross-platform GPU backend.
//
// Metal is faster on Apple silicon and stays the default there, but it exists
// only on Apple. This backend is what makes GPU acceleration available on
// NVIDIA, AMD, and Intel: OpenCL 1.2 is supported by every one of their drivers,
// and the kernel below is compiled AT RUNTIME by whichever driver is present,
// so a single binary runs on all of them with no vendor SDK at build time.
//
// The exported symbols are plain C for the same reason the Metal ones are: this
// library is routinely built with Homebrew GCC while some backends must be
// compiled by Clang, so the two objects can be linked against different standard
// libraries. Only trivially-copyable C types may cross that line — see the note
// in device.cpp, which was written after a std::string across the boundary
// produced an empty device name and an abort on exit.
//
// A MEASURED WARNING ABOUT APPLE. This backend is correct on macOS and its
// kernel is genuinely fast there, but Apple's OpenCL driver is deprecated and
// adds enormous per-dispatch latency. Profiled on an M1 with a 200k x 384
// corpus and 64 queries, using CL_QUEUE_PROFILING_ENABLE to separate the two:
//
//   kernel execution (device timestamps)   16.1 ms
//   readback of the 48 MB result            4.3 ms
//   wall time of the same dispatch         672   ms      <- the driver
//
// The kernel is faster than the Metal backend's 41 ms; the ~650 ms is pure ICD
// overhead, reproducible on every dispatch rather than just the first, and
// unaffected by work-group size (16..256 all measure 16 ms of kernel). That is
// why device.cpp prefers Metal on Apple and this backend is the fallback there.
// On NVIDIA/AMD/Intel ICDs the overhead is not present and this is the path that
// makes GPU acceleration available at all.

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 120
#define CL_SILENCE_DEPRECATION
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace {

// One tile of the score matrix per work-item.
//
// out[q * n + i] = dot(queries[q], corpus[i])
//
// Each work-item owns one (query, candidate) pair and walks `dim` floats of
// each. The corpus row is the hot operand: consecutive work-items in a
// work-group share the same query and stride across neighbouring candidates, so
// the corpus reads coalesce, which is the access pattern every GPU memory system
// is built for.
//
// Kept deliberately simple — no local-memory tiling, no vectorised loads. A
// naive coalesced kernel already saturates bandwidth for this shape (the
// arithmetic intensity is one multiply-add per two loaded floats), and a
// complicated kernel that must compile on four vendors' drivers is a liability,
// not an optimisation.
const char* kKernelSource = R"CLC(
__kernel void score_batch(__global const float* corpus,
                          __global const float* queries,
                          const unsigned int dim,
                          const unsigned int n,
                          const unsigned int nq,
                          __global float* out) {
    const size_t i = get_global_id(0);   // candidate
    const size_t q = get_global_id(1);   // query
    if (i >= n || q >= nq) return;

    const __global float* c = corpus  + i * (size_t)dim;
    const __global float* v = queries + q * (size_t)dim;

    float acc = 0.0f;
    for (unsigned int d = 0; d < dim; ++d) acc += c[d] * v[d];
    out[q * (size_t)n + i] = acc;
}
)CLC";

struct Ctx {
    cl_platform_id   platform = nullptr;
    cl_device_id     device   = nullptr;
    cl_context       context  = nullptr;
    cl_command_queue queue    = nullptr;
    cl_program       program  = nullptr;
    cl_kernel        kernel   = nullptr;
    char             name[256] = {0};
    cl_ulong         max_alloc = 0;
    cl_bool          unified   = CL_FALSE;
    bool             ok        = false;
};

Ctx& ctx() {
    static Ctx c;
    return c;
}

// Pick the best device across every platform: a real GPU if one exists,
// otherwise nothing. We deliberately do NOT fall back to a CPU OpenCL device —
// the library already has a well-tuned threaded NEON/AVX path, and routing
// through a driver to reach the same silicon would be slower, not faster.
bool pick_device(Ctx& c) {
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return false;
    std::vector<cl_platform_id> plats(nplat);
    if (clGetPlatformIDs(nplat, plats.data(), nullptr) != CL_SUCCESS) return false;

    cl_ulong best_mem = 0;
    for (cl_platform_id p : plats) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0)
            continue;
        std::vector<cl_device_id> devs(ndev);
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, ndev, devs.data(), nullptr) != CL_SUCCESS)
            continue;
        for (cl_device_id d : devs) {
            cl_bool avail = CL_FALSE;
            clGetDeviceInfo(d, CL_DEVICE_AVAILABLE, sizeof avail, &avail, nullptr);
            if (!avail) continue;
            cl_ulong gmem = 0;
            clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof gmem, &gmem, nullptr);
            // More memory is a decent proxy for "the discrete card, not the
            // integrated one" on the mixed-GPU laptops where this matters.
            if (gmem > best_mem) { best_mem = gmem; c.platform = p; c.device = d; }
        }
    }
    return c.device != nullptr;
}

} // namespace

extern "C" {

bool opencl_init() noexcept {
    Ctx& c = ctx();
    if (c.ok) return true;
    if (!pick_device(c)) return false;

    cl_int err = CL_SUCCESS;
    c.context = clCreateContext(nullptr, 1, &c.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !c.context) return false;

    // clCreateCommandQueue is deprecated in OpenCL 2.0 but is the only call
    // available on 1.2-only drivers, which is most of the installed base.
    c.queue = clCreateCommandQueue(c.context, c.device, 0, &err);
    if (err != CL_SUCCESS || !c.queue) return false;

    c.program = clCreateProgramWithSource(c.context, 1, &kKernelSource, nullptr, &err);
    if (err != CL_SUCCESS || !c.program) return false;

    // -cl-fast-relaxed-math is NOT used: it permits reassociation, and this
    // kernel's results must match the CPU dot product closely enough that a
    // GPU-scored batch and a CPU-scored one rank identically. A faster kernel
    // that reorders results is a correctness bug wearing a performance costume.
    err = clBuildProgram(c.program, 1, &c.device, "-cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    c.kernel = clCreateKernel(c.program, "score_batch", &err);
    if (err != CL_SUCCESS || !c.kernel) return false;

    clGetDeviceInfo(c.device, CL_DEVICE_NAME, sizeof c.name - 1, c.name, nullptr);
    clGetDeviceInfo(c.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof c.max_alloc, &c.max_alloc, nullptr);
    clGetDeviceInfo(c.device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof c.unified, &c.unified, nullptr);

    c.ok = true;
    return true;
}

void opencl_info(char* name_out, unsigned long name_cap,
                 int* unified_out, unsigned long long* max_buffer_out) noexcept {
    const Ctx& c = ctx();
    if (name_out && name_cap) {
        std::snprintf(name_out, name_cap, "%s", c.name[0] ? c.name : "opencl");
    }
    if (unified_out)    *unified_out    = c.unified ? 1 : 0;
    if (max_buffer_out) *max_buffer_out = static_cast<unsigned long long>(c.max_alloc);
}

bool opencl_score_batch(const float* corpus, const float* queries,
                        unsigned long dim, unsigned long n, unsigned long nq,
                        float* out) noexcept {
    Ctx& c = ctx();
    if (!c.ok || !corpus || !queries || !out) return false;

    const std::size_t corpus_bytes = static_cast<std::size_t>(n) * dim * sizeof(float);
    const std::size_t query_bytes  = static_cast<std::size_t>(nq) * dim * sizeof(float);
    const std::size_t out_bytes    = static_cast<std::size_t>(n) * nq * sizeof(float);
    if (c.max_alloc && (corpus_bytes > c.max_alloc || out_bytes > c.max_alloc)) return false;

    cl_int err = CL_SUCCESS;
    // On a UNIFIED-MEMORY device the host and the GPU address the same physical
    // RAM, so COPY_HOST_PTR stages a full copy of the corpus for nothing. That
    // is not a small effect: with a 200k x 384 corpus it measured 747 ms against
    // a 482 ms CPU scan — the copy alone turned a win into a 0.67x loss. Where
    // the driver reports unified memory, alias the host allocation instead.
    //
    // USE_HOST_PTR requires the pointer to remain valid for the buffer's
    // lifetime, which holds here: the buffers are released before this function
    // returns, and the caller owns `corpus`/`queries` across the call.
    const cl_mem_flags in_flags = c.unified
        ? (CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR)
        : (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    cl_mem d_corpus = clCreateBuffer(c.context, in_flags,
                                     corpus_bytes, const_cast<float*>(corpus), &err);
    if (err != CL_SUCCESS) return false;
    cl_mem d_query = clCreateBuffer(c.context, in_flags,
                                    query_bytes, const_cast<float*>(queries), &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_corpus); return false; }
    // The output is written by the device and read once; on unified memory we
    // can let the device write straight into the caller's buffer.
    cl_mem d_out = c.unified
        ? clCreateBuffer(c.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, out_bytes, out, &err)
        : clCreateBuffer(c.context, CL_MEM_WRITE_ONLY, out_bytes, nullptr, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_corpus); clReleaseMemObject(d_query); return false; }

    const cl_uint u_dim = static_cast<cl_uint>(dim);
    const cl_uint u_n   = static_cast<cl_uint>(n);
    const cl_uint u_nq  = static_cast<cl_uint>(nq);
    bool ok = true;
    ok &= clSetKernelArg(c.kernel, 0, sizeof(cl_mem), &d_corpus) == CL_SUCCESS;
    ok &= clSetKernelArg(c.kernel, 1, sizeof(cl_mem), &d_query)  == CL_SUCCESS;
    ok &= clSetKernelArg(c.kernel, 2, sizeof(cl_uint), &u_dim)   == CL_SUCCESS;
    ok &= clSetKernelArg(c.kernel, 3, sizeof(cl_uint), &u_n)     == CL_SUCCESS;
    ok &= clSetKernelArg(c.kernel, 4, sizeof(cl_uint), &u_nq)    == CL_SUCCESS;
    ok &= clSetKernelArg(c.kernel, 5, sizeof(cl_mem), &d_out)    == CL_SUCCESS;

    if (ok) {
        // Round the candidate dimension up to a work-group multiple; the kernel
        // bounds-checks, so the padding work-items simply return.
        constexpr std::size_t kTile = 64;
        const std::size_t gws0 = ((static_cast<std::size_t>(n) + kTile - 1) / kTile) * kTile;
        const std::size_t global[2] = { gws0, static_cast<std::size_t>(nq) };
        const std::size_t local[2]  = { kTile, 1 };
        err = clEnqueueNDRangeKernel(c.queue, c.kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Some drivers reject an explicit local size for this shape; let the
            // implementation choose rather than failing the whole dispatch.
            err = clEnqueueNDRangeKernel(c.queue, c.kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        }
        ok = (err == CL_SUCCESS);
    }
    if (ok) {
        if (c.unified) {
            // The device wrote into the caller's memory; map to synchronise
            // rather than copying the results back out of a device allocation.
            void* p = clEnqueueMapBuffer(c.queue, d_out, CL_TRUE, CL_MAP_READ, 0, out_bytes,
                                         0, nullptr, nullptr, &err);
            ok = (err == CL_SUCCESS);
            if (ok) clEnqueueUnmapMemObject(c.queue, d_out, p, 0, nullptr, nullptr);
            clFinish(c.queue);
        } else {
            ok = clEnqueueReadBuffer(c.queue, d_out, CL_TRUE, 0, out_bytes, out, 0, nullptr, nullptr)
                 == CL_SUCCESS;
        }
    }

    clReleaseMemObject(d_corpus);
    clReleaseMemObject(d_query);
    clReleaseMemObject(d_out);
    return ok;
}

} // extern "C"
