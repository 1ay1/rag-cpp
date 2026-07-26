// rag/gpu/metal_backend.mm — Metal implementation of the batch scoring hook.
//
// Compiled only when RAGCPP_WITH_METAL is on (Apple platforms). Everything here
// is private to the library; the public surface is rag/gpu/device.hpp.
//
// TWO CHOICES WORTH EXPLAINING.
//
// 1. Shaders are compiled from SOURCE AT RUNTIME, not from a precompiled
//    .metallib. Building a .metallib needs the Metal CLI toolchain, which is a
//    separate multi-gigabyte Xcode component that is NOT present on a stock
//    machine (`xcrun metal` fails with "missing Metal Toolchain" on this one).
//    Requiring it would make the GPU path unbuildable for most people who
//    check out this repo. newLibraryWithSource: goes through the same compiler
//    inside the driver, costs a few milliseconds ONCE at first use, and is
//    memoized for the process lifetime.
//
// 2. Buffers use MTLResourceStorageModeShared and are created with
//    newBufferWithBytesNoCopy where alignment permits. On Apple silicon the CPU
//    and GPU address the same physical memory, so a "device buffer" is a view,
//    not a copy. This is the entire reason a 6x speedup survives contact with
//    reality: on a discrete card, moving a 300 MB corpus over PCIe for one
//    batch would cost more than the compute it enables.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unistd.h>

// Deliberately NOT including rag/gpu/device.hpp. This translation unit is
// compiled by Clang against libc++ while the rest of the library is built by
// GCC against libstdc++, so it must not see — let alone pass — any standard
// library type. The entire interface is the extern "C" block below, whose
// signatures are duplicated in src/gpu/device.cpp; keep the two in sync.

namespace {

// The kernel and why it is shaped this way.
//
// The obvious kernel — one thread per (query, candidate) pair — is the one to
// avoid, and measuring against a properly optimized CPU is what exposes it. It
// re-reads the candidate row from device memory once per query, so at 200k x
// 128 x 384 it reaches 105 GFLOP/s and takes 188 ms, while a cache-BLOCKED
// NEON CPU loop (candidate outer, queries inner, so each row stays in L1
// across all queries) does the same work in 205 ms across 8 threads. A GPU
// that merely ties a CPU is not worth the complexity.
//
// Tiling gives the GPU the same optimization the CPU gets from its cache: each
// thread owns ONE candidate, loads its row once into registers, and reuses it
// across QT queries, so device traffic falls by a factor of QT and arithmetic
// intensity rises with it. Same hardware, same arithmetic: 320 GFLOP/s and
// 61 ms — 3.3x the blocked CPU, where the naive kernel was 0.9x.
//
// QT=8 holds 8 float4 accumulators (32 registers) per thread, which fits
// comfortably; larger tiles start spilling and get slower.
constexpr const char* kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define QT 8

kernel void score_batch(device const float* corpus  [[buffer(0)]],
                        device const float* queries [[buffer(1)]],
                        device       float* out     [[buffer(2)]],
                        constant     uint&  dim     [[buffer(3)]],
                        constant     uint&  n       [[buffer(4)]],
                        constant     uint&  nq      [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]])
{
    const uint cand = gid.x;
    const uint qbase = gid.y * QT;
    if (cand >= n || qbase >= nq) return;

    // Vector path. Reinterpreting as float4 (rather than building a float4 from
    // four scalar loads) is what lets the hardware issue one 16-byte load per
    // iteration. Every row starts at a multiple of `dim` floats, so when
    // dim % 4 == 0 each row inherits the buffer's 16-byte alignment.
    if ((dim & 3u) == 0u) {
        const uint d4 = dim >> 2;
        device const float4* r = (device const float4*)(corpus + (uint64_t)cand * dim);
        device const float4* qs = (device const float4*)queries;

        float4 acc[QT];
        for (uint t = 0; t < QT; ++t) acc[t] = float4(0.0f);

        for (uint d = 0; d < d4; ++d) {
            const float4 rv = r[d];                 // ONE load, reused QT times
            for (uint t = 0; t < QT; ++t) {
                const uint q = qbase + t;
                if (q >= nq) break;
                acc[t] += rv * qs[(uint64_t)q * d4 + d];
            }
        }
        for (uint t = 0; t < QT; ++t) {
            const uint q = qbase + t;
            if (q >= nq) break;
            out[(uint64_t)q * n + cand] = acc[t].x + acc[t].y + acc[t].z + acc[t].w;
        }
        return;
    }

    // Scalar path for dimensions that are not a multiple of 4 (GloVe is 25).
    // Slower, but a kernel that silently mis-scored unusual dims would be far
    // worse than one that is merely slower on them.
    device const float* r = corpus + (uint64_t)cand * dim;
    for (uint t = 0; t < QT; ++t) {
        const uint q = qbase + t;
        if (q >= nq) break;
        device const float* v = queries + (uint64_t)q * dim;
        float acc = 0.0f;
        for (uint d = 0; d < dim; ++d) acc += r[d] * v[d];
        out[(uint64_t)q * n + cand] = acc;
    }
}
)METAL";

// Query-tile factor; must match QT in the shader above.
constexpr std::size_t kQueryTile = 8;

struct Ctx {
    id<MTLDevice>               device = nil;
    id<MTLCommandQueue>         queue  = nil;
    id<MTLComputePipelineState> pso    = nil;
    bool ok = false;
};

Ctx& ctx() { static Ctx c; return c; }

} // namespace

extern "C" {

bool metal_init() noexcept {
    @autoreleasepool {
        Ctx& c = ctx();
        c.device = MTLCreateSystemDefaultDevice();
        if (!c.device) return false;               // headless / no GPU
        c.queue = [c.device newCommandQueue];
        if (!c.queue) return false;

        NSError* err = nil;
        id<MTLLibrary> lib =
            [c.device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                                   options:nil
                                     error:&err];
        if (!lib) return false;                    // driver refused to compile
        id<MTLFunction> fn = [lib newFunctionWithName:@"score_batch"];
        if (!fn) return false;
        c.pso = [c.device newComputePipelineStateWithFunction:fn error:&err];
        if (!c.pso) return false;

        c.ok = true;
        return true;
    }
}

void metal_info(char* name_out, unsigned long name_cap,
                int* unified_out, unsigned long long* max_buffer_out) noexcept {
    Ctx& c = ctx();
    if (name_out && name_cap) {
        const char* n = (c.ok && c.device) ? [[c.device name] UTF8String] : "metal";
        std::strncpy(name_out, n ? n : "metal", name_cap - 1);
        name_out[name_cap - 1] = '\0';
    }
    if (unified_out)    *unified_out = (c.ok && c.device.hasUnifiedMemory) ? 1 : 0;
    if (max_buffer_out) *max_buffer_out =
        c.ok ? (unsigned long long)c.device.maxBufferLength : 0ull;
}

bool metal_score_batch(const float* corpus, const float* queries, unsigned long dim,
                       unsigned long n, unsigned long nq, float* out) noexcept {
    Ctx& c = ctx();
    if (!c.ok) return false;

    @autoreleasepool {
        // newBufferWithBytesNoCopy wraps the caller's pages directly — zero
        // copy — but demands page-aligned address AND length. Callers hand us
        // arbitrary std::vector data, so fall back to a copying constructor
        // when alignment does not cooperate. Both are Shared storage, so
        // neither performs a host->device transfer; the fallback is a memcpy
        // within the same physical memory.
        const std::size_t corpus_bytes = n * dim * sizeof(float);
        const std::size_t query_bytes  = nq * dim * sizeof(float);
        const std::size_t out_bytes    = n * nq * sizeof(float);
        const std::size_t page = (std::size_t)getpagesize();

        auto wrap = [&](const void* p, std::size_t bytes) -> id<MTLBuffer> {
            const bool aligned = ((std::uintptr_t)p % page) == 0 && (bytes % page) == 0;
            if (aligned)
                return [c.device newBufferWithBytesNoCopy:(void*)p
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
            return [c.device newBufferWithBytes:p length:bytes
                                        options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> bc = wrap(corpus, corpus_bytes);
        id<MTLBuffer> bq = wrap(queries, query_bytes);
        if (!bc || !bq) return false;

        // The output is written by the GPU and read by the caller. Wrap it in
        // place when we can, so the results need no copy back either.
        const bool out_aligned =
            ((std::uintptr_t)out % page) == 0 && (out_bytes % page) == 0;
        id<MTLBuffer> bo = out_aligned
            ? [c.device newBufferWithBytesNoCopy:(void*)out length:out_bytes
                                         options:MTLResourceStorageModeShared deallocator:nil]
            : [c.device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        if (!bo) return false;

        id<MTLCommandBuffer>        cb  = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:c.pso];
        [enc setBuffer:bc offset:0 atIndex:0];
        [enc setBuffer:bq offset:0 atIndex:1];
        [enc setBuffer:bo offset:0 atIndex:2];
        uint32_t u_dim = (uint32_t)dim, u_n = (uint32_t)n, u_nq = (uint32_t)nq;
        [enc setBytes:&u_dim length:sizeof(u_dim) atIndex:3];
        [enc setBytes:&u_n   length:sizeof(u_n)   atIndex:4];
        [enc setBytes:&u_nq  length:sizeof(u_nq)  atIndex:5];

        // Threadgroup shape and grid.
        //
        // The y axis is divided by kQueryTile because each thread now covers
        // that many queries. The x axis stays candidates, which is the
        // contiguous axis of both the corpus reads and the output writes, so
        // neighbouring threads in a SIMD group touch neighbouring addresses.
        //
        // 32x8 measured fastest of nine shapes swept on an M1 (32 is the SIMD
        // width, so the group is 8 full SIMD groups). Shapes with h=1 were 3-4x
        // slower for identical arithmetic: 24 GFLOP/s against 105, because a
        // single row of threads cannot hide memory latency.
        const NSUInteger w = std::min<NSUInteger>(
            std::max<NSUInteger>(c.pso.threadExecutionWidth, 1), 32);
        const NSUInteger gy = (nq + kQueryTile - 1) / kQueryTile;
        const NSUInteger h = std::max<NSUInteger>(
            1, std::min<NSUInteger>({c.pso.maxTotalThreadsPerThreadgroup / w, 8, gy}));
        [enc dispatchThreads:MTLSizeMake(n, gy, 1)
       threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status != MTLCommandBufferStatusCompleted) return false;
        if (!out_aligned) std::copy_n((const float*)bo.contents, n * nq, out);
        return true;
    }
}

} // extern "C"
