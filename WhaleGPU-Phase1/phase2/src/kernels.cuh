#pragma once
#include "common.cuh"
#include <math_constants.h>
#include <algorithm>

struct Config {
    int threads = 256, items = 1, grid = 0; // grid=0 natural; otherwise absolute cap
    bool shuffle = true, contiguous = false;
};
struct Resources { int registers = 0, static_shared = 0, dynamic_shared = 0, active_blocks = 0; };
template<Op O> __device__ float neutral() {
    if constexpr (O == Op::Sum) return 0;
    if constexpr (O == Op::Max) return -CUDART_INF_F;
    return CUDART_INF_F;
}
template<Op O> __device__ float merge_value(float a, float b) {
    if constexpr (O == Op::Sum) return a + b;
    if constexpr (O == Op::Max) return fmaxf(a, b);
    return fminf(a, b);
}
template<Op O> __device__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v = merge_value<O>(v, __shfl_down_sync(0xffffffffu, v, d));
    return v;
}
template<Op O, bool Shuffle> __global__ void tuned_reduce(const float* in, float* out, size_t n, int items, bool contiguous) {
    extern __shared__ float sm[];
    unsigned t = threadIdx.x;
    size_t tile = size_t(blockDim.x) * items;
    float v = neutral<O>();
    for (size_t base = size_t(blockIdx.x) * tile; base < n; base += size_t(gridDim.x) * tile) {
        for (int e = 0; e < items; ++e) {
            size_t i = base + (contiguous ? size_t(t) * items + e : size_t(e) * blockDim.x + t);
            if (i < n) v = merge_value<O>(v, in[i]);
        }
    }
    if constexpr (Shuffle) {
        v = warp_sum<O>(v);
        if (t % 32 == 0) sm[t / 32] = v;
        __syncthreads();
        if (t < 32) {
            v = t < blockDim.x / 32 ? sm[t] : neutral<O>();
            v = warp_sum<O>(v);
            if (t == 0) out[blockIdx.x] = v;
        }
    } else {
        sm[t] = v;
        __syncthreads();
        for (unsigned d = blockDim.x / 2; d; d >>= 1) {
            if (t < d) sm[t] = merge_value<O>(sm[t], sm[t + d]);
            __syncthreads();
        }
        if (t == 0) out[blockIdx.x] = sm[0];
    }
}
inline size_t grid_count(size_t n, const Config& c) {
    size_t blocks = std::max(size_t(1), (n + size_t(c.threads) * c.items - 1) / (size_t(c.threads) * c.items));
    return c.grid ? std::min(blocks, size_t(c.grid)) : blocks;
}
template<Op O, bool S> const float* execute_tuned(const float* in, size_t n, float* a, float* b, const Config& c) {
    float* out = a;
    do {
        size_t blocks = grid_count(n, c);
        tuned_reduce<O,S><<<unsigned(blocks), c.threads, (S ? c.threads / 32 : c.threads) * sizeof(float)>>>(in, out, n, c.items, c.contiguous);
        CUDA_CHECK(cudaGetLastError());
        in = out; out = out == a ? b : a; n = blocks;
    } while (n > 1);
    return in;
}
template<Op O> const float* dispatch_version(const float* in, size_t n, float* a, float* b, const Config& c) {
    return c.shuffle ? execute_tuned<O,true>(in,n,a,b,c) : execute_tuned<O,false>(in,n,a,b,c);
}
inline const float* reduce_tuned(const float* in, size_t n, float* a, float* b, Op op, const Config& c) {
    if (op == Op::Sum) return dispatch_version<Op::Sum>(in,n,a,b,c);
    if (op == Op::Max) return dispatch_version<Op::Max>(in,n,a,b,c);
    return dispatch_version<Op::Min>(in,n,a,b,c);
}
template<Op O, bool S> Resources resources_for(const Config& c) {
    Resources r;
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, tuned_reduce<O,S>));
    r.registers = attr.numRegs; r.static_shared = int(attr.sharedSizeBytes);
    r.dynamic_shared = (S ? c.threads / 32 : c.threads) * sizeof(float);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks, tuned_reduce<O,S>, c.threads, r.dynamic_shared));
    return r;
}
template<Op O> Resources version_resources(const Config& c) {
    return c.shuffle ? resources_for<O,true>(c) : resources_for<O,false>(c);
}
inline Resources get_resources(Op op, const Config& c) {
    if (op == Op::Sum) return version_resources<Op::Sum>(c);
    if (op == Op::Max) return version_resources<Op::Max>(c);
    return version_resources<Op::Min>(c);
}
