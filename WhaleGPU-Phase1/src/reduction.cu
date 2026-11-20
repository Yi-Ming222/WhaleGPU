#include "common.cuh"
#include <math_constants.h>

template<Op op> __device__ float identity() {
    if constexpr (op == Op::Sum) return 0.0f;
    if constexpr (op == Op::Max) return -CUDART_INF_F;
    return CUDART_INF_F;
}
template<Op op> __device__ float combine(float a, float b) {
    if constexpr (op == Op::Sum) return a + b;
    if constexpr (op == Op::Max) return fmaxf(a, b);
    return fminf(a, b);
}

template<Op op> __global__ void shared_reduce(const float* in, float* out, size_t n) {
    __shared__ float values[BLOCK];
    unsigned t = threadIdx.x;
    size_t i = static_cast<size_t>(blockIdx.x) * BLOCK + t;
    values[t] = i < n ? in[i] : identity<op>();
    __syncthreads();
    for (unsigned stride = BLOCK / 2; stride; stride >>= 1) {
        if (t < stride) values[t] = combine<op>(values[t], values[t + stride]);
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = values[0];
}

template<Op op> __device__ float warp_reduce(float value) {
    // Every lane in each full warp executes every shuffle, including padded lanes.
    for (int offset = 16; offset; offset >>= 1)
        value = combine<op>(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}
template<Op op> __global__ void shuffle_reduce(const float* in, float* out, size_t n) {
    __shared__ float warp_results[BLOCK / 32];
    unsigned t = threadIdx.x, lane = t % 32, warp = t / 32;
    size_t i = static_cast<size_t>(blockIdx.x) * BLOCK + t;
    float value = warp_reduce<op>(i < n ? in[i] : identity<op>());
    if (lane == 0) warp_results[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < BLOCK / 32 ? warp_results[lane] : identity<op>();
        value = warp_reduce<op>(value);
        if (lane == 0) out[blockIdx.x] = value;
    }
}

template<Op op> const float* reduce_impl(const float* in, size_t n, float* a, float* b, Version version) {
    float* out = a;
    // At least one launch also handles empty and singleton inputs consistently.
    do {
        size_t blocks = n ? (n + BLOCK - 1) / BLOCK : 1;
        if (version == Version::Shared)
            shared_reduce<op><<<static_cast<unsigned>(blocks), BLOCK>>>(in, out, n);
        else shuffle_reduce<op><<<static_cast<unsigned>(blocks), BLOCK>>>(in, out, n);
        CUDA_CHECK(cudaGetLastError());
        in = out;
        out = out == a ? b : a;
        n = blocks;
    } while (n > 1);
    return in;
}
const float* launch_reduce(const float* in, size_t n, float* a, float* b, Op op, Version version) {
    switch (op) {
        case Op::Sum: return reduce_impl<Op::Sum>(in, n, a, b, version);
        case Op::Max: return reduce_impl<Op::Max>(in, n, a, b, version);
        case Op::Min: return reduce_impl<Op::Min>(in, n, a, b, version);
    }
    throw std::runtime_error("Invalid reduction operation");
}
