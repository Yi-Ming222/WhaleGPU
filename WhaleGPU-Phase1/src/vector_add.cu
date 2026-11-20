#include "common.cuh"

__global__ void vector_add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
void launch_add(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) return;
    vector_add_kernel<<<static_cast<unsigned>((n + BLOCK - 1) / BLOCK), BLOCK>>>(a, b, out, n);
    CUDA_CHECK(cudaGetLastError());
}
