#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void check_cuda(cudaError_t e, const char* expr) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(e));
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr)
constexpr int BLOCK = 256;
enum class Op { Sum, Max, Min };
enum class Version { Shared, Shuffle };

struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, (n ? n : 1) * sizeof(float))); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
};
void launch_add(const float* a, const float* b, float* out, size_t n);
// Scratch buffers require at least max(1, ceil(n / BLOCK)) floats each.
// n == 0 produces the operation's identity; all launches use the default stream.
const float* launch_reduce(const float* in, size_t n, float* scratch_a,
                           float* scratch_b, Op op, Version version);
