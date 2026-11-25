#include "../WhaleGPU-Phase34/register.cuh"
#include "../WhaleGPU-Phase1/phase3/src/reference.hpp"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>

namespace {
constexpr int m = 513, n = 769, k = 257, launches = 10;
constexpr cudaStream_t stream = nullptr;  // Same thread and default stream as launch_register.

std::string sha256(const std::vector<float>& data) {
    BCRYPT_ALG_HANDLE alg;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0)
        throw std::runtime_error("BCryptOpenAlgorithmProvider failed");
    unsigned char hash[32];
    auto status = BCryptHash(alg, nullptr, 0, (PUCHAR)data.data(),
                             static_cast<ULONG>(data.size()*sizeof(float)), hash, sizeof(hash));
    BCryptCloseAlgorithmProvider(alg, 0);
    if (status != 0) throw std::runtime_error("BCryptHash failed");
    std::ostringstream hex;
    for (auto byte : hash) hex << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return hex.str();
}

void upload_checked(const std::vector<float>& src, float* dst) {
    auto bytes = src.size()*sizeof(float);
    std::vector<float> back(src.size());
    CUDA_CHECK(cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(back.data(), src.data(), bytes)) throw std::runtime_error("input roundtrip mismatch");
}

struct Target {
    const char* name;
    RegisterVariant variant;
};

void metric(double value) {
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}

void check(const Target& target, const char* phase, const float* a, const float* b,
           float* c, const Reference& ref) {
    std::vector<float> out(size_t(m)*n);
    auto bytes = out.size()*sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(c, 0xff, bytes, stream));
    target.variant.launch(a,b,c,m,n,k);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(out.data(), c, bytes, cudaMemcpyDeviceToHost));
    auto errors = compare(out,ref,k,false);
    std::cout << "{\"type\":\"check\",\"phase\":" << std::quoted(phase)
              << ",\"kernel_id\":" << std::quoted(target.name)
              << ",\"poison_byte\":255,\"elements\":" << out.size()
              << ",\"mismatches\":" << errors.mismatches << ",\"max_abs_error\":";
    metric(errors.max_error);
    std::cout << ",\"max_error_ratio\":";
    metric(errors.max_error_ratio);
    std::cout << "}" << std::endl;
    if (errors.mismatches) throw std::runtime_error("CPU reference mismatch");
}

void batch(int index, const Target& target, const float* a, const float* b, float* c,
           Event& start, Event& stop) {
    std::cout << "{\"type\":\"begin\",\"index\":" << index << ",\"kernel_id\":"
              << std::quoted(target.name) << ",\"planned\":" << launches << "}" << std::endl;
    int submitted = 0;
    bool synced = false;
    float ms = 0;
    try {
        CUDA_CHECK(cudaEventRecord(start.value, stream));
        for (; submitted < launches; ++submitted) target.variant.launch(a,b,c,m,n,k);
        CUDA_CHECK(cudaEventRecord(stop.value, stream));
        CUDA_CHECK(cudaEventSynchronize(stop.value));
        synced = true;
        CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));
    } catch (...) {
        // submitted counts only wrapper calls which returned after cudaGetLastError.
        std::cout << "{\"type\":\"batch_error\",\"index\":" << index
                  << ",\"submitted_acknowledged\":" << submitted << ",\"completed\":"
                  << (synced ? std::to_string(submitted) : "null") << "}" << std::endl;
        throw;
    }
    std::cout << "{\"type\":\"batch\",\"index\":" << index << ",\"kernel_id\":"
              << std::quoted(target.name) << ",\"stream\":0,\"stop_synchronized\":true"
              << ",\"planned\":" << launches << ",\"submitted\":" << submitted
              << ",\"completed\":" << submitted << ",\"event_total_ms\":" << ms << "}" << std::endl;
}
}

int main(int argc, char** argv) try {
    bool multi = argc==3 && std::string(argv[2])=="--multi";
    if (!multi && argc != 2) throw std::runtime_error("expected sealed order");
    std::string order(argv[1]);
    if (order.empty() || order.size()%4) throw std::runtime_error("incomplete unit");
    if (!multi && order.size()!=4) throw std::runtime_error("expected one unit");
    for (size_t i=0;i<order.size();i+=4)
        if (order.substr(i,4)!="ABBA" && order.substr(i,4)!="BAAB")
            throw std::runtime_error("invalid unit order");
    Target targets[] = {{"T32_R4_row", select_register("T32_R4_row")},
                        {"T64_R4_row", select_register("T64_R4_row")}};
    if (targets[0].variant.launch != launch_register<32,4,false> ||
        targets[1].variant.launch != launch_register<64,4,false>)
        throw std::runtime_error("dispatch mapping mismatch");
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    if (std::string(prop.name) != "NVIDIA GeForce RTX 5070 Laptop GPU")
        throw std::runtime_error("unexpected device");
    int runtime, driver;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    auto input = make_input(m,n,k,20261117u,"random");
    auto ref = cpu_reference(input,m,n,k);
    DeviceBuffer a(input.a.size()), b(input.b.size()), c(size_t(m)*n);
    upload_checked(input.a,a.ptr);
    upload_checked(input.b,b.ptr);
    std::cout << std::setprecision(17) << "{\"type\":\"ready\",\"device\":" << std::quoted(prop.name)
              << ",\"runtime\":" << runtime << ",\"driver\":" << driver << ",\"cc_major\":" << prop.major
              << ",\"cc_minor\":" << prop.minor << ",\"os_pid\":" << GetCurrentProcessId()
              << ",\"input_roundtrip\":true,\"a_sha256\":\"" << sha256(input.a)
              << "\",\"b_sha256\":\"" << sha256(input.b) << "\"}" << std::endl;
    for (auto& t : targets) check(t,"pre",a.ptr,b.ptr,c.ptr,ref);
    Event start, stop;
    for (size_t i=0;i<order.size();i+=4) {
        auto utc = std::chrono::system_clock::now().time_since_epoch();
        auto begin = std::chrono::steady_clock::now();
        for (size_t j=i;j<i+4;++j) batch(static_cast<int>(j),targets[order[j]=='B'],a.ptr,b.ptr,c.ptr,start,stop);
        auto elapsed = std::chrono::steady_clock::now()-begin;
        if (multi) std::cout << "{\"type\":\"unit\",\"index\":" << i/4
                  << ",\"start_utc_us\":" << std::chrono::duration_cast<std::chrono::microseconds>(utc).count()
                  << ",\"wall_us\":" << std::chrono::duration<double,std::micro>(elapsed).count() << "}" << std::endl;
    }
    for (auto& t : targets) check(t,"post",a.ptr,b.ptr,c.ptr,ref);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
