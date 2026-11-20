#include "common.cuh"
#include "reference.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <numeric>

constexpr int WARMUP = 2, SAMPLES = 5, REPEATS = 10;

std::string quoted(const std::string& s) {
    std::string result = "\"";
    for (char c : s) { if (c == '"') result += '"'; result += c; }
    return result + '"';
}

std::vector<float> measure(const std::function<void()>& launch) {
    for (int i = 0; i < WARMUP; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    Event begin, end;
    std::vector<float> timings;
    for (int sample = 0; sample < SAMPLES; ++sample) {
        CUDA_CHECK(cudaEventRecord(begin.value));
        for (int i = 0; i < REPEATS; ++i) launch();
        CUDA_CHECK(cudaEventRecord(end.value));
        CUDA_CHECK(cudaEventSynchronize(end.value));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin.value, end.value));
        timings.push_back(ms / REPEATS);
    }
    return timings;
}

void make_input(std::vector<float>& a, std::vector<float>& b, const std::string& pattern) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (pattern == "mixed_exact") a[i] = float(int(i % 1021) - 510) / 16.0f;
        else if (pattern == "negative") a[i] = -float(1 + i % 3);
        else if (pattern == "positive") a[i] = float(1 + i % 3);
        else if (pattern == "constant") a[i] = 0.25f;
        else if (pattern == "tail_extrema") a[i] = 0.25f;
        else {
            // Deterministic 32-bit LCG, no library-dependent random distribution.
            unsigned x = unsigned(i) * 1664525u + 1013904223u;
            a[i] = float(int((x >> 8) % 20001) - 10000) / 10000.0f;
        }
        b[i] = float(int(i % 37) - 18) / 8.0f;
    }
    if (pattern == "tail_extrema" && !a.empty()) {
        a.back() = 32768.0f;
        if (a.size() > 1) a[a.size() - 2] = -32768.0f;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: whalegpu_tests <smoke|vector|shared|shuffle|all> <output-directory>");
        std::string stage = argv[1], dir = argv[2];
        if (stage != "smoke" && stage != "vector" && stage != "shared" && stage != "shuffle" && stage != "all")
            throw std::runtime_error("Unknown test stage");
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::ofstream hardware(dir + "/hardware.txt");
        if (!hardware) throw std::runtime_error("Cannot write hardware.txt");
        hardware << "gpu=" << p.name << "\ncompute_capability=" << p.major << '.' << p.minor
                 << "\nglobal_memory_bytes=" << p.totalGlobalMem << "\nSM_count=" << p.multiProcessorCount
                 << "\nwarp_size=" << p.warpSize << "\ndriver_api_version=" << driver
                 << "\nruntime_version=" << runtime << "\nnvcc=" << __CUDACC_VER_MAJOR__ << '.'
                 << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
                 << "\nmsvc=" << _MSC_VER << "\ncompiled_arch=sm_120\nblock_size=" << BLOCK
                 << "\nwarmups=" << WARMUP << "\nsamples=" << SAMPLES << "\nrepeats_per_sample=" << REPEATS << '\n';
        hardware.close();
        if (!hardware) throw std::runtime_error("Failed writing hardware metadata");
        std::cout << "GPU: " << p.name << "; CC " << p.major << '.' << p.minor << '\n';
        if (stage == "smoke") {
            DeviceBuffer a(1), b(1), c(1);
            float x = 2, y = 3, z = 0;
            CUDA_CHECK(cudaMemcpy(a.ptr, &x, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.ptr, &y, sizeof(float), cudaMemcpyHostToDevice));
            launch_add(a.ptr, b.ptr, c.ptr, 1);
            CUDA_CHECK(cudaMemcpy(&z, c.ptr, sizeof(float), cudaMemcpyDeviceToHost));
            if (z != 5) throw std::runtime_error("Smoke mismatch");
            std::cout << "PASS smoke: nvcc build, device allocation, launch, copy, 2+3=5\n";
            return 0;
        }
        std::ofstream csv(dir + "/results.csv"), samples(dir + "/timings.csv");
        if (!csv || !samples) throw std::runtime_error("Cannot write CSV output");
        csv << "operator,version,n,pattern,mean_ms,min_ms,max_ms,correct,mismatches,max_abs_error,tolerance,expected,actual,kernel_passes,gpu,cc,global_memory_bytes,driver_api,runtime,block_size,warmups,samples,repeats\n";
        samples << "operator,version,n,pattern,sample,kernel_ms\n";
        csv << std::setprecision(17);
        samples << std::setprecision(9);
        const std::vector<size_t> sizes = {0,1,2,3,31,32,33,255,256,257,513,1024,65537,1048576,1048579,2097155};
        const std::vector<std::string> patterns = {"mixed_exact","negative","positive","constant","tail_extrema","fractional"};
        size_t cases = 0, failed = 0;
        auto record = [&](const char* op, const char* version, size_t n, const std::string& pattern,
                          const std::vector<float>& times, size_t mismatches, double error,
                          double tolerance, double expected, double actual, int passes) {
            double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            csv << op << ',' << version << ',' << n << ',' << pattern << ',' << mean << ','
                << *std::min_element(times.begin(), times.end()) << ',' << *std::max_element(times.begin(), times.end())
                << ',' << (mismatches ? "FAIL" : "PASS") << ',' << mismatches << ',' << error << ',' << tolerance
                << ',' << expected << ',' << actual << ',' << passes << ',' << quoted(p.name) << ','
                << p.major << '.' << p.minor << ',' << p.totalGlobalMem << ',' << driver << ',' << runtime
                << ',' << BLOCK << ',' << WARMUP << ',' << SAMPLES << ',' << REPEATS << '\n';
            for (size_t i = 0; i < times.size(); ++i)
                samples << op << ',' << version << ',' << n << ',' << pattern << ',' << i << ',' << times[i] << '\n';
            ++cases;
            if (mismatches) { ++failed; std::cerr << "FAIL " << op << '/' << version << " n=" << n << " pattern=" << pattern << " expected=" << expected << " actual=" << actual << '\n'; }
        };
        for (size_t n : sizes) {
            for (const auto& pattern : patterns) {
                std::vector<float> a(n), b(n), out(n);
                make_input(a, b, pattern);
                DeviceBuffer da(n), db(n), dc(n), scratch_a((n + BLOCK - 1) / BLOCK), scratch_b((n + BLOCK - 1) / BLOCK);
                if (n) {
                    CUDA_CHECK(cudaMemcpy(da.ptr, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(db.ptr, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
                }
                if (stage == "vector" || stage == "all") {
                    auto times = n ? measure([&] { launch_add(da.ptr, db.ptr, dc.ptr, n); }) : std::vector<float>(SAMPLES, 0);
                    if (n) CUDA_CHECK(cudaMemcpy(out.data(), dc.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
                    auto expected = reference_add(a, b);
                    size_t mismatches = 0; double error = 0;
                    for (size_t i = 0; i < n; ++i) {
                        if (out[i] != expected[i]) ++mismatches;
                        error = std::max(error, std::abs(double(out[i]) - expected[i]));
                    }
                    record("vector_add", "native", n, pattern, times, mismatches, error, 0, 0, 0, n ? 1 : 0);
                }
                if (stage == "vector") continue;
                for (Version version : {Version::Shared, Version::Shuffle}) {
                    if (stage == "shared" && version != Version::Shared) continue;
                    if (stage == "shuffle" && version != Version::Shuffle) continue;
                    for (Op op : {Op::Sum, Op::Max, Op::Min}) {
                        const float* result = nullptr;
                        auto times = measure([&] { result = launch_reduce(da.ptr, n, scratch_a.ptr, scratch_b.ptr, op, version); });
                        float actual = 0;
                        CUDA_CHECK(cudaMemcpy(&actual, result, sizeof(float), cudaMemcpyDeviceToHost));
                        double expected = reference_reduce(a, op), tolerance = 0;
                        int passes = 0;
                        size_t remaining = n;
                        do { remaining = remaining ? (remaining + BLOCK - 1) / BLOCK : 1; ++passes; } while (remaining > 1);
                        if (op == Op::Sum && pattern == "fractional") {
                            double l1 = 0;
                            for (float x : a) l1 += std::abs(double(x));
                            // Tree depth upper bound: eight additions per kernel pass.
                            double k_eps = passes * 8 * std::numeric_limits<float>::epsilon();
                            tolerance = k_eps / (1 - k_eps) * l1;
                        }
                        double error = actual == expected ? 0 : std::abs(double(actual) - expected);
                        bool correct = actual == expected || (std::isfinite(actual) && std::isfinite(expected) && error <= tolerance);
                        record(op == Op::Sum ? "sum" : op == Op::Max ? "max" : "min",
                               version == Version::Shared ? "shared" : "shuffle", n, pattern,
                               times, correct ? 0 : 1, error, tolerance, expected, actual, passes);
                    }
                }
            }
            std::cout << "Completed n=" << n << "; cases=" << cases << "; failed=" << failed << std::endl;
        }
        csv.close(); samples.close();
        if (!csv || !samples) throw std::runtime_error("Failed flushing CSV output");
        std::cout << (failed ? "FAIL" : "PASS") << " stage=" << stage << " cases=" << cases << " failed=" << failed << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}
