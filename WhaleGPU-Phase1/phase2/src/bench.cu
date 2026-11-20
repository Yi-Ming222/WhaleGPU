#include "kernels.cuh"
#include "reference.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <memory>

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point begin) { return std::chrono::duration<double,std::milli>(Clock::now()-begin).count(); }
struct Job {
    std::string id, operation, algorithm, allocation, pattern;
    size_t n;
    Config c;
    int rounds, warmups, repeats;
    Op op;
    bool baseline;
};
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> v; std::stringstream s(line); std::string x;
    while (std::getline(s,x,',')) v.push_back(x);
    return v;
}
std::vector<Job> read_jobs(const char* path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("Cannot read plan");
    std::string line; std::getline(f,line); std::vector<Job> jobs;
    while (std::getline(f,line)) {
        if (line.empty()) continue;
        auto v = split(line); if (v.size()!=14) throw std::runtime_error("Plan requires 14 columns");
        Job j; j.id=v[0]; j.operation=v[1]; j.n=std::stoull(v[2]); j.algorithm=v[3];
        j.c.threads=std::stoi(v[4]); j.c.items=std::stoi(v[5]); j.c.grid=std::stoi(v[6]);
        j.c.contiguous=v[7]=="contiguous"; j.allocation=v[8]; j.pattern=v[9];
        j.rounds=std::stoi(v[10]); j.warmups=std::stoi(v[11]); j.repeats=std::stoi(v[12]);
        j.baseline=j.algorithm.rfind("phase1_",0)==0;
        j.c.shuffle=j.algorithm=="shuffle" || j.algorithm=="phase1_shuffle";
        if (j.operation!="sum" && j.operation!="max" && j.operation!="min") throw std::runtime_error("Invalid op");
        j.op=j.operation=="sum"?Op::Sum:j.operation=="max"?Op::Max:Op::Min;
        if (j.algorithm!="shared" && j.algorithm!="shuffle" && j.algorithm!="phase1_shared" && j.algorithm!="phase1_shuffle") throw std::runtime_error("Invalid algorithm");
        if (j.c.threads!=128 && j.c.threads!=256 && j.c.threads!=512) throw std::runtime_error("Invalid block size");
        if (j.c.items!=1 && j.c.items!=4 && j.c.items!=8) throw std::runtime_error("Invalid items");
        if (j.c.grid<0 || j.n>size_t(1)<<29 || j.rounds<1 || j.rounds>1000 || j.warmups<1 || j.repeats<1) throw std::runtime_error("Invalid plan limits");
        if (v[7]!="striped" && v[7]!="contiguous") throw std::runtime_error("Invalid layout");
        if (j.allocation!="pageable_reuse" && j.allocation!="pinned_reuse" && j.allocation!="pageable_fresh") throw std::runtime_error("Invalid allocation");
        if (j.baseline && (j.c.threads!=256 || j.c.items!=1 || j.c.grid!=0 || j.c.contiguous)) throw std::runtime_error("Baseline parameters are fixed");
        jobs.push_back(j);
    }
    return jobs;
}
void fill(std::vector<float>& a, const std::string& pattern) {
    for (size_t i=0;i<a.size();++i) {
        if (pattern=="mixed_exact") a[i]=float(int(i%127)-63); // bounded, exact integer sums
        else if (pattern=="negative") a[i]=-float(1+i%3);
        else if (pattern=="positive") a[i]=float(1+i%3);
        else if (pattern=="constant") a[i]=0.25f;
        else if (pattern=="tail_extrema") a[i]=0;
        else if (pattern=="fractional") {
            unsigned x=unsigned(i)*1664525u+1013904223u;
            a[i]=float(int((x>>8)%20001)-10000)/10000.f;
        } else throw std::runtime_error("Invalid pattern");
    }
    if (pattern=="tail_extrema" && !a.empty()) {
        a.back()=32768; if(a.size()>1) a[a.size()-2]=-32768;
    }
}
struct HostMemory {
    float* input=nullptr; float* output=nullptr;
    explicit HostMemory(size_t n) {
        CUDA_CHECK(cudaMallocHost(&input,std::max(size_t(1),n)*sizeof(float)));
        try { CUDA_CHECK(cudaMallocHost(&output,sizeof(float))); } catch(...) { cudaFreeHost(input); throw; }
    }
    ~HostMemory(){cudaFreeHost(input); cudaFreeHost(output);}
};
const float* launch(const Job& j, const float* in, float* a, float* b) {
    return j.baseline ? launch_reduce(in,j.n,a,b,j.op,j.c.shuffle?Version::Shuffle:Version::Shared)
                      : reduce_tuned(in,j.n,a,b,j.op,j.c);
}
struct Shape { size_t first=0, traffic=0; int passes=0; double additions=0; };
Shape shape(const Job& j) {
    Shape s; size_t n=j.n;
    do {
        size_t blocks=j.baseline?std::max(size_t(1),(n+255)/256):grid_count(n,j.c);
        if (!s.passes) s.first=blocks;
        s.traffic+=(n+blocks)*sizeof(float); ++s.passes;
        size_t tile=size_t(j.c.threads)*j.c.items;
        size_t tiles=n ? (n+blocks*tile-1)/(blocks*tile) : 0;
        s.additions+=double(tiles*j.c.items)+std::log2(j.c.threads);
        n=blocks;
    }while(n>1);
    return s;
}
struct Check { double expected, error, tolerance; bool ok; };
Check check(float actual,double expected,double tolerance) {
    double err=actual==expected?0:std::abs(double(actual)-expected);
    return {expected,err,tolerance,actual==expected || (std::isfinite(actual)&&std::isfinite(expected)&&err<=tolerance)};
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: whalegpu_bench <validate|bench|profile> <plan.csv> <output-dir>");
        std::string mode=argv[1],dir=argv[3];
        if(mode!="validate" && mode!="bench" && mode!="profile") throw std::runtime_error("Invalid mode");
        auto jobs=read_jobs(argv[2]);
        auto init=Clock::now(); CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaFree(nullptr));
        double init_ms=elapsed(init); cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
        int driver=0,runtime=0; CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::ofstream meta(dir+"/hardware.json");
        meta << "{\"gpu\":\"" << p.name << "\",\"cc\":\"" << p.major << '.' << p.minor
             << "\",\"sm_count\":" << p.multiProcessorCount << ",\"l2_bytes\":" << p.l2CacheSize
             << ",\"memory_bytes\":" << p.totalGlobalMem << ",\"max_threads_per_sm\":" << p.maxThreadsPerMultiProcessor
             << ",\"driver_api\":" << driver << ",\"runtime\":" << runtime
             << ",\"nvcc\":\"" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
             << "\",\"msvc\":" << _MSC_VER << ",\"cuda_init_host_ms\":" << init_ms << "}\n";
        meta.close(); if(!meta) throw std::runtime_error("Metadata write failure");
        std::ofstream raw(dir+"/samples.csv"), checks(dir+"/correctness.csv"), cold(dir+"/first_calls.csv"), resource(dir+"/resources.csv");
        raw<<"id,round,kernel_ms,e2e_ms,correct\n";
        checks<<"id,expected,actual,abs_error,tolerance,correct\n";
        cold<<"id,first_call_host_ms\n";
        resource<<"id,first_grid,passes,logical_traffic_bytes,registers_per_thread,static_shared_bytes,dynamic_shared_bytes,theoretical_blocks_per_sm,theoretical_occupancy\n";
        for(auto* f:{&raw,&checks,&cold,&resource}) { if(!*f) throw std::runtime_error("CSV open failure"); *f<<std::setprecision(17); }
        size_t done=0,failures=0;
        // Cache CPU references by current data; host preparation is outside all timers.
        std::vector<float> host; std::string cached_pattern; size_t cached_n=size_t(-1);
        double expected_sum=0,expected_max=0,expected_min=0,l1=0;
        for(const auto& j:jobs) {
            if(cached_n!=j.n || cached_pattern!=j.pattern) {
                host.resize(j.n); fill(host,j.pattern); l1=0;
                expected_sum=reference_reduce(host,Op::Sum); expected_max=reference_reduce(host,Op::Max); expected_min=reference_reduce(host,Op::Min);
                for(float x:host) l1+=std::abs(double(x)); cached_n=j.n; cached_pattern=j.pattern;
            }
            double expected=j.op==Op::Sum?expected_sum:j.op==Op::Max?expected_max:expected_min;
            Shape sh=shape(j); double tolerance=0;
            if(j.op==Op::Sum && j.pattern=="fractional") {
                double ke=sh.additions*std::numeric_limits<float>::epsilon();
                if(ke>=1) throw std::runtime_error("Error bound not meaningful");
                tolerance=ke/(1-ke)*l1;
            }
            DeviceBuffer input(j.n),a(sh.first),b(sh.first);
            const float* host_input=host.data(); float output=0; float* host_output=&output;
            std::unique_ptr<HostMemory> pinned;
            if(j.allocation=="pinned_reuse") {
                pinned=std::make_unique<HostMemory>(j.n);
                std::copy(host.begin(),host.end(),pinned->input); host_input=pinned->input; host_output=pinned->output;
            }
            if(j.n) CUDA_CHECK(cudaMemcpy(input.ptr,host_input,j.n*sizeof(float),cudaMemcpyHostToDevice));
            auto first=Clock::now(); const float* result=launch(j,input.ptr,a.ptr,b.ptr); CUDA_CHECK(cudaDeviceSynchronize());
            cold<<j.id<<','<<elapsed(first)<<'\n';
            CUDA_CHECK(cudaMemcpy(&output,result,sizeof(float),cudaMemcpyDeviceToHost));
            Check c=check(output,expected,tolerance);
            checks<<j.id<<','<<expected<<','<<output<<','<<c.error<<','<<tolerance<<','<<(c.ok?"PASS":"FAIL")<<'\n';
            if(!c.ok) {++failures; continue;}
            Resources r;
            if(!j.baseline) r=get_resources(j.op,j.c);
            resource<<j.id<<','<<sh.first<<','<<sh.passes<<','<<sh.traffic<<',';
            if(j.baseline) resource<<"NA,NA,NA,NA,NA\n";
            else resource<<r.registers<<','<<r.static_shared<<','<<r.dynamic_shared<<','<<r.active_blocks<<','<<double(r.active_blocks*j.c.threads)/p.maxThreadsPerMultiProcessor<<'\n';
            if(mode=="validate" || mode=="profile") { ++done; continue; }
            Event start,stop;
            // Warm both measurement paths, including transfers/allocation strategy.
            auto end_to_end=[&]() {
                auto begin=Clock::now();
                if(j.allocation=="pageable_fresh") {
                    { DeviceBuffer fresh(j.n),x(sh.first),y(sh.first);
                      if(j.n) CUDA_CHECK(cudaMemcpy(fresh.ptr,host_input,j.n*sizeof(float),cudaMemcpyHostToDevice));
                      const float* out=launch(j,fresh.ptr,x.ptr,y.ptr);
                      CUDA_CHECK(cudaMemcpy(host_output,out,sizeof(float),cudaMemcpyDeviceToHost));
                      CUDA_CHECK(cudaDeviceSynchronize()); }
                } else {
                    if(j.n) CUDA_CHECK(cudaMemcpy(input.ptr,host_input,j.n*sizeof(float),cudaMemcpyHostToDevice));
                    const float* out=launch(j,input.ptr,a.ptr,b.ptr);
                    CUDA_CHECK(cudaMemcpy(host_output,out,sizeof(float),cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                return elapsed(begin);
            };
            for(int w=0;w<j.warmups;++w) { launch(j,input.ptr,a.ptr,b.ptr); end_to_end(); }
            CUDA_CHECK(cudaDeviceSynchronize());
            for(int round=0;round<j.rounds;++round) {
                CUDA_CHECK(cudaEventRecord(start.value));
                for(int i=0;i<j.repeats;++i) result=launch(j,input.ptr,a.ptr,b.ptr);
                CUDA_CHECK(cudaEventRecord(stop.value)); CUDA_CHECK(cudaEventSynchronize(stop.value));
                float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));
                CUDA_CHECK(cudaMemcpy(&output,result,sizeof(float),cudaMemcpyDeviceToHost));
                bool ok=check(output,expected,tolerance).ok;
                double e2e=end_to_end(); ok=ok && check(*host_output,expected,tolerance).ok;
                raw<<j.id<<','<<round<<','<<ms/j.repeats<<','<<e2e<<','<<(ok?"PASS":"FAIL")<<'\n';
                if(!ok) ++failures;
            }
            ++done;
            if(done%100==0) std::cout<<"completed="<<done<<"/"<<jobs.size()<<" failures="<<failures<<std::endl;
        }
        for(auto* f:{&raw,&checks,&cold,&resource}) { f->close(); if(!*f) throw std::runtime_error("CSV flush failure"); }
        std::cout<<(failures?"FAIL":"PASS")<<" mode="<<mode<<" jobs="<<jobs.size()<<" completed="<<done<<" failures="<<failures<<std::endl;
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<std::endl; return 2; }
}
