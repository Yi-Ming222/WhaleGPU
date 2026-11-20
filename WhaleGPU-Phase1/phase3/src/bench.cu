#include "gemm.cuh"
#include "reference.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <omp.h>

using Clock=std::chrono::steady_clock;
double ms_since(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Job {std::string id,pattern,group,role;Variant v;int m,n,k,rounds,warmups,repeats;uint32_t seed;};
std::vector<Job> read_plan(const std::string& path) {
    std::ifstream f(path);if(!f) throw std::runtime_error("Cannot read plan");
    std::string line;std::getline(f,line);std::vector<Job> result;
    while(std::getline(f,line)) {
        if(line.empty()) continue;
        std::stringstream ss(line);std::vector<std::string> x;std::string v;
        while(std::getline(ss,v,',')) x.push_back(v);
        if(x.size()!=13) throw std::runtime_error("Expected 13 columns");
        Job j{x[0],x[6],x[11],x[12],variant(x[1]),std::stoi(x[2]),std::stoi(x[3]),std::stoi(x[4]),
              std::stoi(x[8]),std::stoi(x[9]),std::stoi(x[10]),uint32_t(std::stoul(x[5]))};
        if(j.m<0||j.n<0||j.k<0||j.m>2048||j.n>2048||j.k>2048||j.rounds<1||j.rounds>1000||j.warmups<1||j.repeats<1)
            throw std::runtime_error("Plan exceeds supported limits");
        if(x[7]!="pageable_reuse") throw std::runtime_error("Only pageable_reuse supported");
        result.push_back(j);
    }
    return result;
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: whalegpu_gemm <validate|bench|profile> <plan.csv> <output-dir>");
        std::string mode=argv[1],dir=argv[3];
        if(mode!="validate"&&mode!="bench"&&mode!="profile") throw std::runtime_error("Invalid mode");
        auto plan=read_plan(argv[2]);
        omp_set_num_threads(std::min(8,omp_get_num_procs()));
        auto init=Clock::now();CUDA_CHECK(cudaSetDevice(0));CUDA_CHECK(cudaFree(nullptr));double init_ms=ms_since(init);
        cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));int driver=0,runtime=0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::ofstream meta(dir+"/hardware.json");
        meta<<"{\"gpu\":\""<<p.name<<"\",\"cc\":\""<<p.major<<'.'<<p.minor<<"\",\"sm_count\":"<<p.multiProcessorCount
            <<",\"memory_bytes\":"<<p.totalGlobalMem<<",\"l2_bytes\":"<<p.l2CacheSize<<",\"driver_api\":"<<driver
            <<",\"runtime\":"<<runtime<<",\"nvcc\":\""<<__CUDACC_VER_MAJOR__<<'.'<<__CUDACC_VER_MINOR__<<'.'<<__CUDACC_VER_BUILD__
            <<"\",\"msvc\":"<<_MSC_VER<<",\"cpu_reference_threads\":"<<omp_get_max_threads()<<",\"cuda_init_host_ms\":"<<init_ms<<"}\n";
        meta.close();if(!meta) throw std::runtime_error("Cannot write metadata");
        std::ofstream raw(dir+"/samples.csv"),checkfile(dir+"/correctness.csv"),firstfile(dir+"/first_calls.csv"),resources(dir+"/resources.csv"),refs(dir+"/references.csv");
        raw<<"id,round,kernel_ms,e2e_ms,pack_ms,mismatches,max_abs_error,max_error_ratio,correct\n";
        checkfile<<"id,elements,mismatches,max_abs_error,max_tolerance,max_error_ratio,correct\n";
        firstfile<<"id,first_call_host_ms\n";
        resources<<"id,threads,registers_per_thread,shared_bytes,theoretical_blocks_per_sm,theoretical_occupancy\n";
        refs<<"m,n,k,pattern,seed,cpu_reference_ms\n";
        for(auto f:{&raw,&checkfile,&firstfile,&resources,&refs}) {if(!*f) throw std::runtime_error("Cannot open output");*f<<std::setprecision(17);}
        Input host;Reference ref;std::vector<float> packed,output;
        int last_m=-1,last_n=-1,last_k=-1;uint32_t last_seed=0;std::string last_pattern;
        size_t done=0,failures=0;
        for(const Job& j:plan) {
            if(mode=="bench" && (!j.m||!j.n||!j.k)) throw std::runtime_error("Empty workloads are validation only");
            if(j.m!=last_m||j.n!=last_n||j.k!=last_k||j.seed!=last_seed||j.pattern!=last_pattern) {
                host=make_input(j.m,j.n,j.k,j.seed,j.pattern);packed.resize(host.b.size());output.resize(size_t(j.m)*j.n);
                auto begin=Clock::now();ref=cpu_reference(host,j.m,j.n,j.k);
                refs<<j.m<<','<<j.n<<','<<j.k<<','<<j.pattern<<','<<j.seed<<','<<ms_since(begin)<<'\n';
                last_m=j.m;last_n=j.n;last_k=j.k;last_seed=j.seed;last_pattern=j.pattern;
            }
            bool exact=j.pattern=="zero"||j.pattern=="tail"||j.pattern=="identity"||j.k==0;
            if(j.v.transposed) pack_b(host.b,packed,j.n,j.k);
            const auto& device_b=j.v.transposed?packed:host.b;
            DeviceBuffer a(host.a.size()),b(host.b.size()),c(output.size());
            if(!host.a.empty()) CUDA_CHECK(cudaMemcpy(a.ptr,host.a.data(),host.a.size()*sizeof(float),cudaMemcpyHostToDevice));
            if(!device_b.empty()) CUDA_CHECK(cudaMemcpy(b.ptr,device_b.data(),device_b.size()*sizeof(float),cudaMemcpyHostToDevice));
            // Poison C so omitted boundary writes cannot silently match initialized zeros.
            if(!output.empty()) CUDA_CHECK(cudaMemset(c.ptr,0xff,output.size()*sizeof(float)));
            auto first=Clock::now();gemm(j.v,a.ptr,b.ptr,c.ptr,j.m,j.n,j.k);CUDA_CHECK(cudaDeviceSynchronize());
            firstfile<<j.id<<','<<ms_since(first)<<'\n';
            if(!output.empty()) CUDA_CHECK(cudaMemcpy(output.data(),c.ptr,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
            auto initial=compare(output,ref,j.k,exact);
            checkfile<<j.id<<','<<output.size()<<','<<initial.mismatches<<','<<initial.max_error<<','<<initial.max_tolerance<<','<<initial.max_error_ratio<<','<<(initial.mismatches?"FAIL":"PASS")<<'\n';
            if(initial.mismatches) {++failures;continue;}
            auto r=resource(j.v);
            resources<<j.id<<','<<r.threads<<','<<r.registers<<','<<r.shared_bytes<<','<<r.active_blocks<<','<<double(r.active_blocks*r.threads)/p.maxThreadsPerMultiProcessor<<'\n';
            if(mode=="bench") {
                auto execute=[&]() {gemm(j.v,a.ptr,b.ptr,c.ptr,j.m,j.n,j.k);};
                auto end_to_end=[&]() {
                    auto begin=Clock::now();double pack_ms=0;
                    if(j.v.transposed) {auto t=Clock::now();pack_b(host.b,packed,j.n,j.k);pack_ms=ms_since(t);}
                    CUDA_CHECK(cudaMemcpy(a.ptr,host.a.data(),host.a.size()*sizeof(float),cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(b.ptr,device_b.data(),device_b.size()*sizeof(float),cudaMemcpyHostToDevice));
                    execute();
                    CUDA_CHECK(cudaMemcpy(output.data(),c.ptr,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    return std::make_pair(ms_since(begin),pack_ms);
                };
                Event begin,end;
                for(int i=0;i<j.warmups;++i) {execute();end_to_end();}
                CUDA_CHECK(cudaDeviceSynchronize());
                for(int round=0;round<j.rounds;++round) {
                    CUDA_CHECK(cudaEventRecord(begin.value));
                    for(int i=0;i<j.repeats;++i) execute();
                    CUDA_CHECK(cudaEventRecord(end.value));CUDA_CHECK(cudaEventSynchronize(end.value));
                    float kernel_ms=0;CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,begin.value,end.value));
                    CUDA_CHECK(cudaMemcpy(output.data(),c.ptr,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
                    auto x=compare(output,ref,j.k,exact);
                    auto times=end_to_end();auto y=compare(output,ref,j.k,exact);
                    size_t errors=x.mismatches+y.mismatches;
                    raw<<j.id<<','<<round<<','<<kernel_ms/j.repeats<<','<<times.first<<','<<times.second<<','<<errors<<','
                       <<std::max(x.max_error,y.max_error)<<','<<std::max(x.max_error_ratio,y.max_error_ratio)<<','<<(errors?"FAIL":"PASS")<<'\n';
                    if(errors) ++failures;
                }
            }
            ++done;
            if(done%50==0) std::cout<<"completed="<<done<<'/'<<plan.size()<<" failures="<<failures<<std::endl;
        }
        for(auto f:{&raw,&checkfile,&firstfile,&resources,&refs}) {f->close();if(!*f) throw std::runtime_error("Output flush failed");}
        std::cout<<(failures?"FAIL":"PASS")<<" mode="<<mode<<" jobs="<<plan.size()<<" completed="<<done<<" failures="<<failures<<std::endl;
        return failures?1:0;
    }catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<std::endl;return 2;}
}
