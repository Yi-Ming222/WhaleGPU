#pragma once
#include "../WhaleGPU-Phase1/include/common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <filesystem>
using Clock43=std::chrono::steady_clock;
struct Logs{
    bool bench; std::mt19937 rng; std::ofstream raw,checks,res,allocation,first;
    Logs(std::string mode,std::string path,unsigned seed):bench(mode=="bench"),rng(seed),
      raw(path+"/samples.csv"),checks(path+"/checks.csv"),res(path+"/resources.csv"),allocation(path+"/allocation.csv"),first(path+"/first_call.csv"){
        for(auto f:{&raw,&checks,&res,&allocation,&first}){f->exceptions(std::ios::badbit|std::ios::failbit);*f<<std::setprecision(14);}
        auto begin=Clock43::now();CUDA_CHECK(cudaFree(nullptr));
        double init=std::chrono::duration<double,std::milli>(Clock43::now()-begin).count();
        cudaDeviceProp device{};CUDA_CHECK(cudaGetDeviceProperties(&device,0));
        std::ofstream info(path+"/device.csv");
        info<<"gpu,major,minor,sm,l2_bytes,warp,max_threads_per_sm,init_host_ms\n"
            <<device.name<<','<<device.major<<','<<device.minor<<','<<device.multiProcessorCount<<','
            <<device.l2CacheSize<<','<<device.warpSize<<','<<device.maxThreadsPerMultiProcessor<<','<<init<<'\n';
        raw<<"case,variant,pair,order,kernel_ms,e2e_ms\n";
        checks<<"case,variant,n,max_abs,rmse\n";
        res<<"case,variant,kernel,grid_x,grid_y,threads,regs,static_shared,dynamic_shared,blocks_per_sm,theoretical_occupancy\n";
        first<<"case,variant,post_setup_host_launch_to_sync_ms\n";
        allocation<<"bytes,malloc_free_host_ms\n";
    }
};
template<class K>void resource43(Logs& l,std::string key,std::string v,std::string name,K kernel,dim3 grid,int threads){
    cudaFuncAttributes a{};cudaDeviceProp p{};int blocks;
    CUDA_CHECK(cudaGetDeviceProperties(&p,0));CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,0));
    l.res<<key<<','<<v<<','<<name<<','<<grid.x<<','<<grid.y<<','<<threads<<','<<a.numRegs<<','<<a.sharedSizeBytes<<",0,"<<blocks<<','<<double(blocks*((threads+p.warpSize-1)/p.warpSize))/(p.maxThreadsPerMultiProcessor/p.warpSize)<<'\n';
}
template<class T>void transfer43(T* dst,const T* src,size_t n,cudaMemcpyKind way){if(n)CUDA_CHECK(cudaMemcpy(dst,src,sizeof(T)*n,way));}
__global__ void empty43(){}
__global__ void affine43(const float* src,float* dst,size_t n,float a,float b){
    size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<n)dst[i]=fmaf(a,src[i],b);
}
__global__ void copy43(const float* src,float* dst,size_t n){
    size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<n)dst[i]=src[i];
}
void post43(int id,float* out,const float* fixed,size_t n,float a,float b){
    if(!n)return;
    int grid=int((n+255)/256);
    if(id==0)affine43<<<grid,256>>>(out,out,n,a,b);
    if(id==2)empty43<<<1,1>>>();
    if(id==4)affine43<<<grid,256>>>(fixed,out,n,a,b);
    if(id==5)copy43<<<grid,256>>>(fixed,out,n);
    CUDA_CHECK(cudaGetLastError());
}
void post_resources(Logs& l,std::string key,std::vector<std::string> labels,size_t n){
    for(int id:{0,4})resource43(l,key,labels[id],"affine43",affine43,(n+255)/256,256);
    resource43(l,key,labels[2],"empty43",empty43,1,1);
    resource43(l,key,labels[5],"copy43",copy43,(n+255)/256,256);
}
struct Error43{double max=0,squares=0;size_t count=0;
    void add(double actual,double expected,double bound){
        double e=std::abs(actual-expected);
        if(!std::isfinite(actual)||!(e<=bound))throw std::runtime_error("mismatch at "+std::to_string(count)+" expected="+std::to_string(expected)+" actual="+std::to_string(actual));
        max=std::max(max,e);squares+=e*e;++count;
    }
    void save(Logs& l,std::string key,std::string v){
        l.checks<<key<<','<<v<<','<<count<<','<<max<<',';
        if(count)l.checks<<sqrt(squares/count);else l.checks<<"NA";l.checks<<'\n';
    }
};
inline void bits43(float a,float b){if(memcmp(&a,&b,4))throw std::runtime_error("paired FP32 bit mismatch");}
inline double rounding43(double value){return 1.2e-7*std::abs(value)+1.5e-45;}
const std::vector<std::string> pair_names={"split","fused","fused_empty","producer","affine_only","copy_only"};
// Each observation is a complete chain; initialization/poison and checks are outside timers.
template<class Prepare,class Op,class Down,class Check,class Poison>
void measure43(Logs& l,std::string key,const std::vector<std::string>& labels,Prepare upload,Op op,Down down,Check verify,Poison poison){
    std::vector<int> order(labels.size());std::iota(order.begin(),order.end(),0);
    for(int id:order){
        upload(id);poison();CUDA_CHECK(cudaDeviceSynchronize());
        auto begin=Clock43::now();op(id);CUDA_CHECK(cudaDeviceSynchronize());
        l.first<<key<<','<<labels[id]<<','<<std::chrono::duration<double,std::milli>(Clock43::now()-begin).count()<<'\n';
        down();verify(id).save(l,key,labels[id]);
    }
    if(!l.bench)return;
    Event start,stop;
    for(int id:order)for(int w=0;w<10;++w){upload(id);op(id);down();}
    for(int r=0;r<21;++r){
        if(r%2==0)std::shuffle(order.begin(),order.end(),l.rng);else std::reverse(order.begin(),order.end());
        int index=0;
        for(int id:order){
            upload(id);CUDA_CHECK(cudaEventRecord(start.value));
            for(int j=0;j<10;++j)op(id);
            CUDA_CHECK(cudaEventRecord(stop.value));CUDA_CHECK(cudaEventSynchronize(stop.value));
            float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));down();verify(id);
            auto t=Clock43::now();upload(id);op(id);down();double wall=std::chrono::duration<double,std::milli>(Clock43::now()-t).count();
            verify(id);l.raw<<key<<','<<labels[id]<<','<<r<<','<<index++<<','<<ms/10<<','<<wall<<'\n';
        }
    }
}
void allocation43(Logs& l,size_t bytes){
    for(int i=0;i<11;++i){void* p=nullptr;auto t=Clock43::now();CUDA_CHECK(cudaMalloc(&p,bytes));CUDA_CHECK(cudaFree(p));
        l.allocation<<bytes<<','<<std::chrono::duration<double,std::milli>(Clock43::now()-t).count()<<'\n';}
}
