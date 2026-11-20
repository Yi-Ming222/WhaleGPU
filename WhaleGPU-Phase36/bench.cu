#include "../WhaleGPU-Phase1/include/common.cuh"
#include "../WhaleGPU-Phase1/phase3/src/reference.hpp"
#include "kernels.cuh"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstring>
using Clock=std::chrono::steady_clock;
const char* kernel_name[]={"warp16","block32","pipeline32"};
const char* flow_name[]={"cpu","gpu","prepacked"};

template<class T> void pack(const std::vector<float>& src,std::vector<T>& dst){
    std::transform(src.begin(),src.end(),dst.begin(),[](float x){return T(x);});
}
template<class T> void transfer(T* dst,const T* src,size_t count,cudaMemcpyKind kind){
    if(count) CUDA_CHECK(cudaMemcpy(dst,src,count*sizeof(T),kind));
}
struct ErrorStats {
    double absolute=0,relative=0,squares=0;
    size_t bins[5]={};
    void add(double error,double ref){
        absolute=std::max(absolute,error);
        relative=std::max(relative,error/std::max(std::abs(ref),1e-7));
        squares+=error*error;
        ++bins[error<=1e-6?0:error<=1e-4?1:error<=1e-3?2:error<=1e-2?3:4];
    }
    void write(std::ostream& os,size_t count) const {
        os<<','<<absolute<<','<<relative<<','<<std::sqrt(squares/count);
        for(auto v:bins) os<<','<<v;
    }
};
template<class T>
void shape(int m,int n,int k,int seed,bool bench,std::mt19937& rng,std::ostream& raw,std::ostream& errors){
    auto in=make_input(m,n,k,seed,"random");
    auto original=cpu_reference(in,m,n,k);
    std::vector<T> qa(in.a.size()),qb(in.b.size());
    pack(in.a,qa);pack(in.b,qb);
    Input rounded=in;
    std::copy(qa.begin(),qa.end(),rounded.a.begin());std::copy(qb.begin(),qb.end(),rounded.b.begin());
    auto ref=cpu_reference(rounded,m,n,k);
    std::vector<float> out(size_t(m)*n);
    DeviceBuffer fa(qa.size()),fb(qb.size()),a(qa.size()),b(qb.size()),c(out.size());
    T* da=reinterpret_cast<T*>(a.ptr);T* db=reinterpret_cast<T*>(b.ptr);
    auto gpu_pack=[&](const std::vector<float>& src,float* staging,T* dest){
        transfer(staging,src.data(),src.size(),cudaMemcpyHostToDevice);
        if(!src.empty()){
            convert_input<<<(src.size()+255)/256,256>>>(staging,dest,int(src.size()));
            CUDA_CHECK(cudaGetLastError());
        }
    };
    auto verify_bits=[&](const std::vector<float>& src,const std::vector<T>& expected,float* staging,T* dest){
        gpu_pack(src,staging,dest);
        std::vector<T> actual(expected.size());
        transfer(actual.data(),dest,actual.size(),cudaMemcpyDeviceToHost);
        if(!actual.empty() && std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(T)))
            throw std::runtime_error("conversion bit mismatch");
    };
    verify_bits(in.a,qa,fa.ptr,da);verify_bits(in.b,qb,fb.ptr,db);
    std::vector<int> jobs={0,1,2,3,4,5,6,7,8};
    std::shuffle(jobs.begin(),jobs.end(),rng);
    for(int job:jobs){
        int kernel=job/3,flow=job%3;
        auto prepare=[&]{
            if(flow==1){gpu_pack(in.a,fa.ptr,da);gpu_pack(in.b,fb.ptr,db);return;}
            if(flow==0){pack(in.a,qa);pack(in.b,qb);}
            transfer(da,qa.data(),qa.size(),cudaMemcpyHostToDevice);
            transfer(db,qb.data(),qb.size(),cudaMemcpyHostToDevice);
        };
        auto launch=[&]{
            dim3 grid((n+31)/32,(m+31)/32);
            if(kernel==0) tensor_gemm<T><<<dim3((n+15)/16,(m+15)/16),32>>>(da,db,c.ptr,m,n,k);
            else if(kernel==1) block_gemm<T,false><<<grid,128>>>(da,db,c.ptr,m,n,k);
            else block_gemm<T,true><<<grid,128>>>(da,db,c.ptr,m,n,k);
            CUDA_CHECK(cudaGetLastError());
        };
        auto download=[&]{transfer(out.data(),c.ptr,out.size(),cudaMemcpyDeviceToHost);};
        auto validate=[&](bool save){
            ErrorStats accumulation,total;
            double u=std::is_same<T,half>::value?1./2048:1./256;
            double eps=k*double(std::numeric_limits<float>::epsilon());
            for(size_t i=0;i<out.size();++i){
                double err=std::abs(out[i]-ref.value[i]),full=std::abs(out[i]-original.value[i]);
                double limit=k?eps/(1-eps)*ref.absolute[i]+1e-7:0;
                if(!(err<=limit && full<=limit+(2*u+u*u)*original.absolute[i]))
                    throw std::runtime_error("GEMM mismatch");
                if(save){accumulation.add(err,ref.value[i]);total.add(full,original.value[i]);}
            }
            if(save){
                for(auto label:{"rounded","original"}){
                    errors<<(std::is_same<T,half>::value?"FP16":"BF16")<<','<<kernel_name[kernel]<<','<<flow_name[flow]<<','<<m<<','<<n<<','<<k<<','<<seed<<','<<label<<','<<out.size();
                    (label[0]=='r'?accumulation:total).write(errors,out.size());errors<<'\n';
                }
            }
        };
        prepare();CUDA_CHECK(cudaMemset(c.ptr,0xff,out.size()*4));launch();download();validate(true);
        if(!bench) continue;
        for(int w=0;w<10;++w){prepare();launch();download();}
        Event start,end;
        for(int round=0;round<15;++round){
            CUDA_CHECK(cudaEventRecord(start.value));
            for(int r=0;r<10;++r) launch();
            CUDA_CHECK(cudaEventRecord(end.value));CUDA_CHECK(cudaEventSynchronize(end.value));
            float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,end.value));
            download();validate(false);
            auto begin=Clock::now();prepare();launch();download();
            double e2e=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            validate(false);
            raw<<(std::is_same<T,half>::value?"FP16":"BF16")<<','<<kernel_name[kernel]<<','<<flow_name[flow]<<','<<m<<','<<n<<','<<k<<','<<seed<<','<<round<<','<<ms/10<<','<<e2e<<'\n';
        }
    }
}
int main(int argc,char** argv) try{
    if(argc!=3)return 2;
    std::mt19937 rng(std::stoul(argv[2]));
    std::ofstream raw(std::string(argv[1])+"-samples.csv"),errors(std::string(argv[1])+"-errors.csv");
    raw<<std::setprecision(12)<<"dtype,kernel,flow,m,n,k,seed,round,kernel_ms,e2e_ms\n";
    errors<<std::setprecision(12)<<"dtype,kernel,flow,m,n,k,seed,reference,count,max_abs,max_rel,rmse,bin0,bin1,bin2,bin3,bin4\n";
    int shapes[][3]={{1,1,1},{17,31,0},{31,33,15},{33,17,65},{64,64,64},{257,193,129},{512,512,512},{1024,1024,1024}};
    for(auto& dims:shapes)for(int seed:{17,65537}){
        if(seed!=17 && dims[0]>=512)continue;
        bool bench=seed==17 && dims[0]>=64;
        shape<half>(dims[0],dims[1],dims[2],seed,bench,rng,raw,errors);
        shape<__nv_bfloat16>(dims[0],dims[1],dims[2],seed,bench,rng,raw,errors);
    }
    raw.close();errors.close();if(!raw||!errors)return 2;
    std::cout<<"PASS conversion bits, rounded and original references, all benchmark samples\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
