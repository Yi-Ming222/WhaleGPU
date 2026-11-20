#include "../WhaleGPU-Phase34/register.cuh"
#include "../WhaleGPU-Phase1/phase3/src/reference.hpp"
#include "tensor.cuh"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
using Clock=std::chrono::steady_clock;
float tf32_rne(float value) {
    unsigned bits; std::memcpy(&bits,&value,4);
    bits=(bits+4095+((bits>>13)&1))&~8191u;
    std::memcpy(&value,&bits,4); return value;
}
template<class T> T encode(float x){return T(x);}
template<> float encode<float>(float x){return tf32_rne(x);}
template<class T> void pack(const std::vector<float>& in,std::vector<T>& out) {
    std::transform(in.begin(),in.end(),out.begin(),encode<T>);
}
void check_copy(void* to,const void* from,size_t bytes,cudaMemcpyKind kind) {
    if(bytes) CUDA_CHECK(cudaMemcpy(to,from,bytes,kind));
}
const char* names[]={"naive_FP32","shared16_FP32","register64x4_FP32","WMMA_FP16","WMMA_BF16","WMMA_TF32"};
template<class T> void experiment(int id,int m,int n,int k,int seed,bool timed,std::ofstream& log) {
    bool tensor=id>=3;
    auto input=make_input(m,n,k,seed,"random");
    auto original=cpu_reference(input,m,n,k);
    std::vector<T> a(input.a.size()),b(input.b.size());
    Input rounded=input;
    if(tensor) {
        pack(input.a,a);pack(input.b,b);
        std::copy(a.begin(),a.end(),rounded.a.begin());std::copy(b.begin(),b.end(),rounded.b.begin());
    }
    auto reference=cpu_reference(rounded,m,n,k);
    std::vector<float> output(size_t(m)*n);
    DeviceBuffer da(input.a.size()),db(input.b.size()),dc(output.size());
    auto upload=[&] {
        if(tensor){
            pack(input.a,a); pack(input.b,b);
            check_copy(da.ptr,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice);
            check_copy(db.ptr,b.data(),b.size()*sizeof(T),cudaMemcpyHostToDevice);
        }else{
            check_copy(da.ptr,input.a.data(),input.a.size()*4,cudaMemcpyHostToDevice);
            check_copy(db.ptr,input.b.data(),input.b.size()*4,cudaMemcpyHostToDevice);
        }
    };
    Variant baseline=variant(id==0?"naive_col_row":"tiled16_row");
    auto launch=[&] {
        if(tensor) tensor_gemm<T><<<dim3((n+15)/16,(m+15)/16),32>>>((T*)da.ptr,(T*)db.ptr,dc.ptr,m,n,k);
        else if(id==2) launch_register<64,4,false>(da.ptr,db.ptr,dc.ptr,m,n,k);
        else gemm(baseline,da.ptr,db.ptr,dc.ptr,m,n,k);
        CUDA_CHECK(cudaGetLastError());
    };
    auto download=[&] {check_copy(output.data(),dc.ptr,output.size()*4,cudaMemcpyDeviceToHost);};
    double accum_error=0,original_error=0;
    auto validate=[&] {
        auto result=compare(output,reference,k,k==0);
        if(result.mismatches) throw std::runtime_error(std::string(names[id])+" accumulation mismatch");
        accum_error=std::max(accum_error,result.max_error);
        double u=id==4?1./256:tensor?1./2048:0;
        double eps=k*double(std::numeric_limits<float>::epsilon());
        for(size_t i=0;i<output.size();++i){
            double error=std::abs(output[i]-original.value[i]);
            // Rounding both inputs contributes (2u+u^2) times the absolute dot product.
            double limit=(2*u+u*u)*original.absolute[i]+eps/(1-eps)*reference.absolute[i]+1e-7;
            if(!(error<=limit)) throw std::runtime_error("Original-input tolerance exceeded");
            original_error=std::max(original_error,error);
        }
    };
    upload(); CUDA_CHECK(cudaMemset(dc.ptr,0xff,output.size()*4)); launch(); download(); validate();
    Resource res=tensor?query_resource(tensor_gemm<T>,32):id==2?inspect_register<64,4,false>():resource(baseline);
    Event start,end;
    for(int w=0;timed && w<10;++w){upload();launch();download();}
    for(int i=0;i<(timed?15:1);++i){
        CUDA_CHECK(cudaEventRecord(start.value));
        int reps=timed?10:1;
        for(int r=0;r<reps;++r)launch();
        CUDA_CHECK(cudaEventRecord(end.value)); CUDA_CHECK(cudaEventSynchronize(end.value));
        float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,end.value));download();validate();
        auto begin=Clock::now();upload();launch();download();
        double e2e=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();validate();
        log<<(timed?"benchmark":"check")<<','<<names[id]<<','<<m<<','<<n<<','<<k<<','<<seed<<','<<i<<','<<ms/reps<<','<<e2e<<','<<res.registers<<','<<res.shared_bytes<<','<<accum_error<<','<<original_error<<"\n";
    }
}
int main(int argc,char** argv) try {
    if(argc!=2) return 2;
    DeviceBuffer probe(128); native_probe<<<1,32>>>(probe.ptr);
    std::vector<float> p(128);check_copy(p.data(),probe.ptr,512,cudaMemcpyDeviceToHost);
    for(float x:p) if(x!=17) throw std::runtime_error("mma probe failed");
    std::cout<<"native FP16 MMA: 128/128 = 17\n";
    std::ofstream log(argv[1]); log<<std::setprecision(12)<<"stage,variant,m,n,k,seed,round,kernel_ms,e2e_ms,registers,shared_bytes,accum_error,original_error\n";
    int shapes[][3]={{1,1,1},{17,31,0},{15,17,9},{16,16,16},{33,65,31},{64,64,64},{257,193,129},{512,512,512},{1024,1024,1024}};
    std::mt19937 rng(35);
    for(int phase=0;phase<2;++phase){
      for(auto& shape:shapes){
        if(phase && shape[0]<64) continue;
        for(int seed:{17,65537}){
          if((phase||shape[0]>=512)&&seed!=17) continue;
          std::vector<int> ids={0,1,2,3,4,5};std::shuffle(ids.begin(),ids.end(),rng);
          for(int id:ids){
            if(id==3)experiment<half>(id,shape[0],shape[1],shape[2],seed,phase,log);
            else if(id==4)experiment<__nv_bfloat16>(id,shape[0],shape[1],shape[2],seed,phase,log);
            else experiment<float>(id,shape[0],shape[1],shape[2],seed,phase,log);
          }
        }
      }
      std::cout<<"PASS phase "<<phase<<std::endl;
    }
    log.close();return log?0:2;
} catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
