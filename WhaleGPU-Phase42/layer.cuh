#pragma once
#include "../WhaleGPU-Phase1/include/common.cuh"
#include <algorithm>
struct Params {
    const float *x,*gamma,*beta;
    float* y;
    double *means,*vars;
    size_t rows,h;
    float eps;
    bool columns,affine;
    __device__ size_t index(size_t r,size_t j) const {return columns?j*rows+r:r*h+j;}
    __device__ double get(size_t r,size_t j) const {return double(x[index(r,j)]);}
    __device__ void store(size_t r,size_t j,double centered,double inv) const {
        double v=centered*inv;
        if(affine)v=v*gamma[j]+beta[j];
        y[index(r,j)]=float(v);
    }
};
__global__ void serial_stats(Params p){
    for(size_t r=size_t(blockIdx.x)*blockDim.x+threadIdx.x;r<p.rows;r+=size_t(gridDim.x)*blockDim.x){
        double mean=0,var=0;
        for(size_t j=0;j<p.h;++j)mean+=p.get(r,j);
        mean/=p.h;
        for(size_t j=0;j<p.h;++j){double d=p.get(r,j)-mean;var+=d*d;}
        p.means[r]=mean;p.vars[r]=var/p.h;
    }
}
__global__ void apply_stats(Params p){
    size_t count=p.rows*p.h;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x){
        size_t r=i/p.h,j=i%p.h;
        p.store(r,j,p.get(r,j)-p.means[r],1./sqrt(p.vars[r]+p.eps));
    }
}
inline unsigned blocks(size_t n){return unsigned(std::min(n,size_t(65535)));}
#include "parallel.cuh"
#include "register.cuh"
inline const char* names[]={"naive","block","warp","online","merge","register4","fused4"};
inline void launch(int id,Params p){
    if(!p.rows||!p.h)return;
    switch(id){
    case 0:serial_stats<<<blocks((p.rows+127)/128),128>>>(p);break;
    case 1:two_pass<false><<<blocks(p.rows),256>>>(p);break;
    case 2:two_pass<true><<<blocks((p.rows+7)/8),256>>>(p);break;
    case 3:welford<false><<<blocks((p.rows+127)/128),128>>>(p);break;
    case 4:welford<true><<<blocks(p.rows),256>>>(p);break;
    case 5:register_row<false><<<blocks(p.rows),256>>>(p);break;
    case 6:register_row<true><<<blocks(p.rows),256>>>(p);CUDA_CHECK(cudaGetLastError());return;
    default:throw std::invalid_argument("unknown variant");
    }
    apply_stats<<<blocks((p.rows*p.h+255)/256),256>>>(p);
    CUDA_CHECK(cudaGetLastError());
}
struct Resource {int regs;size_t shared;};
template<class K> Resource attributes(K kernel){
    cudaFuncAttributes a{};CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));return {a.numRegs,a.sharedSizeBytes};
}
inline Resource stats_resource(int id){
    switch(id){
        case 0:return attributes(serial_stats);
        case 1:return attributes(two_pass<false>);
        case 2:return attributes(two_pass<true>);
        case 3:return attributes(welford<false>);
        case 4:return attributes(welford<true>);
        case 5:return attributes(register_row<false>);
        default:return attributes(register_row<true>);
    }
}
inline bool fused(int id){return id==6;}
