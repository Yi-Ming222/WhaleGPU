#pragma once
#include "../WhaleGPU-Phase1/include/common.cuh"
#include <math_constants.h>
__device__ inline float weight(float x,float peak){
    return isinf(peak)?float(x==peak):expf(x-peak);
}
__global__ void serial_softmax(const float* x,float* y,size_t n){
    float peak=-CUDART_INF_F;
    for(size_t i=0;i<n;++i) peak=fmaxf(peak,x[i]);
    double total=0;
    for(size_t i=0;i<n;++i) total+=weight(x[i],peak);
    for(size_t i=0;i<n;++i) y[i]=weight(x[i],peak)/total;
}
#include <algorithm>
#include <type_traits>
#include "parallel.cuh"
#include "register.cuh"
constexpr const char* variants[]={"naive","warp","block","shared_tree","multistage","register4"};
inline void launch(int id,const float* x,float* y,size_t n,float* scratch){
    if(!n)return;
    switch(id){
        case 0:serial_softmax<<<1,1>>>(x,y,n);break;
        case 1:fused_softmax<true,false><<<1,32>>>(x,y,n);break;
        case 2:fused_softmax<false,false><<<1,256>>>(x,y,n);break;
        case 3:fused_softmax<false,true><<<1,256>>>(x,y,n);break;
        case 4:multi(x,y,n,scratch);break;
        case 5:register_launch(x,y,n,scratch);break;
        default:throw std::invalid_argument("unknown softmax variant");
    }
    CUDA_CHECK(cudaGetLastError());
}
