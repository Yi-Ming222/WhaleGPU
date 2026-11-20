#pragma once
#include "../WhaleGPU-Phase35/tensor.cuh"
#include <cuda_pipeline.h>
#include <cstdint>

template<class T>
__global__ void convert_input(const float* src,T* dst,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) dst[i]=T(src[i]);
}

// Each participating thread owns one 16-byte segment in each shared tile.
template<class T,bool Async>
__device__ void load_segment(T* dst,const T* matrix,int row,int col,int rows,int cols) {
    if constexpr(Async) {
        if(row<rows && col+8<=cols) {
            const T* src=matrix+row*cols+col;
            if((reinterpret_cast<uintptr_t>(src)&15)==0) {
                __pipeline_memcpy_async(dst,src,16);
                return;
            }
        }
    }
    #pragma unroll
    for(int j=0;j<8;++j)
        dst[j]=(row<rows && col+j<cols)?matrix[row*cols+col+j]:T(0.f);
}
template<class T,bool Async>
__device__ void load_tiles(T* left,T* right,const T* a,const T* b,int m,int n,int k,int base) {
    int t=threadIdx.x;
    if(t<64) {
        load_segment<T,Async>(left+8*t,a,blockIdx.y*32+t/2,base+(t%2)*8,m,k);
        load_segment<T,Async>(right+8*t,b,base+t/4,blockIdx.x*32+(t%4)*8,k,n);
    }
    if constexpr(Async) __pipeline_commit();
}
template<class T,bool Async>
__global__ void block_gemm(const T* a,const T* b,float* c,int m,int n,int k) {
    constexpr int stages=Async?2:1;
    __shared__ __align__(32) T left[stages][512],right[stages][512];
    __shared__ __align__(32) float out[1024];
    int warp=threadIdx.x/32, y=16*(warp/2),x=16*(warp%2);
    wmma::fragment<wmma::accumulator,16,16,16,float> sum;
    wmma::fill_fragment(sum,0.f);
    if constexpr(Async) {
        if(k) load_tiles<T,true>(left[0],right[0],a,b,m,n,k,0);
    }
    for(int base=0;base<k;base+=16) {
        int stage=(base/16)%stages;
        if constexpr(Async) __pipeline_wait_prior(0);
        else load_tiles<T,false>(left[0],right[0],a,b,m,n,k,base);
        __syncthreads();
        if constexpr(Async) {
            if(base+16<k) load_tiles<T,true>(left[1-stage],right[1-stage],a,b,m,n,k,base+16);
        }
        wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> bf;
        wmma::load_matrix_sync(af,left[stage]+y*16,16);
        wmma::load_matrix_sync(bf,right[stage]+x,32);
        wmma::mma_sync(sum,af,bf,sum);
        __syncthreads();
    }
    wmma::store_matrix_sync(out+y*32+x,sum,32,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<1024;i+=128) {
        int row=blockIdx.y*32+i/32,col=blockIdx.x*32+i%32;
        if(row<m && col<n) c[row*n+col]=out[i];
    }
}
