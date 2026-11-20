#pragma once
#include "common.cuh"
#include <string>

template<bool RowLanes, bool TransposedB>
__global__ void naive_gemm(const float* a,const float* b,float* c,int m,int n,int k) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(m)*n) return;
    int row=RowLanes ? int(i%m) : int(i/n);
    int col=RowLanes ? int(i/m) : int(i%n);
    float value=0;
    for(int q=0;q<k;++q)
        value=fmaf(a[size_t(row)*k+q],b[TransposedB ? size_t(col)*k+q : size_t(q)*n+col],value);
    c[size_t(row)*n+col]=value;
}

template<int T, bool TransposedB>
__global__ void tiled_gemm(const float* a,const float* b,float* c,int m,int n,int k) {
    __shared__ float at[T][T];
    __shared__ float bt[T][T];
    int x=threadIdx.x,y=threadIdx.y;
    int row=blockIdx.y*T+y,col=blockIdx.x*T+x;
    float value=0;
    for(int base=0;base<k;base+=T) {
        at[y][x]=(row<m && base+x<k) ? a[size_t(row)*k+base+x] : 0.f;
        if constexpr(TransposedB) {
            int bcol=blockIdx.x*T+y;
            // Consecutive x lanes read consecutive K locations, then transpose in shared.
            bt[x][y]=(bcol<n && base+x<k) ? b[size_t(bcol)*k+base+x] : 0.f;
        } else {
            bt[y][x]=(base+y<k && col<n) ? b[size_t(base+y)*n+col] : 0.f;
        }
        __syncthreads(); // Every shared slot initialized before consumption.
        #pragma unroll
        for(int q=0;q<T;++q) value=fmaf(at[y][q],bt[q][x],value);
        __syncthreads(); // Readers finish before next tile overwrites shared memory.
    }
    if(row<m && col<n) c[size_t(row)*n+col]=value;
}

struct Variant { std::string name; bool row_lanes=false,transposed=false; int tile=0; };
inline Variant variant(const std::string& name) {
    if(name=="naive_col_row") return {name,false,false,0};
    if(name=="naive_row_row") return {name,true,false,0};
    if(name=="naive_col_bt") return {name,false,true,0};
    if(name=="naive_row_bt") return {name,true,true,0};
    for(int t:{8,16,32}) {
        if(name=="tiled"+std::to_string(t)+"_row") return {name,false,false,t};
        if(name=="tiled"+std::to_string(t)+"_bt") return {name,false,true,t};
    }
    throw std::runtime_error("Unknown variant: "+name);
}
template<int T> void tile_launch(const Variant& v,const float* a,const float* b,float* c,int m,int n,int k) {
    dim3 threads(T,T),blocks((n+T-1)/T,(m+T-1)/T);
    if(v.transposed) tiled_gemm<T,true><<<blocks,threads>>>(a,b,c,m,n,k);
    else tiled_gemm<T,false><<<blocks,threads>>>(a,b,c,m,n,k);
}
inline void gemm(const Variant& v,const float* a,const float* b,float* c,int m,int n,int k) {
    if(m==0 || n==0) return;
    if(v.tile==8) tile_launch<8>(v,a,b,c,m,n,k);
    else if(v.tile==16) tile_launch<16>(v,a,b,c,m,n,k);
    else if(v.tile==32) tile_launch<32>(v,a,b,c,m,n,k);
    else {
        unsigned blocks=unsigned((size_t(m)*n+255)/256);
        if(v.row_lanes && v.transposed) naive_gemm<true,true><<<blocks,256>>>(a,b,c,m,n,k);
        else if(v.row_lanes) naive_gemm<true,false><<<blocks,256>>>(a,b,c,m,n,k);
        else if(v.transposed) naive_gemm<false,true><<<blocks,256>>>(a,b,c,m,n,k);
        else naive_gemm<false,false><<<blocks,256>>>(a,b,c,m,n,k);
    }
    CUDA_CHECK(cudaGetLastError());
}
struct Resource { int registers,shared_bytes,active_blocks,threads; };
template<class Kernel> Resource query_resource(Kernel kernel,int threads) {
    cudaFuncAttributes a{}; CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    int active=0; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,threads,0));
    return {a.numRegs,int(a.sharedSizeBytes),active,threads};
}
inline Resource resource(const Variant& v) {
    if(v.tile==8) return v.transposed?query_resource(tiled_gemm<8,true>,64):query_resource(tiled_gemm<8,false>,64);
    if(v.tile==16) return v.transposed?query_resource(tiled_gemm<16,true>,256):query_resource(tiled_gemm<16,false>,256);
    if(v.tile==32) return v.transposed?query_resource(tiled_gemm<32,true>,1024):query_resource(tiled_gemm<32,false>,1024);
    if(v.row_lanes) return v.transposed?query_resource(naive_gemm<true,true>,256):query_resource(naive_gemm<true,false>,256);
    return v.transposed?query_resource(naive_gemm<false,true>,256):query_resource(naive_gemm<false,false>,256);
}
