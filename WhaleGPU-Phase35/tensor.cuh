#pragma once
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using namespace nvcuda;
template<class Storage> struct Operand { using type=Storage; static constexpr int depth=16; };
template<> struct Operand<float> { using type=wmma::precision::tf32; static constexpr int depth=8; };

// One complete warp per tile. WMMA shared-memory addresses are 256-bit aligned.
template<class Storage>
__global__ void tensor_gemm(const Storage* a,const Storage* b,float* c,int m,int n,int k) {
    using Format=typename Operand<Storage>::type;
    constexpr int depth=Operand<Storage>::depth;
    __shared__ __align__(32) Storage sa[16*depth],sb[depth*16];
    __shared__ __align__(32) float output[256];
    int lane=threadIdx.x, row=16*blockIdx.y,col=16*blockIdx.x;
    wmma::fragment<wmma::accumulator,16,16,depth,float> accum;
    wmma::fill_fragment(accum,0.f);
    for(int base=0;base<k;base+=depth) {
        for(int i=lane;i<16*depth;i+=32) {
            int ar=row+i/depth,ak=base+i%depth,bk=base+i/16,bc=col+i%16;
            sa[i]=(ar<m && ak<k)?a[ar*k+ak]:Storage(0.f);
            sb[i]=(bk<k && bc<n)?b[bk*n+bc]:Storage(0.f);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,depth,Format,wmma::row_major> left;
        wmma::fragment<wmma::matrix_b,16,16,depth,Format,wmma::row_major> right;
        wmma::load_matrix_sync(left,sa,depth);
        wmma::load_matrix_sync(right,sb,16);
        wmma::mma_sync(accum,left,right,accum);
        __syncwarp();
    }
    wmma::store_matrix_sync(output,accum,16,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<256;i+=32)
        if(row+i/16<m && col+i%16<n) c[(row+i/16)*n+col+i%16]=output[i];
}
__global__ void native_probe(float* output) {
    unsigned ones=0x3c003c00;
    float d0=1,d1=1,d2=1,d3=1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3};"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(ones));
    int i=threadIdx.x*4;
    output[i]=d0; output[i+1]=d1; output[i+2]=d2; output[i+3]=d3;
}
