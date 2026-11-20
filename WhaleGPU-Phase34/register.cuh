#pragma once
#include "../WhaleGPU-Phase1/phase3/src/gemm.cuh"

// L selects the order of cooperative loads. Both global matrices are row-major.
template<int T,int R,bool L>
__global__ void register_gemm(const float* a,const float* b,float* c,int m,int n,int k) {
    constexpr int K=16, side=T/R, workers=side*side;
    __shared__ float as[T][K], bs[K][T];
    const int lane=threadIdx.x;
    const int row0=blockIdx.y*T, col0=blockIdx.x*T;
    const int y=(lane/side)*R, x=(lane%side)*R;
    float accum[R][R]={};
    for(int base=0;base<k;base+=K) {
        for(int index=lane;index<T*K;index+=workers) {
            int ar=L?index%T:index/K, aq=L?index/T:index%K;
            int bq=L?index%K:index/T, bc=L?index/K:index%T;
            as[ar][aq]=(row0+ar<m && base+aq<k)?a[(row0+ar)*k+base+aq]:0.f;
            bs[bq][bc]=(base+bq<k && col0+bc<n)?b[(base+bq)*n+col0+bc]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int q=0;q<K;++q) {
            #pragma unroll
            for(int i=0;i<R;++i) {
                #pragma unroll
                for(int j=0;j<R;++j)
                    accum[i][j]=fmaf(as[y+i][q],bs[q][x+j],accum[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<R;++i) {
        #pragma unroll
        for(int j=0;j<R;++j)
            if(row0+y+i<m && col0+x+j<n)
                c[(row0+y+i)*n+col0+x+j]=accum[i][j];
    }
}
using Launch=void(*)(const float*,const float*,float*,int,int,int);
struct RegisterVariant { bool transposed=false; Launch launch; Resource (*inspect)(); };
template<int T,int R,bool L>
void launch_register(const float* a,const float* b,float* c,int m,int n,int k) {
    if(m && n) {
        register_gemm<T,R,L><<<dim3((n+T-1)/T,(m+T-1)/T),(T/R)*(T/R)>>>(a,b,c,m,n,k);
        CUDA_CHECK(cudaGetLastError());
    }
}
template<int T,int R,bool L> Resource inspect_register() {
    return query_resource(register_gemm<T,R,L>,(T/R)*(T/R));
}
inline void launch_baseline(const float* a,const float* b,float* c,int m,int n,int k) {
    static const Variant baseline{"tiled16_row",false,false,16};
    gemm(baseline,a,b,c,m,n,k);
}
inline Resource inspect_baseline() {return query_resource(tiled_gemm<16,false>,256);}
inline RegisterVariant select_register(const std::string& name) {
    if(name=="tiled16_row") return {false,launch_baseline,inspect_baseline};
    #include "dispatch.inc"
    throw std::runtime_error("Unsupported configuration: "+name);
}
inline void gemm(const RegisterVariant& v,const float* a,const float* b,float* c,int m,int n,int k) {v.launch(a,b,c,m,n,k);}
inline Resource resource(const RegisterVariant& v) {return v.inspect();}
