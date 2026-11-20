#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"FAIL %s: %s\n", #call,cudaGetErrorString(e)); std::exit(1); } } while(0)
__global__ void add(const float* a,const float* b,float* c,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) c[i]=a[i]+b[i];
}
int main() {
    cudaDeviceProp p{}; CUDA_OK(cudaGetDeviceProperties(&p,0));
    std::printf("GPU: %s; compute capability %d.%d\n",p.name,p.major,p.minor);
    const int n=1048579; const size_t bytes=n*sizeof(float);
    std::vector<float> a(n),b(n),c(n);
    for(int i=0;i<n;++i) { a[i]=float(i%1000); b[i]=float(i%37); }
    float *da=nullptr,*db=nullptr,*dc=nullptr;
    CUDA_OK(cudaMalloc(&da,bytes)); CUDA_OK(cudaMalloc(&db,bytes)); CUDA_OK(cudaMalloc(&dc,bytes));
    CUDA_OK(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
    add<<<(n+255)/256,256>>>(da,db,dc,n);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(c.data(),dc,bytes,cudaMemcpyDeviceToHost));
    int errors=0; for(int i=0;i<n;++i) if(c[i]!=a[i]+b[i]) ++errors;
    CUDA_OK(cudaFree(da)); CUDA_OK(cudaFree(db)); CUDA_OK(cudaFree(dc));
    std::printf("%s: compared %d elements, mismatches=%d\n",errors?"FAIL":"PASS",n,errors);
    return errors?1:0;
}
