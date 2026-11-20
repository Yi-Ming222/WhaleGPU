// Included after weight() and the serial kernel.
template<bool Max,class T> __device__ T merge(T a,T b){
    if constexpr(Max)return a>b?a:b;
    return a+b;
}
template<bool Max,class T> __device__ T warp_reduce(T value){
    for(int d=16;d;d/=2)value=merge<Max>(value,__shfl_down_sync(0xffffffff,value,d));
    return __shfl_sync(0xffffffff,value,0);
}
// All 256 threads call this function. The last barrier protects shared storage on reuse.
template<bool Max,bool Tree,class T> __device__ T block_reduce(T value){
    constexpr int slots=Tree?256:8;
    __shared__ T buf[slots];
    int t=threadIdx.x;
    if constexpr(Tree){
        buf[t]=value;__syncthreads();
        for(int d=128;d;d/=2){
            if(t<d)buf[t]=merge<Max>(buf[t],buf[t+d]);
            __syncthreads();
        }
    }else{
        value=warp_reduce<Max>(value);
        if(t%32==0)buf[t/32]=value;
        __syncthreads();
        if(t<32){
            T v=t<8?buf[t]:(Max?T(-CUDART_INF_F):T(0));
            v=warp_reduce<Max>(v);
            if(t==0)buf[0]=v;
        }
        __syncthreads();
    }
    T answer=buf[0];__syncthreads();return answer;
}
template<bool Warp,bool Tree> __global__ void fused_softmax(const float* x,float* y,size_t n){
    float peak=-CUDART_INF_F;
    for(size_t i=threadIdx.x;i<n;i+=blockDim.x)peak=fmaxf(peak,x[i]);
    if constexpr(Warp)peak=warp_reduce<true>(peak);
    else peak=block_reduce<true,Tree>(peak);
    double sum=0;
    for(size_t i=threadIdx.x;i<n;i+=blockDim.x)sum+=weight(x[i],peak);
    if constexpr(Warp)sum=warp_reduce<false>(sum);
    else sum=block_reduce<false,Tree>(sum);
    for(size_t i=threadIdx.x;i<n;i+=blockDim.x)y[i]=weight(x[i],peak)/sum;
}
constexpr int grid_cap=512;
template<bool Max> __global__ void partial(const float* x,float* work,size_t n){
    using T=typename std::conditional<Max,float,double>::type;
    T value=Max?T(-CUDART_INF_F):T(0);
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256){
        if constexpr(Max)value=fmaxf(value,x[i]);
        else value+=weight(x[i],work[grid_cap]);
    }
    value=block_reduce<Max,false>(value);
    if(!threadIdx.x)work[blockIdx.x]=float(value);
}
template<bool Max> __global__ void finish(float* work,int count){
    double value=Max?-CUDART_INF:0.;
    for(int i=threadIdx.x;i<count;i+=256)value=merge<Max>(value,double(work[i]));
    value=block_reduce<Max,false>(value);
    if(!threadIdx.x)work[grid_cap+(Max?0:1)]=float(value);
}
__global__ void normalize(const float* x,float* y,size_t n,const float* work){
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256)
        y[i]=weight(x[i],work[grid_cap])/work[grid_cap+1];
}
inline void multi(const float* x,float* y,size_t n,float* work){
    int blocks=int(std::min(size_t(grid_cap),1+(n-1)/256));
    partial<true><<<blocks,256>>>(x,work,n);
    finish<true><<<1,256>>>(work,blocks);
    partial<false><<<blocks,256>>>(x,work,n);
    finish<false><<<1,256>>>(work,blocks);
    normalize<<<blocks,256>>>(x,y,n,work);
}
