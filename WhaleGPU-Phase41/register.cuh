// One block owns 1024 consecutive elements. Each lane owns four striped values.
template<bool Single> __global__ void local_tile(const float* x,float* y,size_t n,float* work){
    size_t base=size_t(blockIdx.x)*1024+threadIdx.x;
    float v[4],peak=-CUDART_INF_F;
    #pragma unroll
    for(int j=0;j<4;++j){
        size_t i=base+j*256;
        v[j]=i<n?x[i]:-CUDART_INF_F;
        peak=fmaxf(peak,v[j]);
    }
    peak=block_reduce<true,false>(peak);
    double sum=0;
    #pragma unroll
    for(int j=0;j<4;++j){
        v[j]=base+j*256<n?weight(v[j],peak):0.f;
        sum+=v[j];
    }
    sum=block_reduce<false,false>(sum);
    if constexpr(!Single){
        if(!threadIdx.x){work[blockIdx.x]=peak;work[gridDim.x+blockIdx.x]=float(sum);}
    }
    #pragma unroll
    for(int j=0;j<4;++j)if(base+j*256<n)y[base+j*256]=Single?float(v[j]/sum):v[j];
}
__global__ void merge_tiles(float* work,size_t tiles){
    float peak=-CUDART_INF_F;
    for(size_t i=threadIdx.x;i<tiles;i+=256)peak=fmaxf(peak,work[i]);
    peak=block_reduce<true,false>(peak);
    double sum=0;
    for(size_t i=threadIdx.x;i<tiles;i+=256)sum+=double(weight(work[i],peak))*work[tiles+i];
    sum=block_reduce<false,false>(sum);
    if(!threadIdx.x){work[2*tiles]=peak;work[2*tiles+1]=float(sum);}
}
__global__ void scale_tiles(float* y,size_t n,const float* work){
    size_t tiles=gridDim.x;
    float scale=weight(work[blockIdx.x],work[2*tiles])/work[2*tiles+1];
    for(size_t i=size_t(blockIdx.x)*1024+threadIdx.x,end=min(size_t(n),size_t(blockIdx.x+1)*1024);i<end;i+=256)
        y[i]*=scale;
}
inline void register_launch(const float* x,float* y,size_t n,float* work){
    size_t tiles=1+(n-1)/1024;
    if(tiles==1)local_tile<true><<<1,256>>>(x,y,n,work);
    else if(tiles<=2147483647){
        local_tile<false><<<unsigned(tiles),256>>>(x,y,n,work);
        merge_tiles<<<1,256>>>(work,tiles);
        scale_tiles<<<unsigned(tiles),256>>>(y,n,work);
    }else multi(x,y,n,work);
}
