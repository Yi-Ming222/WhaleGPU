// Reductions are called uniformly by all threads of their participating group.
__device__ double warp_sum42(double x){
    for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);
    return __shfl_sync(0xffffffff,x,0);
}
__device__ double block_sum42(double x){
    __shared__ double partial[8];
    int t=threadIdx.x;
    x=warp_sum42(x);
    if(t%32==0)partial[t/32]=x;
    __syncthreads();
    if(t<32){double v=warp_sum42(t<8?partial[t]:0.);if(t==0)partial[0]=v;}
    __syncthreads();
    double total=partial[0];__syncthreads();return total;
}
template<bool Warp> __global__ void two_pass(Params p){
    int lane=Warp?threadIdx.x%32:threadIdx.x,step=Warp?32:256;
    size_t groups=Warp?8:1;
    for(size_t r=size_t(blockIdx.x)*groups+(Warp?threadIdx.x/32:0);r<p.rows;r+=gridDim.x*groups){
        double sum=0;
        for(size_t j=lane;j<p.h;j+=step)sum+=p.get(r,j);
        double mean=(Warp?warp_sum42(sum):block_sum42(sum))/p.h;
        double sq=0;
        for(size_t j=lane;j<p.h;j+=step){double d=p.get(r,j)-mean;sq+=d*d;}
        double var=(Warp?warp_sum42(sq):block_sum42(sq))/p.h;
        if(!lane){p.means[r]=mean;p.vars[r]=var;}
    }
}
struct Moments {
    size_t count;
    double mean,m2;
    __device__ void add(double x){
        ++count;double delta=x-mean;mean+=delta/count;m2+=delta*(x-mean);
    }
    __device__ void join(Moments b){
        if(!b.count)return;
        if(!count){*this=b;return;}
        double delta=b.mean-mean;
        size_t total=count+b.count;
        m2+=b.m2+delta*delta*(double(count)*b.count/total);
        mean+=delta*(double(b.count)/total);count=total;
    }
};
template<bool Merge> __global__ void welford(Params p){
    __shared__ Moments cache[Merge?256:1];
    int t=threadIdx.x;
    for(size_t r=Merge?blockIdx.x:size_t(blockIdx.x)*blockDim.x+t;r<p.rows;r+=size_t(gridDim.x)*(Merge?1:blockDim.x)){
        Moments state{};
        for(size_t j=Merge?t:0;j<p.h;j+=Merge?256:1)state.add(p.get(r,j));
        if constexpr(Merge){
            cache[t]=state;__syncthreads();
            for(int d=128;d;d/=2){if(t<d)cache[t].join(cache[t+d]);__syncthreads();}
            if(!t){p.means[r]=cache[0].mean;p.vars[r]=cache[0].m2/p.h;}
            __syncthreads();
        }else{p.means[r]=state.mean;p.vars[r]=state.m2/p.h;}
    }
}
