// A block owns a row. Cache its first four striped elements per thread;
// for longer rows, every subsequent element is covered by the strided tail.
template<bool Fuse> __global__ void register_row(Params p){
    int lane=threadIdx.x;
    for(size_t row=blockIdx.x;row<p.rows;row+=gridDim.x){
        float local[4];double sum=0;
        #pragma unroll
        for(int slot=0;slot<4;++slot){
            size_t j=lane+slot*256;local[slot]=j<p.h?p.x[p.index(row,j)]:0.f;sum+=local[slot];
        }
        for(size_t j=1024+lane;j<p.h;j+=256)sum+=p.get(row,j);
        double mean=block_sum42(sum)/p.h, sq=0;
        #pragma unroll
        for(int slot=0;slot<4;++slot){
            if(lane+slot*256<p.h){double d=double(local[slot])-mean;sq+=d*d;}
        }
        for(size_t j=1024+lane;j<p.h;j+=256){double d=p.get(row,j)-mean;sq+=d*d;}
        double var=block_sum42(sq)/p.h;
        if(!lane){p.means[row]=mean;p.vars[row]=var;}
        if constexpr(Fuse){
            double inv=1./sqrt(var+p.eps);
            #pragma unroll
            for(int slot=0;slot<4;++slot){
                size_t j=lane+slot*256;
                if(j<p.h)p.store(row,j,double(local[slot])-mean,inv);
            }
            for(size_t j=1024+lane;j<p.h;j+=256)p.store(row,j,p.get(row,j)-mean,inv);
        }
    }
}
