#include "common.hpp"
#include "rotation.hpp"
#define main phase41_main
#include "../WhaleGPU-Phase41/bench.cu"
#undef main
#include "B_generated.cuh"
void experimentB(Logs& l,size_t n,int pattern,int seed,float a,float b){
    bool rotating=std::getenv("WHALEGPU43_ROTATE")!=nullptr;
    if(rotating && (n!=262147||seed!=17||pattern||a!=-1.25f))return;
    auto x=inputs(n,pattern,seed);
    if(pattern==8 && n){x.front()=INFINITY;x.back()=INFINITY;}
    auto ref=reference(x);
    size_t tiles=(n+1023)/1024;
    DeviceBuffer dx(n),dy(n),fixed(n),work(2*tiles+2);
    std::vector<float> y(n),p(n);
    std::string key=std::to_string(n)+"_"+std::to_string(pattern)+"_"+std::to_string(seed)+"_"+std::to_string(a)+"_"+std::to_string(b);
    auto upload=[&](int id){if(id<4)transfer43(dx.ptr,x.data(),n,cudaMemcpyHostToDevice);else transfer43(fixed.ptr,p.data(),n,cudaMemcpyHostToDevice);};
    auto poison=[&]{if(n)CUDA_CHECK(cudaMemset(dy.ptr,255,4*n));};
    auto down=[&]{transfer43(y.data(),dy.ptr,n,cudaMemcpyDeviceToHost);};
    auto op=[&](int id){
        if(!n)return;
        if(id==1||id==2){
            if(tiles==1)local_affine<true><<<1,256>>>(dx.ptr,dy.ptr,n,work.ptr,a,b);
            else{
                local_tile<false><<<unsigned(tiles),256>>>(dx.ptr,dy.ptr,n,work.ptr);
                merge_tiles<<<1,256>>>(work.ptr,tiles);
                scale_affine<<<unsigned(tiles),256>>>(dy.ptr,n,work.ptr,a,b);
            }
        }else if(id<4)register_launch(dx.ptr,dy.ptr,n,work.ptr);
        post43(id,dy.ptr,fixed.ptr,n,a,b);
    };
    upload(3);poison();op(3);down();verify(y,ref);p=y;
    auto check=[&](int id){
        bool transform=id!=3 && id!=5;
        Error43 e;
        if(!transform||(a==1&&b==0))verify(y,ref);
        for(size_t i=0;i<n;++i){
            float expected=transform?std::fma(a,p[i],b):p[i];bits43(y[i],expected);
            double scale=transform?a:1,bias=transform?b:0;
            e.add(y[i],scale*ref[i]+bias,std::abs(scale)*(2e-8+2e-5*ref[i])+rounding43(scale*p[i]+bias));
        }
        return e;
    };
    if(l.bench){
        for(int id=0;id<4;++id){
            auto v=pair_names[id];bool f=id==1||id==2;
            if(tiles==1){
                if(f)resource43(l,key,v,"local_affine",local_affine<true>,1,256);
                else resource43(l,key,v,"local_tile_single",local_tile<true>,1,256);
            }else{
                resource43(l,key,v,"local_tile_multi",local_tile<false>,unsigned(tiles),256);
                resource43(l,key,v,"merge_tiles",merge_tiles,1,256);
                if(f)resource43(l,key,v,"scale_affine",scale_affine,unsigned(tiles),256);
                else resource43(l,key,v,"scale_tiles",scale_tiles,unsigned(tiles),256);
            }
        }
        post_resources(l,key,pair_names,n);
    }
    if(rotating){
        Ring43 ring(l,n*8);ring.add(dx.ptr,n);ring.add(dy.ptr,n);ring.add(work.ptr,2*tiles+2);
        rotate43(l,key,ring,upload,op,down,check,poison,[]{});return;
    }
    measure43(l,key,pair_names,upload,op,down,check,poison);
}
int main(int argc,char** argv)try{
    if(argc!=4)return 2;Logs l(argv[1],argv[2],std::stoul(argv[3]));
    allocation43(l,1024*1024);
    for(size_t n:{0,1,2,31,32,33,255,256,257,1023,1024,1025,4097,65539,262147,4194305})
        for(int pattern=0;pattern<9;++pattern)for(int seed:{17,65537})
            for(auto ab:{std::pair<float,float>{1,0},{0,-2},{-1.25f,.375f}}){
                if((l.bench||n>65539)&&(pattern||seed!=17||ab.first!=-1.25f))continue;
                if(l.bench && n!=33 && n!=1023 && n!=1024 && n!=1025 && n!=65539 && n!=262147 && n!=4194305)continue;
                experimentB(l,n,pattern,seed,ab.first,ab.second);
            }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
