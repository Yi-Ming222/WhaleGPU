#include "common.hpp"
#include "rotation.hpp"
#include "../WhaleGPU-Phase34/register.cuh"
#include "../WhaleGPU-Phase36/kernels.cuh"
#include "../WhaleGPU-Phase1/phase3/src/reference.hpp"
#include "A_generated.cuh"
template<class T>void experimentA(Logs& l,const char* dtype,int m,int n,int k,float scale,float bias,int seed,std::string pattern){
    bool rotating=std::getenv("WHALEGPU43_ROTATE")!=nullptr;
    if(rotating && (m!=512||n!=512||k!=512||seed!=17||pattern!="random"||scale!=-1.25f))return;
    constexpr bool fp=std::is_same<T,float>::value;
    size_t count=size_t(m)*n;
    std::string key=std::string(dtype)+"_"+std::to_string(m)+"x"+std::to_string(n)+"x"+std::to_string(k)+"_"+std::to_string(scale)+"_"+std::to_string(bias)+"_"+pattern+"_"+std::to_string(seed);
    auto host=make_input(m,n,k,seed,pattern);
    auto original=cpu_reference(host,m,n,k);
    std::vector<T> ha(host.a.begin(),host.a.end()),hb(host.b.begin(),host.b.end());
    std::copy(ha.begin(),ha.end(),host.a.begin());std::copy(hb.begin(),hb.end(),host.b.begin());
    auto rounded=cpu_reference(host,m,n,k);
    DeviceBuffer a(ha.size()),b(hb.size()),c(count),fixed(count);
    T* da=(T*)a.ptr;T* db=(T*)b.ptr;
    dim3 grid((n+(fp?63:31))/(fp?64:32),(m+(fp?63:31))/(fp?64:32));
    std::vector<float> g(count),y(count);
    auto upload=[&](int id){
        if(id>=4){transfer43(fixed.ptr,g.data(),count,cudaMemcpyHostToDevice);return;}
        transfer43(da,ha.data(),ha.size(),cudaMemcpyHostToDevice);
        transfer43(db,hb.data(),hb.size(),cudaMemcpyHostToDevice);
    };
    auto op=[&](int id){
        if(!count)return;
        bool fuse=id==1||id==2;
        if(id<4){
            if constexpr(std::is_same<T,float>::value){
                if(fuse)register_affine<64,4,false><<<grid,256>>>(da,db,c.ptr,m,n,k,scale,bias);
                else register_gemm<64,4,false><<<grid,256>>>(da,db,c.ptr,m,n,k);
            }else{
                if(fuse)pipeline_affine<T,true><<<grid,128>>>(da,db,c.ptr,m,n,k,scale,bias);
                else block_gemm<T,true><<<grid,128>>>(da,db,c.ptr,m,n,k);
            }
        }
        post43(id,c.ptr,fixed.ptr,count,scale,bias);
    };
    auto down=[&]{transfer43(y.data(),c.ptr,count,cudaMemcpyDeviceToHost);};
    auto poison=[&]{if(count)CUDA_CHECK(cudaMemset(c.ptr,255,count*4));};
    upload(3);poison();op(3);down();g=y;
    auto paired=compare(g,rounded,k,pattern!="random"||!k);
    if(paired.mismatches)throw std::runtime_error("invalid G32 producer");
    auto verify=[&](int id){
        Error43 err,quant;
        bool transform=id!=3 && id!=5;
        double alpha=transform?scale:1,beta=transform?bias:0;
        double u=fp?0:std::is_same<T,half>::value?1./2048:1./256;
        double eps=k*double(std::numeric_limits<float>::epsilon());
        for(size_t i=0;i<count;++i){
            float exact=transform?std::fma(scale,g[i],bias):g[i];bits43(y[i],exact);
            double bound=k?eps/(1-eps)*rounded.absolute[i]+1e-7:0;
            double round=transform?rounding43(alpha*g[i]+beta):0;
            quant.add(y[i],alpha*rounded.value[i]+beta,std::abs(alpha)*bound+round);
            err.add(y[i],alpha*original.value[i]+beta,std::abs(alpha)*(bound+(2*u+u*u)*original.absolute[i])+round);
        }
        if(!l.bench)quant.save(l,key,pair_names[id]+"_quantized_input");
        return err;
    };
    if(l.bench){
        for(int id=0;id<4;++id){
            bool f=id==1||id==2;
            if constexpr(std::is_same<T,float>::value){
                if(f)resource43(l,key,pair_names[id],"register_affine",register_affine<64,4,false>,grid,256);
                else resource43(l,key,pair_names[id],"register_gemm",register_gemm<64,4,false>,grid,256);
            }
            else if(f)resource43(l,key,pair_names[id],"pipeline_affine",pipeline_affine<T,true>,grid,128);
            else resource43(l,key,pair_names[id],"block_gemm",block_gemm<T,true>,grid,128);
        }
        post_resources(l,key,pair_names,count);
    }
    if(rotating){
        Ring43 ring(l,(ha.size()+hb.size())*sizeof(T)+count*4);ring.add(da,ha.size());ring.add(db,hb.size());ring.add(c.ptr,count);
        rotate43(l,key,ring,upload,op,down,verify,poison,[]{});return;
    }
    measure43(l,key,pair_names,upload,op,down,verify,poison);
}
int main(int argc,char** argv)try{
    if(argc!=4)return 2;Logs l(argv[1],argv[2],std::stoul(argv[3]));
    allocation43(l,1024*1024);
    int shapes[][3]={{0,7,3},{3,0,7},{7,9,0},{1,1,1},{1,37,17},{31,33,15},{65,63,33},{128,128,128},{257,193,129},{512,512,512},{1024,1024,1024}};
    for(auto& s:shapes)for(int seed:{17,65537})for(auto pat:{"random","tail","identity"})for(auto ab:{std::pair<float,float>{1,0},{0,-2},{-1.25f,.375f},{.5f,-.5f}}){
        if(l.bench && (s[0]<128||seed!=17||std::string(pat)!="random"||ab.first!=-1.25f))continue;
        if(!l.bench && s[0]>=128 && (std::string(pat)!="random"||ab.first!=-1.25f))continue;
        experimentA<float>(l,"FP32",s[0],s[1],s[2],ab.first,ab.second,seed,pat);
        experimentA<half>(l,"FP16",s[0],s[1],s[2],ab.first,ab.second,seed,pat);
        experimentA<__nv_bfloat16>(l,"BF16",s[0],s[1],s[2],ab.first,ab.second,seed,pat);
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}

