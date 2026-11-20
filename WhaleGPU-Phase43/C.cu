#include "common.hpp"
#include "rotation.hpp"
#define main phase42_main
#include "../WhaleGPU-Phase42/bench.cu"
#undef main
#include "C_generated.cuh"
const std::vector<std::string> paths={"split","fused","fused_empty","block_split","reload","apply_only","copy_only"};
void experimentC(Logs& l,size_t rows,size_t h,int pattern,int seed,float eps,bool cols,bool affine){
    bool rotating=std::getenv("WHALEGPU43_ROTATE")!=nullptr;
    if(rotating && (rows!=128||h!=1024||seed!=17||pattern||cols||!affine||eps!=1e-5f))return;
    const size_t n=rows*h;
    std::string key=std::to_string(rows)+"x"+std::to_string(h)+"_"+std::to_string(pattern)+"_"+std::to_string(seed)+"_eps"+(eps==1e-5f?"5":"12")+(cols?"_col":"_row")+(affine?"_aff":"_plain");
    std::vector<float> x(n),logical(n),out(n),gamma(h),beta(h);
    std::vector<double> mu(rows),var(rows);
    std::mt19937 gen(seed);
    for(size_t j=0;j<h;++j){gamma[j]=float(int(j%13)-6)/5;beta[j]=float(int(j%7)-3)/4;}
    for(size_t i=0;i<n;++i){
        int z=int(gen()%17)-8;
        float v=float(int(gen()%20001)-10000)/1000;
        if(pattern==1)v=12345.5f;
        if(pattern==2)v=1e8f+float(z)*8.f;
        if(pattern==3)v=z*1e-12f;
        if(pattern==4)v=z*4e37f;
        if(pattern==5)v=i%h+1==h?std::nextafter(1e8f,INFINITY):1e8f;
        x[cols?(i%h)*rows+i/h:i]=logical[i]=v;
    }
    auto ref=reference(logical,rows,h,gamma,beta,eps,affine);
    DeviceBuffer dx(n),dy(n),dg(h),db(h);Buff64 dm(rows),dv(rows);
    Params p{dx.ptr,dg.ptr,db.ptr,dy.ptr,dm.ptr,dv.ptr,rows,h,eps,cols,affine};
    Buff64 fixed_mu(rows),fixed_var(rows);DeviceBuffer fixed_y(n);
    std::vector<float> frozen(n);
    for(size_t i=0;i<n;++i)frozen[cols?(i%h)*rows+i/h:i]=float(ref.out[i]);
    int active=0;
    auto upload=[&](int id){
        active=id;
        if(id>=5 && n){copy(fixed_mu.ptr,ref.mean.data(),rows,cudaMemcpyHostToDevice);copy(fixed_var.ptr,ref.var.data(),rows,cudaMemcpyHostToDevice);}
        if(id==6){copy(fixed_y.ptr,frozen.data(),n,cudaMemcpyHostToDevice);return;}
        copy(dx.ptr,x.data(),n,cudaMemcpyHostToDevice);
        if(affine){copy(dg.ptr,gamma.data(),h,cudaMemcpyHostToDevice);copy(db.ptr,beta.data(),h,cudaMemcpyHostToDevice);}
    };
    auto poison=[&]{if(n){CUDA_CHECK(cudaMemset(dy.ptr,255,n*4));CUDA_CHECK(cudaMemset(dm.ptr,255,rows*8));CUDA_CHECK(cudaMemset(dv.ptr,255,rows*8));}};
    auto down=[&]{copy(out.data(),dy.ptr,n,cudaMemcpyDeviceToHost);
        if(n){copy(mu.data(),active>=5?fixed_mu.ptr:dm.ptr,rows,cudaMemcpyDeviceToHost);copy(var.data(),active>=5?fixed_var.ptr:dv.ptr,rows,cudaMemcpyDeviceToHost);}
    };
    auto op=[&](int id){
        if(!n)return;
        if(id==6)copy43<<<unsigned((n+255)/256),256>>>(fixed_y.ptr,dy.ptr,n);
        else if(id==5){Params q=p;q.means=fixed_mu.ptr;q.vars=fixed_var.ptr;apply_stats<<<blocks((n+255)/256),256>>>(q);}
        else if(id==4)register_reload<true><<<blocks(rows),256>>>(p);
        else launch(id==0?5:id==3?1:6,p);
        if(id==2)empty43<<<1,1>>>();
        CUDA_CHECK(cudaGetLastError());
    };
    auto check=[&](int id){
        Error43 output,means,variances;
        for(size_t i=0;i<n;++i)output.add(out[cols?(i%h)*rows+i/h:i],ref.out[i],3e-6+2e-5*std::abs(ref.out[i]));
        if(n)for(size_t r=0;r<rows;++r){
            means.add(mu[r],ref.mean[r],1e-14+2e-12*std::max(std::abs(ref.mean[r]),std::sqrt(ref.var[r])));
            variances.add(var[r],ref.var[r],1e-30+1e-5*ref.var[r]);
            if(var[r]<0)throw std::runtime_error("negative variance");
        }
        if(!l.bench){means.save(l,key,paths[id]+"_mean");variances.save(l,key,paths[id]+"_variance");}
        return output;
    };
    if(l.bench){
        for(int id=0;id<(rotating?2:7);++id){
            auto v=paths[id];auto grid=blocks(rows);
            if(id==6)resource43(l,key,v,"copy43",copy43,unsigned((n+255)/256),256);
            else if(id==5)resource43(l,key,v,"apply_stats",apply_stats,blocks((n+255)/256),256);
            else if(id==4)resource43(l,key,v,"register_reload",register_reload<true>,grid,256);
            else if(id==3)resource43(l,key,v,"two_pass",two_pass<false>,grid,256);
            else if(id==0)resource43(l,key,v,"register_stats",register_row<false>,grid,256);
            else resource43(l,key,v,"register_fused",register_row<true>,grid,256);
            if(id==0||id==3)resource43(l,key,v,"apply_stats",apply_stats,blocks((n+255)/256),256);
            if(id==2)resource43(l,key,v,"empty43",empty43,1,1);
        }
    }
    if(rotating){
        Ring43 ring(l,n*8);ring.add(dx.ptr,n);ring.add(dy.ptr,n);ring.add(dm.ptr,rows);ring.add(dv.ptr,rows);ring.add(dg.ptr,h);ring.add(db.ptr,h);
        rotate43(l,key,ring,upload,op,down,check,poison,[&]{p.x=dx.ptr;p.y=dy.ptr;p.means=dm.ptr;p.vars=dv.ptr;p.gamma=dg.ptr;p.beta=db.ptr;});return;
    }
    measure43(l,key,paths,upload,op,down,check,poison);
}
int main(int argc,char** argv)try{
    if(argc!=4)return 2;Logs l(argv[1],argv[2],std::stoul(argv[3]));allocation43(l,1024*1024);
    std::vector<std::pair<int,int>> shapes={{0,17},{3,0},{1,1},{3,31},{7,32},{5,33},{3,255},{5,256},{17,257},{3,1024},{5,1025},{2,4099},{128,1024},{65,4099}};
    for(auto s:shapes)for(int seed:{17,65537})for(int pattern=0;pattern<6;++pattern)for(float eps:{1e-5f,1e-12f})for(bool cols:{false,true})for(bool affine:{true,false}){
        if((l.bench||s.first>=65)&&(seed!=17||pattern||eps!=1e-5f||!affine))continue;
        if(l.bench && s.first!=5 && s.first!=17 && s.first!=65 && s.first!=128)continue;
        experimentC(l,s.first,s.second,pattern,seed,eps,cols,affine);
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
