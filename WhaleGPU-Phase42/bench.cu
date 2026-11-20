#include "layer.cuh"
#include <vector>
#include <random>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <limits>
using Clock=std::chrono::steady_clock;
struct Buff64{
    double* ptr;
    explicit Buff64(size_t n){CUDA_CHECK(cudaMalloc(&ptr,8*std::max(n,size_t(1))));}
    ~Buff64(){cudaFree(ptr);}
};
struct Reference {std::vector<double> mean,var,out;};
Reference reference(const std::vector<float>& x,size_t r,size_t h,const std::vector<float>& g,const std::vector<float>& b,float eps,bool affine){
    Reference ref;ref.mean.resize(r);ref.var.resize(r);ref.out.resize(r*h);
    if(!h)return ref;
    for(size_t i=0;i<r;++i){
        const float* row=x.data()+i*h;
        double mu=std::accumulate(row,row+h,0.)/h;
        double v=0;
        for(size_t j=0;j<h;++j){double d=double(row[j])-mu;v+=d*d;}
        v/=h;ref.mean[i]=mu;ref.var[i]=v;
        for(size_t j=0;j<h;++j){
            double norm=(double(row[j])-mu)/std::sqrt(v+eps);
            ref.out[i*h+j]=affine?norm*g[j]+b[j]:norm;
        }
    }
    return ref;
}
template<class T> void copy(T* dst,const T* src,size_t count,cudaMemcpyKind direction){
    if(count)CUDA_CHECK(cudaMemcpy(dst,src,count*sizeof(T),direction));
}
void test(size_t r,size_t h,int pattern,float eps,bool affine,bool columns,bool bench,std::mt19937& order,std::ostream& os){
    size_t n=r*h;
    std::vector<float> logical(n),physical(n),out(n),g(h),b(h);
    std::mt19937 data(4200);
    for(size_t j=0;j<h;++j){g[j]=float(int(j%13)-6)/5;b[j]=float(int(j%7)-3)/4;}
    for(size_t i=0;i<n;++i){
        int z=int(data()%17)-8;
        float v=float(int(data()%20001)-10000)/1000;
        if(pattern==1)v=12345.5f;
        if(pattern==2)v=1e8f+float(z)*8.f;
        if(pattern==3)v=z*1e-12f;
        if(pattern==4)v=z*4e37f;
        if(pattern==5)v=i%h+1==h?std::nextafter(1e8f,INFINITY):1e8f;
        physical[columns?(i%h)*r+i/h:i]=logical[i]=v;
    }
    auto ref=reference(logical,r,h,g,b,eps,affine);
    DeviceBuffer dx(n),dy(n),dg(h),db(h);Buff64 means(r),vars(r);
    Params p{dx.ptr,dg.ptr,db.ptr,dy.ptr,means.ptr,vars.ptr,r,h,eps,columns,affine};
    auto upload=[&]{copy(dx.ptr,physical.data(),n,cudaMemcpyHostToDevice);if(affine){copy(dg.ptr,g.data(),h,cudaMemcpyHostToDevice);copy(db.ptr,b.data(),h,cudaMemcpyHostToDevice);}};
    std::vector<double> mu(r),var(r);
    auto verify_output=[&]{
        double worst=0;
        for(size_t i=0;i<n;++i){
            double err=std::abs(out[columns?(i%h)*r+i/h:i]-ref.out[i]);
            if(!(err<=3e-6+2e-5*std::abs(ref.out[i])))throw std::runtime_error("output mismatch");
            worst=std::max(worst,err);
        }
        return worst;
    };
    std::vector<int> ids(sizeof(names)/sizeof(*names));std::iota(ids.begin(),ids.end(),0);
    std::shuffle(ids.begin(),ids.end(),order);
    for(int id:ids){
        upload();
        if(n){CUDA_CHECK(cudaMemset(dy.ptr,255,n*4));CUDA_CHECK(cudaMemset(means.ptr,255,r*8));CUDA_CHECK(cudaMemset(vars.ptr,255,r*8));}
        launch(id,p);copy(out.data(),dy.ptr,n,cudaMemcpyDeviceToHost);double err=verify_output(),me=0,ve=0;
        if(n){
            copy(mu.data(),means.ptr,r,cudaMemcpyDeviceToHost);copy(var.data(),vars.ptr,r,cudaMemcpyDeviceToHost);
            for(size_t i=0;i<r;++i){
                double dm=std::abs(mu[i]-ref.mean[i]),dv=std::abs(var[i]-ref.var[i]);
                double scale=std::max(std::abs(ref.mean[i]),std::sqrt(ref.var[i]));
                if(!(dm<=1e-14+2e-12*scale && dv<=1e-30+1e-5*ref.var[i] && var[i]>=0))
                    throw std::runtime_error(std::string(names[id])+" statistics mismatch H="+std::to_string(h)+" pattern="+std::to_string(pattern));
                me=std::max(me,dm);ve=std::max(ve,dv);
            }
        }
        auto a=stats_resource(id), apply=fused(id)?Resource{0,0}:attributes(apply_stats);
        auto e2e=[&]{auto start=Clock::now();upload();launch(id,p);copy(out.data(),dy.ptr,n,cudaMemcpyDeviceToHost);
            return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
        Event start,stop;
        if(bench)for(int w=0;w<5;++w)e2e();
        for(int round=0;round<(bench?11:1);++round){
            double kernel=0,wall=0;
            if(bench){
                CUDA_CHECK(cudaEventRecord(start.value));for(int j=0;j<5;++j)launch(id,p);
                CUDA_CHECK(cudaEventRecord(stop.value));CUDA_CHECK(cudaEventSynchronize(stop.value));
                float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));kernel=ms/5;
                copy(out.data(),dy.ptr,n,cudaMemcpyDeviceToHost);verify_output();wall=e2e();verify_output();
            }
            os<<names[id]<<','<<r<<','<<h<<','<<pattern<<','<<eps<<','<<affine<<','<<columns<<','<<round<<','<<kernel<<','<<wall<<','<<err<<','<<me<<','<<ve<<','<<a.regs<<','<<a.shared<<','<<apply.regs<<','<<apply.shared<<'\n';
        }
    }
}
int main(int argc,char** argv)try{
    if(argc!=4)return 2;
    bool bench=std::string(argv[1])=="bench";std::mt19937 rng(std::stoul(argv[3]));
    std::ofstream os(argv[2]);os<<std::setprecision(15)<<"variant,rows,h,pattern,eps,affine,column_major,round,kernel_ms,e2e_ms,output_error,mean_error,var_error,stats_regs,stats_shared,apply_regs,apply_shared\n";
    std::vector<std::pair<size_t,size_t>> shapes=bench?std::vector<std::pair<size_t,size_t>>{{1,33},{17,257},{128,1024},{65,4099}}:std::vector<std::pair<size_t,size_t>>{{0,17},{3,0},{1,1},{3,31},{7,32},{5,33},{3,255},{5,256},{17,257},{3,1024},{5,1025},{2,4099}};
    for(auto shape:shapes)for(int pattern=0;pattern<(bench?1:6);++pattern)for(float eps:{1e-5f,1e-12f})
        for(bool columns:{false,true})for(bool affine:{false,true})test(shape.first,shape.second,pattern,eps,affine,columns,bench,rng,os);
    os.close();if(!os)return 2;std::cout<<"PASS "<<argv[1]<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
