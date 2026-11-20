#include "softmax.cuh"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using Clock=std::chrono::steady_clock;
std::vector<float> inputs(size_t n,int pattern,unsigned seed){
    std::vector<float> x(n);
    std::mt19937 rng(seed);
    for(size_t i=0;i<n;++i){
        float value=float(int(rng()%40001)-20000)/1000;
        switch(pattern){
            case 1:value=10000.f;break;
            case 2:value+=10000;break;
            case 3:value-=10000;break;
            case 4:value=i%29==0||i+1==n?1000:-1000;break;
            case 5:value=i%7==0?INFINITY:value;break;
            case 6:value=-INFINITY;break;
            case 7:value=i%2?-INFINITY:value;break;
        }
        x[i]=value;
    }
    return x;
}
std::vector<double> reference(const std::vector<float>& x){
    std::vector<double> p(x.size());if(x.empty())return p;
    double peak=*std::max_element(x.begin(),x.end());
    for(size_t i=0;i<x.size();++i)
        p[i]=std::isinf(peak)?double(x[i]==peak):std::exp(double(x[i])-peak);
    double sum=std::accumulate(p.begin(),p.end(),0.);
    for(double& value:p)value/=sum;
    return p;
}
struct Metrics { double error=0,mass=0; };
Metrics verify(const std::vector<float>& out,const std::vector<double>& p){
    Metrics result;
    for(size_t i=0;i<p.size();++i){
        double error=std::abs(out[i]-p[i]);
        if(!std::isfinite(out[i])||out[i]<0||out[i]>1||error>2e-8+2e-5*p[i])
            throw std::runtime_error("invalid probability at "+std::to_string(i));
        result.error=std::max(result.error,error);result.mass+=out[i];
    }
    if(std::abs(result.mass-(p.empty()?0.:1.))>2e-6)throw std::runtime_error("probability mass mismatch");
    return result;
}
int main(int argc,char** argv) try {
    if(argc!=4) return 2;
    bool bench=std::string(argv[1])=="bench";
    std::ofstream csv(argv[2]);csv<<std::setprecision(14)<<"variant,n,pattern,seed,round,kernel_ms,e2e_ms,max_abs,mass\n";
    std::mt19937 order(std::stoul(argv[3]));
    std::vector<size_t> sizes=bench?std::vector<size_t>{33,257,1025,16387,262147}:std::vector<size_t>{0,1,2,31,32,33,255,256,257,1023,1024,1025,4097,65539};
    for(auto n:sizes){
        DeviceBuffer dx(n),dy(n),scratch(std::max(size_t(grid_cap+2),2*((n+1023)/1024)+2));
        for(int pattern=0;pattern<(bench?1:8);++pattern) for(int seed:{17,65537}){
            if((bench||pattern!=0)&&seed!=17)continue;
            auto x=inputs(n,pattern,seed);auto p=reference(x);std::vector<float> y(n);
            auto upload=[&]{if(n)CUDA_CHECK(cudaMemcpy(dx.ptr,x.data(),4*n,cudaMemcpyHostToDevice));};
            auto download=[&]{if(n)CUDA_CHECK(cudaMemcpy(y.data(),dy.ptr,4*n,cudaMemcpyDeviceToHost));};
            std::vector<int> ids(sizeof(variants)/sizeof(*variants));std::iota(ids.begin(),ids.end(),0);std::shuffle(ids.begin(),ids.end(),order);
            for(int id:ids){
                auto op=[&]{launch(id,dx.ptr,dy.ptr,n,scratch.ptr);};
                upload();if(n)CUDA_CHECK(cudaMemset(dy.ptr,255,4*n));op();download();
                auto metrics=verify(y,p);
                Event start,end;
                if(bench)for(int w=0;w<5;++w){upload();op();download();}
                for(int r=0;r<(bench?11:1);++r){
                    double kernel=0,e2e=0;
                    if(bench){
                        CUDA_CHECK(cudaEventRecord(start.value));for(int j=0;j<5;++j)op();
                        CUDA_CHECK(cudaEventRecord(end.value));CUDA_CHECK(cudaEventSynchronize(end.value));
                        float elapsed;CUDA_CHECK(cudaEventElapsedTime(&elapsed,start.value,end.value));kernel=elapsed/5;
                        download();verify(y,p);
                        auto begin=Clock::now();upload();op();download();
                        e2e=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();metrics=verify(y,p);
                    }
                    csv<<variants[id]<<','<<n<<','<<pattern<<','<<seed<<','<<r<<','<<kernel<<','<<e2e<<','<<metrics.error<<','<<metrics.mass<<'\n';
                }
            }
        }
    }
    csv.close();if(!csv)return 2;
    std::cout<<"PASS "<<argv[1]<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
