#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <stdexcept>

struct Input { std::vector<float> a,b; };
inline Input make_input(int m,int n,int k,uint32_t seed,const std::string& pattern) {
    Input in{std::vector<float>(size_t(m)*k),std::vector<float>(size_t(k)*n)};
    uint32_t state=seed;
    auto random=[&]() {state=1664525u*state+1013904223u; return float(int(state>>8)-8388608)/8388608.f;};
    for(auto& x:in.a) x=random();
    for(auto& x:in.b) x=random();
    if(pattern=="zero") {std::fill(in.a.begin(),in.a.end(),0);std::fill(in.b.begin(),in.b.end(),0);}
    else if(pattern=="identity") {
        std::fill(in.a.begin(),in.a.end(),0);
        for(int i=0;i<std::min(m,k);++i) in.a[size_t(i)*k+i]=1;
    } else if(pattern=="tail") {
        std::fill(in.a.begin(),in.a.end(),0);std::fill(in.b.begin(),in.b.end(),0);
        if(!in.a.empty()) in.a.back()=2;
        if(!in.b.empty()) in.b.back()=-3;
    } else if(pattern!="random") throw std::runtime_error("Invalid input pattern");
    return in;
}
inline void pack_b(const std::vector<float>& b,std::vector<float>& bt,int n,int k) {
    // Deliberately serial host packing, separately measured and included in E2E.
    for(int q=0;q<k;++q) for(int col=0;col<n;++col) bt[size_t(col)*k+q]=b[size_t(q)*n+col];
}
struct Reference { std::vector<double> value,absolute; };
inline Reference cpu_reference(const Input& in,int m,int n,int k) {
    Reference ref{std::vector<double>(size_t(m)*n,0),std::vector<double>(size_t(m)*n,0)};
    // Rows independent; each output accumulates K in increasing order in double.
    #pragma omp parallel for schedule(static) if(size_t(m)*n*k>1000000)
    for(int row=0;row<m;++row) {
        for(int q=0;q<k;++q) {
            double a=in.a[size_t(row)*k+q];
            for(int col=0;col<n;++col) {
                double product=a*double(in.b[size_t(q)*n+col]);
                ref.value[size_t(row)*n+col]+=product;
                ref.absolute[size_t(row)*n+col]+=std::abs(product);
            }
        }
    }
    return ref;
}
struct Comparison { size_t mismatches=0; double max_error=0,max_tolerance=0,max_error_ratio=0; };
inline Comparison compare(const std::vector<float>& actual,const Reference& ref,int k,bool exact) {
    Comparison result;
    double ke=k*double(std::numeric_limits<float>::epsilon());
    if(ke>=1) throw std::runtime_error("Unsupported error bound");
    for(size_t i=0;i<actual.size();++i) {
        double tolerance=exact?0:ke/(1-ke)*ref.absolute[i]+1e-7;
        double error=std::abs(double(actual[i])-ref.value[i]);
        bool valid=std::isfinite(actual[i]) && error<=tolerance;
        if(!valid) ++result.mismatches;
        result.max_error=std::max(result.max_error,std::isfinite(error)?error:std::numeric_limits<double>::infinity());
        result.max_tolerance=std::max(result.max_tolerance,tolerance);
        double ratio=tolerance?error/tolerance:(error==0?0:std::numeric_limits<double>::infinity());
        result.max_error_ratio=std::max(result.max_error_ratio,std::isfinite(ratio)?ratio:std::numeric_limits<double>::infinity());
    }
    return result;
}
