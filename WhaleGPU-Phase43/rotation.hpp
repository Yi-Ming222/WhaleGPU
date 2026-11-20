#pragma once
#include <functional>
// Run on distinct, preallocated device address ranges. Restore owner pointers on exit.
struct Ring43 {
    size_t slots,slot=0;
    std::vector<void*> allocations;
    std::vector<std::function<void(size_t)>> binders;
    std::vector<std::function<void()>> restore;
    Ring43(Logs& l,size_t active_bytes){
        cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));
        slots=2*std::max<size_t>(11,size_t(p.l2CacheSize)/active_bytes+2);
        l.allocation<<"# device_l2="<<p.l2CacheSize<<" slots="<<slots<<" active_bytes_per_slot="<<active_bytes<<'\n';
    }
    template<class T>void add(T*& target,size_t count){
        T* orig=target;T* mem=nullptr;
        CUDA_CHECK(cudaMalloc(&mem,slots*count*sizeof(T)));allocations.push_back(mem);
        binders.emplace_back([&target,mem,count](size_t i){target=mem+i*count;});
        restore.emplace_back([&target,orig]{target=orig;});
    }
    void select(size_t i){for(auto& f:binders)f(i);}
    ~Ring43(){for(auto& f:restore)f();for(void* b:allocations)cudaFree(b);}
};
template<class Up,class Op,class Down,class Check,class Poison,class Sync>
void rotate43(Logs& l,std::string key,Ring43& ring,Up upload,Op op,Down down,Check verify,Poison poison,Sync sync){
    auto bind=[&](size_t slot){ring.select(slot);sync();};
    const std::string names[]={"split","fused"};
    for(int id:{0,1})for(size_t slot=0;slot<ring.slots;++slot){
        bind(slot);upload(id);poison();op(id);down();verify(id).save(l,key,names[id]);
    }
    if(!l.bench)return;
    for(int id:{0,1})for(int i=0;i<10;++i){bind(i%ring.slots);op(id);}
    CUDA_CHECK(cudaDeviceSynchronize());
    Event start,stop;
    int first=0;
    for(int r=0;r<21;++r){
        first=r%2?1-first:int(l.rng()%2);
        for(int place:{0,1}){
            int id=first^place;
            // Each variant traverses a disjoint half with the same cyclic offsets.
            size_t base=size_t(r)*10;
            auto slot_at=[&](size_t offset){return id*(ring.slots/2)+(base+offset)%(ring.slots/2);};
            CUDA_CHECK(cudaEventRecord(start.value));
            for(int j=0;j<10;++j){bind(slot_at(j));op(id);}
            CUDA_CHECK(cudaEventRecord(stop.value));CUDA_CHECK(cudaEventSynchronize(stop.value));
            float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));
            for(int j=0;j<10;++j){bind(slot_at(j));down();verify(id);}
            bind(slot_at(0));auto t=Clock43::now();upload(id);op(id);down();
            double wall=std::chrono::duration<double,std::milli>(Clock43::now()-t).count();verify(id);
            l.raw<<key<<','<<names[id]<<','<<r<<','<<place<<','<<ms/10<<','<<wall<<'\n';
        }
    }
}
