#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <vector>
#include <mkl_vml.h>
#include "exp53_hwloc_numa_experiment.hpp"

static inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline double u01(uint64_t h) {
    return (static_cast<double>(h >> 11) + .5) * (1.0 / 9007199254740992.0);
}
static inline uint64_t bits(double x) {
    uint64_t u; std::memcpy(&u,&x,sizeof(u)); return u;
}
static double cpu_ns() {
    timespec t{}; clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);
    return static_cast<double>(t.tv_sec)*1e9 + t.tv_nsec;
}
static double median(std::vector<double> v) {
    std::sort(v.begin(),v.end()); return v[v.size()/2];
}
static void fill(double* x,size_t n) {
    const double e=0x1p-20;
    for(size_t i=0;i<n;++i) {
        const double u=u01(mix(0xd1b54a32d192ed03ULL+(n<<17)+i*0x9e3779b97f4a7c15ULL));
        double v=(i&2)?1.0+u*(99.0-2*e)+e:e+u*(1.0-2*e);
        x[i]=(i&1)?-v:v;
    }
}
static double* aligned_alloc64(size_t n) {
    double* p=nullptr;
    if(posix_memalign(reinterpret_cast<void**>(&p),64,n*sizeof(double))) return nullptr;
    return p;
}

int main(int argc,char** argv) {
    if(argc!=3) return 2;
    const std::string mode=argv[1];
    if(mode!="standard" && mode!="numa") return 2;
    const size_t n=std::strtoull(argv[2],nullptr,10);
    const std::vector<size_t> allowed={100,700,3500,15000,50000,1000000,2000000};
    if(std::find(allowed.begin(),allowed.end(),n)==allowed.end()) return 2;

    Exp53HwlocNumaExperiment ex(2);
    if(mode=="numa" && !ex.usable()) {
        std::cerr<<"NUMA_UNUSABLE same_node="<<ex.same_numa_node()<<" node="<<ex.numa_node()<<"\n";
        return 3;
    }
    double* in=mode=="numa"?ex.allocate(n):aligned_alloc64(n);
    double* out=mode=="numa"?ex.allocate(n):aligned_alloc64(n);
    double* ref=mode=="numa"?ex.allocate(n):aligned_alloc64(n);
    if(!in||!out||!ref) return 4;
    fill(in,n);
    vmdExp(static_cast<MKL_INT>(n),in,ref,VML_HA);

    auto run=[&]{ex.run(out,in,n,2);};
    run();
    uint64_t maxulp=0;
    for(size_t i=0;i<n;++i) {
        if(!std::isfinite(out[i])||out[i]<=0) return 5;
        const uint64_t a=bits(out[i]),b=bits(ref[i]);
        maxulp=std::max(maxulp,a>b?a-b:b-a);
    }
    if(maxulp>2) return 6;
    for(int i=0;i<8;++i) run();

    size_t calls=50000000ULL/n;
    calls=std::max<size_t>(1,calls);
    calls=std::min<size_t>(500000,calls);
    std::vector<double> wall,cpu;
    volatile double sink=0;
    for(int s=0;s<7;++s) {
        const double c0=cpu_ns();
        const auto w0=std::chrono::steady_clock::now();
        for(size_t k=0;k<calls;++k) run();
        const auto w1=std::chrono::steady_clock::now();
        const double c1=cpu_ns();
        wall.push_back(std::chrono::duration<double,std::nano>(w1-w0).count()/(calls*static_cast<double>(n)));
        cpu.push_back((c1-c0)/(calls*static_cast<double>(n)));
        sink+=out[(s*104729ULL)%n]*0x1p-1022;
    }
    rusage ru{};getrusage(RUSAGE_SELF,&ru);
    const double w=median(wall),c=median(cpu);
    std::cout<<std::fixed<<std::setprecision(9)
             <<"RESULT mode="<<mode<<" n="<<n<<" calls="<<calls
             <<" wall_ns="<<w<<" cpu_ns="<<c<<" effective_cores="<<c/w
             <<" maxrss_kib="<<ru.ru_maxrss<<" numa_node="<<ex.numa_node()
             <<" ancestor_type="<<ex.common_ancestor_type()<<" maxulp="<<maxulp<<"\n";
    if(mode=="numa"){ex.deallocate(in,n);ex.deallocate(out,n);ex.deallocate(ref,n);}
    else{std::free(in);std::free(out);std::free(ref);}
    if(sink==123)std::cerr<<sink;
    return 0;
}
