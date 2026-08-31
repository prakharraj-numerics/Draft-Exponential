// Focused low-batch benchmark: frozen EXP kernel + Intel OpenMP hot 2-thread team
// versus plain Intel VML_HA. Sizes n=50..3000 step 50. No FastFlow/custom2/NT.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include <omp.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double *out,
                                                  const double *in,
                                                  size_t n);

struct AlignedDoubles {
    double *p=nullptr;
    explicit AlignedDoubles(size_t n) {
        if (posix_memalign((void**)&p,64,n*sizeof(double))!=0 || !p) std::exit(2);
    }
    ~AlignedDoubles(){ std::free(p); }
};

static inline uint64_t splitmix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
static double unit01(uint64_t h){ return ((double)(h>>11)+0.5)*(1.0/9007199254740992.0); }
static const char* domain_name(int d){ return d==0?"unit":"mid"; }
static void fill_inputs(double* x,size_t n,int domain){
    const uint64_t seed0=0xbb67ae8584caa73bULL ^ ((uint64_t)n<<17) ^ ((uint64_t)domain<<57);
    for(size_t i=0;i<n;++i){
        const double u=unit01(splitmix64(seed0+i*0x9e3779b97f4a7c15ULL));
        const double e=0x1p-20;
        const double m=(domain==0)?(e+u*(1.0-2.0*e)):(1.0+e+u*(99.0-2.0*e));
        x[i]=(i&1)?-m:m;
    }
}
static size_t call_count(size_t n){
    size_t c=4000000ULL/n;
    if(c<2)c=2;
    if(c>100000)c=100000;
    return c;
}
static double median(std::vector<double>& v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

// Intel OpenMP candidate. The libiomp hot-team cache is kept active by environment;
// each call pays the real OpenMP fork/join + barrier cost. Work itself is exactly two
// static 32-element-aligned contiguous pieces, with serial fallback if two full blocks
// do not exist.
static inline void run_omp(double* out,const double* in,size_t n){
    const size_t full32=n/32;
    if(full32<2){
        exp53_n2_vmstyle_u4_0381_frozen(out,in,n);
        return;
    }
    #pragma omp parallel num_threads(2)
    {
        const int tid=omp_get_thread_num();
        const size_t b0=(full32*(size_t)tid)/2;
        const size_t b1=(full32*(size_t)(tid+1))/2;
        const size_t lo=32*b0;
        size_t hi=32*b1;
        if(tid==1) hi=n;
        exp53_n2_vmstyle_u4_0381_frozen(out+lo,in+lo,hi-lo);
    }
}
static inline void run_intel(double* out,const double* in,size_t n){
    vmdExp((MKL_INT)n,in,out,VML_HA);
}

int main(int argc,char** argv){
    if(argc!=2){ std::fprintf(stderr,"usage: %s omp|intel\n",argv[0]); return 2; }
    const std::string stack=argv[1];
    if(stack!="omp" && stack!="intel") return 2;
    const bool ours=stack=="omp";
    const bool reverse=std::getenv("SWEEP_REVERSE") && std::string(std::getenv("SWEEP_REVERSE"))=="1";

    omp_set_dynamic(0);
    omp_set_num_threads(2);

    std::vector<size_t> sizes;
    for(size_t n=50;n<=3000;n+=50)sizes.push_back(n);
    if(reverse)std::reverse(sizes.begin(),sizes.end());
    std::vector<int> domains={0,1};
    if(reverse)std::reverse(domains.begin(),domains.end());

    // Prime Intel OpenMP hot team before measured sizes.
    if(ours){
        AlignedDoubles a(256),b(256); fill_inputs(a.p,256,0);
        for(int k=0;k<64;++k) run_omp(b.p,a.p,256);
    }

    std::cout<<std::fixed<<std::setprecision(9);
    volatile double sink=0.0;
    for(int domain:domains){
        for(size_t n:sizes){
            AlignedDoubles in(n),out(n); fill_inputs(in.p,n,domain);
            const size_t calls=call_count(n);
            auto invoke=[&](){ if(ours)run_omp(out.p,in.p,n); else run_intel(out.p,in.p,n); };
            for(int w=0;w<8;++w)invoke();
            std::vector<double> samples; samples.reserve(7);
            for(int s=0;s<7;++s){
                const auto t0=std::chrono::steady_clock::now();
                for(size_t c=0;c<calls;++c)invoke();
                const auto t1=std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/((double)calls*(double)n));
            }
            sink += out.p[(n*7u+(size_t)domain)%n]*0x1p-1022;
            std::cout<<"RESULT stack="<<stack<<" domain="<<domain_name(domain)
                     <<" n="<<n<<" calls="<<calls<<" ns="<<median(samples)<<"\n";
        }
    }
    if(sink==123456789.0) std::fprintf(stderr,"sink=%g\n",(double)sink);
    return 0;
}
