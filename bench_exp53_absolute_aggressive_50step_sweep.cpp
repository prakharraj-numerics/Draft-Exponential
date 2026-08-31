#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include "exp53_absolute_aggressive_0100_3000_candidate.hpp"
#include "exp53_batch_custom_2core_nt_frozen.hpp"

struct AlignedDoubles {
    double *p=nullptr;
    explicit AlignedDoubles(size_t n){
        if(posix_memalign((void**)&p,64,n*sizeof(double))||!p){std::fprintf(stderr,"alloc fail\n");std::exit(2);}
    }
    ~AlignedDoubles(){std::free(p);}
    AlignedDoubles(const AlignedDoubles&)=delete;
    AlignedDoubles& operator=(const AlignedDoubles&)=delete;
};

static inline uint64_t splitmix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} 
static double unit01(uint64_t h){return ((double)(h>>11)+0.5)*(1.0/9007199254740992.0);} 
static const char* domain_name(int d){return d==0?"unit":"mid";} 
static void fill_inputs(double*x,size_t n,int domain){
    const uint64_t seed0=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)domain<<59);
    for(size_t i=0;i<n;++i){
        double u=unit01(splitmix64(seed0+i*0x9e3779b97f4a7c15ULL));
        double e=0x1p-20;
        double m=(domain==0)?(e+u*(1.0-2.0*e)):(1.0+e+u*(99.0-2.0*e));
        x[i]=(i&1)?-m:m;
    }
}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t target_calls(size_t n,size_t elems,size_t cap){size_t c=elems/n;if(c<2)c=2;if(c>cap)c=cap;return c;}

template<class F> static double one_ns(F&&f,size_t calls,size_t n){
    auto t0=std::chrono::steady_clock::now();
    for(size_t c=0;c<calls;++c)f();
    auto t1=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::nano>(t1-t0).count()/((double)calls*(double)n);
}
template<class F> static double med_ns(F&&f,size_t calls,size_t n,int reps){
    for(int w=0;w<4;++w)f();
    std::vector<double> s; s.reserve(reps);
    for(int k=0;k<reps;++k)s.push_back(one_ns(f,calls,n));
    return median(s);
}

struct Geo { unsigned pct; size_t align; double scout; bool ok; };

int main(){
    std::cout<<std::fixed<<std::setprecision(9);
    std::cout<<"ABSOLUTE_AGGRESSIVE_SEARCH shares=5..50_step1 align=16,32 coarse_then_refine correctness_gate=bitwise\n";
    Exp53AbsoluteAggressive2CoreCandidate cand;
    Exp53CustomPermanent2CoreFrozen custom;
    volatile double sink=0.0;

    for(int domain=0;domain<2;++domain){
        for(size_t n=100;n<=3000;n+=50){
            AlignedDoubles in(n),ref(n),out(n);
            fill_inputs(in.p,n,domain);
            exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);

            std::vector<Geo> geos;
            geos.reserve(92);
            const size_t scout_calls=target_calls(n,180000,4000);
            for(size_t al: {size_t(16),size_t(32)}){
                for(unsigned pct=5;pct<=50;++pct){
                    std::memset(out.p,0,n*sizeof(double));
                    cand.run_share(out.p,in.p,n,pct,al);
                    bool ok=std::memcmp(out.p,ref.p,n*sizeof(double))==0;
                    double ns=std::numeric_limits<double>::infinity();
                    if(ok){
                        auto invoke=[&](){cand.run_share(out.p,in.p,n,pct,al);};
                        ns=one_ns(invoke,scout_calls,n);
                    }
                    geos.push_back({pct,al,ns,ok});
                }
            }
            std::sort(geos.begin(),geos.end(),[](const Geo&a,const Geo&b){return a.scout<b.scout;});

            const size_t refine_calls=target_calls(n,2500000,30000);
            double best=std::numeric_limits<double>::infinity();
            unsigned bestpct=0; size_t bestal=0;
            int refined=0;
            for(const Geo&g:geos){
                if(!g.ok) continue;
                auto invoke=[&](){cand.run_share(out.p,in.p,n,g.pct,g.align);};
                double ns=med_ns(invoke,refine_calls,n,7);
                ++refined;
                if(ns<best){best=ns;bestpct=g.pct;bestal=g.align;}
                if(refined>=8) break;
            }

            auto serial_invoke=[&](){exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);};
            auto custom_invoke=[&](){custom.run(out.p,in.p,n);};
            auto intel_invoke=[&](){vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};
            double serial=med_ns(serial_invoke,refine_calls,n,7);
            double customns=med_ns(custom_invoke,refine_calls,n,7);
            double intel=med_ns(intel_invoke,refine_calls,n,7);

            int invalid=0; for(const auto&g:geos) if(!g.ok) ++invalid;
            const char* winner=(best<intel)?"OURS":"INTEL";
            sink += ref.p[(n*13u+(size_t)domain)%n]*0x1p-1022;
            std::cout<<"FINAL domain="<<domain_name(domain)
                     <<" n="<<n
                     <<" best_ns="<<best
                     <<" best_pct="<<bestpct
                     <<" best_align="<<bestal
                     <<" serial_ns="<<serial
                     <<" custom2_ns="<<customns
                     <<" intel_ns="<<intel
                     <<" intel_over_best="<<(intel/best)
                     <<" invalid_geometries="<<invalid
                     <<" winner="<<winner<<"\n";
        }
    }
    if(sink==123456789.0)std::fprintf(stderr,"sink=%g\n",(double)sink);
    return 0;
}
