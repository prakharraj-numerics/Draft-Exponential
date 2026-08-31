#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <mkl_vml.h>
#include "exp53_absolute_aggressive_0100_3000_candidate.hpp"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

struct Buf {
    double *p = nullptr;
    explicit Buf(size_t n) {
        if (posix_memalign((void**)&p, 64, n * sizeof(double)) || !p) std::abort();
    }
    ~Buf(){ std::free(p); }
};

static inline uint64_t sm64(uint64_t x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline double u01(uint64_t h){ return ((double)(h >> 11) + 0.5) * (1.0/9007199254740992.0); }
static void fill(double *x, size_t n, int domain){
    const uint64_t seed = 0x243f6a8885a308d3ULL ^ ((uint64_t)n << 19) ^ ((uint64_t)domain << 61);
    for(size_t i=0;i<n;++i){
        const double u = u01(sm64(seed + i * 0x9e3779b97f4a7c15ULL));
        const double a = (domain==0) ? (0x1p-20 + u*(1.0-0x1p-19)) : (1.000001 + u*98.999998);
        x[i] = (i & 1) ? -a : a;
    }
}
static double median(std::vector<double> v){
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}
static size_t calls_for(size_t n){
    size_t c = 250000 / n;
    if(c < 24) c = 24;
    if(c > 2500) c = 2500;
    return c;
}

template<class F>
static double time_ns_per_input(F &&fn, size_t n, size_t calls, int samples){
    for(int w=0; w<4; ++w) fn();
    std::vector<double> s;
    s.reserve(samples);
    for(int k=0;k<samples;++k){
        auto t0 = std::chrono::steady_clock::now();
        for(size_t c=0;c<calls;++c) fn();
        auto t1 = std::chrono::steady_clock::now();
        s.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/((double)calls*(double)n));
    }
    return median(s);
}

int main(){
    constexpr size_t MAXN = 3000;
    Buf in(MAXN), ref(MAXN), out(MAXN);
    Exp53AbsoluteAggressive2CoreCandidate cand;

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "SEARCH helper_percent=20..50 step=1 align=16,32 sizes=100..3000 step=50 correctness=bitwise\n";

    volatile double sink = 0.0;
    for(int domain=0; domain<2; ++domain){
        const char *dn = domain==0 ? "unit" : "mid";
        for(size_t n=100; n<=3000; n+=50){
            fill(in.p,n,domain);
            exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);
            const size_t calls = calls_for(n);

            auto frozen_call = [&]{ exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n); };
            auto intel_call  = [&]{ vmdExp((MKL_INT)n,in.p,out.p,VML_HA); };
            const double frozen_ns = time_ns_per_input(frozen_call,n,calls,5);
            const double intel_ns  = time_ns_per_input(intel_call,n,calls,5);

            double best_ns = std::numeric_limits<double>::infinity();
            unsigned best_share = 0;
            size_t best_align = 0;
            int eligible = 0;

            for(size_t align : {size_t(16), size_t(32)}){
                for(unsigned share=20; share<=50; ++share){
                    std::memset(out.p,0,n*sizeof(double));
                    cand.run_share(out.p,in.p,n,share,align);
                    if(std::memcmp(out.p,ref.p,n*sizeof(double)) != 0) {
                        std::cout << "REJECT domain="<<dn<<" n="<<n<<" share="<<share<<" align="<<align<<" reason=bitdiff\n";
                        continue;
                    }
                    ++eligible;
                    auto call = [&]{ cand.run_share(out.p,in.p,n,share,align); };
                    const double ns = time_ns_per_input(call,n,calls,3);
                    if(ns < best_ns){ best_ns=ns; best_share=share; best_align=align; }
                }
            }

            if(!eligible){
                std::cout << "FINAL domain="<<dn<<" n="<<n<<" eligible=0 frozen_ns="<<frozen_ns<<" intel_ns="<<intel_ns<<"\n";
                continue;
            }

            auto best_call = [&]{ cand.run_share(out.p,in.p,n,best_share,best_align); };
            const double final_best = time_ns_per_input(best_call,n,calls,7);
            std::memset(out.p,0,n*sizeof(double));
            cand.run_share(out.p,in.p,n,best_share,best_align);
            const int bitdiff = std::memcmp(out.p,ref.p,n*sizeof(double)) != 0;
            sink += out.p[(n*7u + (size_t)domain) % n] * 0x1p-1022;

            std::cout << "FINAL domain="<<dn
                      <<" n="<<n
                      <<" eligible="<<eligible
                      <<" best_share="<<best_share
                      <<" best_align="<<best_align
                      <<" best_ns="<<final_best
                      <<" frozen_ns="<<frozen_ns
                      <<" intel_ns="<<intel_ns
                      <<" intel_over_best="<<(intel_ns/final_best)
                      <<" frozen_over_best="<<(frozen_ns/final_best)
                      <<" bitdiff="<<bitdiff
                      <<"\n";
            if(bitdiff) return 9;
        }
    }
    if(sink==1234567.0) std::fprintf(stderr,"sink=%g\n",(double)sink);
    return 0;
}
