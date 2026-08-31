#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include "exp53_batch_production.hpp"

struct AlignedDoubles {
    double *p=nullptr;
    explicit AlignedDoubles(size_t n) {
        if (posix_memalign((void**)&p,64,n*sizeof(double))!=0 || !p) std::exit(2);
    }
    ~AlignedDoubles(){ std::free(p); }
};

static inline uint64_t splitmix64(uint64_t x){
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
static double unit01(uint64_t h){ return ((double)(h>>11)+0.5)*(1.0/9007199254740992.0); }

static void fill_extreme(double *x,size_t n){
    uint64_t seed=0x243f6a8885a308d3ULL ^ ((uint64_t)n<<19);
    for(size_t i=0;i<n;i++){
        double u=unit01(splitmix64(seed+i*0x9e3779b97f4a7c15ULL));
        double m=1000.0+0x1p-10+u*(4000.0-0x1p-9);
        x[i]=(i&1)?-m:m;
    }
}

static size_t classify_bad(const double *got,size_t n){
    size_t bad=0;
    for(size_t i=0;i<n;i++){
        if((i&1)==0){
            if(!(std::isinf(got[i]) && !std::signbit(got[i]))) ++bad;
        } else {
            if(!(got[i]==0.0 && !std::signbit(got[i]))) ++bad;
        }
    }
    return bad;
}

static const std::vector<size_t>& sizes(const std::string& p){
    static const std::vector<size_t> low={50,100,250,500,800,1000,2000,3000};
    static const std::vector<size_t> med={5000,8000,15000,25000,35000,50000,65000};
    static const std::vector<size_t> high={200000,500000,1000000,1500000,2500000};
    static const std::vector<size_t> highest={4000000,5000000,8000000};
    if(p=="low") return low; if(p=="medium") return med; if(p=="high") return high; return highest;
}
static size_t calls_for(size_t n){
    size_t c=100000ULL/n;
    if(c<1) c=1;
    if(c>2000) c=2000;
    return c;
}
static int samples_for(const std::string& p){ return (p=="low"||p=="medium")?5:3; }
static double median(std::vector<double>& v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(int argc,char **argv){
    if(argc!=3){ std::fprintf(stderr,"usage: %s custom|intel hot|stream\n",argv[0]); return 2; }
    std::string stack=argv[1], policy=argv[2];
    bool custom=stack=="custom", stream=policy=="stream";
    if((!custom&&stack!="intel")||(policy!="hot"&&policy!="stream")) return 2;
    Exp53BatchProductionExecutor *pex=custom?new Exp53BatchProductionExecutor(2):nullptr;
    constexpr size_t RING_BYTES=256ULL*1024ULL*1024ULL;
    std::cout<<std::fixed<<std::setprecision(9);
    std::cout<<"STACK="<<stack<<" POLICY="<<policy<<" DOMAIN=extreme\n";

    for(const std::string phase: {"low","medium","high","highest"}){
        for(size_t n:sizes(phase)){
            AlignedDoubles in(n), check(n);
            fill_extreme(in.p,n);
            if(custom){
                if(stream) pex->run_streaming_write_once(check.p,in.p,n); else pex->run(check.p,in.p,n);
            } else {
                vmdExp((MKL_INT)n,in.p,check.p,VML_HA);
            }
            size_t bad=classify_bad(check.p,n);
            size_t stride=(n+7u)&~size_t(7u);
            size_t slots=1;
            if(stream){ slots=RING_BYTES/(stride*sizeof(double)); if(slots<2) slots=2; }
            AlignedDoubles out(stride*slots);
            auto invoke=[&](double *dst){
                if(custom){ if(stream) pex->run_streaming_write_once(dst,in.p,n); else pex->run(dst,in.p,n); }
                else vmdExp((MKL_INT)n,in.p,dst,VML_HA);
            };
            for(int w=0;w<2;w++) invoke(out.p+(stream?(size_t)w%slots:0)*stride);
            size_t calls=calls_for(n); int nsamp=samples_for(phase);
            std::vector<double> samp; samp.reserve(nsamp);
            for(int s=0;s<nsamp;s++){
                auto t0=std::chrono::steady_clock::now();
                for(size_t c=0;c<calls;c++){
                    size_t slot=stream?((c+(size_t)s*calls)%slots):0;
                    invoke(out.p+slot*stride);
                }
                auto t1=std::chrono::steady_clock::now();
                double t=std::chrono::duration<double,std::nano>(t1-t0).count()/((double)calls*(double)n);
                samp.push_back(t);
            }
            std::cout<<"RESULT stack="<<stack<<" policy="<<policy<<" phase="<<phase
                     <<" domain=extreme n="<<n<<" calls="<<calls<<" slots="<<slots
                     <<" ns="<<median(samp)<<" valid="<<(bad==0?1:0)<<" bad="<<bad<<"\n";
        }
    }
    delete pex;
    return 0;
}
