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
#include "production/exp53_batch_production.hpp"
#include "exp53_tail_eff_candidate.hpp"

struct Aligned {
    double* p=nullptr;
    explicit Aligned(size_t n){ if(posix_memalign(reinterpret_cast<void**>(&p),64,n*sizeof(double))||!p) std::exit(2); }
    ~Aligned(){ std::free(p); }
};
static inline uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
static inline double u01(uint64_t h){return (static_cast<double>(h>>11)+.5)*(1.0/9007199254740992.0);}
static void fill(double* x,size_t n){const double e=0x1p-20;const uint64_t seed=0xd1b54a32d192ed03ULL^(static_cast<uint64_t>(n)<<17);for(size_t i=0;i<n;++i){double u=u01(mix(seed+i*0x9e3779b97f4a7c15ULL));unsigned k=i&3u;double v=k<2?e+u*(1.0-2*e):1.0+e+u*(99.0-2*e);x[i]=(k&1u)?-v:v;}}
static inline uint64_t bits(double x){uint64_t u;std::memcpy(&u,&x,sizeof(u));return u;}
static uint64_t check(const double* a,const double* b,size_t n){uint64_t m=0;for(size_t i=0;i<n;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i])||a[i]<=0||b[i]<=0)return UINT64_MAX;uint64_t x=bits(a[i]),y=bits(b[i]);m=std::max(m,x>y?x-y:y-x);}return m;}
static double cpu_ns(){timespec t{};clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return static_cast<double>(t.tv_sec)*1e9+t.tv_nsec;}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

int main(int argc,char** argv){
    if(argc!=3)return 2;
    const std::string mode=argv[1];
    if(mode!="production"&&mode!="candidate")return 2;
    const size_t n=std::strtoull(argv[2],nullptr,10);
    const std::vector<size_t> allowed={100,700,3500,15000,50000,1000000,2000000};
    if(std::find(allowed.begin(),allowed.end(),n)==allowed.end())return 2;

    Aligned in(n),out(n),ref(n);fill(in.p,n);vmdExp(static_cast<MKL_INT>(n),in.p,ref.p,VML_HA);
    std::unique_ptr<Exp53BatchProductionExecutor> prod;
    std::unique_ptr<Exp53TailEffExecutorCandidate> cand;
    if(mode=="production")prod=std::make_unique<Exp53BatchProductionExecutor>(2);
    else cand=std::make_unique<Exp53TailEffExecutorCandidate>(2);
    auto invoke=[&]{if(prod)prod->run(out.p,in.p,n,2);else cand->run(out.p,in.p,n,2);};
    invoke();uint64_t maxulp=check(out.p,ref.p,n);if(maxulp>2){std::cerr<<"accuracy maxulp="<<maxulp<<"\n";return 4;}
    for(int i=0;i<8;++i)invoke();
    size_t calls=50000000ULL/n;calls=std::max<size_t>(1,calls);calls=std::min<size_t>(500000,calls);
    std::vector<double> wall,cpu;volatile double sink=0;
    for(int s=0;s<7;++s){double c0=cpu_ns();auto w0=std::chrono::steady_clock::now();for(size_t k=0;k<calls;++k)invoke();auto w1=std::chrono::steady_clock::now();double c1=cpu_ns();wall.push_back(std::chrono::duration<double,std::nano>(w1-w0).count()/(calls*static_cast<double>(n)));cpu.push_back((c1-c0)/(calls*static_cast<double>(n)));sink+=out.p[(s*104729ULL)%n]*0x1p-1022;}
    rusage ru{};getrusage(RUSAGE_SELF,&ru);double w=median(wall),c=median(cpu);
    std::cout<<std::fixed<<std::setprecision(9)<<"RESULT mode="<<mode<<" n="<<n<<" calls="<<calls<<" wall_ns="<<w<<" cpu_ns="<<c<<" effective_cores="<<c/w<<" maxrss_kib="<<ru.ru_maxrss<<" maxulp="<<maxulp<<" remainder="<<(n&31u)<<"\n";
    if(sink==123)std::cerr<<sink;return 0;
}
