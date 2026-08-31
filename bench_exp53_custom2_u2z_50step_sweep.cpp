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
#include "exp53_batch_custom2_u2z_candidate.hpp"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
extern "C" void exp53_small_u2z_0100_frozen(double*, const double*, size_t);

struct AlignedDoubles {
    double *p=nullptr;
    explicit AlignedDoubles(size_t n){ if(posix_memalign((void**)&p,64,n*sizeof(double))||!p){std::fprintf(stderr,"alloc fail\n");std::exit(2);} }
    ~AlignedDoubles(){std::free(p);} AlignedDoubles(const AlignedDoubles&)=delete; AlignedDoubles& operator=(const AlignedDoubles&)=delete;
};
static inline uint64_t splitmix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} 
static double unit01(uint64_t h){return ((double)(h>>11)+0.5)*(1.0/9007199254740992.0);} 
static const char* domain_name(int d){return d==0?"unit":"mid";} 
static void fill_inputs(double*x,size_t n,int domain){const uint64_t seed0=0x6a09e667f3bcc909ULL^((uint64_t)n<<17)^((uint64_t)domain<<57);for(size_t i=0;i<n;++i){double u=unit01(splitmix64(seed0+i*0x9e3779b97f4a7c15ULL));double e=0x1p-20;double m=(domain==0)?(e+u*(1.0-2.0*e)):(1.0+e+u*(99.0-2.0*e));x[i]=(i&1)?-m:m;}}
static size_t call_count(size_t n){size_t c=4000000ULL/n;if(c<2)c=2;if(c>100000)c=100000;return c;} 
static double median(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}

int main(int argc,char**argv){
    if(argc!=2){std::fprintf(stderr,"usage: %s custom2u2z|u2z|frozen|intel\n",argv[0]);return 2;}
    std::string stack=argv[1]; if(stack!="custom2u2z"&&stack!="u2z"&&stack!="frozen"&&stack!="intel")return 2;
    bool reverse=std::getenv("SWEEP_REVERSE")&&std::string(std::getenv("SWEEP_REVERSE"))=="1";
    std::vector<size_t> sizes;for(size_t n=50;n<=3000;n+=50)sizes.push_back(n);if(reverse)std::reverse(sizes.begin(),sizes.end());
    std::vector<int> domains={0,1};if(reverse)std::reverse(domains.begin(),domains.end());
    std::cout<<std::fixed<<std::setprecision(9)<<"STACK="<<stack<<" REVERSE="<<(reverse?1:0)<<"\n";
    volatile double sink=0.0;
    if(stack=="custom2u2z"){
        Exp53Custom2U2ZCandidate ex;
        for(int domain:domains)for(size_t n:sizes){AlignedDoubles in(n),out(n);fill_inputs(in.p,n,domain);size_t calls=call_count(n);auto invoke=[&](){ex.run(out.p,in.p,n);};for(int w=0;w<5;++w)invoke();std::vector<double>s;for(int k=0;k<7;++k){auto t0=std::chrono::steady_clock::now();for(size_t c=0;c<calls;++c)invoke();auto t1=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/((double)calls*n));}sink+=out.p[(n*13u+(size_t)domain)%n]*0x1p-1022;std::cout<<"RESULT stack="<<stack<<" domain="<<domain_name(domain)<<" n="<<n<<" calls="<<calls<<" ns="<<median(s)<<"\n";}
    } else {
        for(int domain:domains)for(size_t n:sizes){AlignedDoubles in(n),out(n);fill_inputs(in.p,n,domain);size_t calls=call_count(n);auto invoke=[&](){if(stack=="u2z")exp53_small_u2z_0100_frozen(out.p,in.p,n);else if(stack=="frozen")exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);else vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};for(int w=0;w<5;++w)invoke();std::vector<double>s;for(int k=0;k<7;++k){auto t0=std::chrono::steady_clock::now();for(size_t c=0;c<calls;++c)invoke();auto t1=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/((double)calls*n));}sink+=out.p[(n*13u+(size_t)domain)%n]*0x1p-1022;std::cout<<"RESULT stack="<<stack<<" domain="<<domain_name(domain)<<" n="<<n<<" calls="<<calls<<" ns="<<median(s)<<"\n";}
    }
    if(sink==123456789.0)std::fprintf(stderr,"sink=%g\n",(double)sink);return 0;
}
