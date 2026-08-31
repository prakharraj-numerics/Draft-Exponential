#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>
#include "exp53_custom2_microscheduler_sweep_0100_3000_candidate.hpp"
#include "exp53_custom2_conservative_safe_0100_3000_candidate.hpp"
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
struct Buf{double*p=nullptr;explicit Buf(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double))||!p)std::abort();}~Buf(){std::free(p);}};
static inline uint64_t sm64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
static inline double u01(uint64_t h){return ((double)(h>>11)+.5)*(1.0/9007199254740992.0);}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;++i){double u=u01(sm64(s+i*0x9e3779b97f4a7c15ULL));double a=d?1.000001+u*98.999998:0x1p-20+u*(1.0-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls_for(size_t n){size_t c=180000/n;if(c<24)c=24;if(c>1800)c=1800;return c;}
template<class F>static double tim(F&&f,size_t n,size_t calls,int samples){for(int w=0;w<4;++w)f();std::vector<double>s;for(int k=0;k<samples;++k){auto a=std::chrono::steady_clock::now();for(size_t c=0;c<calls;++c)f();auto b=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::nano>(b-a).count()/((double)calls*n));}return med(s);}
int main(){constexpr size_t M=3000;Buf in(M),ref(M),out(M);Exp53Custom2MicroschedulerSweepCandidate m;Exp53Custom2ConservativeSafeCandidate base;volatile double sink=0;const unsigned shares[]={0,10,15,20,25,30,34,38,41,44,46,48,50};std::cout<<std::fixed<<std::setprecision(9);std::cout<<"CUSTOM2_MICROSCHED_SWEEP sizes=100..3000 step=50 shares=0,10,15,20,25,30,34,38,41,44,46,48,50 bitwise_gate=required\n";
for(int d=0;d<2;++d){const char*dn=d?"mid":"unit";for(size_t n=100;n<=3000;n+=50){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);size_t calls=calls_for(n);double best=1e300;unsigned bestp=0;for(unsigned p:shares){std::memset(out.p,0,n*sizeof(double));m.run_share(out.p,in.p,n,p);if(std::memcmp(out.p,ref.p,n*sizeof(double))){std::cout<<"FAIL domain="<<dn<<" n="<<n<<" share="<<p<<" bitdiff=1\n";return 9;}auto fn=[&]{m.run_share(out.p,in.p,n,p);};double t=tim(fn,n,calls,5);std::cout<<"SWEEP domain="<<dn<<" n="<<n<<" share="<<p<<" ns="<<t<<" bitdiff=0\n";if(t<best){best=t;bestp=p;}}
auto bc=[&]{base.run(out.p,in.p,n);};auto fc=[&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);};auto ic=[&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};double bt=tim(bc,n,calls,7),ft=tim(fc,n,calls,7),it=tim(ic,n,calls,7);sink+=out.p[(n*7u+d)%n]*0x1p-1022;std::cout<<"FINAL domain="<<dn<<" n="<<n<<" best_share="<<bestp<<" best_ns="<<best<<" conservative_ns="<<bt<<" frozen_ns="<<ft<<" intel_ns="<<it<<" intel_over_best="<<(it/best)<<" conservative_over_best="<<(bt/best)<<" bitdiff=0\n";}}
if(sink==1234567.0)std::fprintf(stderr,"sink=%g\n",(double)sink);return 0;}
