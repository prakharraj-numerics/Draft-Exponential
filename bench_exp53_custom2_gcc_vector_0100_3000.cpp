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
#include "exp53_custom2_conservative_safe_0100_3000_candidate.hpp"
#include "exp53_custom2_gcc_vector_0100_3000_candidate.hpp"
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
extern "C" void exp53_gcc_vector_0100_3000(double*,const double*,size_t);
struct Buf{double*p=nullptr;explicit Buf(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double))||!p)std::abort();}~Buf(){std::free(p);}};
static inline uint64_t sm64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
static inline double u01(uint64_t h){return((double)(h>>11)+0.5)*(1.0/9007199254740992.0);}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;++i){double u=u01(sm64(s+i*0x9e3779b97f4a7c15ULL));double a=d?1.000001+u*98.999998:0x1p-20+u*(1.0-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls_for(size_t n){size_t c=300000/n;if(c<32)c=32;if(c>3000)c=3000;return c;}
template<class F>static double tim(F&&f,size_t n,size_t calls){for(int w=0;w<7;++w)f();std::vector<double>s;for(int k=0;k<9;++k){auto a=std::chrono::steady_clock::now();for(size_t c=0;c<calls;++c)f();auto b=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::nano>(b-a).count()/((double)calls*n));}return med(s);}
int main(){constexpr size_t M=3000;Buf in(M),ref(M),out(M);Exp53Custom2ConservativeSafeCandidate temporal;Exp53Custom2GCCVectorCandidate gcc2;volatile double sink=0;std::cout<<std::fixed<<std::setprecision(9);std::cout<<"GCC_VECTOR_FUNCTION sizes=100..3000 step=50 compare=gcc_single,gcc_custom2,temporal_custom2,intel,frozen bitwise_gate=required\n";
for(int d=0;d<2;++d){const char*dn=d?"mid":"unit";for(size_t n=100;n<=3000;n+=50){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);std::memset(out.p,0,n*sizeof(double));gcc2.run(out.p,in.p,n);int bd=std::memcmp(out.p,ref.p,n*sizeof(double))!=0;if(bd){std::cout<<"FAIL domain="<<dn<<" n="<<n<<" bitdiff=1\n";return 9;}size_t calls=calls_for(n);auto fc=[&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);};auto tc=[&]{temporal.run(out.p,in.p,n);};auto gs=[&]{exp53_gcc_vector_0100_3000(out.p,in.p,n);};auto gc=[&]{gcc2.run(out.p,in.p,n);};auto ic=[&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};double f=tim(fc,n,calls),t=tim(tc,n,calls),s=tim(gs,n,calls),g=tim(gc,n,calls),i=tim(ic,n,calls);sink+=out.p[(n*7u+d)%n]*0x1p-1022;std::cout<<"FINAL domain="<<dn<<" n="<<n<<" share="<<Exp53Custom2GCCVectorCandidate::conservative_share(n)<<" gcc_single_ns="<<s<<" gcc_custom2_ns="<<g<<" temporal_custom2_ns="<<t<<" frozen_ns="<<f<<" intel_ns="<<i<<" intel_over_gcc2="<<(i/g)<<" temporal_over_gcc2="<<(t/g)<<" bitdiff=0\n";}}
if(sink==1234567.0)std::fprintf(stderr,"sink=%g\n",(double)sink);return 0;}
