#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <mkl_vml.h>
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
extern "C" void exp53_attack_gather(double*,const double*,size_t);
extern "C" void exp53_attack_scalef(double*,const double*,size_t);
extern "C" void exp53_attack_full(double*,const double*,size_t);
extern "C" void exp53_attack_factored(double*,const double*,size_t);
using fn_t=void(*)(double*,const double*,size_t);
struct A{double*p=nullptr;explicit A(size_t n){if(posix_memalign((void**)&p,64,n*8)||!p)std::exit(2);}~A(){std::free(p);}};
static inline uint64_t mix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static inline double u01(uint64_t h){return(double(h>>11)+.5)*(1.0/9007199254740992.0);}static void fill(double*x,size_t n){const double e=0x1p-20;uint64_t s=0xd1b54a32d192ed03ULL^(uint64_t(n)<<17);for(size_t i=0;i<n;++i){double u=u01(mix64(s+i*0x9e3779b97f4a7c15ULL));unsigned k=i&3;double v=k<2?e+u*(1-2*e):1+e+u*(99-2*e);if(k&1)v=-v;x[i]=v;}}static inline uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}static uint64_t ulp(const double*a,const double*b,size_t n){uint64_t m=0;for(size_t i=0;i<n;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i])||a[i]<=0||b[i]<=0)return UINT64_MAX;uint64_t x=bits(a[i]),y=bits(b[i]);m=std::max(m,x>y?x-y:y-x);}return m;}static uint64_t diff(const double*a,const double*b,size_t n){uint64_t d=0;for(size_t i=0;i<n;++i)d+=bits(a[i])!=bits(b[i]);return d;}
extern "C" __attribute__((noinline,used)) void exp53_attack_profile_start(){asm volatile("":::"memory");}extern "C" __attribute__((noinline,used)) void exp53_attack_profile_stop(){asm volatile("":::"memory");}
static size_t split_for(size_t n){return((n/32)/2)*32;}static __attribute__((noinline)) void direct(fn_t f,double*out,const double*in,size_t n){if(n<=3000){f(out,in,n);return;}size_t s=split_for(n);f(out,in,s);f(out+s,in+s,n-s);}static __attribute__((noinline)) void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}static __attribute__((noinline)) void noop(double*out,const double*in,size_t n){asm volatile(""::"r"(out),"r"(in),"r"(n):"memory");}
int main(int argc,char**argv){if(argc!=3){std::cerr<<"usage: variant N\n";return 2;}std::string v=argv[1];size_t n=strtoull(argv[2],nullptr,10);fn_t f=nullptr;if(v=="base")f=exp53_n2_vmstyle_u4_0381_frozen;else if(v=="g")f=exp53_attack_gather;else if(v=="s")f=exp53_attack_scalef;else if(v=="full")f=exp53_attack_full;else if(v=="fact")f=exp53_attack_factored;else if(v!="intel"&&v!="noop")return 2;A in(n),out(n),ref(n),base(n);fill(in.p,n);intel(ref.p,in.p,n);direct(exp53_n2_vmstyle_u4_0381_frozen,base.p,in.p,n);auto invoke=[&]{if(f)direct(f,out.p,in.p,n);else if(v=="intel")intel(out.p,in.p,n);else noop(out.p,in.p,n);};if(v!="noop"){for(int i=0;i<4;++i)invoke();uint64_t u=ulp(out.p,ref.p,n);if(v!="fact"&&u>2)return 4;}exp53_attack_profile_start();invoke();exp53_attack_profile_stop();uint64_t u=0,d=0;if(v!="noop"){u=ulp(out.p,ref.p,n);d=diff(out.p,base.p,n);}volatile double sink=(v=="noop"?in.p[17%n]:out.p[17%n]);std::cout<<"ATTACK_SDE variant="<<v<<" n="<<n<<" maxulp="<<u<<" bitdiff="<<d<<" sink="<<sink<<"\n";return 0;}
