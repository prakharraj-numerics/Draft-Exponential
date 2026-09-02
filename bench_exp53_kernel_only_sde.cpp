#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>

extern "C" void exp53_vcl_u2z_0100_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

struct Aligned {
    double* p=nullptr;
    explicit Aligned(size_t n){ if(posix_memalign(reinterpret_cast<void**>(&p),64,n*sizeof(double))||!p) std::exit(2); }
    ~Aligned(){ std::free(p); }
};
static inline uint64_t mix64(uint64_t x){ x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
static inline double u01(uint64_t h){ return (static_cast<double>(h>>11)+0.5)*(1.0/9007199254740992.0); }
static void fill_mixed(double* x,size_t n){ const double e=0x1p-20; const uint64_t s=0xd1b54a32d192ed03ULL^(static_cast<uint64_t>(n)<<17); for(size_t i=0;i<n;++i){ const double u=u01(mix64(s+i*0x9e3779b97f4a7c15ULL)); const unsigned k=i&3u; double v=k<2?e+u*(1.0-2.0*e):1.0+e+u*(99.0-2.0*e); if(k&1u)v=-v; x[i]=v; } }
static inline uint64_t bits(double x){ uint64_t u; std::memcpy(&u,&x,sizeof(u)); return u; }
static uint64_t cross_check(const double* a,const double* b,size_t n){ uint64_t m=0; for(size_t i=0;i<n;++i){ if(!std::isfinite(a[i])||!std::isfinite(b[i])||a[i]<=0||b[i]<=0)return UINT64_MAX; const uint64_t x=bits(a[i]),y=bits(b[i]); m=std::max(m,x>y?x-y:y-x);} return m; }

extern "C" __attribute__((noinline,used)) void exp53_recon_profile_start(){ asm volatile("":::"memory"); }
extern "C" __attribute__((noinline,used)) void exp53_recon_profile_stop(){ asm volatile("":::"memory"); }

static inline size_t split_for_custom(size_t n){ const size_t full32=n/32; return (full32/2)*32; }
static __attribute__((noinline)) void ours_direct(double* out,const double* in,size_t n){
    if(n<=100){ exp53_vcl_u2z_0100_frozen(out,in,n); return; }
    if(n<=3000){ exp53_n2_vmstyle_u4_0381_frozen(out,in,n); return; }
    const size_t split=split_for_custom(n);
    exp53_n2_vmstyle_u4_0381_frozen(out,in,split);
    exp53_n2_vmstyle_u4_0381_frozen(out+split,in+split,n-split);
}
static __attribute__((noinline)) void intel_direct(double* out,const double* in,size_t n){
    vmdExp(static_cast<MKL_INT>(n),in,out,VML_HA);
}
static __attribute__((noinline)) void noop_direct(double* out,const double* in,size_t n){
    asm volatile(""::"r"(out),"r"(in),"r"(n):"memory");
}

int main(int argc,char** argv){
    if(argc!=3){ std::cerr<<"usage: "<<argv[0]<<" ours|intel|noop N\n"; return 2; }
    const std::string stack=argv[1]; const size_t n=std::strtoull(argv[2],nullptr,10);
    const std::vector<size_t> allowed={100,700,3500,15000,50000,1000000,2000000};
    if((stack!="ours"&&stack!="intel"&&stack!="noop")||std::find(allowed.begin(),allowed.end(),n)==allowed.end()) return 2;
    Aligned in(n),out(n),ref(n); fill_mixed(in.p,n); intel_direct(ref.p,in.p,n);
    auto invoke=[&]{ if(stack=="ours") ours_direct(out.p,in.p,n); else if(stack=="intel") intel_direct(out.p,in.p,n); else noop_direct(out.p,in.p,n); };
    if(stack!="noop"){ for(int i=0;i<4;++i)invoke(); const uint64_t u=cross_check(out.p,ref.p,n); if(u>2){ std::cerr<<"correctness maxulp="<<u<<"\n"; return 4; } }
    exp53_recon_profile_start(); invoke(); exp53_recon_profile_stop();
    volatile double sink = stack=="noop" ? in.p[(n*17ULL)%n] : out.p[(n*17ULL)%n];
    uint64_t maxulp=0; if(stack!="noop") maxulp=cross_check(out.p,ref.p,n);
    std::cout<<"KERNEL_SDE_RUN stack="<<stack<<" n="<<n<<" maxulp="<<maxulp<<" sink="<<sink;
    if(stack=="ours"&&n>3000) std::cout<<" split="<<split_for_custom(n)<<" second="<<(n-split_for_custom(n));
    std::cout<<"\n"; return 0;
}
