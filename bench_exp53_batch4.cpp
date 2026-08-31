#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

extern "C" void exp53_n2_fused_u4_038_frozen(double*, const double*, size_t);

#if defined(USE_VCL)
#include "vectorclass.h"
#include "vectormath_exp.h"
static const char* libname="VCL";
static void ext_exp(double* o,const double* in,size_t n){
  size_t i=0;
  for(;i+8<=n;i+=8){ Vec8d x; x.load(in+i); Vec8d y=exp(x); y.store(o+i); }
  for(;i<n;i++) o[i]=std::exp(in[i]);
}
#elif defined(USE_XSIMD)
#include <xsimd/xsimd.hpp>
static const char* libname="XSIMD";
static void ext_exp(double* o,const double* in,size_t n){
  using B=xsimd::batch<double>;
  const size_t W=B::size;
  size_t i=0;
  for(;i+W<=n;i+=W){ B x=B::load_unaligned(in+i); B y=xsimd::exp(x); y.store_unaligned(o+i); }
  for(;i<n;i++) o[i]=std::exp(in[i]);
}
#elif defined(USE_VDT)
#include "exp.h"
static const char* libname="VDT";
static void ext_exp(double* o,const double* in,size_t n){
  const uint32_t nn=(uint32_t)n;
  // VDT's public batch signature. Implementation is generated from its inline fast_exp core.
  for(uint32_t i=0;i<nn;i++) o[i]=vdt::fast_exp(in[i]);
}
#elif defined(USE_MIPP)
#include <mipp.h>
static const char* libname="MIPP";
static void ext_exp(double* o,const double* in,size_t n){
  const size_t W=mipp::N<double>();
  size_t i=0;
  for(;i+W<=n;i+=W){ mipp::Reg<double> x(in+i); auto y=mipp::exp(x); y.store(o+i); }
  for(;i<n;i++) o[i]=std::exp(in[i]);
}
#else
#error choose library macro
#endif

static inline uint64_t ord(double x){
  uint64_t u; memcpy(&u,&x,8);
  return (u>>63) ? ~u : (u | 0x8000000000000000ULL);
}
static inline uint64_t ulpd(double a,double b){
  if(std::isnan(a)||std::isnan(b)) return UINT64_MAX;
  uint64_t x=ord(a), y=ord(b); return x>y?x-y:y-x;
}
static uint64_t rngs=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(){ rngs^=rngs<<7; rngs^=rngs>>9; rngs^=rngs<<8; return rngs; }
static inline double genx(size_t i){
  double u=(double)(xr()>>11)*(1.0/9007199254740992.0);
  double mag;
  if((i&3)<2) mag=0.000001 + u*0.999998; else mag=1.0 + u*99.0;
  return (i&1)?-mag:mag;
}
static void accuracy(){
  const size_t n=200000;
  std::vector<double> in(n),out(n);
  rngs=0x123456789abcdef0ULL;
  for(size_t i=0;i<n;i++) in[i]=genx(i);
  ext_exp(out.data(),in.data(),n);
  uint64_t mx=0,gt1=0,gt2=0,gt4=0;
  for(size_t i=0;i<n;i++){
    double ref=(double)expl((long double)in[i]);
    uint64_t d=ulpd(out[i],ref); mx=std::max(mx,d); gt1+=d>1; gt2+=d>2; gt4+=d>4;
  }
  printf("ACC lib=%s n=%zu maxulp=%llu gt1=%llu gt2=%llu gt4=%llu\n",libname,n,(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)gt2,(unsigned long long)gt4);
}
using fn_t=void(*)(double*,const double*,size_t);
static double bench(fn_t fn,size_t n){
  std::vector<double> in(n),out(n);
  rngs=0xdecafbad12345678ULL;
  for(size_t i=0;i<n;i++) in[i]=genx(i);
  for(int w=0;w<40;w++) fn(out.data(),in.data(),n);
  size_t reps=1; double sec=0;
  do{
    auto a=std::chrono::steady_clock::now();
    for(size_t r=0;r<reps;r++) fn(out.data(),in.data(),n);
    auto b=std::chrono::steady_clock::now();
    sec=std::chrono::duration<double>(b-a).count();
    if(sec<0.20) reps*=2;
  }while(sec<0.20 && reps<(1u<<20));
  volatile double sink=out[n/3]; (void)sink;
  return sec*1e9/((double)reps*n);
}
int main(){
  printf("LIB=%s\n",libname);
#if defined(USE_XSIMD)
  printf("XSIMD_WIDTH=%zu\n",(size_t)xsimd::batch<double>::size);
#elif defined(USE_MIPP)
  printf("MIPP_WIDTH=%d\n",mipp::N<double>());
#endif
  accuracy();
  for(size_t n: {size_t(12288),size_t(65536)}){
    double e=bench(ext_exp,n); double o=bench(exp53_n2_fused_u4_038_frozen,n);
    printf("TIME lib=%s n=%zu external_ns=%.9f ours_ns=%.9f ext_over_ours=%.6f\n",libname,n,e,o,e/o);
  }
  return 0;
}
