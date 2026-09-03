#define _GNU_SOURCE
#include <mkl.h>
#include <mpfr.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include "exp53_power_family_raw_candidates.hpp"
#include "exp53_m32d8_eft_depth_sweep.hpp"

using fn_t=void(*)(double*,const double*,size_t);
static volatile double g_sink=0.0;
extern "C" __attribute__((noinline)) void exp53_depth_profile_start(){asm volatile("":::"memory");}
extern "C" __attribute__((noinline)) void exp53_depth_profile_stop(){asm volatile("":::"memory");}
static void pin0(){cpu_set_t s;CPU_ZERO(&s);CPU_SET(0,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
static void* al64(size_t n){void*p=nullptr;return posix_memalign(&p,64,n)==0?p:nullptr;}
static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static double u01(uint64_t x){return((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static void fill(double*x,size_t n,bool neg){const double edge=0x1p-20;uint64_t seed=UINT64_C(0xa0761d6478bd642f)^(uint64_t)n*UINT64_C(0xe7037ed1a0b428db)^(neg?UINT64_C(0x8ebc6af09c88c6e3):0);for(size_t i=0;i<n;i++){double q=u01(seed+i*UINT64_C(0x9e3779b97f4a7c15));double a=edge+(1.0-2.0*edge)*q;x[i]=neg?-a:a;}}
static void r0(double*o,const double*x,size_t n){exp53_m32d8_eft_depth::d0(o,x,n);}static void r1(double*o,const double*x,size_t n){exp53_m32d8_eft_depth::d1(o,x,n);}static void r2(double*o,const double*x,size_t n){exp53_m32d8_eft_depth::d2(o,x,n);}static void r3(double*o,const double*x,size_t n){exp53_m32d8_eft_depth::d3(o,x,n);}static void r4(double*o,const double*x,size_t n){exp53_m32d8_eft_depth::d4(o,x,n);}static void r5(double*o,const double*x,size_t n){exp53_m32d8_eft_depth::d5(o,x,n);}
static void raw(double*o,const double*x,size_t n){exp53_power_family_raw::m32d8(o,x,n);}static void intel(double*o,const double*x,size_t n){vmdExp((MKL_INT)n,x,o,VML_HA);}static void noop(double*o,const double*x,size_t n){asm volatile(""::"r"(o),"r"(x),"r"(n):"memory");}
static fn_t choose(const std::string&s){if(s=="d0")return r0;if(s=="d1")return r1;if(s=="d2")return r2;if(s=="d3")return r3;if(s=="d4")return r4;if(s=="d5")return r5;if(s=="raw")return raw;if(s=="intel")return intel;if(s=="noop")return noop;return nullptr;}
static uint64_t ordered(double v){uint64_t u;std::memcpy(&u,&v,8);return(u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulpdist(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=ordered(a),y=ordered(b);return x>y?x-y:y-x;}
static double ns(clockid_t id){timespec t{};clock_gettime(id,&t);return(double)t.tv_sec*1e9+t.tv_nsec;}
static size_t reps_for(size_t n){size_t r=UINT64_C(12000000)/n;if(r<2)r=2;if(r>200000)r=200000;return r;}
static int accuracy(const std::string&s){fn_t fn=choose(s);if(!fn||s=="noop")return 2;constexpr size_t N=9600;double*x=(double*)al64(N*8),*y=(double*)al64(N*8),*iv=(double*)al64(N*8);for(size_t i=0;i<N;i++){double a;if(i==0)a=0;else if(i==1)a=std::nextafter(1.0,0.0);else if(i==2)a=1.0;else if(i==3)a=0x1p-20;else{double q=u01(UINT64_C(0xd1b54a32d192ed03)+i*UINT64_C(0x9e3779b97f4a7c15));a=0x1p-20+(1.0-0x1p-20)*q;}x[i]=(i&1)?-a:a;}fn(y,x,N);intel(iv,x,N);mpfr_t mx,mr;mpfr_init2(mx,256);mpfr_init2(mr,256);uint64_t mm=0,mi=0;size_t g1=0,g2=0;double wx=0,wy=0,wr=0;for(size_t i=0;i<N;i++){mpfr_set_d(mx,x[i],MPFR_RNDN);mpfr_exp(mr,mx,MPFR_RNDN);double ref=mpfr_get_d(mr,MPFR_RNDN);uint64_t d=ulpdist(y[i],ref),di=ulpdist(y[i],iv[i]);if(d>mm){mm=d;wx=x[i];wy=y[i];wr=ref;}mi=std::max(mi,di);g1+=d>1;g2+=d>2;}std::printf("DEPTH_ACCURACY stack=%s maxulp_vs_mpfr=%llu count_gt1=%zu count_gt2=%zu maxulp_vs_intel=%llu worst_x=%.17g worst_y=%.17g worst_ref=%.17g\n",s.c_str(),(unsigned long long)mm,g1,g2,(unsigned long long)mi,wx,wy,wr);mpfr_clear(mx);mpfr_clear(mr);free(iv);free(y);free(x);return 0;}
static int native(const std::string&s,size_t n,bool neg){fn_t fn=choose(s);if(!fn)return 2;double*x=(double*)al64(n*8),*y=(double*)al64(n*8);fill(x,n,neg);for(int w=0;w<5;w++)fn(y,x,n);size_t reps=reps_for(n);double wv[7],cv[7];for(int t=0;t<7;t++){double w0=ns(CLOCK_MONOTONIC_RAW),c0=ns(CLOCK_PROCESS_CPUTIME_ID);for(size_t r=0;r<reps;r++)fn(y,x,n);double c1=ns(CLOCK_PROCESS_CPUTIME_ID),w1=ns(CLOCK_MONOTONIC_RAW);wv[t]=(w1-w0)/((double)reps*n);cv[t]=(c1-c0)/((double)reps*n);g_sink+=y[(n*7u/11u+t)%n];}std::sort(wv,wv+7);std::sort(cv,cv+7);std::printf("DEPTH_NATIVE stack=%s domain=%s n=%zu wall_ns_el=%.9f cpu_ns_el=%.9f sink=%.17g\n",s.c_str(),neg?"neg":"pos",n,wv[3],cv[3],(double)g_sink);free(y);free(x);return 0;}
static int sde(const std::string&s,size_t n,bool neg,size_t calls){fn_t fn=choose(s);if(!fn)return 2;double*x=(double*)al64(n*8),*y=(double*)al64(n*8);fill(x,n,neg);fn(y,x,n);exp53_depth_profile_start();for(size_t r=0;r<calls;r++)fn(y,x,n);exp53_depth_profile_stop();if(s!="noop")g_sink+=y[(n*5u/13u)%n];std::printf("DEPTH_SDE stack=%s n=%zu calls=%zu sink=%.17g\n",s.c_str(),n,calls,(double)g_sink);free(y);free(x);return 0;}
int main(int argc,char**argv){pin0();mkl_set_num_threads_local(1);if(argc<3)return 2;std::string m=argv[1],s=argv[2];if(m=="accuracy")return accuracy(s);if(argc<5)return 2;size_t n=strtoull(argv[3],nullptr,10);bool neg=std::string(argv[4])=="neg";if(m=="native")return native(s,n,neg);if(m=="sde"&&argc==6)return sde(s,n,neg,strtoull(argv[5],nullptr,10));return 2;}
