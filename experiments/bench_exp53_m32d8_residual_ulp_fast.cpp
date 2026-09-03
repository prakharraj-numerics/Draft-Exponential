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
#include "exp53_m32d8_residual_ulp_fast_candidate.hpp"

using fn_t=void(*)(double*,const double*,size_t); static volatile double g_sink=0.0;
extern "C" __attribute__((noinline)) void exp53_fast_profile_start(){asm volatile("":::"memory");}
extern "C" __attribute__((noinline)) void exp53_fast_profile_stop(){asm volatile("":::"memory");}
static void pin0(){cpu_set_t s;CPU_ZERO(&s);CPU_SET(0,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);} static void* al64(size_t n){void*p=nullptr;return posix_memalign(&p,64,n)==0?p:nullptr;}
static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);} static double u01(uint64_t x){return((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static void fill(double*x,size_t n,bool neg){const double edge=0x1p-20;uint64_t seed=UINT64_C(0xa0761d6478bd642f)^(uint64_t)n*UINT64_C(0xe7037ed1a0b428db)^(neg?UINT64_C(0x8ebc6af09c88c6e3):0);for(size_t i=0;i<n;i++){double q=u01(seed+i*UINT64_C(0x9e3779b97f4a7c15));double a=edge+(1.0-2.0*edge)*q;x[i]=neg?-a:a;}}
static void run_candidate(double*o,const double*x,size_t n){exp53_m32d8_residual_ulp_fast::kernel(o,x,n);} static void run_raw(double*o,const double*x,size_t n){exp53_power_family_raw::m32d8(o,x,n);} static void run_intel(double*o,const double*x,size_t n){vmdExp((MKL_INT)n,x,o,VML_HA);} static void run_noop(double*o,const double*x,size_t n){asm volatile(""::"r"(o),"r"(x),"r"(n):"memory");}
static fn_t choose(const std::string&s){if(s=="candidate")return run_candidate;if(s=="raw")return run_raw;if(s=="intel")return run_intel;if(s=="noop")return run_noop;return nullptr;}
static uint64_t ordered(double v){uint64_t u;std::memcpy(&u,&v,8);return(u>>63)?~u:(u|UINT64_C(0x8000000000000000));} static uint64_t ulpdist(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=ordered(a),y=ordered(b);return x>y?x-y:y-x;} static double ns(clockid_t id){timespec t{};clock_gettime(id,&t);return(double)t.tv_sec*1e9+t.tv_nsec;} static size_t reps_for(size_t n){size_t r=UINT64_C(12000000)/n;if(r<2)r=2;if(r>200000)r=200000;return r;}
static int accuracy(const std::string&stack){fn_t fn=choose(stack);if(!fn||stack=="noop")return 2;constexpr size_t N=9600;double*x=(double*)al64(N*8),*y=(double*)al64(N*8),*intel=(double*)al64(N*8);if(!x||!y||!intel)return 3;for(size_t i=0;i<N;i++){double a;if(i==0)a=0.0;else if(i==1)a=std::nextafter(1.0,0.0);else if(i==2)a=1.0;else if(i==3)a=0x1p-20;else{double q=u01(UINT64_C(0xd1b54a32d192ed03)+i*UINT64_C(0x9e3779b97f4a7c15));a=0x1p-20+(1.0-0x1p-20)*q;}x[i]=(i&1)?-a:a;}fn(y,x,N);run_intel(intel,x,N);mpfr_t mx,mr;mpfr_init2(mx,256);mpfr_init2(mr,256);uint64_t maxm=0,maxi=0;size_t gt1=0,gt2=0;double wx=0,wy=0,wr=0;for(size_t i=0;i<N;i++){mpfr_set_d(mx,x[i],MPFR_RNDN);mpfr_exp(mr,mx,MPFR_RNDN);double ref=mpfr_get_d(mr,MPFR_RNDN);uint64_t dm=ulpdist(y[i],ref),di=ulpdist(y[i],intel[i]);if(dm>maxm){maxm=dm;wx=x[i];wy=y[i];wr=ref;}if(di>maxi)maxi=di;if(dm>1)gt1++;if(dm>2)gt2++;}std::printf("FAST_ACCURACY stack=%s cases=%zu reference=MPFR256 maxulp_vs_mpfr=%llu count_gt1_mpfr=%zu count_gt2_mpfr=%zu maxulp_vs_intel=%llu worst_x=%.17g worst_y=%.17g worst_ref=%.17g\n",stack.c_str(),N,(unsigned long long)maxm,gt1,gt2,(unsigned long long)maxi,wx,wy,wr);mpfr_clear(mx);mpfr_clear(mr);std::free(intel);std::free(y);std::free(x);return 0;}
static int native(const std::string&stack,size_t n,bool neg){fn_t fn=choose(stack);if(!fn)return 2;double*x=(double*)al64(n*8),*y=(double*)al64(n*8),*ref=(double*)al64(n*8);if(!x||!y||!ref)return 3;fill(x,n,neg);run_intel(ref,x,n);fn(y,x,n);uint64_t mx=0;for(size_t i=0;i<n;i++)mx=std::max(mx,ulpdist(y[i],ref[i]));for(int w=0;w<5;w++)fn(y,x,n);size_t reps=reps_for(n);double wall[7],cpu[7];for(int t=0;t<7;t++){double w0=ns(CLOCK_MONOTONIC_RAW),c0=ns(CLOCK_PROCESS_CPUTIME_ID);for(size_t r=0;r<reps;r++)fn(y,x,n);double c1=ns(CLOCK_PROCESS_CPUTIME_ID),w1=ns(CLOCK_MONOTONIC_RAW);wall[t]=(w1-w0)/((double)reps*n);cpu[t]=(c1-c0)/((double)reps*n);g_sink+=y[(n*7u/11u+t)%n];}std::sort(wall,wall+7);std::sort(cpu,cpu+7);std::printf("FAST_NATIVE stack=%s domain=%s n=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f maxulp_vs_intel=%llu sink=%.17g\n",stack.c_str(),neg?"neg":"pos",n,reps,wall[3],cpu[3],(unsigned long long)mx,(double)g_sink);std::free(ref);std::free(y);std::free(x);return 0;}
static int sde(const std::string&stack,size_t n,bool neg,size_t calls){fn_t fn=choose(stack);if(!fn)return 2;double*x=(double*)al64(n*8),*y=(double*)al64(n*8);if(!x||!y)return 3;fill(x,n,neg);fn(y,x,n);exp53_fast_profile_start();for(size_t r=0;r<calls;r++)fn(y,x,n);exp53_fast_profile_stop();if(stack!="noop")g_sink+=y[(n*5u/13u)%n];std::printf("FAST_SDE stack=%s domain=%s n=%zu calls=%zu sink=%.17g\n",stack.c_str(),neg?"neg":"pos",n,calls,(double)g_sink);std::free(y);std::free(x);return 0;}
int main(int argc,char**argv){pin0();mkl_set_num_threads_local(1);if(argc<3)return 2;std::string mode=argv[1],stack=argv[2];if(mode=="accuracy")return accuracy(stack);if(argc<5)return 2;size_t n=(size_t)std::strtoull(argv[3],nullptr,10);bool neg=std::string(argv[4])=="neg";if(mode=="native")return native(stack,n,neg);if(mode=="sde"&&argc==6)return sde(stack,n,neg,(size_t)std::strtoull(argv[5],nullptr,10));return 2;}
