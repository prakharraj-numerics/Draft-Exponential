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
#include "exp53_m32d8_estrin_candidate.hpp"

using fn_t=void(*)(double*,const double*,size_t);
static volatile double g_sink=0.0;
static void pin0(){cpu_set_t s;CPU_ZERO(&s);CPU_SET(0,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
static void* al64(size_t n){void*p=nullptr;return posix_memalign(&p,64,n)==0?p:nullptr;}
static uint64_t mix64(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static double u01(uint64_t x){return((double)(mix64(x)>>11)+0.5)*0x1p-53;}
static uint64_t ordered(double v){uint64_t u;std::memcpy(&u,&v,8);return(u>>63)?~u:(u|UINT64_C(0x8000000000000000));}
static uint64_t ulpdist(double a,double b){if(std::isnan(a)||std::isnan(b))return UINT64_MAX;uint64_t x=ordered(a),y=ordered(b);return x>y?x-y:y-x;}
static double ns(clockid_t id){timespec t{};clock_gettime(id,&t);return(double)t.tv_sec*1e9+t.tv_nsec;}
static size_t reps_for(size_t n){size_t r=UINT64_C(12000000)/n;if(r<2)r=2;if(r>200000)r=200000;return r;}
static void fill(double*x,size_t n,bool neg){const double edge=0x1p-20;uint64_t seed=UINT64_C(0xa0761d6478bd642f)^(uint64_t)n*UINT64_C(0xe7037ed1a0b428db)^(neg?UINT64_C(0x8ebc6af09c88c6e3):0);for(size_t i=0;i<n;i++){double q=u01(seed+i*UINT64_C(0x9e3779b97f4a7c15));double a=edge+(1.0-2.0*edge)*q;x[i]=neg?-a:a;}}

static void run_estrin(double*o,const double*x,size_t n){exp53_m32d8_estrin::kernel(o,x,n);}
static void run_horner(double*o,const double*x,size_t n){exp53_power_family_raw::m32d8(o,x,n);}
static void run_intel(double*o,const double*x,size_t n){vmdExp((MKL_INT)n,x,o,VML_HA);}
static fn_t choose(const std::string&s){if(s=="estrin")return run_estrin;if(s=="horner")return run_horner;if(s=="intel")return run_intel;return nullptr;}

static int accuracy(const std::string&stack){fn_t fn=choose(stack);if(!fn)return 2;constexpr size_t N=9600;double*x=(double*)al64(N*8),*y=(double*)al64(N*8);if(!x||!y)return 3;for(size_t i=0;i<N;i++){double a;if(i==0)a=0.0;else if(i==1)a=std::nextafter(1.0,0.0);else if(i==2)a=1.0;else if(i==3)a=0x1p-20;else{double q=u01(UINT64_C(0xd1b54a32d192ed03)+i*UINT64_C(0x9e3779b97f4a7c15));a=0x1p-20+(1.0-0x1p-20)*q;}x[i]=(i&1)?-a:a;}fn(y,x,N);mpfr_t mx,mr;mpfr_init2(mx,256);mpfr_init2(mr,256);uint64_t maxm=0;size_t gt1=0,gt2=0;double wx=0,wy=0,wr=0;for(size_t i=0;i<N;i++){mpfr_set_d(mx,x[i],MPFR_RNDN);mpfr_exp(mr,mx,MPFR_RNDN);double ref=mpfr_get_d(mr,MPFR_RNDN);uint64_t d=ulpdist(y[i],ref);if(d>maxm){maxm=d;wx=x[i];wy=y[i];wr=ref;}if(d>1)gt1++;if(d>2)gt2++;}std::printf("ESTRIN_ACCURACY stack=%s cases=%zu reference=MPFR256 maxulp=%llu count_gt1=%zu count_gt2=%zu worst_x=%.17g worst_y=%.17g worst_ref=%.17g\n",stack.c_str(),N,(unsigned long long)maxm,gt1,gt2,wx,wy,wr);mpfr_clear(mx);mpfr_clear(mr);std::free(y);std::free(x);return 0;}

static int native(const std::string&stack,size_t n,bool neg){fn_t fn=choose(stack);if(!fn)return 2;double*x=(double*)al64(n*8),*y=(double*)al64(n*8);if(!x||!y)return 3;fill(x,n,neg);for(int w=0;w<5;w++)fn(y,x,n);size_t reps=reps_for(n);double wall[7];for(int t=0;t<7;t++){double w0=ns(CLOCK_MONOTONIC_RAW);for(size_t r=0;r<reps;r++)fn(y,x,n);double w1=ns(CLOCK_MONOTONIC_RAW);wall[t]=(w1-w0)/((double)reps*n);g_sink+=y[(n*7u/11u+t)%n];}std::sort(wall,wall+7);std::printf("ESTRIN_NATIVE stack=%s domain=%s n=%zu reps=%zu wall_ns_el=%.9f sink=%.17g\n",stack.c_str(),neg?"neg":"pos",n,reps,wall[3],(double)g_sink);std::free(y);std::free(x);return 0;}

int main(int argc,char**argv){pin0();mkl_set_num_threads_local(1);if(argc<3)return 2;std::string mode=argv[1],stack=argv[2];if(mode=="accuracy")return accuracy(stack);if(mode=="native"&&argc==5)return native(stack,(size_t)std::strtoull(argv[3],nullptr,10),std::string(argv[4])=="neg");return 2;}
