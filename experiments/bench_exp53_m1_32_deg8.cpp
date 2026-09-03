#define _GNU_SOURCE
#include <mkl.h>
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
#include <vector>

#include "exp53_m1_32_deg8_candidate.hpp"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

static volatile double g_sink = 0.0;
using fn_t = void(*)(double*, const double*, size_t);

extern "C" __attribute__((noinline)) void exp53_m132_profile_start(void) { asm volatile("" ::: "memory"); }
extern "C" __attribute__((noinline)) void exp53_m132_profile_stop(void)  { asm volatile("" ::: "memory"); }

static void pin0() {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(0,&s);
    (void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);
}

static void* al64(size_t bytes) {
    void* p=nullptr; if(posix_memalign(&p,64,bytes)!=0) return nullptr; return p;
}

static uint64_t mix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static double u01(uint64_t x) { return ((double)(mix64(x)>>11)+0.5)*0x1p-53; }

static void fill(double* x,size_t n,bool neg) {
    const double edge=0x1p-20;
    const uint64_t seed=UINT64_C(0xd1b54a32d192ed03) ^ (uint64_t)n*UINT64_C(0x94d049bb133111eb) ^ (neg?UINT64_C(0x7263d9bd8409f526):0);
    for(size_t i=0;i<n;i++) {
        const double q=u01(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));
        const double a=edge+(1.0-2.0*edge)*q;
        x[i]=neg?-a:a;
    }
}

static void run_horner(double* o,const double* x,size_t n){ exp53_m1_32_deg8::horner(o,x,n); }
static void run_estrin(double* o,const double* x,size_t n){ exp53_m1_32_deg8::estrin(o,x,n); }
static void run_serial(double* o,const double* x,size_t n){ exp53_n2_vmstyle_u4_0381_frozen(o,x,n); }
static void run_intel(double* o,const double* x,size_t n){ vmdExp((MKL_INT)n,x,o,VML_HA); }
static void run_noop(double* o,const double* x,size_t n){ asm volatile("" : : "r"(o),"r"(x),"r"(n) : "memory"); }

static fn_t choose(const std::string& s) {
    if(s=="horner") return run_horner;
    if(s=="estrin") return run_estrin;
    if(s=="serial") return run_serial;
    if(s=="intel") return run_intel;
    if(s=="noop") return run_noop;
    return nullptr;
}

static uint64_t ordered(double v){
    uint64_t u; std::memcpy(&u,&v,8);
    return (u>>63)?~u:(u|UINT64_C(0x8000000000000000));
}
static uint64_t ulpdist(double a,double b){
    if(std::isnan(a)||std::isnan(b)) return UINT64_MAX;
    uint64_t x=ordered(a),y=ordered(b); return x>y?x-y:y-x;
}

static double ns(clockid_t id){ timespec t{}; clock_gettime(id,&t); return (double)t.tv_sec*1e9+t.tv_nsec; }
static double median7(double a[7]){ std::sort(a,a+7); return a[3]; }

static size_t reps_for(size_t n){
    size_t r=UINT64_C(12000000)/n; if(r<2)r=2; if(r>200000)r=200000; return r;
}

static int native(const std::string& stack,size_t n,bool neg){
    fn_t fn=choose(stack); if(!fn) return 2;
    double* x=(double*)al64(n*8),*y=(double*)al64(n*8),*ref=(double*)al64(n*8);
    if(!x||!y||!ref) return 3;
    fill(x,n,neg);
    run_intel(ref,x,n); fn(y,x,n);
    uint64_t mx=0;
    for(size_t i=0;i<n;i++) mx=std::max(mx,ulpdist(y[i],ref[i]));
    for(int w=0;w<5;w++) fn(y,x,n);
    const size_t reps=reps_for(n);
    double wall[7],cpu[7];
    for(int t=0;t<7;t++){
        const double w0=ns(CLOCK_MONOTONIC_RAW),c0=ns(CLOCK_PROCESS_CPUTIME_ID);
        for(size_t r=0;r<reps;r++) fn(y,x,n);
        const double c1=ns(CLOCK_PROCESS_CPUTIME_ID),w1=ns(CLOCK_MONOTONIC_RAW);
        wall[t]=(w1-w0)/((double)reps*n); cpu[t]=(c1-c0)/((double)reps*n);
        g_sink += y[(n*7u/11u+t)%n];
    }
    const double mw=median7(wall),mc=median7(cpu);
    std::printf("M132_NATIVE stack=%s domain=%s n=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxulp_vs_intel=%llu sink=%.17g\n",
                stack.c_str(),neg?"neg":"pos",n,reps,mw,mc,mc/mw,(unsigned long long)mx,(double)g_sink);
    std::free(ref);std::free(y);std::free(x); return 0;
}

static int sde(const std::string& stack,size_t n,bool neg,size_t calls){
    fn_t fn=choose(stack); if(!fn) return 2;
    double* x=(double*)al64(n*8),*y=(double*)al64(n*8); if(!x||!y)return 3;
    fill(x,n,neg); fn(y,x,n);
    exp53_m132_profile_start();
    for(size_t r=0;r<calls;r++) fn(y,x,n);
    exp53_m132_profile_stop();
    if(stack!="noop") g_sink += y[(n*5u/13u)%n];
    std::printf("M132_SDE stack=%s domain=%s n=%zu calls=%zu sink=%.17g\n",stack.c_str(),neg?"neg":"pos",n,calls,(double)g_sink);
    std::free(y);std::free(x); return 0;
}

int main(int argc,char**argv){
    if(argc<5) return 2;
    pin0(); mkl_set_num_threads_local(1);
    const std::string mode=argv[1],stack=argv[2],domain=argv[4];
    const size_t n=(size_t)std::strtoull(argv[3],nullptr,10); const bool neg=(domain=="neg");
    if(mode=="native") return native(stack,n,neg);
    if(mode=="sde" && argc==6) return sde(stack,n,neg,(size_t)std::strtoull(argv[5],nullptr,10));
    return 2;
}
