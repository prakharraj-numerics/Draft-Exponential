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
#include <vector>

#include "exp53_m1_32_deg8_candidate.hpp"
#include "exp53_m1_32_deg8_accuracy_candidate.hpp"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

static volatile double g_sink = 0.0;
using fn_t = void(*)(double*, const double*, size_t);

extern "C" __attribute__((noinline)) void exp53_m132acc_profile_start(void) { asm volatile("" ::: "memory"); }
extern "C" __attribute__((noinline)) void exp53_m132acc_profile_stop(void)  { asm volatile("" ::: "memory"); }

static void pin0() {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(0,&s);
    (void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);
}
static void* al64(size_t bytes) { void* p=nullptr; return posix_memalign(&p,64,bytes)==0?p:nullptr; }

static uint64_t mix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static double u01(uint64_t x) { return ((double)(mix64(x)>>11)+0.5)*0x1p-53; }

static void fill(double* x,size_t n,bool neg) {
    const double edge=0x1p-20;
    const uint64_t seed=UINT64_C(0x91e10da5c79e7b1d) ^ (uint64_t)n*UINT64_C(0xd1b54a32d192ed03) ^ (neg?UINT64_C(0x7263d9bd8409f526):0);
    for(size_t i=0;i<n;i++) {
        const double q=u01(seed+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));
        const double a=edge+(1.0-2.0*edge)*q;
        x[i]=neg?-a:a;
    }
}

static void run_raw(double* o,const double* x,size_t n){ exp53_m1_32_deg8::horner(o,x,n); }
static void run_basic(double* o,const double* x,size_t n){ exp53_m1_32_deg8_accuracy::expm1_basic(o,x,n); }
static void run_comp(double* o,const double* x,size_t n){ exp53_m1_32_deg8_accuracy::expm1_comp(o,x,n); }
static void run_serial(double* o,const double* x,size_t n){ exp53_n2_vmstyle_u4_0381_frozen(o,x,n); }
static void run_intel(double* o,const double* x,size_t n){ vmdExp((MKL_INT)n,x,o,VML_HA); }
static void run_noop(double* o,const double* x,size_t n){ asm volatile("" : : "r"(o),"r"(x),"r"(n) : "memory"); }

static fn_t choose(const std::string& s) {
    if(s=="raw") return run_raw;
    if(s=="basic") return run_basic;
    if(s=="comp") return run_comp;
    if(s=="serial") return run_serial;
    if(s=="intel") return run_intel;
    if(s=="noop") return run_noop;
    return nullptr;
}

static uint64_t ordered(double v){ uint64_t u; std::memcpy(&u,&v,8); return (u>>63)?~u:(u|UINT64_C(0x8000000000000000)); }
static uint64_t ulpdist(double a,double b){ if(std::isnan(a)||std::isnan(b)) return UINT64_MAX; const uint64_t x=ordered(a),y=ordered(b); return x>y?x-y:y-x; }
static double ns(clockid_t id){ timespec t{}; clock_gettime(id,&t); return (double)t.tv_sec*1e9+t.tv_nsec; }
static double median7(double a[7]){ std::sort(a,a+7); return a[3]; }
static size_t reps_for(size_t n){ size_t r=UINT64_C(12000000)/n; if(r<2)r=2; if(r>200000)r=200000; return r; }

static double mpfr_exp_rnd(double x) {
    mpfr_t a,b; mpfr_init2(a,256); mpfr_init2(b,256);
    mpfr_set_d(a,x,MPFR_RNDN); mpfr_exp(b,a,MPFR_RNDN);
    const double y=mpfr_get_d(b,MPFR_RNDN);
    mpfr_clear(b); mpfr_clear(a); return y;
}

static int accuracy(const std::string& stack) {
    fn_t fn=choose(stack); if(!fn || stack=="noop") return 2;
    constexpr size_t N=9600;
    double* x=(double*)al64(N*8),*y=(double*)al64(N*8),*intel=(double*)al64(N*8);
    if(!x||!y||!intel) return 3;
    // Deterministic unit-domain stress: exact endpoints/near-endpoints/zero plus
    // dense pseudorandom signed values.  MPFR-256 is the independent reference.
    x[0]=0.0; x[1]=1.0; x[2]=-1.0;
    x[3]=std::nextafter(1.0,0.0); x[4]=std::nextafter(-1.0,0.0);
    x[5]=0x1p-52; x[6]=-0x1p-52; x[7]=0x1p-20; x[8]=-0x1p-20;
    for(size_t i=9;i<N;i++) {
        const double q=u01(UINT64_C(0x243f6a8885a308d3)+(uint64_t)i*UINT64_C(0x9e3779b97f4a7c15));
        x[i]=2.0*q-1.0;
    }
    fn(y,x,N); run_intel(intel,x,N);
    uint64_t max_mpfr=0,max_intel=0; size_t gt1=0,gt2=0,intel_gt1=0,intel_gt2=0;
    double worst_x=0.0,worst_y=0.0,worst_ref=0.0;
    for(size_t i=0;i<N;i++) {
        const double r=mpfr_exp_rnd(x[i]);
        const uint64_t dm=ulpdist(y[i],r), di=ulpdist(y[i],intel[i]);
        if(dm>max_mpfr){max_mpfr=dm;worst_x=x[i];worst_y=y[i];worst_ref=r;}
        max_intel=std::max(max_intel,di);
        gt1 += dm>1; gt2 += dm>2; intel_gt1 += di>1; intel_gt2 += di>2;
    }
    std::printf("M132ACC_ACCURACY stack=%s cases=%zu reference=MPFR256 maxulp_vs_mpfr=%llu count_gt1_mpfr=%zu count_gt2_mpfr=%zu maxulp_vs_intel=%llu count_gt1_intel=%zu count_gt2_intel=%zu worst_x=%.17g worst_y=%.17g worst_ref=%.17g\n",
      stack.c_str(),N,(unsigned long long)max_mpfr,gt1,gt2,(unsigned long long)max_intel,intel_gt1,intel_gt2,worst_x,worst_y,worst_ref);
    std::free(intel);std::free(y);std::free(x); return 0;
}

static int native(const std::string& stack,size_t n,bool neg){
    fn_t fn=choose(stack); if(!fn) return 2;
    double* x=(double*)al64(n*8),*y=(double*)al64(n*8),*ref=(double*)al64(n*8);
    if(!x||!y||!ref) return 3;
    fill(x,n,neg); run_intel(ref,x,n); fn(y,x,n);
    uint64_t mx=0; for(size_t i=0;i<n;i++) mx=std::max(mx,ulpdist(y[i],ref[i]));
    for(int w=0;w<5;w++) fn(y,x,n);
    const size_t reps=reps_for(n); double wall[7],cpu[7];
    for(int t=0;t<7;t++){
        const double w0=ns(CLOCK_MONOTONIC_RAW),c0=ns(CLOCK_PROCESS_CPUTIME_ID);
        for(size_t r=0;r<reps;r++) fn(y,x,n);
        const double c1=ns(CLOCK_PROCESS_CPUTIME_ID),w1=ns(CLOCK_MONOTONIC_RAW);
        wall[t]=(w1-w0)/((double)reps*n); cpu[t]=(c1-c0)/((double)reps*n);
        g_sink += y[(n*7u/11u+t)%n];
    }
    const double mw=median7(wall),mc=median7(cpu);
    std::printf("M132ACC_NATIVE stack=%s domain=%s n=%zu reps=%zu wall_ns_el=%.9f cpu_ns_el=%.9f effective_cores=%.6f maxulp_vs_intel=%llu sink=%.17g\n",
      stack.c_str(),neg?"neg":"pos",n,reps,mw,mc,mc/mw,(unsigned long long)mx,(double)g_sink);
    std::free(ref);std::free(y);std::free(x); return 0;
}

static int sde(const std::string& stack,size_t n,bool neg,size_t calls){
    fn_t fn=choose(stack); if(!fn) return 2;
    double* x=(double*)al64(n*8),*y=(double*)al64(n*8); if(!x||!y)return 3;
    fill(x,n,neg); fn(y,x,n);
    exp53_m132acc_profile_start();
    for(size_t r=0;r<calls;r++) fn(y,x,n);
    exp53_m132acc_profile_stop();
    if(stack!="noop") g_sink += y[(n*5u/13u)%n];
    std::printf("M132ACC_SDE stack=%s domain=%s n=%zu calls=%zu sink=%.17g\n",stack.c_str(),neg?"neg":"pos",n,calls,(double)g_sink);
    std::free(y);std::free(x); return 0;
}

int main(int argc,char**argv){
    if(argc<3) return 2; pin0(); mkl_set_num_threads_local(1);
    const std::string mode=argv[1],stack=argv[2];
    if(mode=="accuracy") return accuracy(stack);
    if(argc<5) return 2;
    const size_t n=(size_t)std::strtoull(argv[3],nullptr,10); const std::string domain=argv[4]; const bool neg=(domain=="neg");
    if(mode=="native") return native(stack,n,neg);
    if(mode=="sde" && argc==6) return sde(stack,n,neg,(size_t)std::strtoull(argv[5],nullptr,10));
    return 2;
}
