#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <mkl_vml.h>

#include "exp53_fastflow_batch_2core_frozen.hpp"

extern "C" void exp53_n2_vmstyle_u4_0381_nt_sfence(double*,const double*,size_t);

static volatile double sink_value=0.0;

static void *xalloc(size_t bytes){
    void *p=nullptr;
    if(posix_memalign(&p,64,bytes)!=0 || !p){ std::perror("posix_memalign"); std::exit(2); }
    return p;
}
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return (double)t.tv_sec*1e9+t.tv_nsec; }
static uint64_t bits(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }
static uint64_t ord(double x){ uint64_t u=bits(x); return (u>>63)?~u:(u|0x8000000000000000ULL); }
static uint64_t ud(double a,double b){ uint64_t x=ord(a),y=ord(b); return x>y?x-y:y-x; }
static uint64_t rng=0x2545f4914f6cdd1dULL;
static uint64_t ru(){ uint64_t x=rng; x^=x>>12; x^=x<<25; x^=x>>27; rng=x; return x*2685821657736338717ULL; }
static double rd(){ return -100.0+200.0*(double)(ru()>>11)*(1.0/9007199254740992.0); }
static void vha(double *o,const double *x,size_t n){ vmdExp((MKL_INT)n,x,o,VML_HA); }

typedef void(*fn_t)(double*,const double*,size_t);
struct Stats { double med,best,call_ns; };

static int calls_for(size_t n){
    uint64_t c=32000000ULL/n;
    if(c<1)c=1; if(c>3000)c=3000;
    return (int)c;
}

static Stats bench_hot(fn_t fn,double *out,const double *in,size_t n){
    fn(out,in,n); fn(out,in,n);
    int calls=calls_for(n);
    std::vector<double> v;
    for(int q=0;q<7;q++){
        double a=now_ns();
        for(int r=0;r<calls;r++) fn(out,in,n);
        double b=now_ns();
        v.push_back((b-a)/((double)calls*n));
        sink_value+=out[((size_t)q*157u)%n];
    }
    std::sort(v.begin(),v.end());
    return {v[3],v[0],v[3]*(double)n};
}

static Stats bench_ring(fn_t fn,std::vector<double*>& outs,const double *in,size_t n){
    fn(outs[0],in,n); fn(outs[0],in,n);
    int calls=calls_for(n);
    std::vector<double> v;
    for(int q=0;q<7;q++){
        double a=now_ns();
        for(int r=0;r<calls;r++) fn(outs[(size_t)(r+q)%outs.size()],in,n);
        double b=now_ns();
        v.push_back((b-a)/((double)calls*n));
        sink_value+=outs[(size_t)q%outs.size()][((size_t)q*157u)%n];
    }
    std::sort(v.begin(),v.end());
    return {v[3],v[0],v[3]*(double)n};
}

static Stats bench_ff_ring(fn_t fn,std::vector<double*>& outs,const double *in,size_t n){
    Exp53FastFlowFrozenExecutor ex(2,fn);
    ex.run_static(outs[0],in,n,2); ex.run_static(outs[0],in,n,2);
    int calls=calls_for(n);
    std::vector<double> v;
    for(int q=0;q<7;q++){
        double a=now_ns();
        for(int r=0;r<calls;r++) ex.run_static(outs[(size_t)(r+q)%outs.size()],in,n,2);
        double b=now_ns();
        v.push_back((b-a)/((double)calls*n));
        sink_value+=outs[(size_t)q%outs.size()][((size_t)q*157u)%n];
    }
    std::sort(v.begin(),v.end());
    return {v[3],v[0],v[3]*(double)n};
}

static size_t ring_slots(size_t n){
    const size_t target=256ULL*1024ULL*1024ULL;
    size_t one=n*sizeof(double);
    size_t s=one?target/one:1;
    if(s<2)s=2; if(s>8)s=8;
    return s;
}

static void accuracy(){
    const size_t n=200000;
    double *in=(double*)xalloc(n*8);
    double *ref=(double*)xalloc(n*8);
    double *nt=(double*)xalloc(n*8);
    double *raw=(double*)xalloc((n+8)*8);
    double *mis=raw+1;
    for(size_t i=0;i<n;i++)in[i]=rd();
    exp53_n2_vmstyle_u4_0381_frozen(ref,in,n);
    exp53_n2_vmstyle_u4_0381_nt_sfence(nt,in,n);
    exp53_n2_vmstyle_u4_0381_nt_sfence(mis,in,n);
    uint64_t bd=0,bdmis=0,mx=0,gt1=0;
    for(size_t i=0;i<n;i++){
        if(bits(nt[i])!=bits(ref[i]))bd++;
        if(bits(mis[i])!=bits(ref[i]))bdmis++;
        double r=(double)std::exp((long double)in[i]);
        uint64_t d=ud(nt[i],r); if(d>mx)mx=d; if(d>1)gt1++;
    }
    std::printf("NTCACHE_ACCURACY aligned_bitdiff=%llu misaligned_bitdiff=%llu maxULP=%llu gt1=%llu\n",
        (unsigned long long)bd,(unsigned long long)bdmis,(unsigned long long)mx,(unsigned long long)gt1);
    std::free(in);std::free(ref);std::free(nt);std::free(raw);
}

int main(){
    accuracy();
    const size_t ns[]={4096,8192,12288,32768,65536,131072,262144,524288,1000000,2000000,4000000,8000000,16000000};
    for(size_t n:ns){
        double *in=(double*)xalloc(n*8);
        for(size_t i=0;i<n;i++)in[i]=rd();
        size_t slots=ring_slots(n);
        std::vector<double*> outs(slots);
        for(size_t s=0;s<slots;s++){
            outs[s]=(double*)xalloc(n*8+64);
            std::memset(outs[s],0,n*8);
        }

        Stats th=bench_hot(exp53_n2_vmstyle_u4_0381_frozen,outs[0],in,n);
        Stats nh=bench_hot(exp53_n2_vmstyle_u4_0381_nt_sfence,outs[0],in,n);
        Stats tr=bench_ring(exp53_n2_vmstyle_u4_0381_frozen,outs,in,n);
        Stats nr=bench_ring(exp53_n2_vmstyle_u4_0381_nt_sfence,outs,in,n);
        Stats vr=bench_ring(vha,outs,in,n);
        Stats tff=bench_ff_ring(exp53_n2_vmstyle_u4_0381_frozen,outs,in,n);
        Stats nff=bench_ff_ring(exp53_n2_vmstyle_u4_0381_nt_sfence,outs,in,n);
        Stats vff=bench_ff_ring(vha,outs,in,n);

        std::printf("NTCACHE n=%zu slots=%zu temporal_hot=%.9f nt_hot=%.9f hot_speedup=%.6f temporal_ring=%.9f nt_ring=%.9f ring_speedup=%.6f vml_ring=%.9f temporal_ff=%.9f nt_ff=%.9f ff_speedup=%.6f vml_ff=%.9f vmlff_over_ntff=%.6f\n",
            n,slots,th.med,nh.med,th.med/nh.med,tr.med,nr.med,tr.med/nr.med,vr.med,
            tff.med,nff.med,tff.med/nff.med,vff.med,vff.med/nff.med);

        if(n==1000000){
            double *mr=(double*)xalloc((n+8)*8);
            double *mo=mr+1;
            Stats ma=bench_hot(exp53_n2_vmstyle_u4_0381_nt_sfence,mo,in,n);
            std::printf("NTCACHE_MISALIGN n=%zu nt_aligned_hot=%.9f nt_plus8_hot=%.9f ratio=%.6f\n",n,nh.med,ma.med,ma.med/nh.med);
            std::free(mr);
        }

        for(double *p:outs)std::free(p);
        std::free(in);
    }
    return sink_value==1234567.0;
}
