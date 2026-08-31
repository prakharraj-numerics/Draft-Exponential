#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
#include <mkl_vml.h>
#include <ff/parallel_for.hpp>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_nt_sfence(double*,const double*,size_t);

typedef void(*fn_t)(double*,const double*,size_t);
static volatile double sink_value=0.0;

static void *xalloc(size_t bytes){
    void *p=nullptr;
    if(posix_memalign(&p,64,bytes)!=0 || !p){std::perror("posix_memalign"); std::exit(2);}
    return p;
}
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static uint64_t bits(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }
static uint64_t ord(double x){ uint64_t u=bits(x); return (u>>63)?~u:(u|0x8000000000000000ULL); }
static uint64_t ud(double a,double b){ uint64_t x=ord(a),y=ord(b); return x>y?x-y:y-x; }
static uint64_t rng=0x9e3779b97f4a7c15ULL;
static uint64_t ru(){ uint64_t x=rng; x^=x>>12; x^=x<<25; x^=x>>27; rng=x; return x*2685821657736338717ULL; }
static double rd(){ return -100.0+200.0*(double)(ru()>>11)*(1.0/9007199254740992.0); }
static void vha(double *o,const double *x,size_t n){ vmdExp((MKL_INT)n,x,o,VML_HA); }

class SharedFF {
public:
    SharedFF():pf_(2,true,true){}
    void run(fn_t fn,double*out,const double*in,size_t n){
        if(!n)return;
        const size_t full32=n/32;
        if(full32<2){fn(out,in,n);return;}
        pf_.parallel_for_static(0,2,1,0,[&](const long w){
            const size_t b0=(full32*(size_t)w)/2u;
            const size_t b1=(full32*(size_t)(w+1))/2u;
            const size_t lo=32*b0;
            size_t hi=32*b1;
            if(w==1) hi=n;
            fn(out+lo,in+lo,hi-lo);
        },2);
    }
private:
    ff::ParallelFor pf_;
};

static void accuracy(){
    const size_t n=200000;
    double *in=(double*)xalloc(n*8), *a=(double*)xalloc(n*8), *b=(double*)xalloc(n*8);
    for(size_t i=0;i<n;i++) in[i]=rd();
    exp53_n2_vmstyle_u4_0381_frozen(a,in,n);
    exp53_n2_vmstyle_u4_0381_nt_sfence(b,in,n);
    uint64_t bd=0,mx=0,gt1=0;
    for(size_t i=0;i<n;i++){
        if(bits(a[i])!=bits(b[i])) bd++;
        double ref=std::exp(in[i]);
        uint64_t d=ud(b[i],ref); if(d>mx)mx=d; if(d>1)gt1++;
    }
    std::printf("TRI_ACCURACY bitdiff=%llu maxULP=%llu gt1=%llu\n",
        (unsigned long long)bd,(unsigned long long)mx,(unsigned long long)gt1);
    std::free(in);std::free(a);std::free(b);
}

struct Ring {
    std::vector<double*> p;
    Ring(size_t slots,size_t n):p(slots){
        for(size_t s=0;s<slots;s++){p[s]=(double*)xalloc(n*8); std::memset(p[s],0,n*8);}
    }
    ~Ring(){for(double*q:p)std::free(q);}
};

static size_t ring_slots(size_t n){
    const size_t target=256ULL*1024ULL*1024ULL;
    size_t one=n*8;
    size_t s=(target+one-1)/one;
    if(s<8)s=8;
    if(s>192)s=192;
    return s;
}
static int calls_for(size_t n){
    uint64_t c=128000000ULL/n;
    if(c<96)c=96;
    if(c>640)c=640;
    return (int)c;
}
static double median(std::vector<double> v){std::sort(v.begin(),v.end()); return v[v.size()/2];}

static double measure_ff_hot(SharedFF&ex,fn_t fn,double*out,const double*in,size_t n,int calls){
    double a=now_ns();
    for(int r=0;r<calls;r++)ex.run(fn,out,in,n);
    double b=now_ns();
    sink_value+=out[(size_t)calls%n];
    return (b-a)/((double)calls*(double)n);
}
static double measure_vml_hot(double*out,const double*in,size_t n,int calls){
    double a=now_ns();
    for(int r=0;r<calls;r++)vha(out,in,n);
    double b=now_ns();
    sink_value+=out[(size_t)calls%n];
    return (b-a)/((double)calls*(double)n);
}
static double measure_ff_ring(SharedFF&ex,fn_t fn,Ring&ring,const double*in,size_t n,int calls,int phase){
    double a=now_ns();
    for(int r=0;r<calls;r++)ex.run(fn,ring.p[(size_t)(r+phase)%ring.p.size()],in,n);
    double b=now_ns();
    sink_value+=ring.p[(size_t)phase%ring.p.size()][((size_t)phase*157u)%n];
    return (b-a)/((double)calls*(double)n);
}
static double measure_vml_ring(Ring&ring,const double*in,size_t n,int calls,int phase){
    double a=now_ns();
    for(int r=0;r<calls;r++)vha(ring.p[(size_t)(r+phase)%ring.p.size()],in,n);
    double b=now_ns();
    sink_value+=ring.p[(size_t)phase%ring.p.size()][((size_t)phase*157u)%n];
    return (b-a)/((double)calls*(double)n);
}

int main(){
    accuracy();
    const size_t ns[]={200000,500000,800000,1000000,1200000,1500000};
    SharedFF ex;
    for(size_t n:ns){
        double *in=(double*)xalloc(n*8);
        double *hot_t=(double*)xalloc(n*8), *hot_n=(double*)xalloc(n*8), *hot_v=(double*)xalloc(n*8);
        for(size_t i=0;i<n;i++)in[i]=rd();
        const size_t slots=ring_slots(n);
        Ring rt(slots,n), rn(slots,n), rv(slots,n);
        const int calls=calls_for(n);

        for(int w=0;w<12;w++){
            ex.run(exp53_n2_vmstyle_u4_0381_frozen,hot_t,in,n);
            ex.run(exp53_n2_vmstyle_u4_0381_nt_sfence,hot_n,in,n);
            vha(hot_v,in,n);
        }

        std::vector<double> ht,hn,hv, st,sn,sv;
        ht.reserve(18);hn.reserve(18);hv.reserve(18);st.reserve(18);sn.reserve(18);sv.reserve(18);
        const std::array<std::array<int,3>,6> orders={{{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}}};
        for(int q=0;q<18;q++){
            const auto &o=orders[(size_t)q%orders.size()];
            for(int k=0;k<3;k++){
                if(o[k]==0) ht.push_back(measure_ff_hot(ex,exp53_n2_vmstyle_u4_0381_frozen,hot_t,in,n,calls));
                if(o[k]==1) hn.push_back(measure_ff_hot(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,hot_n,in,n,calls));
                if(o[k]==2) hv.push_back(measure_vml_hot(hot_v,in,n,calls));
            }
        }
        for(int q=0;q<18;q++){
            const auto &o=orders[(size_t)(q+2)%orders.size()];
            for(int k=0;k<3;k++){
                if(o[k]==0) st.push_back(measure_ff_ring(ex,exp53_n2_vmstyle_u4_0381_frozen,rt,in,n,calls,q));
                if(o[k]==1) sn.push_back(measure_ff_ring(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,rn,in,n,calls,q));
                if(o[k]==2) sv.push_back(measure_vml_ring(rv,in,n,calls,q));
            }
        }

        const double mht=median(ht), mhn=median(hn), mhv=median(hv);
        const double mst=median(st), msn=median(sn), msv=median(sv);
        std::printf("TRI n=%zu slots=%zu calls=%d hot_ff=%.9f hot_ff_nt=%.9f hot_vml_plain=%.9f hot_vml_over_ff=%.6f hot_vml_over_nt=%.6f stream_ff=%.9f stream_ff_nt=%.9f stream_vml_plain=%.9f stream_vml_over_ff=%.6f stream_vml_over_nt=%.6f nt_over_temporal=%.6f\n",
            n,slots,calls,mht,mhn,mhv,mhv/mht,mhv/mhn,mst,msn,msv,msv/mst,msv/msn,mst/msn);

        std::free(hot_t);std::free(hot_n);std::free(hot_v);std::free(in);
    }
    return sink_value==1234567.0;
}
