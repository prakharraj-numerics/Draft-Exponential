#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
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
static uint64_t rng=0x517cc1b727220a95ULL;
static uint64_t ru(){ uint64_t x=rng; x^=x>>12; x^=x<<25; x^=x>>27; rng=x; return x*2685821657736338717ULL; }
static double rd(){ return -100.0+200.0*(double)(ru()>>11)*(1.0/9007199254740992.0); }

class SharedFF {
public:
    SharedFF():pf_(2,true,true){}
    void run(fn_t fn,double*out,const double*in,size_t n){
        if(!n)return;
        const size_t full32=n/32;
        if(full32<2){fn(out,in,n);return;}
        const long active=2;
        pf_.parallel_for_static(0,active,1,0,[&](const long w){
            const size_t b0=(full32*(size_t)w)/2u;
            const size_t b1=(full32*(size_t)(w+1))/2u;
            const size_t lo=32*b0;
            size_t hi=32*b1;
            if(w==1)hi=n;
            fn(out+lo,in+lo,hi-lo);
        },active);
    }
private:
    ff::ParallelFor pf_;
};

static double one_ring(SharedFF&ex,fn_t fn,std::vector<double*>&outs,const double*in,size_t n,int calls,int phase){
    const double a=now_ns();
    for(int r=0;r<calls;r++) ex.run(fn,outs[(size_t)(r+phase)%outs.size()],in,n);
    const double b=now_ns();
    sink_value += outs[(size_t)phase%outs.size()][((size_t)phase*157u)%n];
    return (b-a)/((double)calls*(double)n);
}
static double one_hot(SharedFF&ex,fn_t fn,double*out,const double*in,size_t n,int calls){
    const double a=now_ns();
    for(int r=0;r<calls;r++) ex.run(fn,out,in,n);
    const double b=now_ns();
    sink_value += out[(size_t)calls%n];
    return (b-a)/((double)calls*(double)n);
}
static double median(std::vector<double> v){std::sort(v.begin(),v.end()); return v[v.size()/2];}

static void accuracy(){
    const size_t n=200000;
    double *in=(double*)xalloc(n*8), *a=(double*)xalloc(n*8), *b=(double*)xalloc(n*8);
    for(size_t i=0;i<n;i++) in[i]=rd();
    exp53_n2_vmstyle_u4_0381_frozen(a,in,n);
    exp53_n2_vmstyle_u4_0381_nt_sfence(b,in,n);
    uint64_t bd=0,mx=0,gt1=0;
    for(size_t i=0;i<n;i++){
        if(bits(a[i])!=bits(b[i])) bd++;
        const double ref=(double)std::exp((long double)in[i]);
        const uint64_t d=ud(b[i],ref); if(d>mx)mx=d; if(d>1)gt1++;
    }
    std::printf("CROSS_ACCURACY bitdiff=%llu maxULP=%llu gt1=%llu\n",
        (unsigned long long)bd,(unsigned long long)mx,(unsigned long long)gt1);
    std::free(in);std::free(a);std::free(b);
}

int main(){
    accuracy();
    const size_t ns[]={200000,500000,800000,1000000,1200000,1500000};
    SharedFF ex;
    const size_t target_ring_bytes=256ULL*1024ULL*1024ULL;

    for(size_t n:ns){
        double *in=(double*)xalloc(n*8), *hot=(double*)xalloc(n*8);
        for(size_t i=0;i<n;i++) in[i]=rd();

        size_t slots=(target_ring_bytes + n*8 - 1)/(n*8);
        if(slots<8) slots=8;
        if(slots>192) slots=192;
        std::vector<double*> ring(slots);
        for(size_t s=0;s<slots;s++){ ring[s]=(double*)xalloc(n*8); std::memset(ring[s],0,n*8); }

        for(int w=0;w<16;w++){
            ex.run(exp53_n2_vmstyle_u4_0381_frozen,hot,in,n);
            ex.run(exp53_n2_vmstyle_u4_0381_nt_sfence,hot,in,n);
        }

        int calls=(int)(96000000ULL/n);
        if(calls<96) calls=96;
        if(calls>640) calls=640;

        std::vector<double> hot_t,hot_n,ring_t,ring_n;
        hot_t.reserve(17);hot_n.reserve(17);ring_t.reserve(17);ring_n.reserve(17);
        for(int q=0;q<17;q++){
            if(q&1){
                hot_n.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,hot,in,n,calls));
                hot_t.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_frozen,hot,in,n,calls));
            }else{
                hot_t.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_frozen,hot,in,n,calls));
                hot_n.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,hot,in,n,calls));
            }
        }
        for(int q=0;q<17;q++){
            if(q&1){
                ring_n.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,ring,in,n,calls,q));
                ring_t.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_frozen,ring,in,n,calls,q));
            }else{
                ring_t.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_frozen,ring,in,n,calls,q));
                ring_n.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,ring,in,n,calls,q));
            }
        }

        const double ht=median(hot_t), hn=median(hot_n), rt=median(ring_t), rn=median(ring_n);
        std::printf("CROSS n=%zu slots=%zu calls=%d hot_temporal=%.9f hot_nt=%.9f hot_speedup=%.6f ring_temporal=%.9f ring_nt=%.9f ring_speedup=%.6f\n",
            n,slots,calls,ht,hn,ht/hn,rt,rn,rt/rn);

        for(double*p:ring)std::free(p);
        std::free(hot); std::free(in);
    }
    return sink_value==1234567.0;
}
