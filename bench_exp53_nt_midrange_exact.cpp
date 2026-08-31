#include <algorithm>
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

static volatile double sink_value=0.0;
static void *xalloc(size_t bytes){ void *p=nullptr; if(posix_memalign(&p,64,bytes)!=0||!p){std::perror("posix_memalign");std::exit(2);} return p; }
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static uint64_t bits(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }
static uint64_t ord(double x){ uint64_t u=bits(x); return (u>>63)?~u:(u|0x8000000000000000ULL); }
static uint64_t ud(double a,double b){ uint64_t x=ord(a),y=ord(b); return x>y?x-y:y-x; }
static uint64_t rng=0x9e3779b97f4a7c15ULL;
static uint64_t ru(){ uint64_t x=rng; x^=x>>12; x^=x<<25; x^=x>>27; rng=x; return x*2685821657736338717ULL; }
static double rd(){ return -100.0+200.0*(double)(ru()>>11)*(1.0/9007199254740992.0); }
static void vha(double *o,const double *x,size_t n){ vmdExp((MKL_INT)n,x,o,VML_HA); }

typedef void(*fn_t)(double*,const double*,size_t);

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

static double one_hot(SharedFF&ex,fn_t fn,double*out,const double*in,size_t n,int calls){
    const double a=now_ns();
    for(int r=0;r<calls;r++)ex.run(fn,out,in,n);
    const double b=now_ns();
    sink_value+=out[(size_t)calls%n];
    return (b-a)/((double)calls*(double)n);
}
static double one_ring(SharedFF&ex,fn_t fn,std::vector<double*>&outs,const double*in,size_t n,int calls,int phase){
    const double a=now_ns();
    for(int r=0;r<calls;r++)ex.run(fn,outs[(size_t)(r+phase)%outs.size()],in,n);
    const double b=now_ns();
    sink_value+=outs[(size_t)phase%outs.size()][((size_t)phase*157u)%n];
    return (b-a)/((double)calls*(double)n);
}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

static void accuracy(){
    const size_t n=100000;
    double*in=(double*)xalloc(n*8),*fr=(double*)xalloc(n*8),*nt=(double*)xalloc(n*8);
    for(size_t i=0;i<n;i++)in[i]=rd();
    exp53_n2_vmstyle_u4_0381_frozen(fr,in,n);
    exp53_n2_vmstyle_u4_0381_nt_sfence(nt,in,n);
    uint64_t bd=0,mx=0,gt1=0;
    for(size_t i=0;i<n;i++){
        if(bits(fr[i])!=bits(nt[i]))bd++;
        double ref=(double)std::exp((long double)in[i]);
        uint64_t d=ud(nt[i],ref); if(d>mx)mx=d; if(d>1)gt1++;
    }
    std::printf("MIDNT_ACCURACY bitdiff=%llu maxULP=%llu gt1=%llu\n",
      (unsigned long long)bd,(unsigned long long)mx,(unsigned long long)gt1);
    std::free(in);std::free(fr);std::free(nt);
}

int main(){
    accuracy();
    const size_t ns[]={7000,12000,20000,35000,65000};
    SharedFF ex;
    for(size_t n:ns){
        double*in=(double*)xalloc(n*8);
        double*out=(double*)xalloc(n*8);
        for(size_t i=0;i<n;i++)in[i]=rd();
        const size_t slots=8;
        std::vector<double*> ring(slots);
        for(size_t s=0;s<slots;s++){ring[s]=(double*)xalloc(n*8);std::memset(ring[s],0,n*8);}

        /* Warm all three implementations through the exact same persistent pool. */
        for(int w=0;w<12;w++){
            ex.run(exp53_n2_vmstyle_u4_0381_nt_sfence,out,in,n);
            ex.run(vha,out,in,n);
            ex.run(exp53_n2_vmstyle_u4_0381_frozen,out,in,n);
        }

        int calls=(int)(24000000ULL/n); if(calls<200)calls=200; if(calls>4000)calls=4000;
        std::vector<double> h_nt,h_vml,h_tmp,r_nt,r_vml,r_tmp;
        h_nt.reserve(15);h_vml.reserve(15);h_tmp.reserve(15);
        r_nt.reserve(15);r_vml.reserve(15);r_tmp.reserve(15);

        /* Rotate method order every sample to cancel systematic frequency/placement drift. */
        for(int q=0;q<15;q++){
            const int order=q%3;
            if(order==0){
                h_nt.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,out,in,n,calls));
                h_vml.push_back(one_hot(ex,vha,out,in,n,calls));
                h_tmp.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_frozen,out,in,n,calls));
            }else if(order==1){
                h_vml.push_back(one_hot(ex,vha,out,in,n,calls));
                h_tmp.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_frozen,out,in,n,calls));
                h_nt.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,out,in,n,calls));
            }else{
                h_tmp.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_frozen,out,in,n,calls));
                h_nt.push_back(one_hot(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,out,in,n,calls));
                h_vml.push_back(one_hot(ex,vha,out,in,n,calls));
            }
        }

        for(int q=0;q<15;q++){
            const int order=q%3;
            if(order==0){
                r_nt.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,ring,in,n,calls,q));
                r_vml.push_back(one_ring(ex,vha,ring,in,n,calls,q));
                r_tmp.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_frozen,ring,in,n,calls,q));
            }else if(order==1){
                r_vml.push_back(one_ring(ex,vha,ring,in,n,calls,q));
                r_tmp.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_frozen,ring,in,n,calls,q));
                r_nt.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,ring,in,n,calls,q));
            }else{
                r_tmp.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_frozen,ring,in,n,calls,q));
                r_nt.push_back(one_ring(ex,exp53_n2_vmstyle_u4_0381_nt_sfence,ring,in,n,calls,q));
                r_vml.push_back(one_ring(ex,vha,ring,in,n,calls,q));
            }
        }

        const double hn=median(h_nt),hv=median(h_vml),ht=median(h_tmp);
        const double rn=median(r_nt),rv=median(r_vml),rt=median(r_tmp);
        std::printf("MIDNT n=%zu calls=%d hot_nt=%.9f hot_vml=%.9f hot_temporal=%.9f hot_vml_over_nt=%.6f hot_temporal_over_nt=%.6f ring_nt=%.9f ring_vml=%.9f ring_temporal=%.9f ring_vml_over_nt=%.6f ring_temporal_over_nt=%.6f\n",
          n,calls,hn,hv,ht,hv/hn,ht/hn,rn,rv,rt,rv/rn,rt/rn);

        for(double*p:ring)std::free(p); std::free(out);std::free(in);
    }
    return sink_value==1234567.0;
}
