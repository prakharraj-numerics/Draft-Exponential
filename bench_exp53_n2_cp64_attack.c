#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mkl_vml.h>

void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
void exp53_n2_cp64_attack(double*,const double*,size_t);

typedef void (*fn_t)(double*,const double*,size_t);
static volatile double sink;

static void *xalloc(size_t bytes){
    void *p=NULL; if(posix_memalign(&p,64,bytes)!=0 || !p){perror("posix_memalign");exit(2);} return p;
}
static uint64_t bits(double x){ uint64_t u; memcpy(&u,&x,8); return u; }
static uint64_t ordered(double x){
    uint64_t u=bits(x);
    return (u>>63) ? ~u : (u | 0x8000000000000000ULL);
}
static uint64_t ulpdiff(double a,double b){
    uint64_t x=ordered(a),y=ordered(b); return x>y?x-y:y-x;
}
static uint64_t rng=0x9e3779b97f4a7c15ULL;
static uint64_t nextu(void){
    uint64_t x=rng; x^=x>>12; x^=x<<25; x^=x>>27; rng=x; return x*2685821657736338717ULL;
}
static double rnd_domain(void){
    uint64_t u=nextu();
    double z=(double)(u>>11)*(1.0/9007199254740992.0);
    return -100.0 + 200.0*z;
}
static void intel_ha(double*out,const double*in,size_t n){ vmdExp((MKL_INT)n,in,out,VML_HA); }

static void accuracy(void){
    const size_t n=200000;
    double *in=xalloc(n*sizeof(double));
    double *a=xalloc(n*sizeof(double));
    double *b=xalloc(n*sizeof(double));
    for(size_t i=0;i<n;i++) in[i]=rnd_domain();
    exp53_n2_vmstyle_u4_0381_frozen(a,in,n);
    exp53_n2_cp64_attack(b,in,n);
    uint64_t mx=0,gt1=0,bitdiff=0;
    for(size_t i=0;i<n;i++){
        double ref=(double)expl((long double)in[i]);
        uint64_t d=ulpdiff(b[i],ref); if(d>mx)mx=d; if(d>1)gt1++;
        if(bits(a[i])!=bits(b[i])) bitdiff++;
    }
    printf("CP64_ACCURACY n=%zu maxULP=%llu gt1=%llu bitdiff_vs_frozen=%llu\n",
           n,(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)bitdiff);
    free(in);free(a);free(b);
}

static double nsnow(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t);
    return (double)t.tv_sec*1e9+(double)t.tv_nsec;
}
static double bench(fn_t f,size_t n){
    double *in=xalloc(n*sizeof(double));
    double *out=xalloc(n*sizeof(double));
    for(size_t i=0;i<n;i++) in[i]=rnd_domain();
    for(int w=0;w<20;w++) f(out,in,n);
    int calls=(int)(12000000ULL/n); if(calls<80)calls=80; if(calls>4000)calls=4000;
    double best=1e300;
    for(int trial=0;trial<11;trial++){
        double t0=nsnow();
        for(int r=0;r<calls;r++) f(out,in,n);
        double t1=nsnow();
        double ns=(t1-t0)/((double)calls*(double)n);
        if(ns<best)best=ns;
        sink+=out[(trial*131u)%n];
    }
    free(in);free(out); return best;
}

int main(void){
    printf("CP64_HARNESS faithful_n2=1\n");
    accuracy();
    const size_t ns[]={10079,12288,65536,262144};
    for(size_t z=0;z<sizeof(ns)/sizeof(ns[0]);z++){
        size_t n=ns[z];
        double f=bench(exp53_n2_vmstyle_u4_0381_frozen,n);
        double c=bench(exp53_n2_cp64_attack,n);
        double v=bench(intel_ha,n);
        printf("CP64_TIME n=%zu frozen=%.9f cp64=%.9f vml_ha=%.9f frozen_over_cp64=%.6f vml_over_cp64=%.6f\n",
               n,f,c,v,f/c,v/c);
    }
    return sink==1234567.0;
}
