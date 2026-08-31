#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BUILD_LABEL
#define BUILD_LABEL "UNKNOWN"
#endif

typedef void (*fn_t)(double*,const double*,size_t);
void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);

static volatile double sink;
static uint64_t rs=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(void){rs^=rs<<7;rs^=rs>>9;rs^=rs<<8;return rs;}
static inline double genx(size_t i){double u=(double)(xr()>>11)*0x1p-53;double m=((i&3)<2)?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);return(i&1)?-m:m;}
static inline uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static inline uint64_t ord(double x){uint64_t u=bits(x);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ulpd(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*(double)t.tv_nsec;}

static void intel_ha(double *o,const double *i,size_t n){vmdExp((MKL_INT)n,i,o,VML_HA);}

/* Positive control for compiler vector-math selection.  This is NOT our kernel.
   It exists only to prove whether the ICX build is actually selecting SVML. */
__attribute__((noinline))
static void compiler_exp_loop(double *restrict o,const double *restrict i,size_t n){
#pragma omp simd
    for(size_t k=0;k<n;k++) o[k]=exp(i[k]);
}

static void screen(const char *name,fn_t f,const double *in,const double *frozen,size_t n,int compare_frozen){
    double *out=aligned_alloc(64,n*sizeof(double));
    f(out,in,n);
    uint64_t mx=0,gt1=0,bd=0;
    for(size_t k=0;k<n;k++){
        double ref=(double)expl((long double)in[k]);
        uint64_t d=ulpd(out[k],ref); if(d>mx)mx=d; if(d>1)gt1++;
        if(compare_frozen && bits(out[k])!=bits(frozen[k])) bd++;
    }
    printf("SVMLSPINE_ACC build=%s name=%s n=%zu maxulp=%llu gt1=%llu bitdiff_vs_frozen=%llu\n",
           BUILD_LABEL,name,n,(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)bd);
    free(out);
}

static double bench(fn_t f,size_t n){
    double *in=aligned_alloc(64,n*sizeof(double)),*out=aligned_alloc(64,n*sizeof(double));
    rs=0xdecafbad12345678ULL;for(size_t k=0;k<n;k++)in[k]=genx(k);
    for(int w=0;w<100;w++)f(out,in,n);
    int reps=(n<=12288)?40000:10000;double best=1e99;
    for(int z=0;z<9;z++){
        double a=now();for(int r=0;r<reps;r++)f(out,in,n);double e=now()-a;
        if(e<best)best=e;sink+=out[(z*97u)%n];
    }
    free(in);free(out);return best*1e9/((double)reps*(double)n);
}

int main(void){
    mkl_set_num_threads_local(1);
    printf("SVMLSPINE_BUILD label=%s\n",BUILD_LABEL);
    const size_t an=200000;
    double *ain=aligned_alloc(64,an*sizeof(double)),*frozen=aligned_alloc(64,an*sizeof(double));
    rs=0x123456789abcdef0ULL;for(size_t k=0;k<an;k++)ain[k]=genx(k);
    exp53_n2_vmstyle_u4_0381_frozen(frozen,ain,an);
    screen("OURS_N2",exp53_n2_vmstyle_u4_0381_frozen,ain,frozen,an,1);
    screen("COMPILER_EXP",compiler_exp_loop,ain,frozen,an,0);
    screen("INTEL_VML_HA",intel_ha,ain,frozen,an,0);
    free(ain);free(frozen);

    struct {const char *n;fn_t f;} a[]={
        {"OURS_N2",exp53_n2_vmstyle_u4_0381_frozen},
        {"COMPILER_EXP",compiler_exp_loop},
        {"INTEL_VML_HA",intel_ha}
    };
    const size_t ns[]={12288,65536};
    for(int z=0;z<2;z++)for(size_t k=0;k<sizeof(a)/sizeof(a[0]);k++)
        printf("SVMLSPINE_TIME build=%s name=%s n=%zu ns_per_input=%.9f\n",BUILD_LABEL,a[k].n,ns[z],bench(a[k].f,ns[z]));
    return sink==1234567.0;
}
