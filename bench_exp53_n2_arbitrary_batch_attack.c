#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*fn_t)(double*,const double*,size_t);
void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
void exp53_n2_pipe3_stream(double*,const double*,size_t);
void exp53_n2_u4_masktail(double*,const double*,size_t);
void exp53_n2_dynamic_batch(double*,const double*,size_t);
static void intel_ha(double *o,const double *i,size_t n){vmdExp((MKL_INT)n,i,o,VML_HA);}

static volatile double sink;
static uint64_t rs=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(void){rs^=rs<<7;rs^=rs>>9;rs^=rs<<8;return rs;}
static inline double genx(size_t i){double u=(double)(xr()>>11)*0x1p-53;double m=((i&3)<2)?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);return(i&1)?-m:m;}
static inline uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static inline uint64_t ord(double x){uint64_t u=bits(x);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ulpd(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*(double)t.tv_nsec;}
static void *a64(size_t n){size_t b=(n*sizeof(double)+63u)&~63u;return aligned_alloc(64,b?b:64);}

static void screen(const char *name,fn_t f,const double *in,const double *frozen,size_t n,int bdcheck){
    double *out=a64(n); f(out,in,n); uint64_t mx=0,gt1=0,bd=0;
    for(size_t k=0;k<n;k++){uint64_t d=ulpd(out[k],(double)expl((long double)in[k]));if(d>mx)mx=d;if(d>1)gt1++;if(bdcheck&&bits(out[k])!=bits(frozen[k]))bd++;}
    printf("ARB_ACC name=%s n=%zu maxulp=%llu gt1=%llu bitdiff_vs_frozen=%llu\n",name,n,(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)bd);free(out);
}

static void short_sweep(const char *name,fn_t f){
    uint64_t mx=0,gt1=0; size_t where=0;
    for(size_t n=1;n<=1025;n++){
        double *in=a64(n),*out=a64(n);rs=0x13579bdf00000000ULL+n;for(size_t k=0;k<n;k++)in[k]=genx(k);f(out,in,n);
        for(size_t k=0;k<n;k++){uint64_t d=ulpd(out[k],(double)expl((long double)in[k]));if(d>mx){mx=d;where=n;}if(d>1)gt1++;}free(in);free(out);
    }
    printf("ARB_SHORT name=%s nrange=1..1025 maxulp=%llu gt1=%llu max_at_n=%zu\n",name,(unsigned long long)mx,(unsigned long long)gt1,where);
}

static double bench(fn_t f,size_t n){
    double *in=a64(n),*out=a64(n);rs=0xdecafbad12345678ULL;for(size_t k=0;k<n;k++)in[k]=genx(k);for(int w=0;w<100;w++)f(out,in,n);
    int reps=(n<=64)?300000:(n<=256?200000:(n<=12288?40000:10000));double best=1e99;
    for(int z=0;z<9;z++){double a=now();for(int r=0;r<reps;r++)f(out,in,n);double e=now()-a;if(e<best)best=e;sink+=out[(z*97u)%n];}
    free(in);free(out);return best*1e9/((double)reps*(double)n);
}

int main(void){
    mkl_set_num_threads_local(1);
    const size_t an=200000;double *ain=a64(an),*fr=a64(an);rs=0x123456789abcdef0ULL;for(size_t k=0;k<an;k++)ain[k]=genx(k);exp53_n2_vmstyle_u4_0381_frozen(fr,ain,an);
    screen("U4_MASKTAIL",exp53_n2_u4_masktail,ain,fr,an,1);screen("DYNAMIC",exp53_n2_dynamic_batch,ain,fr,an,1);free(ain);free(fr);
    short_sweep("U4_MASKTAIL",exp53_n2_u4_masktail);short_sweep("DYNAMIC",exp53_n2_dynamic_batch);
    struct {const char*name;fn_t f;} fs[]={{"FROZEN",exp53_n2_vmstyle_u4_0381_frozen},{"PIPE3",exp53_n2_pipe3_stream},{"U4_MASKTAIL",exp53_n2_u4_masktail},{"DYNAMIC",exp53_n2_dynamic_batch},{"VML_HA",intel_ha}};
    const size_t ns[]={31,32,33,63,64,65,100,253,511,512,513,10079,12288,65536};
    for(size_t z=0;z<sizeof(ns)/sizeof(ns[0]);z++)for(size_t k=0;k<sizeof(fs)/sizeof(fs[0]);k++)printf("ARB_TIME name=%s n=%zu ns_per_input=%.9f\n",fs[k].name,ns[z],bench(fs[k].f,ns[z]));
    return sink==1234567.0;
}
