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
void exp53_n2_critical_u6_48(double*,const double*,size_t);
void exp53_n2_critical_u8_64(double*,const double*,size_t);

static volatile double sink;
static uint64_t rs=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(void){rs^=rs<<7;rs^=rs>>9;rs^=rs<<8;return rs;}
static inline double genx(size_t i){double u=(double)(xr()>>11)*0x1p-53;double m=((i&3)<2)?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);return(i&1)?-m:m;}
static inline uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static inline uint64_t ord(double x){uint64_t u=bits(x);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ulpd(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*(double)t.tv_nsec;}
static void intel_ha(double *o,const double *i,size_t n){vmdExp((MKL_INT)n,i,o,VML_HA);}

static void screen(const char *name,fn_t f,const double *in,const double *frozen,size_t n){
    size_t bytes=((n*sizeof(double)+63)/64)*64;
    double *out=aligned_alloc(64,bytes);
    f(out,in,n); uint64_t mx=0,gt1=0,bd=0;
    for(size_t k=0;k<n;k++){
        double ref=(double)expl((long double)in[k]);
        uint64_t d=ulpd(out[k],ref); if(d>mx)mx=d; if(d>1)gt1++;
        if(bits(out[k])!=bits(frozen[k]))bd++;
    }
    printf("CRITWAVE_ACC name=%s n=%zu maxulp=%llu gt1=%llu bitdiff_frozen=%llu\n",name,n,(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)bd);
    free(out);
}

static double bench(fn_t f,size_t n){
    size_t bytes=((n*sizeof(double)+63)/64)*64;
    double *in=aligned_alloc(64,bytes),*out=aligned_alloc(64,bytes);
    rs=0xdecafbad12345678ULL; for(size_t k=0;k<n;k++)in[k]=genx(k);
    for(int w=0;w<120;w++)f(out,in,n);
    int reps;
    if(n<=253) reps=300000;
    else if(n<=12288) reps=40000;
    else reps=8000;
    double best=1e99;
    for(int z=0;z<9;z++){
        double a=now(); for(int r=0;r<reps;r++)f(out,in,n); double e=now()-a;
        if(e<best)best=e; sink+=out[(z*97u)%n];
    }
    free(in);free(out); return best*1e9/((double)reps*(double)n);
}

int main(void){
    mkl_set_num_threads_local(1);
    const size_t an=200000, abytes=((an*sizeof(double)+63)/64)*64;
    double *ain=aligned_alloc(64,abytes),*fro=aligned_alloc(64,abytes);
    rs=0x123456789abcdef0ULL; for(size_t k=0;k<an;k++)ain[k]=genx(k);
    exp53_n2_vmstyle_u4_0381_frozen(fro,ain,an);
    struct {const char*n;fn_t f;} a[]={
      {"FROZEN",exp53_n2_vmstyle_u4_0381_frozen},
      {"CRIT_U6_48",exp53_n2_critical_u6_48},
      {"CRIT_U8_64",exp53_n2_critical_u8_64},
      {"VML_HA",intel_ha}
    };
    for(size_t k=0;k<3;k++)screen(a[k].n,a[k].f,ain,fro,an);
    free(ain);free(fro);
    const size_t ns[]={100,253,10079,12288,65536};
    for(size_t z=0;z<sizeof(ns)/sizeof(ns[0]);z++)for(size_t k=0;k<4;k++)
      printf("CRITWAVE_TIME name=%s n=%zu ns_per_input=%.9f\n",a[k].n,ns[z],bench(a[k].f,ns[z]));
    return sink==1234567.0;
}
