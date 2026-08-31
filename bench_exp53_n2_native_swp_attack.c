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
void exp53_n2_native_default(double*,const double*,size_t);
void exp53_n2_native_swp_disable(double*,const double*,size_t);
void exp53_n2_native_swp_ii1(double*,const double*,size_t);
void exp53_n2_native_swp_ii2(double*,const double*,size_t);
void exp53_n2_native_swp_ii4(double*,const double*,size_t);
void exp53_n2_native_swp_ii8(double*,const double*,size_t);
void exp53_n2_native_interleave2(double*,const double*,size_t);
void exp53_n2_native_interleave4(double*,const double*,size_t);

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
    double *out=aligned_alloc(64,n*sizeof(double));
    f(out,in,n); uint64_t mx=0,gt1=0,bd=0;
    for(size_t k=0;k<n;k++){
        double ref=(double)expl((long double)in[k]);
        uint64_t d=ulpd(out[k],ref); if(d>mx)mx=d; if(d>1)gt1++;
        if(bits(out[k])!=bits(frozen[k]))bd++;
    }
    printf("NATIVESWP_ACC name=%s n=%zu maxulp=%llu gt1=%llu bitdiff_frozen=%llu\n",name,n,(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)bd);
    free(out);
}

static double bench(fn_t f,size_t n){
    double *in=aligned_alloc(64,n*sizeof(double)),*out=aligned_alloc(64,n*sizeof(double));
    rs=0xdecafbad12345678ULL; for(size_t k=0;k<n;k++)in[k]=genx(k);
    for(int w=0;w<80;w++)f(out,in,n);
    int reps=n<=12288?30000:6000; double best=1e99;
    for(int z=0;z<7;z++){
        double a=now(); for(int r=0;r<reps;r++)f(out,in,n); double e=now()-a;
        if(e<best)best=e; sink+=out[(z*97u)%n];
    }
    free(in);free(out); return best*1e9/((double)reps*(double)n);
}

int main(void){
    mkl_set_num_threads_local(1);
    const size_t an=200000;
    double *ain=aligned_alloc(64,an*sizeof(double)),*fro=aligned_alloc(64,an*sizeof(double));
    rs=0x123456789abcdef0ULL; for(size_t k=0;k<an;k++)ain[k]=genx(k);
    exp53_n2_vmstyle_u4_0381_frozen(fro,ain,an);
    struct {const char*n;fn_t f;} a[]={
      {"FROZEN",exp53_n2_vmstyle_u4_0381_frozen},
      {"NATIVE_DEFAULT",exp53_n2_native_default},
      {"SWP_DISABLE",exp53_n2_native_swp_disable},
      {"SWP_II1",exp53_n2_native_swp_ii1},
      {"SWP_II2",exp53_n2_native_swp_ii2},
      {"SWP_II4",exp53_n2_native_swp_ii4},
      {"SWP_II8",exp53_n2_native_swp_ii8},
      {"INTERLEAVE2",exp53_n2_native_interleave2},
      {"INTERLEAVE4",exp53_n2_native_interleave4},
      {"VML_HA",intel_ha}
    };
    for(size_t k=0;k<9;k++)screen(a[k].n,a[k].f,ain,fro,an);
    free(ain);free(fro);
    const size_t ns[]={10079,12288,65536};
    for(size_t z=0;z<3;z++)for(size_t k=0;k<10;k++)
      printf("NATIVESWP_TIME name=%s n=%zu ns_per_input=%.9f\n",a[k].n,ns[z],bench(a[k].f,ns[z]));
    return sink==1234567.0;
}
