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
void exp53_n2_epstyle_perm8_early(double*,const double*,size_t);
void exp53_n2_epstyle_perm8_afterreduce(double*,const double*,size_t);

static volatile double sink;
static uint64_t rs=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(void){rs^=rs<<7;rs^=rs>>9;rs^=rs<<8;return rs;}
static inline double genx(size_t i){double u=(double)(xr()>>11)*0x1p-53;double m=((i&3)<2)?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);return(i&1)?-m:m;}
static inline uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static inline uint64_t ord(double x){uint64_t u=bits(x);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ulpd(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*(double)t.tv_nsec;}

static void intel_ha(double*o,const double*i,size_t n){vmdExp((MKL_INT)n,i,o,VML_HA);}
static void intel_ep(double*o,const double*i,size_t n){vmdExp((MKL_INT)n,i,o,VML_EP);}

static void screen(const char *name, fn_t f, const double *in, const double *base, size_t n, int compare_base){
    double *out=aligned_alloc(64,n*sizeof(double));
    f(out,in,n);
    uint64_t mx=0,bd=0;
    for(size_t i=0;i<n;i++){
        double ref=(double)expl((long double)in[i]);
        uint64_t d=ulpd(out[i],ref); if(d>mx)mx=d;
        if(compare_base && bits(out[i])!=bits(base[i])) bd++;
    }
    printf("EPATTACK_ACC name=%s n=%zu maxulp=%llu bitdiff_vs_frozen=%llu\n",name,n,(unsigned long long)mx,(unsigned long long)bd);
    free(out);
}

static double bench(fn_t f,size_t n){
    double *in=aligned_alloc(64,n*sizeof(double)),*out=aligned_alloc(64,n*sizeof(double));
    rs=0xdecafbad12345678ULL; for(size_t i=0;i<n;i++)in[i]=genx(i);
    for(int w=0;w<100;w++)f(out,in,n);
    int reps=(n<=12288)?40000:10000; double best=1e99;
    for(int z=0;z<9;z++){
        double a=now(); for(int r=0;r<reps;r++)f(out,in,n); double e=now()-a;
        if(e<best)best=e; sink+=out[(z*97u)%n];
    }
    free(in);free(out);return best*1e9/((double)reps*(double)n);
}

int main(void){
    mkl_set_num_threads_local(1);
    const size_t an=200000;
    double *ain=aligned_alloc(64,an*sizeof(double)),*base=aligned_alloc(64,an*sizeof(double));
    rs=0x123456789abcdef0ULL;for(size_t i=0;i<an;i++)ain[i]=genx(i);
    exp53_n2_vmstyle_u4_0381_frozen(base,ain,an);
    screen("FROZEN",exp53_n2_vmstyle_u4_0381_frozen,ain,base,an,1);
    screen("EP_PERM8_EARLY",exp53_n2_epstyle_perm8_early,ain,base,an,1);
    screen("EP_PERM8_AFTERREDUCE",exp53_n2_epstyle_perm8_afterreduce,ain,base,an,1);
    screen("INTEL_HA",intel_ha,ain,base,an,0);
    screen("INTEL_EP",intel_ep,ain,base,an,0);
    free(ain);free(base);

    struct {const char*n;fn_t f;} a[]={
      {"FROZEN",exp53_n2_vmstyle_u4_0381_frozen},
      {"EP_PERM8_EARLY",exp53_n2_epstyle_perm8_early},
      {"EP_PERM8_AFTERREDUCE",exp53_n2_epstyle_perm8_afterreduce},
      {"INTEL_HA",intel_ha},{"INTEL_EP",intel_ep}};
    const size_t ns[]={12288,65536};
    for(int z=0;z<2;z++) for(size_t k=0;k<sizeof(a)/sizeof(a[0]);k++)
      printf("EPATTACK_TIME name=%s n=%zu ns_per_input=%.9f\n",a[k].n,ns[z],bench(a[k].f,ns[z]));
    return sink==1234567.0;
}
