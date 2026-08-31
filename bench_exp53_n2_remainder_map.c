#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*fn_t)(double*,const double*,size_t);
void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
void exp53_n2_pipe3_stream(double*,const double*,size_t);
void exp53_n2_u4_masktail(double*,const double*,size_t);

static volatile double sink;
static uint64_t rs=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(void){rs^=rs<<7;rs^=rs>>9;rs^=rs<<8;return rs;}
static inline double genx(size_t i){double u=(double)(xr()>>11)*0x1p-53;double m=((i&3)<2)?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);return(i&1)?-m:m;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*(double)t.tv_nsec;}
static void *a64(size_t n){size_t b=(n*sizeof(double)+63u)&~63u;return aligned_alloc(64,b?b:64);}
static double bench(fn_t f,size_t n){
    double *in=a64(n),*out=a64(n);rs=0xdecafbad12345678ULL+n;for(size_t k=0;k<n;k++)in[k]=genx(k);for(int w=0;w<100;w++)f(out,in,n);
    int reps=(n<128)?250000:(n<2048?80000:30000);double best=1e99;
    for(int z=0;z<9;z++){double a=now();for(int r=0;r<reps;r++)f(out,in,n);double e=now()-a;if(e<best)best=e;sink+=out[(z*97u)%n];}
    free(in);free(out);return best*1e9/((double)reps*(double)n);
}
int main(void){
    mkl_set_num_threads_local(1);
    const size_t bases[]={32,256,1024,10048};
    for(size_t b=0;b<sizeof(bases)/sizeof(bases[0]);b++){
        for(unsigned rem=0;rem<32;rem++){
            size_t n=bases[b]+rem;
            double fr=bench(exp53_n2_vmstyle_u4_0381_frozen,n);
            double mt=bench(exp53_n2_u4_masktail,n);
            double p3=bench(exp53_n2_pipe3_stream,n);
            printf("REMMAP base=%zu rem=%u n=%zu frozen=%.9f masktail=%.9f pipe3=%.9f mask_over_frozen=%.6f pipe_over_frozen=%.6f\n",bases[b],rem,n,fr,mt,p3,mt/fr,p3/fr);
        }
    }
    return sink==1234567.0;
}
