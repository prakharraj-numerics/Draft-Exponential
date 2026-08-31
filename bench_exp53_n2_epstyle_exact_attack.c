#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
void exp53_n2_epx_glx_stream1(double*,const double*,size_t);
void exp53_n2_epx_glx_pair2(double*,const double*,size_t);
void exp53_n2_epx_glx_trip3(double*,const double*,size_t);
void exp53_n2_epx_glx_consume4(double*,const double*,size_t);

typedef void(*fn_t)(double*,const double*,size_t);
static volatile double sink;
static uint64_t rng=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(void){rng^=rng<<7;rng^=rng>>9;rng^=rng<<8;return rng;}
static inline double gx(size_t i){double u=(double)(xr()>>11)*0x1p-53;double m=((i&3)<2)?(0x1p-20+u*(1.0-0x1p-20)):(1.0+99.0*u);return(i&1)?-m:m;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*(double)t.tv_nsec;}
static inline uint64_t ord(double x){uint64_t u;memcpy(&u,&x,8);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ulpd(double a,double b){if(isnan(a)||isnan(b))return UINT64_MAX;uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}

static void intel_ep(double*o,const double*i,size_t n){vmdExp((MKL_INT)n,i,o,VML_EP);}
static void intel_ha(double*o,const double*i,size_t n){vmdExp((MKL_INT)n,i,o,VML_HA);}

static void acc(const char*name,fn_t f,const double*in,const double*base,size_t n){
 double*out=aligned_alloc(64,n*sizeof(double));f(out,in,n);uint64_t mx=0;size_t diff=0;
 for(size_t i=0;i<n;i++){double ref=(double)expl((long double)in[i]);uint64_t d=ulpd(out[i],ref);if(d>mx)mx=d;if(base&&memcmp(&out[i],&base[i],8))diff++;}
 printf("EPX_ACC name=%-12s n=%zu maxulp=%llu bitdiff_vs_frozen=%zu\n",name,n,(unsigned long long)mx,diff);free(out);
}

static double bench(fn_t f,const double*in,double*out,size_t n){
 for(int w=0;w<100;w++)f(out,in,n);int reps=(n<=12288)?50000:12000;double best=1e99;
 for(int z=0;z<9;z++){double a=now();for(int r=0;r<reps;r++)f(out,in,n);double e=now()-a;if(e<best)best=e;sink+=out[(z*97u)%n];}
 return best*1e9/((double)reps*(double)n);
}

int main(void){
 mkl_set_num_threads_local(1);
 const size_t na=200000;double*ain=aligned_alloc(64,na*sizeof(double));double*abase=aligned_alloc(64,na*sizeof(double));
 rng=0x123456789abcdef0ULL;for(size_t i=0;i<na;i++)ain[i]=gx(i);exp53_n2_vmstyle_u4_0381_frozen(abase,ain,na);
 acc("frozen",exp53_n2_vmstyle_u4_0381_frozen,ain,abase,na);
 acc("stream1",exp53_n2_epx_glx_stream1,ain,abase,na);
 acc("pair2",exp53_n2_epx_glx_pair2,ain,abase,na);
 acc("trip3",exp53_n2_epx_glx_trip3,ain,abase,na);
 acc("consume4",exp53_n2_epx_glx_consume4,ain,abase,na);
 acc("intel_EP",intel_ep,ain,NULL,na);acc("intel_HA",intel_ha,ain,NULL,na);
 free(ain);free(abase);
 struct{const char*n;fn_t f;} fs[]={{"frozen",exp53_n2_vmstyle_u4_0381_frozen},{"stream1",exp53_n2_epx_glx_stream1},{"pair2",exp53_n2_epx_glx_pair2},{"trip3",exp53_n2_epx_glx_trip3},{"consume4",exp53_n2_epx_glx_consume4},{"intel_EP",intel_ep},{"intel_HA",intel_ha}};
 size_t ns[]={12288,65536};
 for(int q=0;q<2;q++){size_t n=ns[q];double*in=aligned_alloc(64,n*sizeof(double));double*out=aligned_alloc(64,n*sizeof(double));rng=0xdecafbad12345678ULL;for(size_t i=0;i<n;i++)in[i]=gx(i);
  for(size_t j=0;j<sizeof(fs)/sizeof(fs[0]);j++)printf("EPX_TIME n=%zu name=%-12s ns_per_input=%.9f\n",n,fs[j].n,bench(fs[j].f,in,out,n));free(in);free(out);}
 return sink==1234567.0;
}
