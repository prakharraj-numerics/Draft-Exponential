#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*fn_t)(double*,const double*,size_t);
static volatile double sink;

static void f_ha(double*o,const double*i,size_t n){vmdExp((MKL_INT)n,i,o,VML_HA);} 
static void f_la(double*o,const double*i,size_t n){vmdExp((MKL_INT)n,i,o,VML_LA);} 
static void f_ep(double*o,const double*i,size_t n){vmdExp((MKL_INT)n,i,o,VML_EP);} 

static inline uint64_t ord(double x){uint64_t u;memcpy(&u,&x,8);return (u>>63)?~u:(u|0x8000000000000000ULL);} 
static inline uint64_t ulpd(double a,double b){if(isnan(a)||isnan(b))return UINT64_MAX;uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static uint64_t rs=0x9e3779b97f4a7c15ULL;
static inline uint64_t xr(void){rs^=rs<<7;rs^=rs>>9;rs^=rs<<8;return rs;}
static inline double genx(size_t i){double u=(double)(xr()>>11)*0x1p-53;double m=((i&3)<2)?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);return(i&1)?-m:m;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (double)t.tv_sec+1e-9*(double)t.tv_nsec;}

static void accuracy(const char*name,fn_t f){
  const size_t n=200000;double*in=aligned_alloc(64,n*sizeof(double));double*out=aligned_alloc(64,n*sizeof(double));
  rs=0x123456789abcdef0ULL;for(size_t i=0;i<n;i++)in[i]=genx(i);f(out,in,n);uint64_t mx=0;
  for(size_t i=0;i<n;i++){double ref=(double)expl((long double)in[i]);uint64_t d=ulpd(out[i],ref);if(d>mx)mx=d;}
  printf("VML_ACC mode=%s n=%zu maxulp=%llu\n",name,n,(unsigned long long)mx);free(in);free(out);
}

static double bench(fn_t f,size_t n){
  double*in=aligned_alloc(64,n*sizeof(double));double*out=aligned_alloc(64,n*sizeof(double));
  rs=0xdecafbad12345678ULL;for(size_t i=0;i<n;i++)in[i]=genx(i);for(int w=0;w<80;w++)f(out,in,n);
  int reps=(n<=12288)?50000:12000;double best=1e99;
  for(int z=0;z<9;z++){double a=now();for(int r=0;r<reps;r++)f(out,in,n);double e=now()-a;if(e<best)best=e;sink+=out[(z*97u)%n];}
  free(in);free(out);return best*1e9/((double)reps*(double)n);
}

int main(void){
  mkl_set_num_threads_local(1);
  struct {const char*n;fn_t f;} a[]={{"HA",f_ha},{"LA",f_la},{"EP",f_ep}};
  for(int i=0;i<3;i++)accuracy(a[i].n,a[i].f);
  size_t ns[]={12288,65536};
  for(int z=0;z<2;z++)for(int i=0;i<3;i++)printf("VML_TIME mode=%s n=%zu ns_per_input=%.9f\n",a[i].n,ns[z],bench(a[i].f,ns[z]));
  return sink==1234567.0;
}
