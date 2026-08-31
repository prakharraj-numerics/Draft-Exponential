#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void exp53_n2_fused_u4_038_frozen(double*,const double*,size_t);
void exp53_n2_hw_swpipe4(double*,const double*,size_t);
void exp53_n2_hw_splitg256_u4(double*,const double*,size_t);
void exp53_n2_hw_aligned_u4(double*,const double*,size_t);
void exp53_n2_hw_prefetch_u4(double*,const double*,size_t);
void exp53_n2_hw_avx2_u4(double*,const double*,size_t);
typedef void(*fn)(double*,const double*,size_t); static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);} static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;} static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);} static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}
static double tm(fn f,double*out,const double*in,size_t n){
  const uint64_t target=200000000ULL; int reps=(int)(target/n); if(reps<256)reps=256; if(reps>5000)reps=5000;
  for(int k=0;k<24;k++)f(out,in,n); double best=1e9;
  for(int z=0;z<5;z++){double t=now();for(int k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*37)%n];}
  printf("HW_META n=%zu reps=%d trials=5 processed_per_trial=%llu\n",n,reps,(unsigned long long)reps*n);
  return best*1e9/((double)reps*n);
}
static int acc(const char*n,fn f,double*out,const double*in,size_t N){f(out,in,N);uint64_t mx=0;int g1=0;for(size_t i=0;i<N;i++){uint64_t d=D(out[i],exp(in[i]));if(d>mx)mx=d;if(d>1)g1++;}printf("HW_ACC %-12s max=%llu gt1=%d\n",n,(unsigned long long)mx,g1);return g1==0;}
int main(void){enum{NA=200000,NB=65536};size_t M=NA>NB?NA:NB;double*x=aligned_alloc(64,M*sizeof(double)),*y=aligned_alloc(64,M*sizeof(double));uint64_t s=0x123456789abcdef0ULL;for(size_t i=0;i<M/4;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(size_t i=M/4;i<M/2;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(size_t i=M/2;i<3*M/4;i++)x[i]=1.0+uni(&s)*99.0;for(size_t i=3*M/4;i<M;i++)x[i]=-(1.0+uni(&s)*99.0);mkl_set_num_threads_local(1);struct{const char*n;fn f;int safe;}a[]={{"base038",exp53_n2_fused_u4_038_frozen,0},{"swpipe4",exp53_n2_hw_swpipe4,0},{"splitg256",exp53_n2_hw_splitg256_u4,0},{"aligned",exp53_n2_hw_aligned_u4,0},{"prefetch",exp53_n2_hw_prefetch_u4,0},{"avx2",exp53_n2_hw_avx2_u4,0}};int na=sizeof(a)/sizeof(a[0]);for(int i=0;i<na;i++)a[i].safe=acc(a[i].n,a[i].f,y,x,NA);size_t ns[]={12288,65536};for(int z=0;z<2;z++){size_t n=ns[z];double ti=tm(intel,y,x,n);printf("HW_TIME n=%zu INTEL %.6f\n",n,ti);for(int i=0;i<na;i++)if(a[i].safe){double t=tm(a[i].f,y,x,n);printf("HW_TIME n=%zu %-12s %.6f intel_over=%.4fx\n",n,a[i].n,t,ti/t);}}free(x);free(y);return sink==1234567.0;}
