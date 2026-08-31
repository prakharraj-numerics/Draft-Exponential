#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void exp53_n2_fused_u4_038_frozen(double*,const double*,size_t);
void exp53_n2_vm_early_u2(double*,const double*,size_t);
void exp53_n2_vm_early_u3(double*,const double*,size_t);
void exp53_n2_vm_early_u4(double*,const double*,size_t);
void exp53_n2_vm_early_u5(double*,const double*,size_t);
void exp53_n2_vm_early_u4_aligned(double*,const double*,size_t);
typedef void(*fn)(double*,const double*,size_t);
static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);} 
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static inline uint64_t bits(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t ord(double a){uint64_t u=bits(a);return(u>>63)?~u:(u|0x8000000000000000ULL);} 
static inline uint64_t ulpd(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);} 
static double tm(fn f,double*out,const double*in,size_t n){int reps=n<=12288?50000:12000;for(int k=0;k<60;k++)f(out,in,n);double best=1e99;for(int z=0;z<9;z++){double t=now();for(int k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*53)%n];}return best*1e9/(reps*n);} 

int main(void){
  enum{NA=200000,NB=65536}; size_t M=NA>NB?NA:NB;
  double*x=aligned_alloc(64,M*sizeof(double)),*yb=aligned_alloc(64,M*sizeof(double)),*y=aligned_alloc(64,M*sizeof(double));
  uint64_t s=0x123456789abcdef0ULL;
  for(size_t i=0;i<M/4;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);
  for(size_t i=M/4;i<M/2;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));
  for(size_t i=M/2;i<3*M/4;i++)x[i]=1.0+uni(&s)*99.0;
  for(size_t i=3*M/4;i<M;i++)x[i]=-(1.0+uni(&s)*99.0);
  struct{const char*n;fn f;} a[]={
    {"vm_early_u2",exp53_n2_vm_early_u2},{"vm_early_u3",exp53_n2_vm_early_u3},
    {"vm_early_u4",exp53_n2_vm_early_u4},{"vm_early_u5",exp53_n2_vm_early_u5},
    {"vm_u4_aligned",exp53_n2_vm_early_u4_aligned}};
  int na=(int)(sizeof(a)/sizeof(a[0]));
  exp53_n2_fused_u4_038_frozen(yb,x,NA);
  uint64_t bmx=0;int bg1=0;for(size_t i=0;i<NA;i++){uint64_t d=ulpd(yb[i],exp(x[i]));if(d>bmx)bmx=d;if(d>1)bg1++;}
  printf("VM_ACC base038 max=%llu gt1=%d bitdiff_vs_base=0\n",(unsigned long long)bmx,bg1);
  for(int z=0;z<na;z++){
    a[z].f(y,x,NA);uint64_t mx=0,bd=0;int g1=0;
    for(size_t i=0;i<NA;i++){uint64_t d=ulpd(y[i],exp(x[i]));if(d>mx)mx=d;if(d>1)g1++;if(bits(y[i])!=bits(yb[i]))bd++;}
    printf("VM_ACC %-14s max=%llu gt1=%d bitdiff_vs_base=%llu\n",a[z].n,(unsigned long long)mx,g1,(unsigned long long)bd);
  }
  mkl_set_num_threads_local(1);
  size_t ns[]={12288,65536};
  for(int zz=0;zz<2;zz++){
    size_t n=ns[zz];double ti=tm(intel,y,x,n);double tb=tm(exp53_n2_fused_u4_038_frozen,y,x,n);
    printf("VM_TIME n=%zu INTEL %.6f\n",n,ti);
    printf("VM_TIME n=%zu base038 %.6f intel_over=%.4fx\n",n,tb,ti/tb);
    for(int z=0;z<na;z++){double t=tm(a[z].f,y,x,n);printf("VM_TIME n=%zu %-14s %.6f vs_base=%.4fx intel_over=%.4fx\n",n,a[z].n,t,tb/t,ti/t);}
  }
  free(x);free(yb);free(y);return sink==1234567.0;
}
