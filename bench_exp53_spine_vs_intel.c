#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl_vml.h>

void exp53_spine_avx512_batch(double *restrict out,const double *restrict in,size_t n);

enum { N=6400, HALF=3200, REPS=20000 };
static double x[N], y[N], z[N];

static inline uint64_t u64(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t ord(double a){uint64_t u=u64(a);return (u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ulpd(double a,double b){uint64_t A=ord(a),B=ord(b);return A>B?A-B:B-A;}
static inline double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (double)t.tv_sec+1e-9*(double)t.tv_nsec;}

static uint64_t sm64(uint64_t *s){uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static double uni(uint64_t *s){return (sm64(s)>>11)*0x1.0p-53;}
static void make_inputs(void){
  uint64_t s=0x6a09e667f3bcc909ULL;
  for(int i=0;i<1600;i++) x[i]=0x1.0p-20 + uni(&s)*(1.0-0x1.0p-20);
  for(int i=1600;i<3200;i++) x[i]=-(0x1.0p-20 + uni(&s)*(1.0-0x1.0p-20));
  for(int i=3200;i<4800;i++) x[i]=1.0 + uni(&s)*99.0;
  for(int i=4800;i<6400;i++) x[i]=-(1.0 + uni(&s)*99.0);
}

static void check(void){
  exp53_spine_avx512_batch(y,x,N);
  uint64_t maxulp=0; long double maxrel=0.0L; int gt1=0,gt2=0,gt4=0;
  for(int i=0;i<N;i++){
    double r=exp(x[i]); z[i]=r; uint64_t d=ulpd(y[i],r); if(d>maxulp)maxulp=d;
    if(d>1)gt1++; if(d>2)gt2++; if(d>4)gt4++;
    long double rel=fabsl(((long double)y[i]-(long double)r)/(long double)r); if(rel>maxrel)maxrel=rel;
  }
  printf("ACCURACY max_ulp_vs_libm=%llu gt1=%d gt2=%d gt4=%d max_rel=%.3Le\n",(unsigned long long)maxulp,gt1,gt2,gt4,maxrel);
}

static double bench_ours(int lo,int n){
  for(int w=0;w<200;w++)exp53_spine_avx512_batch(y+lo,x+lo,n);
  double t0=sec();
  for(int r=0;r<REPS;r++)exp53_spine_avx512_batch(y+lo,x+lo,n);
  double t1=sec();
  return (t1-t0)*1e9/((double)REPS*n);
}
static double bench_intel(int lo,int n){
  for(int w=0;w<200;w++)vmdExp(n,x+lo,z+lo,VML_HA);
  double t0=sec();
  for(int r=0;r<REPS;r++)vmdExp(n,x+lo,z+lo,VML_HA);
  double t1=sec();
  return (t1-t0)*1e9/((double)REPS*n);
}
int main(void){
  setenv("MKL_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);
  make_inputs();check();
  double os=bench_ours(0,HALF), ow=bench_ours(HALF,HALF), oa=bench_ours(0,N);
  double is=bench_intel(0,HALF), iw=bench_intel(HALF,HALF), ia=bench_intel(0,N);
  printf("EXP53_SPINE_SMALL ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.4fx\n",os,is,is/os);
  printf("EXP53_SPINE_WIDE ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.4fx\n",ow,iw,iw/ow);
  printf("EXP53_SPINE_ALL ours_ns=%.6f intel_ns=%.6f intel_over_ours=%.4fx\n",oa,ia,ia/oa);
  volatile double sink=y[7]+z[11]; (void)sink;
  return 0;
}
