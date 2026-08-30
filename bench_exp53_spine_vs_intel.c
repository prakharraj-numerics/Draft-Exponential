#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>
#include <mkl_vml.h>
typedef void(*fn_t)(double*,const double*,size_t);
void exp53_spine_v128_frozen(double*,const double*,size_t);void exp53_spine_v128_u2(double*,const double*,size_t);void exp53_spine_v128_u4(double*,const double*,size_t);void exp53_spine_v128_decomp(double*,const double*,size_t);void exp53_spine_v128_avx2(double*,const double*,size_t);
enum{N=6400,HALF=3200,REPS=20000};static double x[N],y[N],z[N];
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}static inline double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*t.tv_nsec;}static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static void inputs(void){uint64_t s=0x6a09e667f3bcc909ULL;for(int i=0;i<1600;i++)x[i]=0x1p-20+uni(&s)*(1-0x1p-20);for(int i=1600;i<3200;i++)x[i]=-(0x1p-20+uni(&s)*(1-0x1p-20));for(int i=3200;i<4800;i++)x[i]=1+uni(&s)*99;for(int i=4800;i<6400;i++)x[i]=-(1+uni(&s)*99);}
static void acc(const char*n,fn_t f){f(y,x,N);uint64_t mx=0;int g1=0,g2=0,g4=0;long double mr=0;for(int i=0;i<N;i++){double r=exp(x[i]);uint64_t d=D(y[i],r);if(d>mx)mx=d;if(d>1)g1++;if(d>2)g2++;if(d>4)g4++;long double e=fabsl(((long double)y[i]-r)/r);if(e>mr)mr=e;}printf("ACC %-14s maxulp=%llu gt1=%d gt2=%d gt4=%d maxrel=%.3Le\n",n,(unsigned long long)mx,g1,g2,g4,mr);}
static double bo(fn_t f,int lo,int n){for(int i=0;i<200;i++)f(y+lo,x+lo,n);double a=sec();for(int i=0;i<REPS;i++)f(y+lo,x+lo,n);double b=sec();return(b-a)*1e9/((double)REPS*n);}static double bi(int lo,int n,MKL_INT64 mode){for(int i=0;i<200;i++)vmdExp(n,x+lo,z+lo,mode);double a=sec();for(int i=0;i<REPS;i++)vmdExp(n,x+lo,z+lo,mode);double b=sec();return(b-a)*1e9/((double)REPS*n);}static void rep(const char*n,fn_t f,double ia){double s=bo(f,0,HALF),w=bo(f,HALF,HALF),a=bo(f,0,N);printf("RES %-14s small=%.6f wide=%.6f all=%.6f intelHA_over=%.4fx\n",n,s,w,a,ia/a);}
int main(void){setenv("MKL_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);mkl_set_num_threads_local(1);inputs();fn_t fs[]={exp53_spine_v128_frozen,exp53_spine_v128_u2,exp53_spine_v128_u4,exp53_spine_v128_decomp,exp53_spine_v128_avx2};const char*ns[]={"frozen","u2","u4","decomp16x8","avx2"};for(int i=0;i<5;i++)acc(ns[i],fs[i]);double ha=bi(0,N,VML_HA),la=bi(0,N,VML_LA),ep=bi(0,N,VML_EP);printf("INTEL_MODES HA=%.6f LA=%.6f EP=%.6f\n",ha,la,ep);for(int i=0;i<5;i++)rep(ns[i],fs[i],ha);return 0;}
