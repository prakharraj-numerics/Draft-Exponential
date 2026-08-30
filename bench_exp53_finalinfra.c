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
void exp53_spine_v128_formula6_compfastC(double*,const double*,size_t);
void exp53_all_pipe_u6(double*,const double*,size_t);
void exp53_final_early_u4(double*,const double*,size_t); void exp53_final_early_u5(double*,const double*,size_t); void exp53_final_early_u6(double*,const double*,size_t);
void exp53_final_mid_u4(double*,const double*,size_t); void exp53_final_mid_u5(double*,const double*,size_t); void exp53_final_mid_u6(double*,const double*,size_t);
void exp53_final_split_u6(double*,const double*,size_t);
enum{N=12288,HALF=6144,REPS=30000}; static double x[N],y[N],z[N];
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;} static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);} static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;} static inline double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*t.tv_nsec;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);} static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static void inputs(void){uint64_t s=0x243f6a8885a308d3ULL;for(int i=0;i<HALF/2;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(int i=HALF/2;i<HALF;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(int i=HALF;i<HALF+HALF/2;i++)x[i]=1.0+uni(&s)*99.0;for(int i=HALF+HALF/2;i<N;i++)x[i]=-(1.0+uni(&s)*99.0);}
static void acc(const char*n,fn_t f){f(y,x,N);uint64_t mx=0;int g1=0,g2=0;for(int i=0;i<N;i++){double r=exp(x[i]);uint64_t d=D(y[i],r);if(d>mx)mx=d;if(d>1)g1++;if(d>2)g2++;}printf("ACC %-12s maxulp=%llu gt1=%d gt2=%d\n",n,(unsigned long long)mx,g1,g2);}
static double bo(fn_t f,int lo,int n){for(int i=0;i<700;i++)f(y+lo,x+lo,n);double a=sec();for(int i=0;i<REPS;i++)f(y+lo,x+lo,n);double b=sec();return(b-a)*1e9/((double)REPS*n);} static double bi(int lo,int n){for(int i=0;i<700;i++)vmdExp(n,x+lo,z+lo,VML_HA);double a=sec();for(int i=0;i<REPS;i++)vmdExp(n,x+lo,z+lo,VML_HA);double b=sec();return(b-a)*1e9/((double)REPS*n);}
static void rep(const char*n,fn_t f){printf("RES %-12s small=%.6f wide=%.6f all=%.6f\n",n,bo(f,0,HALF),bo(f,HALF,HALF),bo(f,0,N));}
int main(void){setenv("MKL_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);mkl_set_num_threads_local(1);inputs();struct{const char*n;fn_t f;}v[]={{"baselineC",exp53_spine_v128_formula6_compfastC},{"pipe_u6",exp53_all_pipe_u6},{"early_u4",exp53_final_early_u4},{"early_u5",exp53_final_early_u5},{"early_u6",exp53_final_early_u6},{"mid_u4",exp53_final_mid_u4},{"mid_u5",exp53_final_mid_u5},{"mid_u6",exp53_final_mid_u6},{"split_u6",exp53_final_split_u6}};for(unsigned i=0;i<sizeof(v)/sizeof(v[0]);i++)acc(v[i].n,v[i].f);printf("INTEL small=%.6f wide=%.6f all=%.6f\n",bi(0,HALF),bi(HALF,HALF),bi(0,N));for(unsigned i=0;i<sizeof(v)/sizeof(v[0]);i++)rep(v[i].n,v[i].f);return 0;}
