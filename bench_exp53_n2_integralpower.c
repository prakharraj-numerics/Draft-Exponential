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
void exp53_final_mid_u6(double*,const double*,size_t);
void exp53_n2_fast_u2(double*,const double*,size_t);
void exp53_n2_fast_u4(double*,const double*,size_t);
void exp53_n2_fast_u6(double*,const double*,size_t);
void exp53_n2_pc_u2(double*,const double*,size_t);
void exp53_n2_pc_u4(double*,const double*,size_t);
void exp53_n2_pc_u6(double*,const double*,size_t);
void exp53_n2_rc_u4(double*,const double*,size_t);
void exp53_n2_rc_u6(double*,const double*,size_t);

enum{N=12288,HALF=6144};
static double x[N] __attribute__((aligned(64)));
static double y[N] __attribute__((aligned(64)));
static double z[N] __attribute__((aligned(64)));
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static inline double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*t.tv_nsec;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static void inputs(void){uint64_t s=0x9e3779b97f4a7c15ULL;for(int i=0;i<HALF/2;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(int i=HALF/2;i<HALF;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(int i=HALF;i<HALF+HALF/2;i++)x[i]=1.0+uni(&s)*99.0;for(int i=HALF+HALF/2;i<N;i++)x[i]=-(1.0+uni(&s)*99.0);}
static void acc(const char*n,fn_t f){f(y,x,N);uint64_t mx=0;int g0=0,g1=0,g2=0,g3=0,pos=0,neg=0;for(int i=0;i<N;i++){double r=exp(x[i]);uint64_t d=D(y[i],r);if(d>mx)mx=d;if(d>0)g0++;if(d>1){g1++;if(O(y[i])>O(r))pos++;else neg++;}if(d>2)g2++;if(d>3)g3++;}printf("ACC %-11s maxulp=%llu gt0=%d gt1=%d gt2=%d gt3=%d badpos=%d badneg=%d\n",n,(unsigned long long)mx,g0,g1,g2,g3,pos,neg);}
static long reps_for(int n){long r=240000000L/n;if(r<12000)r=12000;if(r>3000000)r=3000000;return r;}
static double bo_range(fn_t f,int off,int n){long reps=reps_for(n);for(int i=0;i<500;i++)f(y+off,x+off,n);double a=sec();for(long i=0;i<reps;i++)f(y+off,x+off,n);double b=sec();return(b-a)*1e9/((double)reps*n);}
static double bi_range(int off,int n){long reps=reps_for(n);for(int i=0;i<500;i++)vmdExp(n,x+off,z+off,VML_HA);double a=sec();for(long i=0;i<reps;i++)vmdExp(n,x+off,z+off,VML_HA);double b=sec();return(b-a)*1e9/((double)reps*n);}
struct V{const char*n;fn_t f;};
static void category(const char *cat,int off,int n,struct V *v,size_t nv){
 double it=bi_range(off,n);
 printf("CATEGORY %s n=%d INTEL_HA=%.6f\n",cat,n,it);
 for(size_t i=0;i<nv;i++){double t=bo_range(v[i].f,off,n);printf("CATEGORY %s n=%d %-11s=%.6f intel_over=%.4fx\n",cat,n,v[i].n,t,it/t);}
}
int main(void){
 setenv("MKL_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);mkl_set_num_threads_local(1);inputs();
 struct V v[]={
  {"safe_old",exp53_final_mid_u6},
  {"n2_fast_u2",exp53_n2_fast_u2},{"n2_fast_u4",exp53_n2_fast_u4},{"n2_fast_u6",exp53_n2_fast_u6},
  {"n2_pc_u2",exp53_n2_pc_u2},{"n2_pc_u4",exp53_n2_pc_u4},{"n2_pc_u6",exp53_n2_pc_u6},
  {"n2_rc_u4",exp53_n2_rc_u4},{"n2_rc_u6",exp53_n2_rc_u6}
 };
 size_t nv=sizeof(v)/sizeof(v[0]);
 puts("=== ACCURACY vs scalar system exp (screen; MPFR follows for survivors) ===");
 for(size_t i=0;i<nv;i++)acc(v[i].n,v[i].f);
 int bs[]={8,32,128,1024,12288};
 puts("=== BATCH ns/input ===");
 for(unsigned b=0;b<sizeof(bs)/sizeof(bs[0]);b++){
   int n=bs[b]; double it=bi_range(0,n); printf("BATCH n=%d INTEL_HA=%.6f\n",n,it);
   for(size_t i=0;i<nv;i++){double t=bo_range(v[i].f,0,n);printf("BATCH n=%d %-11s=%.6f intel_over=%.4fx\n",n,v[i].n,t,it/t);} }
 puts("=== MAGNITUDE CATEGORY ns/input ===");
 category("absx_lt_1",0,HALF,v,nv);
 category("absx_gt_1",HALF,HALF,v,nv);
 return 0;
}
