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
void exp53_spine_v128_u4_frozen(double*,const double*,size_t);
void exp53_spine_v128_u4_formula6(double*,const double*,size_t);
void exp53_spine_v128_u4_formula6_tabcomp(double*,const double*,size_t);

enum{N=6400,HALF=3200,REPS=20000};
static double x[N],y[N],z[N];
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static inline double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*t.tv_nsec;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static void inputs(void){uint64_t s=0x6a09e667f3bcc909ULL;for(int i=0;i<1600;i++)x[i]=0x1p-20+uni(&s)*(1-0x1p-20);for(int i=1600;i<3200;i++)x[i]=-(0x1p-20+uni(&s)*(1-0x1p-20));for(int i=3200;i<4800;i++)x[i]=1+uni(&s)*99;for(int i=4800;i<6400;i++)x[i]=-(1+uni(&s)*99);}
static void acc(const char*n,fn_t f){f(y,x,N);uint64_t mx=0;int g1=0,g2=0,g4=0;long double mr=0;for(int i=0;i<N;i++){double r=exp(x[i]);uint64_t d=D(y[i],r);if(d>mx)mx=d;if(d>1)g1++;if(d>2)g2++;if(d>4)g4++;long double e=fabsl(((long double)y[i]-r)/r);if(e>mr)mr=e;}printf("ACC %-12s maxulp=%llu gt1=%d gt2=%d gt4=%d maxrel=%.3Le\n",n,(unsigned long long)mx,g1,g2,g4,mr);}
static void diagnose_tabcomp(void){
 const double INV128=0x1.71547652b82fep+7,HI=0x1.62e42fefa39efp-8,MI=0x1.abc9e3b39803fp-63,LO=0x1.7b57a079a1934p-118;
 exp53_spine_v128_u4_formula6_tabcomp(y,x,N);
 for(int i=0;i<N;i++){
   double ref=exp(x[i]); uint64_t d=D(y[i],ref); if(d<=4)continue;
   double kd=nearbyint(x[i]*INV128); long long k=(long long)kd;
   double r=fma(-kd,HI,x[i]); r=fma(-kd,MI,r); r=fma(-kd,LO,r);
   long long j=k&127LL, q=k>>7; long long dir=(y[i]>ref)?1:-1;
   printf("TABCOMP_BAD i=%d x=%a got=%a ref=%a ulp=%llu dir=%+lld k=%lld j=%lld q=%lld r=%a rdec=%.17g\n",i,x[i],y[i],ref,(unsigned long long)d,dir,k,j,q,r,r);
 }
}
static double bo(fn_t f,int lo,int n){for(int i=0;i<200;i++)f(y+lo,x+lo,n);double a=sec();for(int i=0;i<REPS;i++)f(y+lo,x+lo,n);double b=sec();return(b-a)*1e9/((double)REPS*n);}
static double bi(int lo,int n){for(int i=0;i<200;i++)vmdExp(n,x+lo,z+lo,VML_HA);double a=sec();for(int i=0;i<REPS;i++)vmdExp(n,x+lo,z+lo,VML_HA);double b=sec();return(b-a)*1e9/((double)REPS*n);}
static void rep(const char*n,fn_t f,double ia){double s=bo(f,0,HALF),w=bo(f,HALF,HALF),a=bo(f,0,N);printf("RES %-12s small=%.6f wide=%.6f all=%.6f intelHA_over=%.4fx\n",n,s,w,a,ia/a);}
int main(void){setenv("MKL_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);mkl_set_num_threads_local(1);inputs();acc("u4_frozen",exp53_spine_v128_u4_frozen);acc("formula6",exp53_spine_v128_u4_formula6);acc("f6_tabcomp",exp53_spine_v128_u4_formula6_tabcomp);diagnose_tabcomp();double ha=bi(0,N);printf("INTEL_HA all=%.6f\n",ha);rep("u4_frozen",exp53_spine_v128_u4_frozen,ha);rep("formula6",exp53_spine_v128_u4_formula6,ha);rep("f6_tabcomp",exp53_spine_v128_u4_formula6_tabcomp,ha);return 0;}
