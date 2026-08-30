#define _GNU_SOURCE
#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>
#include <mkl_vml.h>

typedef void(*fn_t)(double*,const double*,size_t);
void exp53_final_mid_u6(double*,const double*,size_t);
void exp53_spinepoly_mid_u4(double*,const double*,size_t);
void exp53_ip_n2d3_u4(double*,const double*,size_t);
void exp53_ip_n2d3_u6(double*,const double*,size_t);
void exp53_ip_n2d4_u4(double*,const double*,size_t);
void exp53_ip_n2d4_u5(double*,const double*,size_t);
void exp53_ip_n2d4_u6(double*,const double*,size_t);
void exp53_ip_n2d5_u4(double*,const double*,size_t);
void exp53_ip_n3d4_u4(double*,const double*,size_t);
void exp53_ip_n4d4_u4(double*,const double*,size_t);

enum{N=12288,HALF=6144,REPS=30000};
static double x[N],y[N],z[N],ref[N];
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static inline double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*t.tv_nsec;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static void inputs(void){uint64_t s=0x9e3779b97f4a7c15ULL;for(int i=0;i<HALF/2;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(int i=HALF/2;i<HALF;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(int i=HALF;i<HALF+HALF/2;i++)x[i]=1.0+uni(&s)*99.0;for(int i=HALF+HALF/2;i<N;i++)x[i]=-(1.0+uni(&s)*99.0);}
static int cr_exp(double a,double*out){mpfr_prec_t p=192;for(int q=0;q<4;q++,p*=2){mpfr_t m,l,h;mpfr_init2(m,p);mpfr_init2(l,p);mpfr_init2(h,p);mpfr_set_d(m,a,MPFR_RNDN);mpfr_exp(l,m,MPFR_RNDD);mpfr_exp(h,m,MPFR_RNDU);double dl=mpfr_get_d(l,MPFR_RNDN),dh=mpfr_get_d(h,MPFR_RNDN);mpfr_clears(m,l,h,(mpfr_ptr)0);if(U(dl)==U(dh)){*out=dl;return 1;}}return 0;}
static void refs(void){int bad=0;for(int i=0;i<N;i++)if(!cr_exp(x[i],&ref[i]))bad++;printf("MPFR_REF uncert=%d\n",bad);}
static void acc(const char*n,fn_t f){f(y,x,N);uint64_t mx=0;int g0=0,g1=0,g2=0,g3=0;long long pos=0,neg=0;for(int i=0;i<N;i++){uint64_t d=D(y[i],ref[i]);if(d>mx)mx=d;if(d>0)g0++;if(d>1){g1++;if(O(y[i])>O(ref[i]))pos++;else neg++;}if(d>2)g2++;if(d>3)g3++;}printf("ACC %-12s maxulp=%llu gt0=%d gt1=%d gt2=%d gt3=%d badpos=%lld badneg=%lld\n",n,(unsigned long long)mx,g0,g1,g2,g3,pos,neg);}
static double bo(fn_t f,int lo,int n){for(int i=0;i<800;i++)f(y+lo,x+lo,n);double a=sec();for(int i=0;i<REPS;i++)f(y+lo,x+lo,n);double b=sec();return(b-a)*1e9/((double)REPS*n);}
static double bi(int lo,int n){for(int i=0;i<800;i++)vmdExp(n,x+lo,z+lo,VML_HA);double a=sec();for(int i=0;i<REPS;i++)vmdExp(n,x+lo,z+lo,VML_HA);double b=sec();return(b-a)*1e9/((double)REPS*n);}
static void rep(const char*n,fn_t f){printf("RES %-12s small=%.6f wide=%.6f all=%.6f\n",n,bo(f,0,HALF),bo(f,HALF,HALF),bo(f,0,N));}
int main(void){setenv("MKL_NUM_THREADS","1",1);setenv("OMP_NUM_THREADS","1",1);mkl_set_num_threads_local(1);inputs();refs();struct{const char*n;fn_t f;}v[]={
 {"old_safe",exp53_final_mid_u6},{"spinepoly",exp53_spinepoly_mid_u4},
 {"n2d3_u4",exp53_ip_n2d3_u4},{"n2d3_u6",exp53_ip_n2d3_u6},{"n2d4_u4",exp53_ip_n2d4_u4},{"n2d4_u5",exp53_ip_n2d4_u5},{"n2d4_u6",exp53_ip_n2d4_u6},{"n2d5_u4",exp53_ip_n2d5_u4},{"n3d4_u4",exp53_ip_n3d4_u4},{"n4d4_u4",exp53_ip_n4d4_u4}};
 for(unsigned i=0;i<sizeof(v)/sizeof(v[0]);i++)acc(v[i].n,v[i].f);
 printf("INTEL small=%.6f wide=%.6f all=%.6f\n",bi(0,HALF),bi(HALF,HALF),bi(0,N));
 for(unsigned i=0;i<sizeof(v)/sizeof(v[0]);i++)rep(v[i].n,v[i].f);return 0;}
