#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void exp53_n2_elonly_u4(double*,const double*,size_t);void exp53_n2_bv_u1(double*,const double*,size_t);void exp53_n2_bv_u2(double*,const double*,size_t);void exp53_n2_bv_u3(double*,const double*,size_t);void exp53_n2_bv_u4(double*,const double*,size_t);void exp53_n2_bv_u5(double*,const double*,size_t);void exp53_n2_bv_u6(double*,const double*,size_t);void exp53_n2_bv_mid_u4(double*,const double*,size_t);void exp53_n2_bv_late_u4(double*,const double*,size_t);void exp53_n2_bv_adaptive(double*,const double*,size_t);
typedef void(*fn)(double*,const double*,size_t);static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}static void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}static double tm(fn f,double*out,const double*in,size_t n){int reps=(n<=32)?200000:(n<=128?100000:(n<=1024?60000:40000));for(int k=0;k<30;k++)f(out,in,n);double best=1e9;for(int z=0;z<7;z++){double t=now();for(int k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*7)%n];}return best*1e9/(reps*n);}static void acc(const char*n,fn f,double*out,const double*in,size_t N){f(out,in,N);uint64_t mx=0;int g1=0;for(size_t i=0;i<N;i++){uint64_t d=D(out[i],exp(in[i]));if(d>mx)mx=d;if(d>1)g1++;}printf("BV_ACC %-10s max=%llu gt1=%d\n",n,(unsigned long long)mx,g1);}
int main(void){enum{N=12288};double*x=aligned_alloc(64,N*sizeof(double)),*y=aligned_alloc(64,N*sizeof(double));uint64_t s=0x9e3779b97f4a7c15ULL;for(int i=0;i<N/4;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(int i=N/4;i<N/2;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(int i=N/2;i<3*N/4;i++)x[i]=1.0+uni(&s)*99.0;for(int i=3*N/4;i<N;i++)x[i]=-(1.0+uni(&s)*99.0);mkl_set_num_threads_local(1);
struct{const char*n;fn f;}a[]={{"old_el_u4",exp53_n2_elonly_u4},{"u1",exp53_n2_bv_u1},{"u2",exp53_n2_bv_u2},{"u3",exp53_n2_bv_u3},{"u4",exp53_n2_bv_u4},{"u5",exp53_n2_bv_u5},{"u6",exp53_n2_bv_u6},{"mid_u4",exp53_n2_bv_mid_u4},{"late_u4",exp53_n2_bv_late_u4},{"adaptive",exp53_n2_bv_adaptive}};int na=sizeof(a)/sizeof(a[0]);for(int i=0;i<na;i++)acc(a[i].n,a[i].f,y,x,N);
size_t bs[]={8,16,24,32,40,64,128,256,1024,12288};int nb=sizeof(bs)/sizeof(bs[0]);for(int b=0;b<nb;b++){size_t n=bs[b];double ti=tm(intel,y,x,n);printf("BV_BATCH n=%zu INTEL %.6f\n",n,ti);for(int i=0;i<na;i++){double t=tm(a[i].f,y,x,n);printf("BV_BATCH n=%zu %-10s %.6f intel_over=%.4fx\n",n,a[i].n,t,ti/t);}}
free(x);free(y);return sink==1234567.0;}
