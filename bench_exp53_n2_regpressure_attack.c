#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void exp53_n2_fused_u4_038_frozen(double*,const double*,size_t);
void exp53_n2_rp_pair2(double*,const double*,size_t);
void exp53_n2_rp_tight4(double*,const double*,size_t);
void exp53_n2_rp_lategather4(double*,const double*,size_t);
void exp53_n2_rp_stream1(double*,const double*,size_t);
typedef void(*fn)(double*,const double*,size_t);static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}static void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}static double tm(fn f,double*out,const double*in,size_t n){int reps=n<=12288?50000:(n<=65536?12000:700);for(int k=0;k<50;k++)f(out,in,n);double best=1e99;for(int z=0;z<9;z++){double t=now();for(int k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*97)%n];}return best*1e9/(reps*n);}static void fill(double*x,size_t n,uint64_t*s){for(size_t i=0;i<n/4;i++)x[i]=0x1p-20+uni(s)*(1.0-0x1p-20);for(size_t i=n/4;i<n/2;i++)x[i]=-(0x1p-20+uni(s)*(1.0-0x1p-20));for(size_t i=n/2;i<3*n/4;i++)x[i]=1.0+uni(s)*99.0;for(size_t i=3*n/4;i<n;i++)x[i]=-(1.0+uni(s)*99.0);}static void acc(const char*n,fn f,double*out,const double*in,size_t N){f(out,in,N);uint64_t mx=0;size_t g1=0;for(size_t i=0;i<N;i++){uint64_t d=D(out[i],exp(in[i]));if(d>mx)mx=d;if(d>1)g1++;}printf("RP_ACC %-12s max=%llu gt1=%zu\n",n,(unsigned long long)mx,g1);}
int main(void){enum{NA=200000,NMAX=1048576};double*x=aligned_alloc(64,NMAX*sizeof(double)),*y=aligned_alloc(64,NMAX*sizeof(double));if(!x||!y)return 2;uint64_t s=0x57a91c33b2d104efULL;fill(x,NA,&s);mkl_set_num_threads_local(1);struct{const char*n;fn f;}a[]={{"base038",exp53_n2_fused_u4_038_frozen},{"pair2",exp53_n2_rp_pair2},{"tight4",exp53_n2_rp_tight4},{"lategather4",exp53_n2_rp_lategather4},{"stream1",exp53_n2_rp_stream1}};int na=sizeof(a)/sizeof(a[0]);for(int i=0;i<na;i++)acc(a[i].n,a[i].f,y,x,NA);size_t ns[]={12288,65536,1048576};for(int z=0;z<3;z++){size_t n=ns[z];fill(x,n,&s);double ti=tm(intel,y,x,n);printf("RP_TIME n=%zu INTEL %.6f\n",n,ti);for(int i=0;i<na;i++){double t=tm(a[i].f,y,x,n);printf("RP_TIME n=%zu %-12s %.6f intel_over=%.4fx base_candidate=%.4fx\n",n,a[i].n,t,ti/t,t);}}free(x);free(y);return sink==1234567.0;}
