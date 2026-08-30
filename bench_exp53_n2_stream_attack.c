#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void exp53_stream_mid_u3(double*,const double*,size_t);void exp53_stream_mid_u4(double*,const double*,size_t);void exp53_stream_gearly_slate_u4(double*,const double*,size_t);void exp53_stream_searly_gmid_u4(double*,const double*,size_t);void exp53_stream_gh2_u4(double*,const double*,size_t);void exp53_stream_gh3_u4(double*,const double*,size_t);
typedef void(*fn)(double*,const double*,size_t); static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);} static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static uint64_t O(double a){uint64_t u;memcpy(&u,&a,8);return(u>>63)?~u:(u|0x8000000000000000ULL);} static uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;} static void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);} static double tm(fn f,double*out,const double*in,size_t n){int reps=n>=65536?8000:40000;for(int k=0;k<40;k++)f(out,in,n);double best=1e9;for(int z=0;z<9;z++){double t=now();for(int k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*997)%n];}return best*1e9/(reps*n);} 
int main(void){enum{N=65536};double*x=aligned_alloc(64,N*sizeof(double)),*y=aligned_alloc(64,N*sizeof(double));uint64_t s=0x123456789abcdefULL;for(int i=0;i<N/4;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(int i=N/4;i<N/2;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(int i=N/2;i<3*N/4;i++)x[i]=1+99*uni(&s);for(int i=3*N/4;i<N;i++)x[i]=-(1+99*uni(&s));mkl_set_num_threads_local(1);struct{const char*n;fn f;}a[]={{"mid_u3",exp53_stream_mid_u3},{"mid_u4",exp53_stream_mid_u4},{"gE_sL",exp53_stream_gearly_slate_u4},{"sE_gM",exp53_stream_searly_gmid_u4},{"gH2",exp53_stream_gh2_u4},{"gH3",exp53_stream_gh3_u4}};int na=sizeof(a)/sizeof(a[0]);for(int k=0;k<na;k++){a[k].f(y,x,12288);uint64_t mx=0;int g1=0;for(int i=0;i<12288;i++){uint64_t d=D(y[i],exp(x[i]));if(d>mx)mx=d;if(d>1)g1++;}printf("STREAM_ACC %-7s max=%llu gt1=%d\n",a[k].n,(unsigned long long)mx,g1);}size_t bs[]={12288,65536};for(int b=0;b<2;b++){size_t n=bs[b];double ti=tm(intel,y,x,n);printf("STREAM n=%zu INTEL %.6f\n",n,ti);for(int k=0;k<na;k++){double t=tm(a[k].f,y,x,n);printf("STREAM n=%zu %-7s %.6f intel_over=%.4fx\n",n,a[k].n,t,ti/t);}}free(x);free(y);return sink==1234567.0;}
