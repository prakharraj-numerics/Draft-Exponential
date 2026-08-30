#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void exp53_n2_fused_u4_038_frozen(double*,const double*,size_t);
typedef void(*fn)(double*,const double*,size_t);
static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);} 
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);} 
static double tm(fn f,double*out,const double*in,size_t n){int reps=n<=12288?50000:(n<=65536?12000:900);for(int k=0;k<50;k++)f(out,in,n);double best=1e99;for(int z=0;z<11;z++){double t=now();for(int k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*131)%n];}return best*1e9/(reps*n);} 
static void fill(double*x,size_t n,int neg,uint64_t*s){for(size_t i=0;i<n;i++){double a=0x1p-20+uni(s)*(100.0-0x1p-20);x[i]=neg?-a:a;}}
int main(void){size_t ns[]={12288,65536,1048576};size_t maxn=ns[2];double *xp=aligned_alloc(64,maxn*sizeof(double)),*xn=aligned_alloc(64,maxn*sizeof(double)),*y=aligned_alloc(64,maxn*sizeof(double));uint64_t s=0x123456789abcdef0ULL;fill(xp,maxn,0,&s);fill(xn,maxn,1,&s);mkl_set_num_threads_local(1);for(int z=0;z<3;z++){size_t n=ns[z];double op=tm(exp53_n2_fused_u4_038_frozen,y,xp,n),on=tm(exp53_n2_fused_u4_038_frozen,y,xn,n);double ip=tm(intel,y,xp,n),in=tm(intel,y,xn,n);printf("SIGN_TIME n=%zu OUR_POS %.6f OUR_NEG %.6f NEG_OVER_POS %.6fx DIFF_PCT %.3f\n",n,op,on,on/op,100.0*(on/op-1.0));printf("SIGN_INTEL n=%zu POS %.6f NEG %.6f NEG_OVER_POS %.6fx DIFF_PCT %.3f\n",n,ip,in,in/ip,100.0*(in/ip-1.0));}free(xp);free(xn);free(y);return sink==1234567.0;}
