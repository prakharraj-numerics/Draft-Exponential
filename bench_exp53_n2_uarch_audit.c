#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void exp53_n2_fused_u4_038_frozen(double*,const double*,size_t);
typedef void(*fn)(double*,const double*,size_t);
static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void intel(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}
static void fill(double*x,size_t n){uint64_t s=0x91e10da5c79e7b1dULL;for(size_t i=0;i<n/4;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(size_t i=n/4;i<n/2;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(size_t i=n/2;i<3*n/4;i++)x[i]=1.0+uni(&s)*99.0;for(size_t i=3*n/4;i<n;i++)x[i]=-(1.0+uni(&s)*99.0);}
static double tm(fn f,double*out,const double*in,size_t n,int reps){for(int k=0;k<100;k++)f(out,in,n);double best=1e99;for(int z=0;z<9;z++){double t=now();for(int k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*103)%n];}return best*1e9/(reps*n);}
int main(int argc,char**argv){size_t n=65536;int reps=20000;if(argc>2)n=strtoull(argv[2],0,10);if(argc>3)reps=atoi(argv[3]);double*x=aligned_alloc(64,n*sizeof(double)),*y=aligned_alloc(64,n*sizeof(double));if(!x||!y)return 2;fill(x,n);mkl_set_num_threads_local(1);fn f=exp53_n2_fused_u4_038_frozen;const char*name="BASE038";if(argc>1&&strcmp(argv[1],"intel")==0){f=intel;name="INTEL";}if(argc>1&&strcmp(argv[1],"burn")==0){for(int k=0;k<reps;k++){exp53_n2_fused_u4_038_frozen(y,x,n);sink+=y[k%n];}printf("BURN %g\n",sink);free(x);free(y);return 0;}if(argc>1&&strcmp(argv[1],"iburn")==0){for(int k=0;k<reps;k++){intel(y,x,n);sink+=y[k%n];}printf("IBURN %g\n",sink);free(x);free(y);return 0;}double t=tm(f,y,x,n,reps);printf("UARCH_TIME %s n=%zu ns=%.6f\n",name,n,t);free(x);free(y);return sink==1234567.0;}
