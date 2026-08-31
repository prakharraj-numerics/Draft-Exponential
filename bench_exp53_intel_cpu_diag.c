#define _GNU_SOURCE
#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void exp53_n2_fused_u4_038_frozen(double*,const double*,size_t);
static volatile double sink;
static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (double)t.tv_sec+1e-9*(double)t.tv_nsec;}
static uint64_t sm64(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return (sm64(s)>>11)*0x1p-53;}
static void intel_exp(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}
static void fill(double*x,size_t n){uint64_t s=0x517cc1b727220a95ULL;for(size_t i=0;i<n;i++)x[i]=-100.0+200.0*uni(&s);}
static double bench(void(*f)(double*,const double*,size_t),double*out,const double*in,size_t n,double seconds){for(int i=0;i<50;i++)f(out,in,n);uint64_t calls=0;double t0=now_sec(),t=t0;do{for(int k=0;k<64;k++){f(out,in,n);calls++;}t=now_sec();}while(t-t0<seconds);sink+=out[calls%n];return (t-t0)*1e9/((double)calls*(double)n);}
static int run_kernel(int ours,double seconds,size_t n){double*x=aligned_alloc(64,n*sizeof(double)),*y=aligned_alloc(64,n*sizeof(double));if(!x||!y)return 2;fill(x,n);mkl_set_num_threads_local(1);mkl_set_dynamic(0);double v=bench(ours?exp53_n2_fused_u4_038_frozen:intel_exp,y,x,n,seconds);printf("DIAG_TIME kernel=%s n=%zu ns_per_input=%.9f pid=%d\n",ours?"OURS":"INTEL",n,v,getpid());free(x);free(y);return 0;}
static int memhog(double seconds,size_t mb){size_t bytes=mb*1024ULL*1024ULL,n=bytes/sizeof(double);double*a=aligned_alloc(64,bytes);if(!a)return 2;for(size_t i=0;i<n;i++)a[i]=(double)(i&1023)*0.001;double t0=now_sec();uint64_t it=0;while(now_sec()-t0<seconds){for(size_t i=0;i<n;i+=8){a[i]=a[i]*1.0000000000000002+1.0;}it++;}sink+=a[it%n];printf("MEMHOG mb=%zu sweeps=%llu pid=%d\n",mb,(unsigned long long)it,getpid());free(a);return 0;}
static int spin(double seconds){double t0=now_sec(),x=1.0;while(now_sec()-t0<seconds){for(int i=0;i<100000;i++)x=fma(x,1.0000000000000002,1e-300);}sink+=x;printf("SPIN pid=%d\n",getpid());return 0;}
int main(int argc,char**argv){if(argc<2){fprintf(stderr,"usage: %s intel|ours|memhog|spin [seconds] [n_or_mb]\n",argv[0]);return 2;}double sec=argc>2?atof(argv[2]):2.0;if(!strcmp(argv[1],"intel"))return run_kernel(0,sec,argc>3?strtoull(argv[3],0,0):65536);if(!strcmp(argv[1],"ours"))return run_kernel(1,sec,argc>3?strtoull(argv[3],0,0):65536);if(!strcmp(argv[1],"memhog"))return memhog(sec,argc>3?strtoull(argv[3],0,0):256);if(!strcmp(argv[1],"spin"))return spin(sec);return 2;}
