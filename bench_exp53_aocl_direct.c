#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*arr_fn)(int,const double*,double*);
typedef void (*our_fn)(double*,const double*,size_t);
extern void exp53_n2_fused_u4_038_frozen(double*,const double*,size_t);
static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static arr_fn cur;
static void f_aocl(double*out,const double*in,size_t n){cur((int)n,in,out);}static void f_ours(double*out,const double*in,size_t n){exp53_n2_fused_u4_038_frozen(out,in,n);}typedef void(*fn)(double*,const double*,size_t);
static double tm(fn f,double*out,const double*in,size_t n){long reps=(long)(120000000ULL/n);if(reps<700)reps=700;if(reps>4000)reps=4000;for(int k=0;k<20;k++)f(out,in,n);double best=1e99;for(int z=0;z<5;z++){double t=now();for(long k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*37)%n];}printf("AOCL_META n=%zu reps=%ld trials=5\n",n,reps);return best*1e9/(reps*n);}
static int acc(const char*n,fn f,double*out,const double*in,size_t N){f(out,in,N);uint64_t mx=0;size_t g1=0,g2=0,g4=0;for(size_t i=0;i<N;i++){uint64_t d=D(out[i],exp(in[i]));if(d>mx)mx=d;if(d>1)g1++;if(d>2)g2++;if(d>4)g4++;}printf("AOCL_ACC %-24s max=%llu gt1=%zu gt2=%zu gt4=%zu\n",n,(unsigned long long)mx,g1,g2,g4);return g1==0;}
int main(void){const char*libs[]={"libalm.so","libamdlibm.so","libaocl.so",NULL};void*h=NULL;for(int i=0;libs[i];i++){h=dlopen(libs[i],RTLD_NOW|RTLD_LOCAL);if(h){printf("AOCL_LIB opened=%s\n",libs[i]);break;}}if(!h){fprintf(stderr,"AOCL_ERR dlopen: %s\n",dlerror());return 2;}
 const char*names[]={"amd_vrda_exp","amd_vrda_exp_avx512","amd_vrda_exp_amd","amd_vrda_exp_zn3","amd_vrda_exp_avx2",NULL};
 enum{NA=200000};double*x=aligned_alloc(64,NA*sizeof(double)),*y=aligned_alloc(64,NA*sizeof(double));uint64_t s=0x123456789abcdef0ULL;for(size_t i=0;i<NA/4;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(size_t i=NA/4;i<NA/2;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(size_t i=NA/2;i<3*NA/4;i++)x[i]=1.0+uni(&s)*99.0;for(size_t i=3*NA/4;i<NA;i++)x[i]=-(1.0+uni(&s)*99.0);
 acc("OURS_FROZEN",f_ours,y,x,NA);size_t ns[]={12288,65536};for(int z=0;z<2;z++){double t=tm(f_ours,y,x,ns[z]);printf("AOCL_TIME n=%zu %-24s %.6f\n",ns[z],"OURS_FROZEN",t);} 
 for(int j=0;names[j];j++){dlerror();arr_fn f=(arr_fn)dlsym(h,names[j]);const char*e=dlerror();printf("AOCL_DLSYM %-24s %s ptr=%p\n",names[j],(!e&&f)?"FOUND":"MISSING",(void*)f);if(e||!f)continue;cur=f;acc(names[j],f_aocl,y,x,NA);for(int z=0;z<2;z++){double t=tm(f_aocl,y,x,ns[z]);printf("AOCL_TIME n=%zu %-24s %.6f\n",ns[z],names[j],t);}}
 free(x);free(y);return sink==1234567.0;}
