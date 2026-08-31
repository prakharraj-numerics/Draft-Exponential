#define _GNU_SOURCE
#include <immintrin.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef __m512d (*vec8_fn)(__m512d);
typedef void (*arr_fn)(int,const double*,double*);
static vec8_fn sleef8, glibc8;
static arr_fn aocl_arr;
static volatile double sink;
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static void wrap_vec(vec8_fn f,double*out,const double*in,size_t n){size_t i=0;for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i);__m512d y=f(x);_mm512_storeu_pd(out+i,y);}for(;i<n;i++)out[i]=exp(in[i]);}
static void f_sleef(double*out,const double*in,size_t n){wrap_vec(sleef8,out,in,n);}static void f_glibc(double*out,const double*in,size_t n){wrap_vec(glibc8,out,in,n);}static void f_aocl(double*out,const double*in,size_t n){aocl_arr((int)n,in,out);}typedef void(*fn)(double*,const double*,size_t);
static double tm(fn f,double*out,const double*in,size_t n){long reps=(long)(150000000ULL/n);if(reps<800)reps=800;if(reps>5000)reps=5000;for(int k=0;k<20;k++)f(out,in,n);double best=1e99;for(int z=0;z<5;z++){double t=now();for(long k=0;k<reps;k++)f(out,in,n);double e=now()-t;if(e<best)best=e;sink+=out[(z*37)%n];}printf("EXT_META n=%zu reps=%ld trials=5\n",n,reps);return best*1e9/(reps*n);}static void acc(const char*n,fn f,double*out,const double*in,size_t N){f(out,in,N);uint64_t mx=0;size_t g1=0,g2=0,g4=0;for(size_t i=0;i<N;i++){uint64_t d=D(out[i],exp(in[i]));if(d>mx)mx=d;if(d>1)g1++;if(d>2)g2++;if(d>4)g4++;}printf("EXT_ACC %-8s max=%llu gt1=%zu gt2=%zu gt4=%zu\n",n,(unsigned long long)mx,g1,g2,g4);}
static void* open_any(const char**xs){for(int i=0;xs[i];i++){void*h=dlopen(xs[i],RTLD_NOW|RTLD_LOCAL);if(h){printf("EXT_LIB opened=%s\n",xs[i]);return h;}}return NULL;}
int main(void){const char*slibs[]={"libsleef.so.3","libsleef.so",NULL};const char*alibs[]={"libalm.so","libamdlibm.so","libaocl.so",NULL};void*hs=open_any(slibs),*ha=open_any(alibs),*hm=dlopen("libmvec.so.1",RTLD_NOW|RTLD_LOCAL);if(!hs||!ha||!hm){fprintf(stderr,"EXT_ERR dlopen sleef=%p aocl=%p mvec=%p err=%s\n",hs,ha,hm,dlerror());return 2;}sleef8=(vec8_fn)dlsym(hs,"Sleef_expd8_u10");glibc8=(vec8_fn)dlsym(hm,"_ZGVeN8v_exp");aocl_arr=(arr_fn)dlsym(ha,"amd_vrda_exp");printf("EXT_SYM sleef=%p aocl=%p glibc=%p\n",(void*)sleef8,(void*)aocl_arr,(void*)glibc8);if(!sleef8||!aocl_arr||!glibc8)return 3;enum{NA=200000,NB=65536};double*x=aligned_alloc(64,NA*sizeof(double)),*y=aligned_alloc(64,NA*sizeof(double));uint64_t s=0x123456789abcdef0ULL;for(size_t i=0;i<NA/4;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(size_t i=NA/4;i<NA/2;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(size_t i=NA/2;i<3*NA/4;i++)x[i]=1.0+uni(&s)*99.0;for(size_t i=3*NA/4;i<NA;i++)x[i]=-(1.0+uni(&s)*99.0);struct{const char*n;fn f;}a[]={{"SLEEF",f_sleef},{"AOCL",f_aocl},{"LIBMVEC",f_glibc}};for(int i=0;i<3;i++)acc(a[i].n,a[i].f,y,x,NA);size_t ns[]={12288,65536};for(int z=0;z<2;z++){size_t n=ns[z];for(int i=0;i<3;i++){double t=tm(a[i].f,y,x,n);printf("EXT_TIME n=%zu %-8s %.6f\n",n,a[i].n,t);}}free(x);free(y);return sink==1234567.0;}
