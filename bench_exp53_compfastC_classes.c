#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>
#include <mkl_vml.h>

void exp53_spine_v128_formula6_compfastC_frozen(double*,const double*,size_t);

enum { N=6400, HALF=3200, REPS=30000 };
static double xs[HALF], xw[HALF], y[HALF], z[HALF];

static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static inline double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec+1e-9*t.tv_nsec;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}

static void inputs(void){
    uint64_t s=0x6a09e667f3bcc909ULL;
    for(int i=0;i<HALF/2;i++) xs[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);
    for(int i=HALF/2;i<HALF;i++) xs[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));
    for(int i=0;i<HALF/2;i++) xw[i]=1.0+uni(&s)*99.0;
    for(int i=HALF/2;i<HALF;i++) xw[i]=-(1.0+uni(&s)*99.0);
}

static void acc_class(const char*name,const double*x){
    exp53_spine_v128_formula6_compfastC_frozen(y,x,HALF);
    uint64_t mx=0; int g1=0,g2=0;
    for(int i=0;i<HALF;i++){
        double r=exp(x[i]); uint64_t d=D(y[i],r);
        if(d>mx)mx=d; if(d>1)g1++; if(d>2)g2++;
    }
    printf("ACC_CLASS %s maxulp=%llu gt1=%d gt2=%d\n",name,(unsigned long long)mx,g1,g2);
}

static double bench_ours(const double*x){
    for(int i=0;i<300;i++) exp53_spine_v128_formula6_compfastC_frozen(y,x,HALF);
    double a=sec();
    for(int i=0;i<REPS;i++) exp53_spine_v128_formula6_compfastC_frozen(y,x,HALF);
    double b=sec();
    return (b-a)*1e9/((double)REPS*HALF);
}

static double bench_intel(const double*x){
    for(int i=0;i<300;i++) vmdExp(HALF,x,z,VML_HA);
    double a=sec();
    for(int i=0;i<REPS;i++) vmdExp(HALF,x,z,VML_HA);
    double b=sec();
    return (b-a)*1e9/((double)REPS*HALF);
}

int main(void){
    setenv("MKL_NUM_THREADS","1",1); setenv("OMP_NUM_THREADS","1",1); mkl_set_num_threads_local(1);
    inputs();
    acc_class("absx_lt_1",xs);
    acc_class("absx_gt_1",xw);
    double os=bench_ours(xs), ow=bench_ours(xw);
    double is=bench_intel(xs), iw=bench_intel(xw);
    printf("CLASS absx_lt_1 ours_ns=%.6f intelHA_ns=%.6f intelHA_over=%.4fx\n",os,is,is/os);
    printf("CLASS absx_gt_1 ours_ns=%.6f intelHA_ns=%.6f intelHA_over=%.4fx\n",ow,iw,iw/ow);
    return 0;
}
