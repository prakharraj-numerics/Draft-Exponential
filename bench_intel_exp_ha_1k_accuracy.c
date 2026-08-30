#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <mkl_vml.h>

enum { N = 1000 };
static double x[N], y[N];

static inline uint64_t U(double a){uint64_t u; memcpy(&u,&a,8); return u;}
static inline uint64_t O(double a){uint64_t u=U(a); return (u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b); return A>B?A-B:B-A;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return (sm(s)>>11)*0x1p-53;}

int main(void){
    setenv("MKL_NUM_THREADS","1",1); setenv("OMP_NUM_THREADS","1",1); mkl_set_num_threads_local(1);
    uint64_t s=0x243f6a8885a308d3ULL;
    for(int i=0;i<N/4;i++) x[i]=0x1p-20 + uni(&s)*(1.0-0x1p-20);
    for(int i=N/4;i<N/2;i++) x[i]=-(0x1p-20 + uni(&s)*(1.0-0x1p-20));
    for(int i=N/2;i<3*N/4;i++) x[i]=1.0 + uni(&s)*99.0;
    for(int i=3*N/4;i<N;i++) x[i]=-(1.0 + uni(&s)*99.0);

    vmdExp(N,x,y,VML_HA);
    uint64_t mx=0; int gt0=0,gt1=0,gt2=0; int wi=-1;
    for(int i=0;i<N;i++){
        double r=exp(x[i]);
        uint64_t d=D(y[i],r);
        if(d>mx){mx=d;wi=i;}
        if(d>0)gt0++; if(d>1)gt1++; if(d>2)gt2++;
    }
    printf("INTEL_HA_1K maxulp=%llu gt0=%d gt1=%d gt2=%d\n",(unsigned long long)mx,gt0,gt1,gt2);
    if(wi>=0) printf("WORST i=%d x=%a got=%a ref=%a ulp=%llu\n",wi,x[wi],y[wi],exp(x[wi]),(unsigned long long)D(y[wi],exp(x[wi])));
    return 0;
}
