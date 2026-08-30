#define _GNU_SOURCE
#include <mkl.h>
#include <mkl_vml.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static inline uint64_t splitmix64(uint64_t *s){
    uint64_t z=(*s+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
static inline double u01(uint64_t *s){
    return (splitmix64(s)>>11)*(1.0/9007199254740992.0);
}
static inline uint64_t ns_now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts);
    return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
}
static void fill_inputs(double *small,double *wide){
    uint64_t s=UINT64_C(0x6a09e667f3bcc909);
    for(int i=0;i<1600;i++){
        double a=0.000001 + u01(&s)*(0.999999-0.000001);
        small[2*i]=a; small[2*i+1]=-a;
        double b=1.000001 + u01(&s)*(99.999999-1.000001);
        wide[2*i]=b; wide[2*i+1]=-b;
    }
}
static double bench_band(const double *x,double *y,int n,int reps){
    for(int i=0;i<200;i++) vmdExp(n,x,y,VML_HA);
    uint64_t best=UINT64_MAX;
    volatile double sink=0.0;
    for(int t=0;t<9;t++){
        uint64_t a=ns_now();
        for(int r=0;r<reps;r++) vmdExp(n,x,y,VML_HA);
        uint64_t b=ns_now();
        if(b-a<best) best=b-a;
        sink += y[(t*353)%n];
    }
    if(!isfinite(sink)) fprintf(stderr,"sink=%g\n",(double)sink);
    return (double)best/((double)n*(double)reps);
}
int main(void){
    enum { N=3200, REPS=20000 };
    double *small=(double*)mkl_malloc(sizeof(double)*N,64);
    double *wide=(double*)mkl_malloc(sizeof(double)*N,64);
    double *out=(double*)mkl_malloc(sizeof(double)*N,64);
    if(!small||!wide||!out) return 2;
    mkl_set_num_threads_local(1);
    fill_inputs(small,wide);
    double ns_small=bench_band(small,out,N,REPS);
    double ns_wide=bench_band(wide,out,N,REPS);
    double ns_all=(ns_small+ns_wide)/2.0;
    printf("INTEL_XEON_EXP_ONLY total_inputs=6400 small=3200 wide=3200 pos_small=1600 neg_small=1600 pos_wide=1600 neg_wide=1600 mode=VML_HA threads=1 reps=%d\n",REPS);
    printf("INTEL_XEON_EXP_SMALL abs_x_lt_1 avg_ns_per_input=%.6f avg_us_per_input=%.9f\n",ns_small,ns_small/1000.0);
    printf("INTEL_XEON_EXP_WIDE abs_x_1_to_100 avg_ns_per_input=%.6f avg_us_per_input=%.9f\n",ns_wide,ns_wide/1000.0);
    printf("INTEL_XEON_EXP_ALL avg_ns_per_input=%.6f avg_us_per_input=%.9f\n",ns_all,ns_all/1000.0);
    mkl_free(out); mkl_free(wide); mkl_free(small);
    return 0;
}
