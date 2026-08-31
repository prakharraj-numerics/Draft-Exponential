/* EXPERIMENTAL ONLY — GCC vector-function infrastructure attack for 100..3000.
   Frozen production files are untouched. The scalar point body preserves the
   frozen n=2 ER-low mathematical spine and operation order; GCC is asked to
   manufacture/use SIMD clones through OpenMP declare simd + loop vectorization. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#pragma omp declare simd notinbranch simdlen(8)
__attribute__((noinline))
static double exp53_gcc_point(double x)
{
    double biased=fma(x,N2F_INV128,N2F_MAGIC);
    double k=biased-N2F_MAGIC;
    uint64_t bb; memcpy(&bb,&biased,8);
    int64_t kn=(int64_t)(bb-N2F_MAGIC_BITS);
    uint64_t j=(uint64_t)kn & 127u;
    int64_t q=kn>>7;
    double r=fma(-k,N2F_L128_HI,x);
    r=fma(-k,N2F_L128_MI,r);
    r=fma(-k,N2F_L128_LO,r);
    double h=fma(N2F_Q4,r,N2F_Q3);
    h=fma(h,r,N2F_Q2);
    h=fma(h,r,N2F_Q1);
    h=fma(h,r,1.0);
    double s=h*h;
    double er=fma(r,s,1.0);
    double el=fma(r,s,1.0-er);
    uint64_t tb; memcpy(&tb,&N2_FROZEN_TAB128[j],8);
    uint64_t sb=tb+((uint64_t)q<<52);
    double scale; memcpy(&scale,&sb,8);
    double ph=er*scale;
    return fma(el,scale,ph);
}

__attribute__((noinline))
void exp53_gcc_vector_0100_3000(double *restrict out,const double *restrict in,size_t n)
{
    size_t i;
    #pragma omp simd simdlen(8)
    for(i=0;i<n;i++) out[i]=exp53_gcc_point(in[i]);
}
