#pragma once
#include <immintrin.h>
#include <cstddef>

/* EXP53 EXPERIMENT ONLY. Same raw (32,8) Horner polynomial and same five
   squarings. Only the last Horner step is replaced by an error-free-ish
   mul+Fast2Sum decomposition, followed by one first-order end correction.
   Production/frozen code untouched. */

namespace exp53_m32d8_eft_base {

static inline __m512d base_hi_lo(__m512d x, __m512d &lo) {
    __m512d p = _mm512_set1_pd(0x1.a01a01a01a01ap-56);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-48));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-40));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-32));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-25));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-18));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-11));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-5));

    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d t = _mm512_mul_pd(p,x);
    const __m512d emul = _mm512_fmsub_pd(p,x,t);
    const __m512d y0 = _mm512_add_pd(t,one);
    const __m512d z = _mm512_sub_pd(y0,one);
    const __m512d eadd = _mm512_sub_pd(t,z);
    lo = _mm512_add_pd(emul,eadd);
    return y0;
}

static inline __m512d pow32(__m512d y) {
    y=_mm512_mul_pd(y,y);
    y=_mm512_mul_pd(y,y);
    y=_mm512_mul_pd(y,y);
    y=_mm512_mul_pd(y,y);
    y=_mm512_mul_pd(y,y);
    return y;
}

static inline __m512d eval_rcp14(__m512d x) {
    __m512d lo;
    const __m512d y0=base_hi_lo(x,lo);
    __m512d y=pow32(y0);
    const __m512d rcp=_mm512_rcp14_pd(y0);
    const __m512d k=_mm512_mul_pd(rcp,_mm512_set1_pd(32.0));
    const __m512d rel=_mm512_mul_pd(lo,k);
    return _mm512_fmadd_pd(rel,y,y);
}

static inline __m512d eval_linrecip(__m512d x) {
    __m512d lo;
    const __m512d y0=base_hi_lo(x,lo);
    __m512d y=pow32(y0);
    const __m512d inv=_mm512_sub_pd(_mm512_set1_pd(2.0),y0); // 1/y0 ~= 2-y0
    const __m512d k=_mm512_mul_pd(inv,_mm512_set1_pd(32.0));
    const __m512d rel=_mm512_mul_pd(lo,k);
    return _mm512_fmadd_pd(rel,y,y);
}

template<int V>
__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void kernel(double* __restrict out,const double* __restrict in,size_t n) {
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d x0=_mm512_loadu_pd(in+i+0),x1=_mm512_loadu_pd(in+i+8);
        __m512d x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24);
        __m512d y0=V==0?eval_rcp14(x0):eval_linrecip(x0);
        __m512d y1=V==0?eval_rcp14(x1):eval_linrecip(x1);
        __m512d y2=V==0?eval_rcp14(x2):eval_linrecip(x2);
        __m512d y3=V==0?eval_rcp14(x3):eval_linrecip(x3);
        _mm512_storeu_pd(out+i+0,y0);_mm512_storeu_pd(out+i+8,y1);
        _mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
    }
    for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i);_mm512_storeu_pd(out+i,V==0?eval_rcp14(x):eval_linrecip(x));}
    if(i<n){unsigned rem=(unsigned)(n-i);__mmask8 m=(__mmask8)((1u<<rem)-1u);__m512d x=_mm512_maskz_loadu_pd(m,in+i);_mm512_mask_storeu_pd(out+i,m,V==0?eval_rcp14(x):eval_linrecip(x));}
}

static inline void rcp14(double*o,const double*x,size_t n){kernel<0>(o,x,n);} 
static inline void linrecip(double*o,const double*x,size_t n){kernel<1>(o,x,n);} 

} // namespace
