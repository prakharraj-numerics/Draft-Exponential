#pragma once
#include <immintrin.h>
#include <cstddef>

namespace exp53_m32d8_eft_depth {

static inline __m512d base_hi_lo(__m512d x, __m512d &lo) {
    __m512d p=_mm512_set1_pd(0x1.a01a01a01a01ap-56);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-48));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-40));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-32));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-25));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-18));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-11));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-5));
    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d t=_mm512_mul_pd(p,x);
    const __m512d em=_mm512_fmsub_pd(p,x,t);
    const __m512d y0=_mm512_add_pd(t,one);
    const __m512d z=_mm512_sub_pd(y0,one);
    const __m512d ea=_mm512_sub_pd(t,z);
    lo=_mm512_add_pd(em,ea);
    return y0;
}

template<int DEPTH>
static inline __m512d eval(__m512d x) {
    __m512d lo;
    __m512d y0=base_hi_lo(x,lo);
    // Base low-part correction: 1/y0 ~= 2-y0 is sufficient because lo is tiny.
    __m512d rel=_mm512_mul_pd(lo,_mm512_mul_pd(_mm512_sub_pd(_mm512_set1_pd(2.0),y0),_mm512_set1_pd(32.0)));
    __m512d y=y0;
    const double w[5]={16.0,8.0,4.0,2.0,1.0};
    for(int s=0;s<5;s++) {
        __m512d yn=_mm512_mul_pd(y,y);
        if constexpr (DEPTH>0) {
            if(s<DEPTH) {
                __m512d er=_mm512_fmsub_pd(y,y,yn);
                __m512d inv=_mm512_rcp14_pd(yn);
                rel=_mm512_fmadd_pd(er,_mm512_mul_pd(inv,_mm512_set1_pd(w[s])),rel);
            }
        }
        y=yn;
    }
    return _mm512_fmadd_pd(rel,y,y);
}

template<int DEPTH>
__attribute__((target("avx512f,avx512dq,fma"),noinline,hot))
static void kernel(double* __restrict out,const double* __restrict in,size_t n){
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24);
        _mm512_storeu_pd(out+i,eval<DEPTH>(x0));_mm512_storeu_pd(out+i+8,eval<DEPTH>(x1));
        _mm512_storeu_pd(out+i+16,eval<DEPTH>(x2));_mm512_storeu_pd(out+i+24,eval<DEPTH>(x3));
    }
    for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i);_mm512_storeu_pd(out+i,eval<DEPTH>(x));}
    if(i<n){unsigned r=(unsigned)(n-i);__mmask8 m=(__mmask8)((1u<<r)-1u);__m512d x=_mm512_maskz_loadu_pd(m,in+i);_mm512_mask_storeu_pd(out+i,m,eval<DEPTH>(x));}
}

static inline void d0(double*o,const double*x,size_t n){kernel<0>(o,x,n);} // base EFT only
static inline void d1(double*o,const double*x,size_t n){kernel<1>(o,x,n);} // + first squaring residual
static inline void d2(double*o,const double*x,size_t n){kernel<2>(o,x,n);} // + first two
static inline void d3(double*o,const double*x,size_t n){kernel<3>(o,x,n);} // + first three
static inline void d4(double*o,const double*x,size_t n){kernel<4>(o,x,n);} // + first four
static inline void d5(double*o,const double*x,size_t n){kernel<5>(o,x,n);} // + all five

} // namespace exp53_m32d8_eft_depth
