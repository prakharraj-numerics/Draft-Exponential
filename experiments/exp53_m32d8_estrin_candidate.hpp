#pragma once

/* EXP53 EXPERIMENT ONLY — pure Estrin polynomial evaluation.
   Same mathematical formula as raw m32d8:
       exp(x) ~= P8(x)^32
       P8(x)=sum_{j=0}^8 x^j/(32^j j!)
   Difference from raw candidate: P8 is evaluated with a balanced Estrin tree,
   then exactly five binary64 squarings. No Horner, no compensation, no repair.
*/

#include <immintrin.h>
#include <cstddef>

namespace exp53_m32d8_estrin {

static inline __m512d p8_estrin(__m512d x) {
    const __m512d a0 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d a1 = _mm512_set1_pd(0x1.0000000000000p-5);
    const __m512d a2 = _mm512_set1_pd(0x1.0000000000000p-11);
    const __m512d a3 = _mm512_set1_pd(0x1.5555555555555p-18);
    const __m512d a4 = _mm512_set1_pd(0x1.5555555555555p-25);
    const __m512d a5 = _mm512_set1_pd(0x1.1111111111111p-32);
    const __m512d a6 = _mm512_set1_pd(0x1.6c16c16c16c17p-40);
    const __m512d a7 = _mm512_set1_pd(0x1.a01a01a01a01ap-48);
    const __m512d a8 = _mm512_set1_pd(0x1.a01a01a01a01ap-56);

    const __m512d x2 = _mm512_mul_pd(x,x);
    const __m512d x4 = _mm512_mul_pd(x2,x2);
    const __m512d x8 = _mm512_mul_pd(x4,x4);

    const __m512d b0 = _mm512_fmadd_pd(a1,x,a0);
    const __m512d b1 = _mm512_fmadd_pd(a3,x,a2);
    const __m512d b2 = _mm512_fmadd_pd(a5,x,a4);
    const __m512d b3 = _mm512_fmadd_pd(a7,x,a6);

    const __m512d c0 = _mm512_fmadd_pd(b1,x2,b0);
    const __m512d c1 = _mm512_fmadd_pd(b3,x2,b2);
    const __m512d d0 = _mm512_fmadd_pd(c1,x4,c0);
    return _mm512_fmadd_pd(a8,x8,d0);
}

static inline __m512d eval(__m512d x) {
    __m512d y = p8_estrin(x);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    return y;
}

__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void kernel(double* __restrict out, const double* __restrict in, size_t n) {
    size_t i=0;
    for (; i+32<=n; i+=32) {
        __m512d x0=_mm512_loadu_pd(in+i+0),  x1=_mm512_loadu_pd(in+i+8);
        __m512d x2=_mm512_loadu_pd(in+i+16), x3=_mm512_loadu_pd(in+i+24);
        __m512d y0=eval(x0), y1=eval(x1), y2=eval(x2), y3=eval(x3);
        _mm512_storeu_pd(out+i+0,y0); _mm512_storeu_pd(out+i+8,y1);
        _mm512_storeu_pd(out+i+16,y2); _mm512_storeu_pd(out+i+24,y3);
    }
    for (; i+8<=n; i+=8) {
        __m512d x=_mm512_loadu_pd(in+i);
        _mm512_storeu_pd(out+i,eval(x));
    }
    if (i<n) {
        unsigned rem=(unsigned)(n-i);
        __mmask8 m=(__mmask8)((1u<<rem)-1u);
        __m512d x=_mm512_maskz_loadu_pd(m,in+i);
        _mm512_mask_storeu_pd(out+i,m,eval(x));
    }
}

} // namespace exp53_m32d8_estrin
