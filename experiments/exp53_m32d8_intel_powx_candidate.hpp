#pragma once

/* EXP53 EXPERIMENT ONLY.

   Keep the user's m=32,d=8 base polynomial unchanged:
       P8(x) = sum_{j=0}^8 x^j/(32^j j!)

   Compare two outer-power realizations:
       raw:       five AVX-512 squarings
       intelpowx: Intel oneMKL VML_HA vmdPowx(P8, 32.0)

   Intel VML supports in-place VM operations, so P8 is first written to out
   and vmdPowx then transforms out -> out. Production/frozen code untouched.
*/

#include <mkl.h>
#include <immintrin.h>
#include <cstddef>

namespace exp53_m32d8_intel_powx {

static inline __m512d p32d8(__m512d x) {
    __m512d p = _mm512_set1_pd(0x1.a01a01a01a01ap-56);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-48));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-40));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-32));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-25));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-18));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-11));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-5));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p+0));
    return p;
}

__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void base_p8(double* __restrict out, const double* __restrict in, size_t n) {
    size_t i=0;
    for (; i+32<=n; i+=32) {
        __m512d x0=_mm512_loadu_pd(in+i+0),  x1=_mm512_loadu_pd(in+i+8);
        __m512d x2=_mm512_loadu_pd(in+i+16), x3=_mm512_loadu_pd(in+i+24);
        _mm512_storeu_pd(out+i+0,p32d8(x0));
        _mm512_storeu_pd(out+i+8,p32d8(x1));
        _mm512_storeu_pd(out+i+16,p32d8(x2));
        _mm512_storeu_pd(out+i+24,p32d8(x3));
    }
    for (; i+8<=n; i+=8) {
        __m512d x=_mm512_loadu_pd(in+i);
        _mm512_storeu_pd(out+i,p32d8(x));
    }
    if (i<n) {
        unsigned rem=(unsigned)(n-i);
        __mmask8 m=(__mmask8)((1u<<rem)-1u);
        __m512d x=_mm512_maskz_loadu_pd(m,in+i);
        _mm512_mask_storeu_pd(out+i,m,p32d8(x));
    }
}

static inline void intel_powx(double* out, const double* in, size_t n) {
    base_p8(out,in,n);
    vmdPowx((MKL_INT)n, out, 32.0, out, VML_HA);
}

} // namespace exp53_m32d8_intel_powx
