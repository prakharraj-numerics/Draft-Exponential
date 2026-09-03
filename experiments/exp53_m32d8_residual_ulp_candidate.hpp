#pragma once

/* EXP53 EXPERIMENT ONLY — production and frozen candidates untouched.

   Formula remains exactly the user's (m,d)=(32,8):
       exp(x) ~= P8(x)^32
       P8(x) = sum_{j=0}^8 x^j/(32^j j!)

   Accuracy repair is confined to the binary64 realization:
   1) form q=P8-1 without prematurely rounding 1+q;
   2) reconstruct (1+q)^32 in residual form;
   3) propagate an estimate of the discarded rounding residuals;
   4) if the estimated missed amount exceeds one output ULP, move the
      positive final result by exactly one representable double toward it.

   No tables, no interval split, no higher precision, no extra Taylor term.
*/

#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace exp53_m32d8_residual_ulp {

static inline __m512d q32d8(__m512d x) {
    const __m512d a8 = _mm512_set1_pd(0x1.a01a01a01a01ap-56);
    const __m512d a7 = _mm512_set1_pd(0x1.a01a01a01a01ap-48);
    const __m512d a6 = _mm512_set1_pd(0x1.6c16c16c16c17p-40);
    const __m512d a5 = _mm512_set1_pd(0x1.1111111111111p-32);
    const __m512d a4 = _mm512_set1_pd(0x1.5555555555555p-25);
    const __m512d a3 = _mm512_set1_pd(0x1.5555555555555p-18);
    const __m512d a2 = _mm512_set1_pd(0x1.0000000000000p-11);
    const __m512d inv32 = _mm512_set1_pd(0x1.0000000000000p-5);

    __m512d t = a8;
    t = _mm512_fmadd_pd(t,x,a7);
    t = _mm512_fmadd_pd(t,x,a6);
    t = _mm512_fmadd_pd(t,x,a5);
    t = _mm512_fmadd_pd(t,x,a4);
    t = _mm512_fmadd_pd(t,x,a3);
    t = _mm512_fmadd_pd(t,x,a2);

    // P8-1 = x/32 + x^2*(a2 + a3*x + ... + a8*x^6).
    // x/32 is exact binary scaling for the tested finite normal domain.
    const __m512d x2 = _mm512_mul_pd(x,x);
    const __m512d x_over_32 = _mm512_mul_pd(x,inv32);
    return _mm512_fmadd_pd(x2,t,x_over_32);
}

static inline __m512d eval(__m512d x) {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512i one_i = _mm512_set1_epi64(1);

    __m512d r = q32d8(x);
    __m512d err = _mm512_setzero_pd();

    // Four residual squarings: 1+r -> (1+r)^2 while retaining an estimate
    // of the exact quantity discarded by binary64 rounding at each step.
    for (int k=0;k<4;k++) {
        const __m512d two_r = _mm512_add_pd(r,r);
        const __m512d rn = _mm512_fmadd_pd(r,r,two_r);

        // Since two_r and rn are close, two_r-rn is exact by Sterbenz in
        // this domain. FMA then recovers the local missed residual closely.
        const __m512d local = _mm512_fmadd_pd(r,r,_mm512_sub_pd(two_r,rn));

        const __m512d one_plus_r = _mm512_add_pd(one,r);
        const __m512d coef = _mm512_add_pd(one_plus_r,one_plus_r);
        const __m512d err_sq_plus_local = _mm512_fmadd_pd(err,err,local);
        err = _mm512_fmadd_pd(coef,err,err_sq_plus_local);
        r = rn;
    }

    // Final square directly: (1+r)^2 = 1 + 2r + r^2.
    const __m512d t = _mm512_fmadd_pd(two,r,one);
    const __m512d d1 = _mm512_fmadd_pd(two,r,_mm512_sub_pd(one,t));
    __m512d y = _mm512_fmadd_pd(r,r,t);
    const __m512d d2 = _mm512_fmadd_pd(r,r,_mm512_sub_pd(t,y));

    const __m512d one_plus_r = _mm512_add_pd(one,r);
    const __m512d coef = _mm512_add_pd(one_plus_r,one_plus_r);
    const __m512d local_final = _mm512_add_pd(d1,d2);
    const __m512d err_sq_plus_local = _mm512_fmadd_pd(err,err,local_final);
    const __m512d total_err = _mm512_fmadd_pd(coef,err,err_sq_plus_local);

    // y is positive normal on the signed-unit benchmark domain.  For positive
    // doubles, increment/decrement of the encoding is exactly nextafter by 1.
    const __m512i yi = _mm512_castpd_si512(y);
    const __m512d yup = _mm512_castsi512_pd(_mm512_add_epi64(yi,one_i));
    const __m512d ydn = _mm512_castsi512_pd(_mm512_sub_epi64(yi,one_i));
    const __m512d ulp_up = _mm512_sub_pd(yup,y);
    const __m512d ulp_dn = _mm512_sub_pd(y,ydn);

    const __mmask8 move_up = _mm512_cmp_pd_mask(total_err,ulp_up,_CMP_GT_OQ);
    const __mmask8 move_dn = _mm512_cmp_pd_mask(total_err,_mm512_sub_pd(_mm512_setzero_pd(),ulp_dn),_CMP_LT_OQ);

    __m512i yo = yi;
    yo = _mm512_mask_add_epi64(yo,move_up,yo,one_i);
    yo = _mm512_mask_sub_epi64(yo,move_dn,yo,one_i);
    return _mm512_castsi512_pd(yo);
}

__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void kernel(double* __restrict out, const double* __restrict in, size_t n) {
    size_t i=0;
    for (; i+32<=n; i+=32) {
        __m512d x0=_mm512_loadu_pd(in+i+0),  x1=_mm512_loadu_pd(in+i+8);
        __m512d x2=_mm512_loadu_pd(in+i+16), x3=_mm512_loadu_pd(in+i+24);
        _mm512_storeu_pd(out+i+0,eval(x0));
        _mm512_storeu_pd(out+i+8,eval(x1));
        _mm512_storeu_pd(out+i+16,eval(x2));
        _mm512_storeu_pd(out+i+24,eval(x3));
    }
    for (; i+8<=n; i+=8) {
        __m512d x=_mm512_loadu_pd(in+i);
        _mm512_storeu_pd(out+i,eval(x));
    }
    if (i<n) {
        const unsigned rem=(unsigned)(n-i);
        const __mmask8 m=(__mmask8)((1u<<rem)-1u);
        const __m512d x=_mm512_maskz_loadu_pd(m,in+i);
        _mm512_mask_storeu_pd(out+i,m,eval(x));
    }
}

} // namespace exp53_m32d8_residual_ulp
