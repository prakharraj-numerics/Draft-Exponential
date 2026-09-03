#pragma once

/* EXP53 EXPERIMENT ONLY — user's m = 1/32 degree-8 series.

   Mathematical formula:

       P8(x) = sum_{j=0}^8 x^j / (32^j j!)
       exp(x) ~= P8(x)^32

   Final power is exactly five squarings.

   This file deliberately contains NO range-reduction table, NO TAB128 gather,
   NO exponent-bit reconstruction, NO helper thread, and NO old EXP53 scaling
   machinery.  It is a clean unit-domain test of the new mathematical spine.

   Two algebraically identical binary64 AVX-512 realizations are provided:
     1) Horner  : minimum arithmetic count / long dependency chain.
     2) Estrin  : shallower dependency graph / slightly more multiplies.

   Coefficients below are the correctly-rounded binary64 values of
   a_j = 1/(32^j j!).
*/

#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace exp53_m1_32_deg8 {

static inline __m512d pow32_5sq(__m512d y) {
    y = _mm512_mul_pd(y, y); // ^2
    y = _mm512_mul_pd(y, y); // ^4
    y = _mm512_mul_pd(y, y); // ^8
    y = _mm512_mul_pd(y, y); // ^16
    y = _mm512_mul_pd(y, y); // ^32
    return y;
}

static inline __m512d p8_horner(__m512d x) {
    const __m512d a0 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d a1 = _mm512_set1_pd(0x1.0000000000000p-5);
    const __m512d a2 = _mm512_set1_pd(0x1.0000000000000p-11);
    const __m512d a3 = _mm512_set1_pd(0x1.5555555555555p-18);
    const __m512d a4 = _mm512_set1_pd(0x1.5555555555555p-25);
    const __m512d a5 = _mm512_set1_pd(0x1.1111111111111p-32);
    const __m512d a6 = _mm512_set1_pd(0x1.6c16c16c16c17p-40);
    const __m512d a7 = _mm512_set1_pd(0x1.a01a01a01a01ap-48);
    const __m512d a8 = _mm512_set1_pd(0x1.a01a01a01a01ap-56);

    __m512d p = a8;
    p = _mm512_fmadd_pd(p, x, a7);
    p = _mm512_fmadd_pd(p, x, a6);
    p = _mm512_fmadd_pd(p, x, a5);
    p = _mm512_fmadd_pd(p, x, a4);
    p = _mm512_fmadd_pd(p, x, a3);
    p = _mm512_fmadd_pd(p, x, a2);
    p = _mm512_fmadd_pd(p, x, a1);
    p = _mm512_fmadd_pd(p, x, a0);
    return p;
}

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

    const __m512d u  = _mm512_mul_pd(x, x);
    const __m512d u2 = _mm512_mul_pd(u, u);
    const __m512d u4 = _mm512_mul_pd(u2, u2);

    // Pair adjacent coefficients first; four FMAs are mutually independent.
    const __m512d t0 = _mm512_fmadd_pd(a1, x, a0);
    const __m512d t1 = _mm512_fmadd_pd(a3, x, a2);
    const __m512d t2 = _mm512_fmadd_pd(a5, x, a4);
    const __m512d t3 = _mm512_fmadd_pd(a7, x, a6);

    const __m512d q0 = _mm512_fmadd_pd(t1, u, t0);
    const __m512d q1 = _mm512_fmadd_pd(t3, u, t2);
    __m512d p = _mm512_fmadd_pd(q1, u2, q0);
    p = _mm512_fmadd_pd(a8, u4, p);
    return p;
}

static inline __m512d eval_horner_vec(__m512d x) {
    return pow32_5sq(p8_horner(x));
}

static inline __m512d eval_estrin_vec(__m512d x) {
    return pow32_5sq(p8_estrin(x));
}

template <bool Estrin>
__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void kernel(double* __restrict out, const double* __restrict in, size_t n) {
    size_t i = 0;
    // Four independent vectors per body let the out-of-order core overlap the
    // five-squaring dependency chains across vectors.
    for (; i + 32 <= n; i += 32) {
        __m512d x0 = _mm512_loadu_pd(in + i + 0);
        __m512d x1 = _mm512_loadu_pd(in + i + 8);
        __m512d x2 = _mm512_loadu_pd(in + i + 16);
        __m512d x3 = _mm512_loadu_pd(in + i + 24);
        __m512d y0 = Estrin ? eval_estrin_vec(x0) : eval_horner_vec(x0);
        __m512d y1 = Estrin ? eval_estrin_vec(x1) : eval_horner_vec(x1);
        __m512d y2 = Estrin ? eval_estrin_vec(x2) : eval_horner_vec(x2);
        __m512d y3 = Estrin ? eval_estrin_vec(x3) : eval_horner_vec(x3);
        _mm512_storeu_pd(out + i + 0, y0);
        _mm512_storeu_pd(out + i + 8, y1);
        _mm512_storeu_pd(out + i + 16, y2);
        _mm512_storeu_pd(out + i + 24, y3);
    }
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(in + i);
        __m512d y = Estrin ? eval_estrin_vec(x) : eval_horner_vec(x);
        _mm512_storeu_pd(out + i, y);
    }
    if (i < n) {
        const unsigned rem = (unsigned)(n - i);
        const __mmask8 m = (__mmask8)((1u << rem) - 1u);
        __m512d x = _mm512_maskz_loadu_pd(m, in + i);
        __m512d y = Estrin ? eval_estrin_vec(x) : eval_horner_vec(x);
        _mm512_mask_storeu_pd(out + i, m, y);
    }
}

static inline void horner(double* out, const double* in, size_t n) {
    kernel<false>(out, in, n);
}

static inline void estrin(double* out, const double* in, size_t n) {
    kernel<true>(out, in, n);
}

} // namespace exp53_m1_32_deg8
