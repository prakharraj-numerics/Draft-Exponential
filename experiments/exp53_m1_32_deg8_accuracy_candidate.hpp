#pragma once

/* EXP53 EXPERIMENT ONLY — accuracy repair for the user's

       exp(x) ~= P8(x)^32,
       P8(x) = sum_{j=0}^8 x^j/(32^j j!).

   Production and frozen v3 are intentionally unrelated/untouched.

   Variant 1, expm1_basic:
     z = x/32 (exact binary scaling)
     u = z + z^2/2! + ... + z^8/8!
     repeat 5 times: u <- fma(u,u,2u)
     result = 1 + u

   This is algebraically identical to P8(x)^32, but it never rounds the
   initial 1+u before the five squarings.

   Variant 2, expm1_comp:
     same formula, with a twofold (hi,lo) correction carried through the
     Horner evaluation and the five u <- 2u+u^2 recurrences.  It is an
     accuracy diagnostic / repair candidate; speed can be optimized later.

   No tables, gathers, exponent-bit construction, range-reduction table,
   helper thread, or old EXP53 scale machinery are used.
*/

#include <immintrin.h>
#include <cstddef>

namespace exp53_m1_32_deg8_accuracy {

static inline __m512d u8_from_z_basic(__m512d x) {
    const __m512d z = _mm512_mul_pd(x, _mm512_set1_pd(0x1.0p-5));
    const __m512d c2 = _mm512_set1_pd(0x1.0000000000000p-1);  // 1/2!
    const __m512d c3 = _mm512_set1_pd(0x1.5555555555555p-3);  // 1/3!
    const __m512d c4 = _mm512_set1_pd(0x1.5555555555555p-5);  // 1/4!
    const __m512d c5 = _mm512_set1_pd(0x1.1111111111111p-7);  // 1/5!
    const __m512d c6 = _mm512_set1_pd(0x1.6c16c16c16c17p-10); // 1/6!
    const __m512d c7 = _mm512_set1_pd(0x1.a01a01a01a01ap-13); // 1/7!
    const __m512d c8 = _mm512_set1_pd(0x1.a01a01a01a01ap-16); // 1/8!
    const __m512d one = _mm512_set1_pd(1.0);

    // u = z * (1 + z/2! + z^2/3! + ... + z^7/8!).
    __m512d h = c8;
    h = _mm512_fmadd_pd(h, z, c7);
    h = _mm512_fmadd_pd(h, z, c6);
    h = _mm512_fmadd_pd(h, z, c5);
    h = _mm512_fmadd_pd(h, z, c4);
    h = _mm512_fmadd_pd(h, z, c3);
    h = _mm512_fmadd_pd(h, z, c2);
    h = _mm512_fmadd_pd(h, z, one);
    return _mm512_mul_pd(z, h);
}

static inline __m512d reconstruct_basic(__m512d u) {
    for (int k = 0; k < 5; ++k) {
        const __m512d two_u = _mm512_add_pd(u, u); // exact doubling
        u = _mm512_fmadd_pd(u, u, two_u);          // u^2 + 2u, one rounding
    }
    return _mm512_add_pd(_mm512_set1_pd(1.0), u);
}

static inline __m512d eval_basic_vec(__m512d x) {
    return reconstruct_basic(u8_from_z_basic(x));
}

struct twofold512 { __m512d hi, lo; };

// Compensated Horner for h(z)=1+z/2!+...+z^7/8!.
// For each fused step newh=RN(h*z+c), c-newh is exact here by Sterbenz
// (the new value remains close to c on |z|<=1/32), so the second FMA captures
// the local rounding residual while lo propagates the previous residual.
static inline twofold512 u8_from_z_comp(__m512d x) {
    const __m512d z = _mm512_mul_pd(x, _mm512_set1_pd(0x1.0p-5));
    const __m512d cs[7] = {
        _mm512_set1_pd(0x1.a01a01a01a01ap-13), // 1/7!
        _mm512_set1_pd(0x1.6c16c16c16c17p-10),// 1/6!
        _mm512_set1_pd(0x1.1111111111111p-7),  // 1/5!
        _mm512_set1_pd(0x1.5555555555555p-5),  // 1/4!
        _mm512_set1_pd(0x1.5555555555555p-3),  // 1/3!
        _mm512_set1_pd(0x1.0000000000000p-1),  // 1/2!
        _mm512_set1_pd(1.0)                    // 1
    };

    __m512d h = _mm512_set1_pd(0x1.a01a01a01a01ap-16); // 1/8!
    __m512d l = _mm512_setzero_pd();
    for (int i = 0; i < 7; ++i) {
        const __m512d nh = _mm512_fmadd_pd(h, z, cs[i]);
        const __m512d d  = _mm512_sub_pd(cs[i], nh);
        const __m512d r  = _mm512_fmadd_pd(h, z, d); // exact-step residual, rounded
        l = _mm512_fmadd_pd(l, z, r);
        h = nh;
    }

    const __m512d uh = _mm512_mul_pd(z, h);
    const __m512d ur = _mm512_fmadd_pd(z, h, _mm512_sub_pd(_mm512_setzero_pd(), uh));
    const __m512d ul = _mm512_fmadd_pd(z, l, ur);
    return {uh, ul};
}

static inline twofold512 step_comp(twofold512 a) {
    const __m512d two_h = _mm512_add_pd(a.hi, a.hi);
    const __m512d nh = _mm512_fmadd_pd(a.hi, a.hi, two_h);

    // Residual of h^2+2h relative to nh.  two_h-nh is exact in this domain.
    const __m512d d = _mm512_sub_pd(two_h, nh);
    const __m512d r = _mm512_fmadd_pd(a.hi, a.hi, d);

    // f'(h)=2+2h.  l^2 is below binary64 relevance for this twofold track.
    const __m512d deriv = _mm512_add_pd(_mm512_set1_pd(2.0), two_h);
    const __m512d nl = _mm512_fmadd_pd(a.lo, deriv, r);
    return {nh, nl};
}

static inline __m512d add_one_twofold(twofold512 a) {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d s = _mm512_add_pd(one, a.hi);
    // General TwoSum residual for 1 + hi (works also when hi > 1).
    const __m512d bb = _mm512_sub_pd(s, one);
    const __m512d e1 = _mm512_sub_pd(one, _mm512_sub_pd(s, bb));
    const __m512d e2 = _mm512_sub_pd(a.hi, bb);
    const __m512d e = _mm512_add_pd(e1, e2);
    return _mm512_add_pd(s, _mm512_add_pd(e, a.lo));
}

static inline __m512d eval_comp_vec(__m512d x) {
    twofold512 u = u8_from_z_comp(x);
    u = step_comp(u);
    u = step_comp(u);
    u = step_comp(u);
    u = step_comp(u);
    u = step_comp(u);
    return add_one_twofold(u);
}

template <bool Comp>
__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void kernel(double* __restrict out, const double* __restrict in, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m512d x0 = _mm512_loadu_pd(in + i + 0);
        const __m512d x1 = _mm512_loadu_pd(in + i + 8);
        const __m512d x2 = _mm512_loadu_pd(in + i + 16);
        const __m512d x3 = _mm512_loadu_pd(in + i + 24);
        const __m512d y0 = Comp ? eval_comp_vec(x0) : eval_basic_vec(x0);
        const __m512d y1 = Comp ? eval_comp_vec(x1) : eval_basic_vec(x1);
        const __m512d y2 = Comp ? eval_comp_vec(x2) : eval_basic_vec(x2);
        const __m512d y3 = Comp ? eval_comp_vec(x3) : eval_basic_vec(x3);
        _mm512_storeu_pd(out + i + 0, y0);
        _mm512_storeu_pd(out + i + 8, y1);
        _mm512_storeu_pd(out + i + 16, y2);
        _mm512_storeu_pd(out + i + 24, y3);
    }
    for (; i + 8 <= n; i += 8) {
        const __m512d x = _mm512_loadu_pd(in + i);
        _mm512_storeu_pd(out + i, Comp ? eval_comp_vec(x) : eval_basic_vec(x));
    }
    if (i < n) {
        const unsigned rem = (unsigned)(n - i);
        const __mmask8 m = (__mmask8)((1u << rem) - 1u);
        const __m512d x = _mm512_maskz_loadu_pd(m, in + i);
        const __m512d y = Comp ? eval_comp_vec(x) : eval_basic_vec(x);
        _mm512_mask_storeu_pd(out + i, m, y);
    }
}

static inline void expm1_basic(double* out, const double* in, size_t n) {
    kernel<false>(out, in, n);
}
static inline void expm1_comp(double* out, const double* in, size_t n) {
    kernel<true>(out, in, n);
}

} // namespace exp53_m1_32_deg8_accuracy
