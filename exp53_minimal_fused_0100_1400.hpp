#pragma once

/* Minimal process-map composition for EXP53 small/mid batches.

   NO scheduler, thread pool, partitioning layer, or library representation
   boundary. The process map contributes only operations that compose inside
   one AVX-512 register pipeline.

   Correctness note: the frozen VM-style baseline processes only complete
   32-value blocks with AVX-512 and delegates the remainder to the older frozen
   kernel, whose <32 path is scalar exp(). We preserve that exact tail policy;
   vectorizing the remainder was the sole source of prior bitdiffs.
*/

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "exp53_highway_sync_1600_3000_constants_frozen.hpp"

extern "C" void exp53_n2_fused_u4_038_frozen(double *out,
                                               const double *in,
                                               size_t n);

namespace exp53_minimal_fused_0100_1400 {
namespace c = exp53_hwy_const_frozen;

static inline void vec8(double* out, const double* in) {
    const __m512d inv   = _mm512_set1_pd(c::INV128);
    const __m512d hi    = _mm512_set1_pd(c::L128_HI);
    const __m512d mi    = _mm512_set1_pd(c::L128_MI);
    const __m512d lo    = _mm512_set1_pd(c::L128_LO);
    const __m512d magic = _mm512_set1_pd(c::MAGIC);
    const __m512d one   = _mm512_set1_pd(1.0);
    const __m512d q1    = _mm512_set1_pd(c::Q1);
    const __m512d q2    = _mm512_set1_pd(c::Q2);
    const __m512d q3    = _mm512_set1_pd(c::Q3);
    const __m512d q4    = _mm512_set1_pd(c::Q4);
    const __m512i mb    = _mm512_set1_epi64((long long)c::MAGIC_BITS);
    const __m512i mask  = _mm512_set1_epi64(127);
    const auto* tab = reinterpret_cast<const long long*>(c::TAB128);

    const __m512d x = _mm512_loadu_pd(in);
    const __m512d biased = _mm512_fmadd_pd(x, inv, magic);
    const __m512d k = _mm512_sub_pd(biased, magic);

    const __m512i kn = _mm512_sub_epi64(_mm512_castpd_si512(biased), mb);
    const __m512i j  = _mm512_and_epi64(kn, mask);
    const __m512i q  = _mm512_srai_epi64(kn, 7);
    const __m512i tb = _mm512_i64gather_epi64(j, tab, 8);

    __m512d r = _mm512_fnmadd_pd(k, hi, x);
    r = _mm512_fnmadd_pd(k, mi, r);
    r = _mm512_fnmadd_pd(k, lo, r);

    __m512d h = _mm512_fmadd_pd(q4, r, q3);
    h = _mm512_fmadd_pd(h, r, q2);
    h = _mm512_fmadd_pd(h, r, q1);
    h = _mm512_fmadd_pd(h, r, one);

    const __m512d s = _mm512_mul_pd(h, h);
    const __m512d er = _mm512_fmadd_pd(r, s, one);
    const __m512d el = _mm512_fmadd_pd(r, s, _mm512_sub_pd(one, er));

    const __m512i sb = _mm512_add_epi64(tb, _mm512_slli_epi64(q, 52));
    const __m512d scale = _mm512_castsi512_pd(sb);
    const __m512d ph = _mm512_mul_pd(er, scale);
    const __m512d y = _mm512_fmadd_pd(el, scale, ph);
    _mm512_storeu_pd(out, y);
}

static inline void run(double* out, const double* in, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        vec8(out + i,      in + i);
        vec8(out + i + 8,  in + i + 8);
        vec8(out + i + 16, in + i + 16);
        vec8(out + i + 24, in + i + 24);
    }
    if (i < n) {
        exp53_n2_fused_u4_038_frozen(out + i, in + i, n - i);
    }
}

} // namespace exp53_minimal_fused_0100_1400
