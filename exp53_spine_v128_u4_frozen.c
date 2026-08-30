/* Frozen winning EXP53 kernel.
   Architecture: v128 magic-round reduction + one gather + 4x AVX-512 unroll.
   Accuracy/performance reference on Intel Xeon 6973P-C (2026-08-30 sweep):
     max ULP = 3, >2 ULP = 32/6400, >4 ULP = 0
     all-domain throughput = 0.460289 ns/input
     Intel oneMKL VML_HA = 0.328993 ns/input
   Mathematical spine is inherited unchanged from exp53_spine_v128_frozen.c.
*/
#include "exp53_spine_v128_frozen.c"

static inline __m512d exp53_v128_u4_block(
    __m512d x,
    const __m512d inv,
    const __m512d hi,
    const __m512d mi,
    const __m512d lo,
    const __m512d magic,
    const __m512i mb,
    const __m512i mask)
{
    __m512d biased = _mm512_fmadd_pd(x, inv, magic);
    __m512d k = _mm512_sub_pd(biased, magic);
    __m512i kn = _mm512_sub_epi64(_mm512_castpd_si512(biased), mb);

    __m512d r = _mm512_fnmadd_pd(k, hi, x);
    r = _mm512_fnmadd_pd(k, mi, r);
    r = _mm512_fnmadd_pd(k, lo, r);

    __m512d er = spine_residual128(r);
    __m512i j = _mm512_and_epi64(kn, mask);
    __m512i q = _mm512_srai_epi64(kn, 7);
    __m512d t = _mm512_i64gather_pd(j, TAB128, 8);

    return _mm512_mul_pd(_mm512_mul_pd(er, t), exp2_from_q(q));
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_u4_frozen(double *restrict out,
                                const double *restrict in,
                                size_t n)
{
    const __m512d inv   = _mm512_set1_pd(INV128);
    const __m512d hi    = _mm512_set1_pd(L128_HI);
    const __m512d mi    = _mm512_set1_pd(L128_MI);
    const __m512d lo    = _mm512_set1_pd(L128_LO);
    const __m512d magic = _mm512_set1_pd(MAGIC);
    const __m512i mb    = _mm512_set1_epi64((long long)MAGIC_BITS);
    const __m512i mask  = _mm512_set1_epi64(127);

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512d x0 = _mm512_loadu_pd(in + i);
        __m512d x1 = _mm512_loadu_pd(in + i + 8);
        __m512d x2 = _mm512_loadu_pd(in + i + 16);
        __m512d x3 = _mm512_loadu_pd(in + i + 24);

        __m512d y0 = exp53_v128_u4_block(x0, inv, hi, mi, lo, magic, mb, mask);
        __m512d y1 = exp53_v128_u4_block(x1, inv, hi, mi, lo, magic, mb, mask);
        __m512d y2 = exp53_v128_u4_block(x2, inv, hi, mi, lo, magic, mb, mask);
        __m512d y3 = exp53_v128_u4_block(x3, inv, hi, mi, lo, magic, mb, mask);

        _mm512_storeu_pd(out + i,      y0);
        _mm512_storeu_pd(out + i + 8,  y1);
        _mm512_storeu_pd(out + i + 16, y2);
        _mm512_storeu_pd(out + i + 24, y3);
    }

    for (; i < n;) {
        if (n - i >= 8) {
            __m512d x = _mm512_loadu_pd(in + i);
            _mm512_storeu_pd(out + i,
                exp53_v128_u4_block(x, inv, hi, mi, lo, magic, mb, mask));
            i += 8;
        } else {
            for (; i < n; ++i)
                out[i] = exp(in[i]);
        }
    }
}
