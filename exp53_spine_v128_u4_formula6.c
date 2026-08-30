/* Formula-level optimization of the frozen winning EXP53 architecture.
   Outer architecture is unchanged: v128 magic reduction + one gather + 4x AVX-512 unroll.
   User spine is preserved:
      C(r/2) = cosh(r/2)-1
      S = sqrt(C(C+2)) = sinh(r/2)
      exp(r) = (1 + C + S)^2
   On |r| <= ln2/256, use t=r^2 and interval-specialized degree-1 minimax forms:
      C ~= t*(A0 + A1*t)
      S ~= r*(B0 + B1*t)
   Then fuse h = 1 + S + C and return h^2.
   Residual core cost: 4 FMA + 2 MUL.
*/
#include "exp53_spine_v128_frozen.c"

#define F6_A0 0x1.ffffffffffff5p-4
#define F6_A1 0x1.555556b330444p-9
#define F6_B0 0x1.fffffffffffe0p-2
#define F6_B1 0x1.555557621dbbbp-6

static inline __m512d spine_residual128_formula6(__m512d r)
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d A0 = _mm512_set1_pd(F6_A0);
    const __m512d A1 = _mm512_set1_pd(F6_A1);
    const __m512d B0 = _mm512_set1_pd(F6_B0);
    const __m512d B1 = _mm512_set1_pd(F6_B1);

    __m512d t = _mm512_mul_pd(r, r);
    __m512d Pc = _mm512_fmadd_pd(A1, t, A0);
    __m512d Ps = _mm512_fmadd_pd(B1, t, B0);
    __m512d h = _mm512_fmadd_pd(r, Ps, one);
    h = _mm512_fmadd_pd(t, Pc, h);
    return _mm512_mul_pd(h, h);
}

static inline __m512d exp53_v128_u4_formula6_block(
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

    __m512d er = spine_residual128_formula6(r);
    __m512i j = _mm512_and_epi64(kn, mask);
    __m512i q = _mm512_srai_epi64(kn, 7);
    __m512d tab = _mm512_i64gather_pd(j, TAB128, 8);
    return _mm512_mul_pd(_mm512_mul_pd(er, tab), exp2_from_q(q));
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_u4_formula6(double *restrict out,
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
        __m512d x0 = _mm512_loadu_pd(in+i);
        __m512d x1 = _mm512_loadu_pd(in+i+8);
        __m512d x2 = _mm512_loadu_pd(in+i+16);
        __m512d x3 = _mm512_loadu_pd(in+i+24);

        __m512d y0 = exp53_v128_u4_formula6_block(x0,inv,hi,mi,lo,magic,mb,mask);
        __m512d y1 = exp53_v128_u4_formula6_block(x1,inv,hi,mi,lo,magic,mb,mask);
        __m512d y2 = exp53_v128_u4_formula6_block(x2,inv,hi,mi,lo,magic,mb,mask);
        __m512d y3 = exp53_v128_u4_formula6_block(x3,inv,hi,mi,lo,magic,mb,mask);

        _mm512_storeu_pd(out+i,y0);
        _mm512_storeu_pd(out+i+8,y1);
        _mm512_storeu_pd(out+i+16,y2);
        _mm512_storeu_pd(out+i+24,y3);
    }

    for (; i < n;) {
        if (n-i >= 8) {
            _mm512_storeu_pd(out+i,
                exp53_v128_u4_formula6_block(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));
            i += 8;
        } else {
            for (; i<n; ++i) out[i] = exp(in[i]);
        }
    }
}
