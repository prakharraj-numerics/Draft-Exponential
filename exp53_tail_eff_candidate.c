/* Isolated compute-efficiency candidate. production/ is untouched.
   Only change under test: replace scalar exp() remainder work with faithful
   AVX-512 vector/masked-tail work. The 32-value hot body is the frozen VM-style
   body under private symbol names. */
#include <immintrin.h>
#include <stddef.h>

#define exp53_n2_fused_u4_038_frozen exp53_n2_fused_u4_038_tailbase
#define exp53_n2_vmstyle_u4_0381_frozen exp53_n2_vmstyle_u4_0381_tailbase
#include "production/exp53_n2_vmstyle_u4_0381_frozen.c"
#undef exp53_n2_vmstyle_u4_0381_frozen
#undef exp53_n2_fused_u4_038_frozen

static __attribute__((target("avx512f,avx512dq,fma"),always_inline)) inline
__m512d exp53_tail_eff_vec(__m512d x)
{
    const __m512d inv=_mm512_set1_pd(N2F_INV128),
                  hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI),
                  lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC),
                  one=_mm512_set1_pd(1.0),
                  q1=_mm512_set1_pd(N2F_Q1),
                  q2=_mm512_set1_pd(N2F_Q2),
                  q3=_mm512_set1_pd(N2F_Q3),
                  q4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),
                  mask127=_mm512_set1_epi64(127);

    __m512d biased=_mm512_fmadd_pd(x,inv,magic);
    __m512d k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512i j=_mm512_and_epi64(kn,mask127);
    __m512i q=_mm512_srai_epi64(kn,7);
    __m512i tb=_mm512_i64gather_epi64(j,(const long long*)N2_FROZEN_TAB128,8);

    __m512d r=_mm512_fnmadd_pd(k,hi,x);
    r=_mm512_fnmadd_pd(k,mi,r);
    r=_mm512_fnmadd_pd(k,lo,r);

    __m512d h=_mm512_fmadd_pd(q4,r,q3);
    h=_mm512_fmadd_pd(h,r,q2);
    h=_mm512_fmadd_pd(h,r,q1);
    h=_mm512_fmadd_pd(h,r,one);
    __m512d s=_mm512_mul_pd(h,h);
    __m512d er=_mm512_fmadd_pd(r,s,one);
    __m512d el=_mm512_fmadd_pd(r,s,_mm512_sub_pd(one,er));

    __m512i sb=_mm512_add_epi64(tb,_mm512_slli_epi64(q,52));
    __m512d scale=_mm512_castsi512_pd(sb);
    __m512d ph=_mm512_mul_pd(er,scale);
    return _mm512_fmadd_pd(el,scale,ph);
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_vmstyle_u4_0381_masktail_candidate(double *restrict out,
                                                  const double *restrict in,
                                                  size_t n)
{
    const size_t bulk=n & ~(size_t)31;
    if (bulk) exp53_n2_vmstyle_u4_0381_tailbase(out,in,bulk);
    size_t i=bulk;
    for (; i+8<=n; i+=8) {
        __m512d x=_mm512_loadu_pd(in+i);
        _mm512_storeu_pd(out+i,exp53_tail_eff_vec(x));
    }
    if (i<n) {
        const unsigned rem=(unsigned)(n-i);
        const __mmask8 km=(__mmask8)((1u<<rem)-1u);
        __m512d x=_mm512_maskz_loadu_pd(km,in+i);
        __m512d y=exp53_tail_eff_vec(x);
        _mm512_mask_storeu_pd(out+i,km,y);
    }
}
