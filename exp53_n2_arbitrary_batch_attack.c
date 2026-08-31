/* Arbitrary-size faithful n=2 EXP attack.

   Observation from the first rolling-pipeline benchmark: the long-array u4
   frozen body is already better scheduled than a one-vector conveyor, while
   irregular sizes suffer because its inherited remainder path eventually
   uses scalar exp().

   This candidate therefore keeps the frozen 32-input hot body unchanged and
   replaces only the final 0..31 values with faithful AVX-512 vectors plus a
   masked 1..7 tail. No mathematical approximation changes.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

static __attribute__((target("avx512f,avx512dq,fma"),always_inline)) inline
__m512d n2arb_vec(__m512d x)
{
    const __m512d inv=_mm512_set1_pd(N2F_INV128),
                  hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI),
                  lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC),
                  one=_mm512_set1_pd(1.0),
                  nq1=_mm512_set1_pd(N2F_Q1),
                  nq2=_mm512_set1_pd(N2F_Q2),
                  nq3=_mm512_set1_pd(N2F_Q3),
                  nq4=_mm512_set1_pd(N2F_Q4);
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
    __m512d h=_mm512_fmadd_pd(nq4,r,nq3);
    h=_mm512_fmadd_pd(h,r,nq2);
    h=_mm512_fmadd_pd(h,r,nq1);
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
void exp53_n2_u4_masktail(double *restrict out,const double *restrict in,size_t n)
{
    size_t bulk=n&~(size_t)31;
    if(bulk) exp53_n2_vmstyle_u4_0381_frozen(out,in,bulk);
    size_t i=bulk;
    for(;i+8<=n;i+=8){
        __m512d x=_mm512_loadu_pd(in+i);
        _mm512_storeu_pd(out+i,n2arb_vec(x));
    }
    if(i<n){
        unsigned rem=(unsigned)(n-i);
        __mmask8 km=(__mmask8)((1u<<rem)-1u);
        __m512d x=_mm512_maskz_loadu_pd(km,in+i);
        __m512d y=n2arb_vec(x);
        _mm512_mask_storeu_pd(out+i,km,y);
    }
}

/* Dynamic production-shaped experiment: rolling PIPE3 only for small batches
   where its lower setup/tail cost may win; otherwise use the proven u4 body
   plus faithful masked tail. Thresholds are benchmark hypotheses, not frozen. */
void exp53_n2_pipe3_stream(double*,const double*,size_t);
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_dynamic_batch(double *restrict out,const double *restrict in,size_t n)
{
    if(n < 512) exp53_n2_pipe3_stream(out,in,n);
    else exp53_n2_u4_masktail(out,in,n);
}
