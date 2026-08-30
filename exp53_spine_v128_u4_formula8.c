/* Formula8: full original truncated spine polynomials, fused in t=r^2 form.
   Outer architecture unchanged: v128 magic reduction + one gather + 4x AVX-512 unroll.
   User spine preserved:
      C(r/2)=cosh(r/2)-1
      S=sqrt(C(C+2))=sinh(r/2)
      exp(r)=(1+C+S)^2
   With t=r^2:
      C ~= t*(1/8 + t*(1/384 + t/46080))
      S ~= r*(1/2 + t*(1/48 + t/3840))
   Residual core: 6 FMA + 2 MUL.
*/
#include "exp53_spine_v128_u4_frozen.c"

#define F8_A0 0x1.0000000000000p-3
#define F8_A1 0x1.5555555555555p-9
#define F8_A2 0x1.6c16c16c16c17p-16
#define F8_B0 0x1.0000000000000p-1
#define F8_B1 0x1.5555555555555p-6
#define F8_B2 0x1.1111111111111p-12

static inline __m512d spine_residual128_formula8(__m512d r)
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d A0 = _mm512_set1_pd(F8_A0);
    const __m512d A1 = _mm512_set1_pd(F8_A1);
    const __m512d A2 = _mm512_set1_pd(F8_A2);
    const __m512d B0 = _mm512_set1_pd(F8_B0);
    const __m512d B1 = _mm512_set1_pd(F8_B1);
    const __m512d B2 = _mm512_set1_pd(F8_B2);

    __m512d t = _mm512_mul_pd(r, r);
    __m512d Pc = _mm512_fmadd_pd(A2, t, A1);
    Pc = _mm512_fmadd_pd(Pc, t, A0);
    __m512d Ps = _mm512_fmadd_pd(B2, t, B1);
    Ps = _mm512_fmadd_pd(Ps, t, B0);
    __m512d h = _mm512_fmadd_pd(r, Ps, one);
    h = _mm512_fmadd_pd(t, Pc, h);
    return _mm512_mul_pd(h, h);
}

static inline __m512d exp53_v128_u4_formula8_block(__m512d x,const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask)
{
    __m512d biased = _mm512_fmadd_pd(x, inv, magic);
    __m512d k = _mm512_sub_pd(biased, magic);
    __m512i kn = _mm512_sub_epi64(_mm512_castpd_si512(biased), mb);
    __m512d r = _mm512_fnmadd_pd(k, hi, x);
    r = _mm512_fnmadd_pd(k, mi, r);
    r = _mm512_fnmadd_pd(k, lo, r);
    __m512d er = spine_residual128_formula8(r);
    __m512i j = _mm512_and_epi64(kn, mask);
    __m512i q = _mm512_srai_epi64(kn, 7);
    __m512d tab = _mm512_i64gather_pd(j, TAB128, 8);
    return _mm512_mul_pd(_mm512_mul_pd(er, tab), exp2_from_q(q));
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_u4_formula8(double *restrict out,const double *restrict in,size_t n)
{
    const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC);
    const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24);
        __m512d y0=exp53_v128_u4_formula8_block(x0,inv,hi,mi,lo,magic,mb,mask),y1=exp53_v128_u4_formula8_block(x1,inv,hi,mi,lo,magic,mb,mask),y2=exp53_v128_u4_formula8_block(x2,inv,hi,mi,lo,magic,mb,mask),y3=exp53_v128_u4_formula8_block(x3,inv,hi,mi,lo,magic,mb,mask);
        _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);_mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
    }
    for(;i<n;){if(n-i>=8){_mm512_storeu_pd(out+i,exp53_v128_u4_formula8_block(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));i+=8;}else{for(;i<n;++i)out[i]=exp(in[i]);}}
}
