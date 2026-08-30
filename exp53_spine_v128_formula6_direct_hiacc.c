/* Accuracy-first Formula6 direct reconstruction.
   Same v128 reduction/table and same direct reconstruction, but evaluate C and S
   through r^8/r^7 terms so polynomial truncation is negligible on |r|<=ln2/256.
   Speed is intentionally secondary in this candidate.
*/
#include "exp53_spine_v128_formula6_direct.c"

static inline __m512d spine_residual128_formula6_direct_hiacc(__m512d r)
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512d A0 = _mm512_set1_pd(1.0/8.0);
    const __m512d A1 = _mm512_set1_pd(1.0/384.0);
    const __m512d A2 = _mm512_set1_pd(1.0/46080.0);
    const __m512d A3 = _mm512_set1_pd(1.0/10321920.0);
    const __m512d B0 = _mm512_set1_pd(1.0/2.0);
    const __m512d B1 = _mm512_set1_pd(1.0/48.0);
    const __m512d B2 = _mm512_set1_pd(1.0/3840.0);
    const __m512d B3 = _mm512_set1_pd(1.0/645120.0);

    __m512d t = _mm512_mul_pd(r, r);
    __m512d Pc = _mm512_fmadd_pd(A3, t, A2);
    Pc = _mm512_fmadd_pd(Pc, t, A1);
    Pc = _mm512_fmadd_pd(Pc, t, A0);
    __m512d Ps = _mm512_fmadd_pd(B3, t, B2);
    Ps = _mm512_fmadd_pd(Ps, t, B1);
    Ps = _mm512_fmadd_pd(Ps, t, B0);
    __m512d C = _mm512_mul_pd(t, Pc);
    __m512d S = _mm512_mul_pd(r, Ps);
    __m512d inner = _mm512_add_pd(two, _mm512_add_pd(C, S));
    __m512d y = _mm512_fmadd_pd(two, S, one);
    return _mm512_fmadd_pd(_mm512_mul_pd(two, C), inner, y);
}

static inline __m512d exp53_v128_formula6_direct_hiacc_block(__m512d x,const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask)
{
    __m512d biased=_mm512_fmadd_pd(x,inv,magic);
    __m512d k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512d r=_mm512_fnmadd_pd(k,hi,x);
    r=_mm512_fnmadd_pd(k,mi,r);
    r=_mm512_fnmadd_pd(k,lo,r);
    __m512d er=spine_residual128_formula6_direct_hiacc(r);
    __m512i j=_mm512_and_epi64(kn,mask);
    __m512i q=_mm512_srai_epi64(kn,7);
    __m512d tab=_mm512_i64gather_pd(j,TAB128,8);
    return _mm512_mul_pd(_mm512_mul_pd(er,tab),exp2_from_q(q));
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_formula6_direct_hiacc(double *restrict out,const double *restrict in,size_t n)
{
    const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC);
    const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24);
        __m512d y0=exp53_v128_formula6_direct_hiacc_block(x0,inv,hi,mi,lo,magic,mb,mask),y1=exp53_v128_formula6_direct_hiacc_block(x1,inv,hi,mi,lo,magic,mb,mask),y2=exp53_v128_formula6_direct_hiacc_block(x2,inv,hi,mi,lo,magic,mb,mask),y3=exp53_v128_formula6_direct_hiacc_block(x3,inv,hi,mi,lo,magic,mb,mask);
        _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);_mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
    }
    for(;i<n;){if(n-i>=8){_mm512_storeu_pd(out+i,exp53_v128_formula6_direct_hiacc_block(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));i+=8;}else{for(;i<n;++i)out[i]=exp(in[i]);}}
}
