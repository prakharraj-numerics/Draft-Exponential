/* Speed optimization sweep for the <=1 ULP compensated Formula6 architecture.
   Includes the proven compacc path and adds progressively cheaper variants.
*/
#include "exp53_spine_v128_formula6_compacc.c"

/* FastTwoSum: exact error when |a| >= |b|, true here for 1+2S and near-1+tiny correction. */
static inline void fast_two_sum_pd(__m512d a,__m512d b,__m512d *s,__m512d *e)
{
    __m512d z=_mm512_add_pd(a,b);
    *s=z;
    *e=_mm512_sub_pd(b,_mm512_sub_pd(z,a));
}

/* Degree-1 audited minimax C/S, compensated reconstruction with fewer instructions. */
static inline void residual_fast_hilo(__m512d r,__m512d *rh,__m512d *rl)
{
    const __m512d one=_mm512_set1_pd(1.0),two=_mm512_set1_pd(2.0);
    const __m512d A0=_mm512_set1_pd(F6_A0),A1=_mm512_set1_pd(F6_A1);
    const __m512d B0=_mm512_set1_pd(F6_B0),B1=_mm512_set1_pd(F6_B1);
    __m512d t=_mm512_mul_pd(r,r);
    __m512d Pc=_mm512_fmadd_pd(A1,t,A0);
    __m512d Ps=_mm512_fmadd_pd(B1,t,B0);
    __m512d C=_mm512_mul_pd(t,Pc);
    __m512d S=_mm512_mul_pd(r,Ps);

    __m512d ah,al;
    fast_two_sum_pd(one,_mm512_add_pd(S,S),&ah,&al);
    __m512d inner=_mm512_add_pd(two,_mm512_add_pd(C,S));
    __m512d twoC=_mm512_add_pd(C,C);
    __m512d bh=_mm512_mul_pd(twoC,inner);
    __m512d bl=_mm512_fmadd_pd(twoC,inner,_mm512_sub_pd(_mm512_setzero_pd(),bh));
    __m512d sh,sl;
    fast_two_sum_pd(ah,bh,&sh,&sl);
    *rh=sh;
    *rl=_mm512_add_pd(_mm512_add_pd(sl,al),bl);
}

/* A: proven architecture, but degree-1 polynomial + FastTwoSum, omit erl*tl (far sub-ulp). */
static inline __m512d block_compfastA(__m512d x,const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask)
{
    __m512d biased=_mm512_fmadd_pd(x,inv,magic),k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512d r=_mm512_fnmadd_pd(k,hi,x); r=_mm512_fnmadd_pd(k,mi,r); r=_mm512_fnmadd_pd(k,lo,r);
    __m512d erh,erl; residual_fast_hilo(r,&erh,&erl);
    __m512i j=_mm512_and_epi64(kn,mask),q=_mm512_srai_epi64(kn,7);
    __m512d th=_mm512_i64gather_pd(j,TAB128,8),tl=_mm512_i64gather_pd(j,TAB128_LO,8);
    __m512d ph=_mm512_mul_pd(erh,th);
    __m512d corr=_mm512_fmadd_pd(erh,th,_mm512_sub_pd(_mm512_setzero_pd(),ph));
    corr=_mm512_fmadd_pd(erh,tl,corr);
    corr=_mm512_fmadd_pd(erl,th,corr);
    return _mm512_mul_pd(_mm512_add_pd(ph,corr),exp2_from_q(q));
}

/* B: single-double direct residual; retain split table + exact product residual. */
static inline __m512d block_compfastB(__m512d x,const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask)
{
    __m512d biased=_mm512_fmadd_pd(x,inv,magic),k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512d r=_mm512_fnmadd_pd(k,hi,x); r=_mm512_fnmadd_pd(k,mi,r); r=_mm512_fnmadd_pd(k,lo,r);
    __m512d er=spine_residual128_formula6_direct(r);
    __m512i j=_mm512_and_epi64(kn,mask),q=_mm512_srai_epi64(kn,7);
    __m512d th=_mm512_i64gather_pd(j,TAB128,8),tl=_mm512_i64gather_pd(j,TAB128_LO,8);
    __m512d ph=_mm512_mul_pd(er,th);
    __m512d corr=_mm512_fmadd_pd(er,th,_mm512_sub_pd(_mm512_setzero_pd(),ph));
    corr=_mm512_fmadd_pd(er,tl,corr);
    return _mm512_mul_pd(_mm512_add_pd(ph,corr),exp2_from_q(q));
}

/* C: compensated residual, canonical one-double table, exact product residual. */
static inline __m512d block_compfastC(__m512d x,const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask)
{
    __m512d biased=_mm512_fmadd_pd(x,inv,magic),k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512d r=_mm512_fnmadd_pd(k,hi,x); r=_mm512_fnmadd_pd(k,mi,r); r=_mm512_fnmadd_pd(k,lo,r);
    __m512d erh,erl; residual_fast_hilo(r,&erh,&erl);
    __m512i j=_mm512_and_epi64(kn,mask),q=_mm512_srai_epi64(kn,7);
    __m512d th=_mm512_i64gather_pd(j,TAB128,8);
    __m512d ph=_mm512_mul_pd(erh,th);
    __m512d corr=_mm512_fmadd_pd(erh,th,_mm512_sub_pd(_mm512_setzero_pd(),ph));
    corr=_mm512_fmadd_pd(erl,th,corr);
    return _mm512_mul_pd(_mm512_add_pd(ph,corr),exp2_from_q(q));
}

#define DEF_KERNEL(NAME,BLOCK) \
__attribute__((target("avx512f,avx512dq,fma"))) \
void NAME(double *restrict out,const double *restrict in,size_t n){ \
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC); \
 const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127); size_t i=0; \
 for(;i+32<=n;i+=32){__m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24); \
 _mm512_storeu_pd(out+i,BLOCK(x0,inv,hi,mi,lo,magic,mb,mask)); _mm512_storeu_pd(out+i+8,BLOCK(x1,inv,hi,mi,lo,magic,mb,mask)); \
 _mm512_storeu_pd(out+i+16,BLOCK(x2,inv,hi,mi,lo,magic,mb,mask)); _mm512_storeu_pd(out+i+24,BLOCK(x3,inv,hi,mi,lo,magic,mb,mask));} \
 for(;i<n;){if(n-i>=8){_mm512_storeu_pd(out+i,BLOCK(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));i+=8;}else{for(;i<n;++i)out[i]=exp(in[i]);}} }

DEF_KERNEL(exp53_spine_v128_formula6_compfastA,block_compfastA)
DEF_KERNEL(exp53_spine_v128_formula6_compfastB,block_compfastB)
DEF_KERNEL(exp53_spine_v128_formula6_compfastC,block_compfastC)
