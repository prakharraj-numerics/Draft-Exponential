/* Batch/vectorization-only audit for the frozen faithful n=2 ER-low kernel.
   Mathematical path is unchanged: Q4 -> Q4^2 -> er=1+rQ4^2 plus er-low repair.
   Variants change only unroll, gather placement, and tail/dispatch behavior. */
#include "exp53_spine_n2_integralpower.c"

#define BV_F1(M) M(0)
#define BV_F2(M) M(0) M(1)
#define BV_F3(M) M(0) M(1) M(2)
#define BV_F4(M) M(0) M(1) M(2) M(3)
#define BV_F5(M) M(0) M(1) M(2) M(3) M(4)
#define BV_F6(M) M(0) M(1) M(2) M(3) M(4) M(5)

#define BV_LOAD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define BV_BIAS(L) __m512d b##L=_mm512_fmadd_pd(x##L,inv,magic);
#define BV_K(L) __m512d k##L=_mm512_sub_pd(b##L,magic); __m512i ki##L=_mm512_sub_epi64(_mm512_castpd_si512(b##L),mb); __m512i j##L=_mm512_and_epi64(ki##L,mask), qq##L=_mm512_srai_epi64(ki##L,7);
#define BV_TAB(L) __m512d th##L=_mm512_i64gather_pd(j##L,TAB128,8), sc##L=exp2_from_q(qq##L);
#define BV_R(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L); r##L=_mm512_fnmadd_pd(k##L,mi,r##L); r##L=_mm512_fnmadd_pd(k##L,lo,r##L);
#define BV_H1(L) __m512d h##L=_mm512_fmadd_pd(nq4,r##L,nq3);
#define BV_H2(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq2);
#define BV_H3(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq1);
#define BV_H4(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define BV_REC(L) __m512d s##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,s##L,one); __m512d el##L=_mm512_fmadd_pd(r##L,s##L,_mm512_sub_pd(one,er##L));
#define BV_MUL(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d p##L=_mm512_fmadd_pd(el##L,th##L,ph##L); __m512d y##L=_mm512_mul_pd(p##L,sc##L);
#define BV_STORE(L) _mm512_storeu_pd(out+i+8*(L),y##L);

#define BV_CONSTS const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);

/* Current strategy: gather immediately after k, then overlap with reduction/Horner. */
#define BV_EARLY(F) F(BV_LOAD) F(BV_BIAS) F(BV_K) F(BV_TAB) F(BV_R) F(BV_H1) F(BV_H2) F(BV_H3) F(BV_H4) F(BV_REC) F(BV_MUL) F(BV_STORE)
/* Mid strategy: reduce first, gather before Horner. */
#define BV_MID(F) F(BV_LOAD) F(BV_BIAS) F(BV_K) F(BV_R) F(BV_TAB) F(BV_H1) F(BV_H2) F(BV_H3) F(BV_H4) F(BV_REC) F(BV_MUL) F(BV_STORE)
/* Late strategy: push gather until after Horner, testing latency overlap tradeoff. */
#define BV_LATE(F) F(BV_LOAD) F(BV_BIAS) F(BV_K) F(BV_R) F(BV_H1) F(BV_H2) F(BV_H3) F(BV_H4) F(BV_TAB) F(BV_REC) F(BV_MUL) F(BV_STORE)

/* Same ER-low math for a single masked vector. This replaces the old RC fallback. */
static __attribute__((target("avx512f,avx512dq,fma"),always_inline)) inline
void bv_el_tail(double *restrict out,const double *restrict in,size_t n)
{
    BV_CONSTS
    while(n){
        unsigned c=n>=8?8u:(unsigned)n; __mmask8 km=(__mmask8)((1u<<c)-1u);
        __m512d x=_mm512_maskz_loadu_pd(km,in);
        __m512d b=_mm512_fmadd_pd(x,inv,magic),k=_mm512_sub_pd(b,magic);
        __m512i ki=_mm512_sub_epi64(_mm512_castpd_si512(b),mb),j=_mm512_and_epi64(ki,mask),qq=_mm512_srai_epi64(ki,7);
        __m512d th=_mm512_i64gather_pd(j,TAB128,8),sc=exp2_from_q(qq);
        __m512d r=_mm512_fnmadd_pd(k,hi,x); r=_mm512_fnmadd_pd(k,mi,r); r=_mm512_fnmadd_pd(k,lo,r);
        __m512d h=_mm512_fmadd_pd(nq4,r,nq3); h=_mm512_fmadd_pd(h,r,nq2); h=_mm512_fmadd_pd(h,r,nq1); h=_mm512_fmadd_pd(h,r,one);
        __m512d s=_mm512_mul_pd(h,h),er=_mm512_fmadd_pd(r,s,one),el=_mm512_fmadd_pd(r,s,_mm512_sub_pd(one,er));
        __m512d ph=_mm512_mul_pd(er,th),p=_mm512_fmadd_pd(el,th,ph),y=_mm512_mul_pd(p,sc);
        _mm512_mask_storeu_pd(out,km,y); in+=c;out+=c;n-=c;
    }
}

#define DEF_BV(NAME,U,F,SCHED) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){BV_CONSTS size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){SCHED(F)}if(i<n)bv_el_tail(out+i,in+i,n-i);}
DEF_BV(exp53_n2_bv_u1,1,BV_F1,BV_EARLY)
DEF_BV(exp53_n2_bv_u2,2,BV_F2,BV_EARLY)
DEF_BV(exp53_n2_bv_u3,3,BV_F3,BV_EARLY)
DEF_BV(exp53_n2_bv_u4,4,BV_F4,BV_EARLY)
DEF_BV(exp53_n2_bv_u5,5,BV_F5,BV_EARLY)
DEF_BV(exp53_n2_bv_u6,6,BV_F6,BV_EARLY)
DEF_BV(exp53_n2_bv_mid_u4,4,BV_F4,BV_MID)
DEF_BV(exp53_n2_bv_late_u4,4,BV_F4,BV_LATE)

/* Batch-adaptive wrapper: exact same vector kernel, just avoid over-unroll for short calls. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_bv_adaptive(double*restrict out,const double*restrict in,size_t n){
    if(n<16) return exp53_n2_bv_u1(out,in,n);
    if(n<24) return exp53_n2_bv_u2(out,in,n);
    if(n<32) return exp53_n2_bv_u3(out,in,n);
    return exp53_n2_bv_u4(out,in,n);
}
