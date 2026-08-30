/* n=2 table representation attack. Mathematical spine is frozen:
   er = 1 + r Q4(r)^2.  Only the representation of 2^(j/128) changes. */
#include <mpfr.h>
#include "exp53_spine_n2_integralpower.c"
static double N2_TLO[128] __attribute__((aligned(64)));
void exp53_n2_init_tlo(void){
    mpfr_t a,e,h; mpfr_inits2(256,a,e,h,(mpfr_ptr)0);
    for(int j=0;j<128;j++){
        mpfr_set_si(a,j,MPFR_RNDN); mpfr_div_ui(a,a,128,MPFR_RNDN); mpfr_const_log2(e,MPFR_RNDN); mpfr_mul(a,a,e,MPFR_RNDN); mpfr_exp(e,a,MPFR_RNDN);
        mpfr_set_d(h,TAB128[j],MPFR_RNDN); mpfr_sub(e,e,h,MPFR_RNDN); N2_TLO[j]=mpfr_get_d(e,MPFR_RNDN);
    }
    mpfr_clears(a,e,h,(mpfr_ptr)0);
}
#define TF2(M) M(0) M(1)
#define TF3(M) M(0) M(1) M(2)
#define TF4(M) M(0) M(1) M(2) M(3)
#define T_LOAD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define T_BIAS(L) __m512d b##L=_mm512_fmadd_pd(x##L,inv,magic);
#define T_K(L) __m512d k##L=_mm512_sub_pd(b##L,magic); __m512i ki##L=_mm512_sub_epi64(_mm512_castpd_si512(b##L),mb);
#define T_TAB(L) __m512i j##L=_mm512_and_epi64(ki##L,mask),qq##L=_mm512_srai_epi64(ki##L,7); __m512d th##L=_mm512_i64gather_pd(j##L,TAB128,8), sc##L=exp2_from_q(qq##L);
#define T_TLO(L) __m512d tl##L=_mm512_i64gather_pd(j##L,N2_TLO,8);
#define T_R(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L); r##L=_mm512_fnmadd_pd(k##L,mi,r##L); r##L=_mm512_fnmadd_pd(k##L,lo,r##L);
#define T_H1(L) __m512d h##L=_mm512_fmadd_pd(nq4,r##L,nq3);
#define T_H2(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq2);
#define T_H3(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq1);
#define T_H4(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define T_ER(L) __m512d s##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,s##L,one);
#define T_PLAIN(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d y##L=_mm512_mul_pd(ph##L,sc##L);
#define T_PC(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d pl##L=_mm512_fmadd_pd(er##L,th##L,_mm512_sub_pd(_mm512_setzero_pd(),ph##L)); __m512d y##L=_mm512_mul_pd(_mm512_add_pd(ph##L,pl##L),sc##L);
#define T_TL1(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d p##L=_mm512_fmadd_pd(er##L,tl##L,ph##L); __m512d y##L=_mm512_mul_pd(p##L,sc##L);
#define T_TLPC(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d pl##L=_mm512_fmadd_pd(er##L,th##L,_mm512_sub_pd(_mm512_setzero_pd(),ph##L)); __m512d p##L=_mm512_fmadd_pd(er##L,tl##L,_mm512_add_pd(ph##L,pl##L)); __m512d y##L=_mm512_mul_pd(p##L,sc##L);
#define T_STORE(L) _mm512_storeu_pd(out+i+8*(L),y##L);
#define TC const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);
#define PRE(F) F(T_LOAD) F(T_BIAS) F(T_K) F(T_TAB) F(T_R) F(T_H1) F(T_H2) F(T_H3) F(T_H4) F(T_ER)
#define PLAIN_ST(F) PRE(F) F(T_PLAIN) F(T_STORE)
#define PC_ST(F) PRE(F) F(T_PC) F(T_STORE)
#define TL1_ST(F) F(T_LOAD) F(T_BIAS) F(T_K) F(T_TAB) F(T_TLO) F(T_R) F(T_H1) F(T_H2) F(T_H3) F(T_H4) F(T_ER) F(T_TL1) F(T_STORE)
#define TLPC_ST(F) F(T_LOAD) F(T_BIAS) F(T_K) F(T_TAB) F(T_TLO) F(T_R) F(T_H1) F(T_H2) F(T_H3) F(T_H4) F(T_ER) F(T_TLPC) F(T_STORE)
#define DEF(NAME,U,F,ST) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){TC size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){ST(F)}if(i<n)exp53_n2_fast_u2(out+i,in+i,n-i);}
DEF(exp53_n2_tplain_u4,4,TF4,PLAIN_ST)
DEF(exp53_n2_tpc_u2,2,TF2,PC_ST)
DEF(exp53_n2_tpc_u3,3,TF3,PC_ST)
DEF(exp53_n2_tpc_u4,4,TF4,PC_ST)
DEF(exp53_n2_tl1_u2,2,TF2,TL1_ST)
DEF(exp53_n2_tl1_u3,3,TF3,TL1_ST)
DEF(exp53_n2_tl1_u4,4,TF4,TL1_ST)
DEF(exp53_n2_tlpc_u2,2,TF2,TLPC_ST)
DEF(exp53_n2_tlpc_u3,3,TF3,TLPC_ST)
