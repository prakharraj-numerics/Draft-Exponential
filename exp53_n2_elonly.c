/* Minimal accuracy repair for faithful n=2 EXP spine.
   Diagnosis showed all FAST >1-ULP failures are fixed by preserving only the
   low part of er = fma(r,Q4(r)^2,1); product residual compensation is omitted.

   Spine remains: Q4(r) -> Q4(r)^2 -> 1+rQ4(r)^2.
*/
#include "exp53_spine_n2_integralpower.c"

#define EL_F2(M) M(0) M(1)
#define EL_F3(M) M(0) M(1) M(2)
#define EL_F4(M) M(0) M(1) M(2) M(3)
#define EL_LOAD(L) __m512d ex##L=_mm512_loadu_pd(in+i+8*(L));
#define EL_BIAS(L) __m512d eb##L=_mm512_fmadd_pd(ex##L,inv,magic);
#define EL_K(L) __m512d ek##L=_mm512_sub_pd(eb##L,magic); __m512i eki##L=_mm512_sub_epi64(_mm512_castpd_si512(eb##L),mb);
#define EL_TAB(L) __m512i ej##L=_mm512_and_epi64(eki##L,mask), eq##L=_mm512_srai_epi64(eki##L,7); __m512d eth##L=_mm512_i64gather_pd(ej##L,TAB128,8), esc##L=exp2_from_q(eq##L);
#define EL_R(L) __m512d erd##L=_mm512_fnmadd_pd(ek##L,hi,ex##L); erd##L=_mm512_fnmadd_pd(ek##L,mi,erd##L); erd##L=_mm512_fnmadd_pd(ek##L,lo,erd##L);
#define EL_H1(L) __m512d eh##L=_mm512_fmadd_pd(nq4,erd##L,nq3);
#define EL_H2(L) eh##L=_mm512_fmadd_pd(eh##L,erd##L,nq2);
#define EL_H3(L) eh##L=_mm512_fmadd_pd(eh##L,erd##L,nq1);
#define EL_H4(L) eh##L=_mm512_fmadd_pd(eh##L,erd##L,one);
#define EL_REC(L) __m512d es##L=_mm512_mul_pd(eh##L,eh##L); __m512d eer##L=_mm512_fmadd_pd(erd##L,es##L,one); __m512d eel##L=_mm512_fmadd_pd(erd##L,es##L,_mm512_sub_pd(one,eer##L));
#define EL_MUL(L) __m512d eph##L=_mm512_mul_pd(eer##L,eth##L); __m512d ep##L=_mm512_fmadd_pd(eel##L,eth##L,eph##L); __m512d ey##L=_mm512_mul_pd(ep##L,esc##L);
#define EL_STORE(L) _mm512_storeu_pd(out+i+8*(L),ey##L);
#define EL_PRE(F) F(EL_LOAD) F(EL_BIAS) F(EL_K) F(EL_TAB) F(EL_R) F(EL_H1) F(EL_H2) F(EL_H3) F(EL_H4) F(EL_REC) F(EL_MUL) F(EL_STORE)
#define EL_CONSTS const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);
#define DEF_EL(NAME,U,F) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){EL_CONSTS size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){EL_PRE(F)}if(i<n)exp53_n2_rc_u4(out+i,in+i,n-i);}
DEF_EL(exp53_n2_elonly_u2,2,EL_F2)
DEF_EL(exp53_n2_elonly_u3,3,EL_F3)
DEF_EL(exp53_n2_elonly_u4,4,EL_F4)
