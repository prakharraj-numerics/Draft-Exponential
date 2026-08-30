/* Focused streaming attack for faithful n=2 ER-low kernel.
   Math unchanged: Q4 -> Q4^2 -> er=1+rQ4^2 plus er-low repair.
   Only scheduling/unroll/scale placement is varied. */
#include "exp53_spine_n2_integralpower.c"

#define SA_F3(M) M(0) M(1) M(2)
#define SA_F4(M) M(0) M(1) M(2) M(3)

#define SA_LOAD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define SA_BIAS(L) __m512d b##L=_mm512_fmadd_pd(x##L,inv,magic);
#define SA_K(L) __m512d k##L=_mm512_sub_pd(b##L,magic); __m512i ki##L=_mm512_sub_epi64(_mm512_castpd_si512(b##L),mb); __m512i j##L=_mm512_and_epi64(ki##L,mask), qq##L=_mm512_srai_epi64(ki##L,7);
#define SA_R(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L); r##L=_mm512_fnmadd_pd(k##L,mi,r##L); r##L=_mm512_fnmadd_pd(k##L,lo,r##L);
#define SA_TAB(L) __m512d th##L=_mm512_i64gather_pd(j##L,TAB128,8);
#define SA_SCALE(L) __m512d sc##L=exp2_from_q(qq##L);
#define SA_H1(L) __m512d h##L=_mm512_fmadd_pd(nq4,r##L,nq3);
#define SA_H2(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq2);
#define SA_H3(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq1);
#define SA_H4(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define SA_REC(L) __m512d s##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,s##L,one); __m512d el##L=_mm512_fmadd_pd(r##L,s##L,_mm512_sub_pd(one,er##L));
#define SA_PROD(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d p##L=_mm512_fmadd_pd(el##L,th##L,ph##L);
#define SA_FINAL(L) __m512d y##L=_mm512_mul_pd(p##L,sc##L);
#define SA_STORE(L) _mm512_storeu_pd(out+i+8*(L),y##L);
#define SA_CONSTS const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);

static __attribute__((target("avx512f,avx512dq,fma"),always_inline)) inline void sa_tail(double*out,const double*in,size_t n){
 SA_CONSTS while(n){unsigned c=n>=8?8u:(unsigned)n;__mmask8 km=(__mmask8)((1u<<c)-1u);__m512d x=_mm512_maskz_loadu_pd(km,in);__m512d b=_mm512_fmadd_pd(x,inv,magic),k=_mm512_sub_pd(b,magic);__m512i ki=_mm512_sub_epi64(_mm512_castpd_si512(b),mb),j=_mm512_and_epi64(ki,mask),qq=_mm512_srai_epi64(ki,7);__m512d r=_mm512_fnmadd_pd(k,hi,x);r=_mm512_fnmadd_pd(k,mi,r);r=_mm512_fnmadd_pd(k,lo,r);__m512d th=_mm512_i64gather_pd(j,TAB128,8);__m512d h=_mm512_fmadd_pd(nq4,r,nq3);h=_mm512_fmadd_pd(h,r,nq2);h=_mm512_fmadd_pd(h,r,nq1);h=_mm512_fmadd_pd(h,r,one);__m512d s=_mm512_mul_pd(h,h),er=_mm512_fmadd_pd(r,s,one),el=_mm512_fmadd_pd(r,s,_mm512_sub_pd(one,er));__m512d p=_mm512_fmadd_pd(el,th,_mm512_mul_pd(er,th));__m512d y=_mm512_mul_pd(p,exp2_from_q(qq));_mm512_mask_storeu_pd(out,km,y);in+=c;out+=c;n-=c;}}

#define SA_BODY_BASE(F) F(SA_LOAD) F(SA_BIAS) F(SA_K) F(SA_R) F(SA_TAB) F(SA_SCALE) F(SA_H1) F(SA_H2) F(SA_H3) F(SA_H4) F(SA_REC) F(SA_PROD) F(SA_FINAL) F(SA_STORE)
#define SA_BODY_SCALEEARLY(F) F(SA_LOAD) F(SA_BIAS) F(SA_K) F(SA_SCALE) F(SA_R) F(SA_TAB) F(SA_H1) F(SA_H2) F(SA_H3) F(SA_H4) F(SA_REC) F(SA_PROD) F(SA_FINAL) F(SA_STORE)
#define SA_BODY_SCALELATE(F) F(SA_LOAD) F(SA_BIAS) F(SA_K) F(SA_R) F(SA_TAB) F(SA_H1) F(SA_H2) F(SA_H3) F(SA_H4) F(SA_REC) F(SA_PROD) F(SA_SCALE) F(SA_FINAL) F(SA_STORE)
#define SA_BODY_TABEARLY(F) F(SA_LOAD) F(SA_BIAS) F(SA_K) F(SA_TAB) F(SA_R) F(SA_H1) F(SA_H2) F(SA_H3) F(SA_H4) F(SA_REC) F(SA_PROD) F(SA_SCALE) F(SA_FINAL) F(SA_STORE)
#define SA_BODY_TABLATE(F) F(SA_LOAD) F(SA_BIAS) F(SA_K) F(SA_R) F(SA_H1) F(SA_H2) F(SA_H3) F(SA_H4) F(SA_TAB) F(SA_REC) F(SA_PROD) F(SA_SCALE) F(SA_FINAL) F(SA_STORE)

#define DEF_SA(NAME,U,F,BODY) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){SA_CONSTS size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){BODY(F)}if(i<n)sa_tail(out+i,in+i,n-i);}
DEF_SA(exp53_n2_sa_mid_u4,4,SA_F4,SA_BODY_BASE)
DEF_SA(exp53_n2_sa_scalearly_u4,4,SA_F4,SA_BODY_SCALEEARLY)
DEF_SA(exp53_n2_sa_scalelate_u4,4,SA_F4,SA_BODY_SCALELATE)
DEF_SA(exp53_n2_sa_tabearly_u4,4,SA_F4,SA_BODY_TABEARLY)
DEF_SA(exp53_n2_sa_tablate_u4,4,SA_F4,SA_BODY_TABLATE)
DEF_SA(exp53_n2_sa_mid_u3,3,SA_F3,SA_BODY_BASE)
DEF_SA(exp53_n2_sa_scalelate_u3,3,SA_F3,SA_BODY_SCALELATE)
