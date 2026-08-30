/* Clean-room audit matrix for the faithful n=2 integral-power spine.
   Preserve e^r = 1 + r Q(r)^2.  Compare degrees/evaluation DAGs/unrolls only. */
#include "exp53_spine_n2_integralpower.c"

#define N2_Q5 0x1.3333333333333p-14 /* 3/40960 */
#define NF1(M) M(0)
#define NF2(M) M(0) M(1)
#define NF3(M) M(0) M(1) M(2)
#define NF4(M) M(0) M(1) M(2) M(3)
#define NF5(M) M(0) M(1) M(2) M(3) M(4)
#define NF6(M) M(0) M(1) M(2) M(3) M(4) M(5)
#define NF8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

#define A_LOAD(L) __m512d ax##L=_mm512_loadu_pd(in+i+8*(L));
#define A_BIAS(L) __m512d ab##L=_mm512_fmadd_pd(ax##L,inv,magic);
#define A_K(L) __m512d ak##L=_mm512_sub_pd(ab##L,magic); __m512i aki##L=_mm512_sub_epi64(_mm512_castpd_si512(ab##L),mb);
#define A_TAB(L) __m512i aj##L=_mm512_and_epi64(aki##L,mask), aq##L=_mm512_srai_epi64(aki##L,7); __m512d ath##L=_mm512_i64gather_pd(aj##L,TAB128,8), asc##L=exp2_from_q(aq##L);
#define A_R(L) __m512d ar##L=_mm512_fnmadd_pd(ak##L,hi,ax##L); ar##L=_mm512_fnmadd_pd(ak##L,mi,ar##L); ar##L=_mm512_fnmadd_pd(ak##L,lo,ar##L);
#define A_D3_1(L) __m512d ah##L=_mm512_fmadd_pd(nq3,ar##L,nq2);
#define A_D3_2(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,nq1);
#define A_D3_3(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,one);
#define A_D4_1(L) __m512d ah##L=_mm512_fmadd_pd(nq4,ar##L,nq3);
#define A_D4_2(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,nq2);
#define A_D4_3(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,nq1);
#define A_D4_4(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,one);
#define A_D5_1(L) __m512d ah##L=_mm512_fmadd_pd(nq5,ar##L,nq4);
#define A_D5_2(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,nq3);
#define A_D5_3(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,nq2);
#define A_D5_4(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,nq1);
#define A_D5_5(L) ah##L=_mm512_fmadd_pd(ah##L,ar##L,one);
#define A_E4_T(L) __m512d at##L=_mm512_mul_pd(ar##L,ar##L);
#define A_E4_E(L) __m512d ae##L=_mm512_fmadd_pd(nq4,at##L,nq2); ae##L=_mm512_fmadd_pd(ae##L,at##L,one);
#define A_E4_O(L) __m512d ao##L=_mm512_fmadd_pd(nq3,at##L,nq1);
#define A_E4_H(L) __m512d ah##L=_mm512_fmadd_pd(ar##L,ao##L,ae##L);
#define A_REC(L) __m512d as##L=_mm512_mul_pd(ah##L,ah##L); __m512d aer##L=_mm512_fmadd_pd(ar##L,as##L,one);
#define A_MUL(L) __m512d ap##L=_mm512_mul_pd(aer##L,ath##L); __m512d ay##L=_mm512_mul_pd(ap##L,asc##L);
#define A_RC(L) __m512d al##L=_mm512_fmadd_pd(ar##L,as##L,_mm512_sub_pd(one,aer##L)); __m512d ap##L=_mm512_mul_pd(aer##L,ath##L); __m512d apl##L=_mm512_fmadd_pd(aer##L,ath##L,_mm512_sub_pd(_mm512_setzero_pd(),ap##L)); apl##L=_mm512_fmadd_pd(al##L,ath##L,apl##L); __m512d ay##L=_mm512_mul_pd(_mm512_add_pd(ap##L,apl##L),asc##L);
#define A_STORE(L) _mm512_storeu_pd(out+i+8*(L),ay##L);
#define A_STORE_R(L) _mm512_storeu_pd(out+i+8*(L),ar##L);
#define A_STORE_H(L) _mm512_storeu_pd(out+i+8*(L),ah##L);
#define A_STORE_ER(L) _mm512_storeu_pd(out+i+8*(L),aer##L);

#define AC const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4),nq5=_mm512_set1_pd(N2_Q5); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);

#define COMMON_PRE(F) F(A_LOAD) F(A_BIAS) F(A_K) F(A_TAB) F(A_R)
#define D3_FAST(F) COMMON_PRE(F) F(A_D3_1) F(A_D3_2) F(A_D3_3) F(A_REC) F(A_MUL) F(A_STORE)
#define D4_FAST(F) COMMON_PRE(F) F(A_D4_1) F(A_D4_2) F(A_D4_3) F(A_D4_4) F(A_REC) F(A_MUL) F(A_STORE)
#define D5_FAST(F) COMMON_PRE(F) F(A_D5_1) F(A_D5_2) F(A_D5_3) F(A_D5_4) F(A_D5_5) F(A_REC) F(A_MUL) F(A_STORE)
#define E4_FAST(F) COMMON_PRE(F) F(A_E4_T) F(A_E4_E) F(A_E4_O) F(A_E4_H) F(A_REC) F(A_MUL) F(A_STORE)
#define D4_RC(F) COMMON_PRE(F) F(A_D4_1) F(A_D4_2) F(A_D4_3) F(A_D4_4) F(A_REC) F(A_RC) F(A_STORE)

#define DEF(NAME,U,F,STAGES) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){AC size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){STAGES(F)}if(i<n)exp53_n2_fast_u2(out+i,in+i,n-i);}
DEF(exp53_n2_d3_u4,4,NF4,D3_FAST)
DEF(exp53_n2_d4_u1,1,NF1,D4_FAST)
DEF(exp53_n2_d4_u2a,2,NF2,D4_FAST)
DEF(exp53_n2_d4_u3,3,NF3,D4_FAST)
DEF(exp53_n2_d4_u4a,4,NF4,D4_FAST)
DEF(exp53_n2_d4_u5,5,NF5,D4_FAST)
DEF(exp53_n2_d4_u6a,6,NF6,D4_FAST)
DEF(exp53_n2_d4_u8,8,NF8,D4_FAST)
DEF(exp53_n2_d5_u4,4,NF4,D5_FAST)
DEF(exp53_n2_e4_u4,4,NF4,E4_FAST)
DEF(exp53_n2_d4rc_u2,2,NF2,D4_RC)
DEF(exp53_n2_d4rc_u3,3,NF3,D4_RC)
DEF(exp53_n2_d4rc_u4,4,NF4,D4_RC)
DEF(exp53_n2_d4rc_u5,5,NF5,D4_RC)
DEF(exp53_n2_d4rc_u6,6,NF6,D4_RC)

/* Stage-stop kernels for throughput deltas. */
#define RED_ST(F) F(A_LOAD) F(A_BIAS) F(A_K) F(A_R) F(A_STORE_R)
#define Q_ST(F) COMMON_PRE(F) F(A_D4_1) F(A_D4_2) F(A_D4_3) F(A_D4_4) F(A_STORE_H)
#define ER_ST(F) COMMON_PRE(F) F(A_D4_1) F(A_D4_2) F(A_D4_3) F(A_D4_4) F(A_REC) F(A_STORE_ER)
DEF(exp53_n2_stage_reduce_u4,4,NF4,RED_ST)
DEF(exp53_n2_stage_q_u4,4,NF4,Q_ST)
DEF(exp53_n2_stage_er_u4,4,NF4,ER_ST)
