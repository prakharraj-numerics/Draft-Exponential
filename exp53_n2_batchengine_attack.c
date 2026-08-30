/* Batch-engine attack for faithful n=2 ER-low EXP.
   Math unchanged. Main change: fuse TAB128[j] and 2^q into one exact scale
   by adding q directly to the gathered IEEE-754 exponent bits. Domain [-100,100]
   keeps scale normal, so this is exact and removes exp2_from_q + one multiply. */
#include "exp53_spine_n2_integralpower.c"

#define BE_F2(M) M(0) M(1)
#define BE_F3(M) M(0) M(1) M(2)
#define BE_F4(M) M(0) M(1) M(2) M(3)

#define BE_LOAD(L) __m512d bx##L=_mm512_loadu_pd(in+i+8*(L));
#define BE_BIAS(L) __m512d bb##L=_mm512_fmadd_pd(bx##L,inv,magic);
#define BE_K(L) __m512d bk##L=_mm512_sub_pd(bb##L,magic); __m512i bkn##L=_mm512_sub_epi64(_mm512_castpd_si512(bb##L),mb); __m512i bj##L=_mm512_and_epi64(bkn##L,mask), bq##L=_mm512_srai_epi64(bkn##L,7);
#define BE_R(L) __m512d br##L=_mm512_fnmadd_pd(bk##L,hi,bx##L); br##L=_mm512_fnmadd_pd(bk##L,mi,br##L); br##L=_mm512_fnmadd_pd(bk##L,lo,br##L);
#define BE_SCALE(L) __m512i btbits##L=_mm512_i64gather_epi64(bj##L,(const long long*)TAB128,8); __m512i bsbits##L=_mm512_add_epi64(btbits##L,_mm512_slli_epi64(bq##L,52)); __m512d bscale##L=_mm512_castsi512_pd(bsbits##L);
#define BE_TAB(L) __m512d btab##L=_mm512_i64gather_pd(bj##L,TAB128,8);
#define BE_EXP2(L) __m512d bexp##L=exp2_from_q(bq##L);
#define BE_H1(L) __m512d bh##L=_mm512_fmadd_pd(nq4,br##L,nq3);
#define BE_H2(L) bh##L=_mm512_fmadd_pd(bh##L,br##L,nq2);
#define BE_H3(L) bh##L=_mm512_fmadd_pd(bh##L,br##L,nq1);
#define BE_H4(L) bh##L=_mm512_fmadd_pd(bh##L,br##L,one);
#define BE_REC(L) __m512d bs##L=_mm512_mul_pd(bh##L,bh##L); __m512d ber##L=_mm512_fmadd_pd(br##L,bs##L,one); __m512d bel##L=_mm512_fmadd_pd(br##L,bs##L,_mm512_sub_pd(one,ber##L));
#define BE_FIN(L) __m512d bph##L=_mm512_mul_pd(ber##L,bscale##L); __m512d by##L=_mm512_fmadd_pd(bel##L,bscale##L,bph##L); _mm512_storeu_pd(out+i+8*(L),by##L);
#define BE_BASEFIN(L) __m512d bph##L=_mm512_mul_pd(ber##L,btab##L); __m512d bp##L=_mm512_fmadd_pd(bel##L,btab##L,bph##L); __m512d by##L=_mm512_mul_pd(bp##L,bexp##L); _mm512_storeu_pd(out+i+8*(L),by##L);

#define BE_CONSTS const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);

#define BE_BASE(F) F(BE_LOAD) F(BE_BIAS) F(BE_K) F(BE_R) F(BE_TAB) F(BE_EXP2) F(BE_H1) F(BE_H2) F(BE_H3) F(BE_H4) F(BE_REC) F(BE_BASEFIN)
#define BE_EARLY(F) F(BE_LOAD) F(BE_BIAS) F(BE_K) F(BE_SCALE) F(BE_R) F(BE_H1) F(BE_H2) F(BE_H3) F(BE_H4) F(BE_REC) F(BE_FIN)
#define BE_MID(F) F(BE_LOAD) F(BE_BIAS) F(BE_K) F(BE_R) F(BE_SCALE) F(BE_H1) F(BE_H2) F(BE_H3) F(BE_H4) F(BE_REC) F(BE_FIN)
#define BE_H1G(F) F(BE_LOAD) F(BE_BIAS) F(BE_K) F(BE_R) F(BE_H1) F(BE_SCALE) F(BE_H2) F(BE_H3) F(BE_H4) F(BE_REC) F(BE_FIN)

#define DEF_BE(NAME,U,F,SCHED) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){BE_CONSTS size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){SCHED(F)}for(;i<n;i++)out[i]=exp(in[i]);}
DEF_BE(exp53_n2_be_base_u4,4,BE_F4,BE_BASE)
DEF_BE(exp53_n2_be_fused_u2,2,BE_F2,BE_MID)
DEF_BE(exp53_n2_be_fused_u3,3,BE_F3,BE_MID)
DEF_BE(exp53_n2_be_fused_u4,4,BE_F4,BE_MID)
DEF_BE(exp53_n2_be_fused_early_u4,4,BE_F4,BE_EARLY)
DEF_BE(exp53_n2_be_fused_h1g_u4,4,BE_F4,BE_H1G)

/* Lifetime-reduced 4-vector block: issue all scales early, but complete vectors in
   two pairs so H/ER/EL/final temporaries for 0-1 die before 2-3 are created. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_be_pair_u4(double*restrict out,const double*restrict in,size_t n){
 BE_CONSTS size_t i=0;
 for(;i+32<=n;i+=32){
  BE_F4(BE_LOAD) BE_F4(BE_BIAS) BE_F4(BE_K) BE_F4(BE_R) BE_F4(BE_SCALE)
  BE_H1(0) BE_H1(1) BE_H2(0) BE_H2(1) BE_H3(0) BE_H3(1) BE_H4(0) BE_H4(1) BE_REC(0) BE_REC(1) BE_FIN(0) BE_FIN(1)
  BE_H1(2) BE_H1(3) BE_H2(2) BE_H2(3) BE_H3(2) BE_H3(3) BE_H4(2) BE_H4(3) BE_REC(2) BE_REC(3) BE_FIN(2) BE_FIN(3)
 }
 for(;i<n;i++)out[i]=exp(in[i]);
}
