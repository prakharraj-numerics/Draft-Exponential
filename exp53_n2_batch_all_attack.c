/* ALL batch-engine attack for faithful n=2 ER-low EXP.
   Mathematical spine remains Q4 -> Q4^2 -> er=1+rQ4^2 plus ER-low repair.
   This file applies the full batch research stack in one benchmarkable sweep:
     (1) fused table+2^q scale via IEEE exponent-bit adjustment,
     (2) true staggered/manual cross-vector scheduling,
     (3) gather-free j=8a+b register-table scale construction,
     (4) parallel even/odd Q4 evaluation (same Q4 polynomial),
     (5) lifetime-reduced pair completion / higher in-flight depth.
   Domain for benchmark/candidate is [-100,100], so fused normal scale is exact.
*/
#include "exp53_spine_n2_integralpower.c"

#define BA_F2(M) M(0) M(1)
#define BA_F3(M) M(0) M(1) M(2)
#define BA_F4(M) M(0) M(1) M(2) M(3)
#define BA_F6(M) BA_F4(M) M(4) M(5)

#define BA_CONSTS \
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4); \
 const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127),seven=_mm512_set1_epi64(7),eight=_mm512_set1_epi64(8);

#define BA_LOAD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define BA_BIAS(L) __m512d bz##L=_mm512_fmadd_pd(x##L,inv,magic);
#define BA_K(L) __m512d k##L=_mm512_sub_pd(bz##L,magic); __m512i kn##L=_mm512_sub_epi64(_mm512_castpd_si512(bz##L),mb); __m512i j##L=_mm512_and_epi64(kn##L,mask), q##L=_mm512_srai_epi64(kn##L,7);
#define BA_R(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L); r##L=_mm512_fnmadd_pd(k##L,mi,r##L); r##L=_mm512_fnmadd_pd(k##L,lo,r##L);

/* Gather raw TAB128 bits, add q directly to exponent field: exact scale TAB128[j]*2^q. */
#define BA_FSCALE(L) __m512i tb##L=_mm512_i64gather_epi64(j##L,(const long long*)TAB128,8); tb##L=_mm512_add_epi64(tb##L,_mm512_slli_epi64(q##L,52)); __m512d scale##L=_mm512_castsi512_pd(tb##L);

/* Canonical Horner Q4. */
#define BA_QH1(L) __m512d h##L=_mm512_fmadd_pd(nq4,r##L,nq3);
#define BA_QH2(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq2);
#define BA_QH3(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq1);
#define BA_QH4(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);

/* Same Q4, algebraically split for ILP:
     Q4 = (1 + r^2(q2+q4 r^2)) + r(q1+q3 r^2).
   Still Q4 -> Q4^2; not an exp Taylor expansion. */
#define BA_QS1(L) __m512d r2_##L=_mm512_mul_pd(r##L,r##L); __m512d ev##L=_mm512_fmadd_pd(nq4,r2_##L,nq2); __m512d od##L=_mm512_fmadd_pd(nq3,r2_##L,nq1);
#define BA_QS2(L) ev##L=_mm512_fmadd_pd(ev##L,r2_##L,one); __m512d h##L=_mm512_fmadd_pd(od##L,r##L,ev##L);

#define BA_REC(L) __m512d s##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,s##L,one); __m512d el##L=_mm512_fmadd_pd(r##L,s##L,_mm512_sub_pd(one,er##L));
#define BA_FIN(L) __m512d ph##L=_mm512_mul_pd(er##L,scale##L); __m512d y##L=_mm512_fmadd_pd(el##L,scale##L,ph##L); _mm512_storeu_pd(out+i+8*(L),y##L);

/* ---------- gather-free scale: j=8a+b, 2^(j/128)=A[a]*B[b] ---------- */
static const double BA_REGA[16]={
  0x1.0000000000000p+0,0x1.0b5586cf9890fp+0,0x1.172b83c7d517bp+0,0x1.2387a6e756238p+0,
  0x1.306fe0a31b715p+0,0x1.3dea64c123422p+0,0x1.4bfdad5362a27p+0,0x1.5ab07dd485429p+0,
  0x1.6a09e667f3bcdp+0,0x1.7a11473eb0187p+0,0x1.8ace5422aa0dbp+0,0x1.9c49182a3f090p+0,
  0x1.ae89f995ad3adp+0,0x1.c199bdd85529cp+0,0x1.d5818dcfba487p+0,0x1.ea4afa2a490dap+0};
static const double BA_REGB[8]={
  0x1.0000000000000p+0,0x1.0163da9fb3335p+0,0x1.02c9a3e778061p+0,0x1.04315e86e7f85p+0,
  0x1.059b0d3158574p+0,0x1.0706b29ddf6dep+0,0x1.0874518759bc8p+0,0x1.09e3ecac6f383p+0};
#define BA_REGCONSTS const __m512d RA0=_mm512_loadu_pd(BA_REGA),RA1=_mm512_loadu_pd(BA_REGA+8),RB=_mm512_loadu_pd(BA_REGB);
#define BA_RSCALE(L) __m512i aa##L=_mm512_srli_epi64(j##L,3), bb##L=_mm512_and_epi64(j##L,seven), ai##L=_mm512_and_epi64(aa##L,seven); __m512d a0_##L=_mm512_permutexvar_pd(ai##L,RA0), a1_##L=_mm512_permutexvar_pd(ai##L,RA1), bv##L=_mm512_permutexvar_pd(bb##L,RB); __mmask8 am##L=_mm512_cmp_epi64_mask(aa##L,eight,_MM_CMPINT_GE); __m512d av##L=_mm512_mask_blend_pd(am##L,a0_##L,a1_##L); __m512i abits##L=_mm512_castpd_si512(av##L); abits##L=_mm512_add_epi64(abits##L,_mm512_slli_epi64(q##L,52)); __m512d as##L=_mm512_castsi512_pd(abits##L); __m512d scale##L=_mm512_mul_pd(as##L,bv##L);

/* Baseline fused winner, same-run reference. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ba_fused_u4(double*restrict out,const double*restrict in,size_t n){BA_CONSTS size_t i=0;for(;i+32<=n;i+=32){BA_F4(BA_LOAD) BA_F4(BA_BIAS) BA_F4(BA_K) BA_F4(BA_R) BA_F4(BA_FSCALE) BA_F4(BA_QH1) BA_F4(BA_QH2) BA_F4(BA_QH3) BA_F4(BA_QH4) BA_F4(BA_REC) BA_F4(BA_FIN)}for(;i<n;i++)out[i]=exp(in[i]);}

/* Parallel-Q candidate. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ba_estrinq_u4(double*restrict out,const double*restrict in,size_t n){BA_CONSTS size_t i=0;for(;i+32<=n;i+=32){BA_F4(BA_LOAD) BA_F4(BA_BIAS) BA_F4(BA_K) BA_F4(BA_R) BA_F4(BA_FSCALE) BA_F4(BA_QS1) BA_F4(BA_QS2) BA_F4(BA_REC) BA_F4(BA_FIN)}for(;i<n;i++)out[i]=exp(in[i]);}

/* Gather-free scale with canonical Horner. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ba_regscale_u4(double*restrict out,const double*restrict in,size_t n){BA_CONSTS BA_REGCONSTS size_t i=0;for(;i+32<=n;i+=32){BA_F4(BA_LOAD) BA_F4(BA_BIAS) BA_F4(BA_K) BA_F4(BA_R) BA_F4(BA_RSCALE) BA_F4(BA_QH1) BA_F4(BA_QH2) BA_F4(BA_QH3) BA_F4(BA_QH4) BA_F4(BA_REC) BA_F4(BA_FIN)}for(;i<n;i++)out[i]=exp(in[i]);}

/* Gather-free + parallel Q. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ba_reg_estrin_u4(double*restrict out,const double*restrict in,size_t n){BA_CONSTS BA_REGCONSTS size_t i=0;for(;i+32<=n;i+=32){BA_F4(BA_LOAD) BA_F4(BA_BIAS) BA_F4(BA_K) BA_F4(BA_R) BA_F4(BA_RSCALE) BA_F4(BA_QS1) BA_F4(BA_QS2) BA_F4(BA_REC) BA_F4(BA_FIN)}for(;i<n;i++)out[i]=exp(in[i]);}

/* Manual cross-domain schedule: integer scale work for later groups is issued while
   FP polynomial work for earlier groups is in flight. Four groups, short lifetimes. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ba_manual_u4(double*restrict out,const double*restrict in,size_t n){BA_CONSTS size_t i=0;for(;i+32<=n;i+=32){
 BA_LOAD(0) BA_LOAD(1) BA_BIAS(0) BA_LOAD(2) BA_K(0) BA_BIAS(1) BA_LOAD(3) BA_R(0) BA_K(1) BA_BIAS(2) BA_FSCALE(0) BA_R(1) BA_K(2) BA_BIAS(3) BA_QS1(0) BA_FSCALE(1) BA_R(2) BA_K(3) BA_QS2(0) BA_QS1(1) BA_FSCALE(2) BA_R(3) BA_REC(0) BA_QS2(1) BA_QS1(2) BA_FSCALE(3) BA_FIN(0) BA_REC(1) BA_QS2(2) BA_QS1(3) BA_FIN(1) BA_REC(2) BA_QS2(3) BA_FIN(2) BA_REC(3) BA_FIN(3)
 }for(;i<n;i++)out[i]=exp(in[i]);}

/* Rotating/deep pipeline approximation: six groups in flight, scales issued early,
   Q stages staggered and groups retired in pairs to cap register pressure. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ba_rotpipe_u6(double*restrict out,const double*restrict in,size_t n){BA_CONSTS size_t i=0;for(;i+48<=n;i+=48){
 BA_F6(BA_LOAD) BA_F6(BA_BIAS) BA_F6(BA_K)
 BA_R(0) BA_FSCALE(0) BA_R(1) BA_FSCALE(1) BA_QS1(0) BA_R(2) BA_FSCALE(2) BA_QS1(1) BA_R(3) BA_FSCALE(3) BA_QS2(0) BA_QS1(2) BA_R(4) BA_FSCALE(4) BA_QS2(1) BA_QS1(3) BA_R(5) BA_FSCALE(5)
 BA_REC(0) BA_QS2(2) BA_QS1(4) BA_FIN(0) BA_REC(1) BA_QS2(3) BA_QS1(5) BA_FIN(1)
 BA_REC(2) BA_QS2(4) BA_FIN(2) BA_REC(3) BA_QS2(5) BA_FIN(3) BA_REC(4) BA_FIN(4) BA_REC(5) BA_FIN(5)
 }for(;i<n;i++)out[i]=exp(in[i]);}

/* ALL-combo: gather-free scale + parallel Q + pairwise lifetime reduction. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ba_allcombo_u4(double*restrict out,const double*restrict in,size_t n){BA_CONSTS BA_REGCONSTS size_t i=0;for(;i+32<=n;i+=32){
 BA_F4(BA_LOAD) BA_F4(BA_BIAS) BA_F4(BA_K) BA_F4(BA_R) BA_F4(BA_RSCALE)
 BA_QS1(0) BA_QS1(1) BA_QS2(0) BA_QS2(1) BA_REC(0) BA_REC(1) BA_FIN(0) BA_FIN(1)
 BA_QS1(2) BA_QS1(3) BA_QS2(2) BA_QS2(3) BA_REC(2) BA_REC(3) BA_FIN(2) BA_FIN(3)
 }for(;i<n;i++)out[i]=exp(in[i]);}
