/* ALL infrastructure sweep. Frozen compfastC is untouched.
   Tests explicit cross-vector software pipelining, unroll 2/4/6/8,
   gather-free register-permute tables, and compensated register-table products. */
#include "exp53_spine_v128_formula6_compfast.c"

static const double REG_A[16] = {
  0x1.0000000000000p+0,0x1.0b5586cf9890fp+0,0x1.172b83c7d517bp+0,0x1.2387a6e756238p+0,
  0x1.306fe0a31b715p+0,0x1.3dea64c123422p+0,0x1.4bfdad5362a27p+0,0x1.5ab07dd485429p+0,
  0x1.6a09e667f3bcdp+0,0x1.7a11473eb0187p+0,0x1.8ace5422aa0dbp+0,0x1.9c49182a3f090p+0,
  0x1.ae89f995ad3adp+0,0x1.c199bdd85529cp+0,0x1.d5818dcfba487p+0,0x1.ea4afa2a490dap+0
};
static const double REG_B[8] = {
  0x1.0000000000000p+0,0x1.0163da9fb3335p+0,0x1.02c9a3e778061p+0,0x1.04315e86e7f85p+0,
  0x1.059b0d3158574p+0,0x1.0706b29ddf6dep+0,0x1.0874518759bc8p+0,0x1.09e3ecac6f383p+0
};

#define F2(M) M(0) M(1)
#define F4(M) F2(M) M(2) M(3)
#define F6(M) F4(M) M(4) M(5)
#define F8(M) F6(M) M(6) M(7)

#define S_LOAD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define S_BIAS(L) __m512d bz##L=_mm512_fmadd_pd(x##L,inv,magic);
#define S_K(L) __m512d k##L=_mm512_sub_pd(bz##L,magic); __m512i kn##L=_mm512_sub_epi64(_mm512_castpd_si512(bz##L),mb);
#define S_RH(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L);
#define S_RM(L) r##L=_mm512_fnmadd_pd(k##L,mi,r##L);
#define S_RL(L) r##L=_mm512_fnmadd_pd(k##L,lo,r##L);
#define S_T(L) __m512d t##L=_mm512_mul_pd(r##L,r##L);
#define S_P(L) __m512d pc##L=_mm512_fmadd_pd(A1,t##L,A0), ps##L=_mm512_fmadd_pd(B1,t##L,B0);
#define S_CS(L) __m512d c##L=_mm512_mul_pd(t##L,pc##L), ss##L=_mm512_mul_pd(r##L,ps##L);
#define S_TS(L) __m512d ts##L=_mm512_add_pd(ss##L,ss##L);
#define S_A(L) __m512d ah##L=_mm512_add_pd(one,ts##L); __m512d al##L=_mm512_sub_pd(ts##L,_mm512_sub_pd(ah##L,one));
#define S_INNER(L) __m512d inner##L=_mm512_add_pd(two,_mm512_add_pd(c##L,ss##L)), tc##L=_mm512_add_pd(c##L,c##L);
#define S_BH(L) __m512d bh##L=_mm512_mul_pd(tc##L,inner##L);
#define S_BL(L) __m512d bl##L=_mm512_fmadd_pd(tc##L,inner##L,_mm512_sub_pd(_mm512_setzero_pd(),bh##L));
#define S_ER(L) __m512d erh##L=_mm512_add_pd(ah##L,bh##L); __m512d sl##L=_mm512_sub_pd(bh##L,_mm512_sub_pd(erh##L,ah##L)); __m512d erl##L=_mm512_add_pd(_mm512_add_pd(sl##L,al##L),bl##L);
#define S_JQ(L) __m512i jj##L=_mm512_and_epi64(kn##L,mask), q##L=_mm512_srai_epi64(kn##L,7);
#define S_GATHER(L) __m512d th##L=_mm512_i64gather_pd(jj##L,TAB128,8);
#define S_PH(L) __m512d ph##L=_mm512_mul_pd(erh##L,th##L);
#define S_CR(L) __m512d cr##L=_mm512_fmadd_pd(erh##L,th##L,_mm512_sub_pd(_mm512_setzero_pd(),ph##L));
#define S_CRL(L) cr##L=_mm512_fmadd_pd(erl##L,th##L,cr##L);
#define S_Y(L) __m512d y##L=_mm512_mul_pd(_mm512_add_pd(ph##L,cr##L),exp2_from_q(q##L));
#define S_STORE(L) _mm512_storeu_pd(out+i+8*(L),y##L);

#define COMMON_CONSTS \
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC); \
 const __m512d one=_mm512_set1_pd(1.0),two=_mm512_set1_pd(2.0),A0=_mm512_set1_pd(F6_A0),A1=_mm512_set1_pd(F6_A1),B0=_mm512_set1_pd(F6_B0),B1=_mm512_set1_pd(F6_B1); \
 const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);

#define PIPE_STAGES(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(S_P) F(S_CS) F(S_TS) F(S_A) F(S_INNER) F(S_BH) F(S_BL) F(S_ER) F(S_JQ) F(S_GATHER) F(S_PH) F(S_CR) F(S_CRL) F(S_Y) F(S_STORE)

#define DEF_PIPE(NAME,U,F) \
__attribute__((target("avx512f,avx512dq,fma"))) \
void NAME(double *restrict out,const double *restrict in,size_t n){ COMMON_CONSTS size_t i=0; for(;i+8*(U)<=n;i+=8*(U)){ PIPE_STAGES(F) } if(i<n) exp53_spine_v128_formula6_compfastC(out+i,in+i,n-i); }

DEF_PIPE(exp53_all_pipe_u2,2,F2)
DEF_PIPE(exp53_all_pipe_u4,4,F4)
DEF_PIPE(exp53_all_pipe_u6,6,F6)
DEF_PIPE(exp53_all_pipe_u8,8,F8)

/* Register table: j=8a+b; select A[a] from two ZMM registers and B[b] from one ZMM. */
#define R_AB(L) __m512i aa##L=_mm512_srli_epi64(jj##L,3), bb##L=_mm512_and_epi64(jj##L,seven), ai##L=_mm512_and_epi64(aa##L,seven);
#define R_PERM(L) __m512d ta0_##L=_mm512_permutexvar_pd(ai##L,TA0), ta1_##L=_mm512_permutexvar_pd(ai##L,TA1), tb##L=_mm512_permutexvar_pd(bb##L,TB);
#define R_SEL(L) __mmask8 mk##L=_mm512_cmp_epi64_mask(aa##L,eight,_MM_CMPINT_GE); __m512d ta##L=_mm512_mask_blend_pd(mk##L,ta0_##L,ta1_##L);
#define R_TH(L) __m512d th##L=_mm512_mul_pd(ta##L,tb##L);
#define R_TL(L) __m512d tl##L=_mm512_fmadd_pd(ta##L,tb##L,_mm512_sub_pd(_mm512_setzero_pd(),th##L));
#define R_CRT(L) cr##L=_mm512_fmadd_pd(erh##L,tl##L,cr##L);

#define REG_CONSTS const __m512d TA0=_mm512_loadu_pd(REG_A),TA1=_mm512_loadu_pd(REG_A+8),TB=_mm512_loadu_pd(REG_B); const __m512i seven=_mm512_set1_epi64(7),eight=_mm512_set1_epi64(8);

#define REG_STAGES_FAST(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(S_P) F(S_CS) F(S_TS) F(S_A) F(S_INNER) F(S_BH) F(S_BL) F(S_ER) F(S_JQ) F(R_AB) F(R_PERM) F(R_SEL) F(R_TH) F(S_PH) F(S_CR) F(S_CRL) F(S_Y) F(S_STORE)
#define REG_STAGES_COMP(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(S_P) F(S_CS) F(S_TS) F(S_A) F(S_INNER) F(S_BH) F(S_BL) F(S_ER) F(S_JQ) F(R_AB) F(R_PERM) F(R_SEL) F(R_TH) F(R_TL) F(S_PH) F(S_CR) F(R_CRT) F(S_CRL) F(S_Y) F(S_STORE)

#define DEF_REG(NAME,U,F,STAGES) \
__attribute__((target("avx512f,avx512dq,fma"))) \
void NAME(double *restrict out,const double *restrict in,size_t n){ COMMON_CONSTS REG_CONSTS size_t i=0; for(;i+8*(U)<=n;i+=8*(U)){ STAGES(F) } if(i<n) exp53_spine_v128_formula6_compfastC(out+i,in+i,n-i); }

DEF_REG(exp53_all_regtab_fast_u4,4,F4,REG_STAGES_FAST)
DEF_REG(exp53_all_regtab_comp_u4,4,F4,REG_STAGES_COMP)
DEF_REG(exp53_all_regtab_comp_u8,8,F8,REG_STAGES_COMP)
