/* ALL pass 2: register-pressure sweet spot + vpermi2pd register tables.
   Includes pass-1 implementations and adds u5 plus direct two-source table selection.
*/
#include "exp53_spine_v128_formula6_allinfra.c"

#define F5(M) F4(M) M(4)
DEF_PIPE(exp53_all_pipe_u5,5,F5)

static const double REG_A_LO[16] = {
  0x0.0p+0,0x1.8a62e4adc610bp-54,-0x1.19041b9d78a76p-55,0x1.9b07eb6c70573p-54,
  0x1.6f46ad23182e4p-55,0x1.ada0911f09ebcp-55,0x1.d4397afec42e2p-56,0x1.6324c054647adp-54,
  -0x1.bdd3413b26456p-54,-0x1.41577ee04992fp-55,0x1.6e9f156864b27p-54,0x1.c7c46b071f2bep-56,
  0x1.7a1cd345dcc81p-54,0x1.11065895048ddp-55,0x1.2ed02d75b3707p-55,-0x1.e9c23179c2893p-54
};
static const double REG_B_LO[8] = {
  0x0.0p+0,0x1.b61299ab8cdb7p-54,-0x1.19083535b085dp-56,-0x1.0a31c1977c96ep-54,
  0x1.d73e2a475b465p-55,-0x1.c91dfe2b13c27p-55,0x1.186be4bb284ffp-57,0x1.1487818316136p-54
};

#define V2_AB(L) __m512i aa2_##L=_mm512_srli_epi64(jj##L,3), bb2_##L=_mm512_and_epi64(jj##L,seven2);
#define V2_SEL(L) __m512d ta2_##L=_mm512_permutex2var_pd(TA20,aa2_##L,TA21), tb2_##L=_mm512_permutexvar_pd(bb2_##L,TB2);
#define V2_TH(L) __m512d th##L=_mm512_mul_pd(ta2_##L,tb2_##L);
#define V2_TL_FAST(L) __m512d tl##L=_mm512_fmadd_pd(ta2_##L,tb2_##L,_mm512_sub_pd(_mm512_setzero_pd(),th##L));
#define V2_SEL_LO(L) __m512d tal2_##L=_mm512_permutex2var_pd(TAL20,aa2_##L,TAL21), tbl2_##L=_mm512_permutexvar_pd(bb2_##L,TBL2);
#define V2_TL_DD(L) __m512d tl##L=_mm512_fmadd_pd(ta2_##L,tb2_##L,_mm512_sub_pd(_mm512_setzero_pd(),th##L)); tl##L=_mm512_fmadd_pd(tal2_##L,tb2_##L,tl##L); tl##L=_mm512_fmadd_pd(ta2_##L,tbl2_##L,tl##L);

#define V2_CONSTS const __m512d TA20=_mm512_loadu_pd(REG_A),TA21=_mm512_loadu_pd(REG_A+8),TB2=_mm512_loadu_pd(REG_B); const __m512i seven2=_mm512_set1_epi64(7);
#define V2_DD_CONSTS V2_CONSTS const __m512d TAL20=_mm512_loadu_pd(REG_A_LO),TAL21=_mm512_loadu_pd(REG_A_LO+8),TBL2=_mm512_loadu_pd(REG_B_LO);

#define V2_STAGES_FAST(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(S_P) F(S_CS) F(S_TS) F(S_A) F(S_INNER) F(S_BH) F(S_BL) F(S_ER) F(S_JQ) F(V2_AB) F(V2_SEL) F(V2_TH) F(V2_TL_FAST) F(S_PH) F(S_CR) F(R_CRT) F(S_CRL) F(S_Y) F(S_STORE)
#define V2_STAGES_DD(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(S_P) F(S_CS) F(S_TS) F(S_A) F(S_INNER) F(S_BH) F(S_BL) F(S_ER) F(S_JQ) F(V2_AB) F(V2_SEL) F(V2_SEL_LO) F(V2_TH) F(V2_TL_DD) F(S_PH) F(S_CR) F(R_CRT) F(S_CRL) F(S_Y) F(S_STORE)

#define DEF_V2(NAME,U,F,CONST,STAGES) \
__attribute__((target("avx512f,avx512dq,fma"))) \
void NAME(double *restrict out,const double *restrict in,size_t n){ COMMON_CONSTS CONST size_t i=0; for(;i+8*(U)<=n;i+=8*(U)){ STAGES(F) } if(i<n) exp53_spine_v128_formula6_compfastC(out+i,in+i,n-i); }

DEF_V2(exp53_all_v2fast_u4,4,F4,V2_CONSTS,V2_STAGES_FAST)
DEF_V2(exp53_all_v2fast_u5,5,F5,V2_CONSTS,V2_STAGES_FAST)
DEF_V2(exp53_all_v2dd_u4,4,F4,V2_DD_CONSTS,V2_STAGES_DD)
DEF_V2(exp53_all_v2dd_u5,5,F5,V2_DD_CONSTS,V2_STAGES_DD)
