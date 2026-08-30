/* Final infrastructure assault: accuracy-safe C baseline plus explicit latency-hiding schedules.
   Mathematical path unchanged. Main experiment: issue TAB128 gather as soon as j is available,
   then execute residual/compensation while gather latency is outstanding. Also test a mid-gather
   placement to trade latency hiding against register pressure. */
#include "exp53_spine_v128_formula6_allinfra2.c"

/* q/j become available immediately after S_K. */
#define E_QJTAB(L) __m512i jj##L=_mm512_and_epi64(kn##L,mask), q##L=_mm512_srai_epi64(kn##L,7); __m512d th##L=_mm512_i64gather_pd(jj##L,TAB128,8);
#define E_SCALE(L) __m512d sc##L=exp2_from_q(q##L);
#define E_Y(L) __m512d y##L=_mm512_mul_pd(_mm512_add_pd(ph##L,cr##L),sc##L);

#define EARLY_STAGES(F) F(S_LOAD) F(S_BIAS) F(S_K) F(E_QJTAB) F(E_SCALE) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(S_P) F(S_CS) F(S_TS) F(S_A) F(S_INNER) F(S_BH) F(S_BL) F(S_ER) F(S_PH) F(S_CR) F(S_CRL) F(E_Y) F(S_STORE)
#define MID_STAGES(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(S_P) F(E_QJTAB) F(E_SCALE) F(S_CS) F(S_TS) F(S_A) F(S_INNER) F(S_BH) F(S_BL) F(S_ER) F(S_PH) F(S_CR) F(S_CRL) F(E_Y) F(S_STORE)

#define DEF_SCHED(NAME,U,F,STAGES) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void NAME(double *restrict out,const double *restrict in,size_t n){ COMMON_CONSTS size_t i=0; for(;i+8*(U)<=n;i+=8*(U)){ STAGES(F) } if(i<n) exp53_spine_v128_formula6_compfastC(out+i,in+i,n-i); }

DEF_SCHED(exp53_final_early_u4,4,F4,EARLY_STAGES)
DEF_SCHED(exp53_final_early_u5,5,F5,EARLY_STAGES)
DEF_SCHED(exp53_final_early_u6,6,F6,EARLY_STAGES)
DEF_SCHED(exp53_final_mid_u4,4,F4,MID_STAGES)
DEF_SCHED(exp53_final_mid_u5,5,F5,MID_STAGES)
DEF_SCHED(exp53_final_mid_u6,6,F6,MID_STAGES)

/* A pressure-reduced split-u6 schedule: preserve six-way overlap for reduction, then complete
   compensation/table stages as two groups of three. This intentionally trades some ILP for fewer
   simultaneously-live temporaries. */
#define PRE6 F6(S_LOAD) F6(S_BIAS) F6(S_K) F6(S_RH) F6(S_RM) F6(S_RL) F6(S_T) F6(S_P) F6(S_CS)
#define POST3A F4(S_TS) /* labels 0..3; label3 intentionally completed here */

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_final_split_u6(double *restrict out,const double *restrict in,size_t n)
{
    COMMON_CONSTS
    size_t i=0;
    for(; i+48<=n; i+=48){
        F6(S_LOAD) F6(S_BIAS) F6(S_K) F6(S_RH) F6(S_RM) F6(S_RL) F6(S_T) F6(S_P) F6(S_CS)
        /* complete lanes 0..2 */
        S_TS(0) S_TS(1) S_TS(2) S_A(0) S_A(1) S_A(2) S_INNER(0) S_INNER(1) S_INNER(2)
        S_BH(0) S_BH(1) S_BH(2) S_BL(0) S_BL(1) S_BL(2) S_ER(0) S_ER(1) S_ER(2)
        S_JQ(0) S_JQ(1) S_JQ(2) S_GATHER(0) S_GATHER(1) S_GATHER(2)
        S_PH(0) S_PH(1) S_PH(2) S_CR(0) S_CR(1) S_CR(2) S_CRL(0) S_CRL(1) S_CRL(2)
        S_Y(0) S_Y(1) S_Y(2) S_STORE(0) S_STORE(1) S_STORE(2)
        /* complete lanes 3..5 */
        S_TS(3) S_TS(4) S_TS(5) S_A(3) S_A(4) S_A(5) S_INNER(3) S_INNER(4) S_INNER(5)
        S_BH(3) S_BH(4) S_BH(5) S_BL(3) S_BL(4) S_BL(5) S_ER(3) S_ER(4) S_ER(5)
        S_JQ(3) S_JQ(4) S_JQ(5) S_GATHER(3) S_GATHER(4) S_GATHER(5)
        S_PH(3) S_PH(4) S_PH(5) S_CR(3) S_CR(4) S_CR(5) S_CRL(3) S_CRL(4) S_CRL(5)
        S_Y(3) S_Y(4) S_Y(5) S_STORE(3) S_STORE(4) S_STORE(5)
    }
    if(i<n) exp53_spine_v128_formula6_compfastC(out+i,in+i,n-i);
}
