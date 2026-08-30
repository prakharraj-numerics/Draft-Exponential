/* EXP residual polynomial derived algebraically from the C-spine.
   C(r/2)=t P(t), t=r^2; S=4r(P+tP');
   e^r=1+4C+2C^2+2S(1+C)=1+r A(t)+t B(t).
   Truncating the C-derived A,B through t^2 gives
     A=1+t/6+t^2/120, B=1/2+t/24+t^2/720.
   No independently fitted exp polynomial is introduced here.
*/
#include "exp53_spine_v128_formula6_finalinfra.c"

#define SP_A2 0x1.1111111111111p-7   /* 1/120 */
#define SP_A1 0x1.5555555555555p-3   /* 1/6   */
#define SP_B2 0x1.6c16c16c16c17p-10  /* 1/720 */
#define SP_B1 0x1.5555555555555p-5   /* 1/24  */
#define SP_B0 0x1.0000000000000p-1   /* 1/2   */

#define SP_CONSTS \
 const __m512d spa2=_mm512_set1_pd(SP_A2),spa1=_mm512_set1_pd(SP_A1),spb2=_mm512_set1_pd(SP_B2),spb1=_mm512_set1_pd(SP_B1),spb0=_mm512_set1_pd(SP_B0);

#define SP_AB1(L) __m512d ap##L=_mm512_fmadd_pd(spa2,t##L,spa1), bp##L=_mm512_fmadd_pd(spb2,t##L,spb1);
#define SP_AB2(L) ap##L=_mm512_fmadd_pd(ap##L,t##L,one); bp##L=_mm512_fmadd_pd(bp##L,t##L,spb0);
#define SP_Y0(L) __m512d ey##L=_mm512_fmadd_pd(r##L,ap##L,one);
#define SP_ER(L) __m512d erp##L=_mm512_fmadd_pd(t##L,bp##L,ey##L);
#define SP_PH(L) __m512d phsp##L=_mm512_mul_pd(erp##L,th##L);
#define SP_CR(L) __m512d crsp##L=_mm512_fmadd_pd(erp##L,th##L,_mm512_sub_pd(_mm512_setzero_pd(),phsp##L));
#define SP_SCALE(L) __m512d scsp##L=exp2_from_q(q##L);
#define SP_STORE_COMP(L) _mm512_storeu_pd(out+i+8*(L),_mm512_mul_pd(_mm512_add_pd(phsp##L,crsp##L),scsp##L));
#define SP_STORE_PLAIN(L) _mm512_storeu_pd(out+i+8*(L),_mm512_mul_pd(phsp##L,scsp##L));

/* Issue gather after the first A/B stage so memory latency overlaps the second A/B stage and reconstruction. */
#define SP_MID_COMP(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(SP_AB1) F(E_QJTAB) F(SP_SCALE) F(SP_AB2) F(SP_Y0) F(SP_ER) F(SP_PH) F(SP_CR) F(SP_STORE_COMP)
#define SP_MID_PLAIN(F) F(S_LOAD) F(S_BIAS) F(S_K) F(S_RH) F(S_RM) F(S_RL) F(S_T) F(SP_AB1) F(E_QJTAB) F(SP_SCALE) F(SP_AB2) F(SP_Y0) F(SP_ER) F(SP_PH) F(SP_STORE_PLAIN)

#define DEF_SP(NAME,U,F,STAGES) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void NAME(double *restrict out,const double *restrict in,size_t n){ COMMON_CONSTS SP_CONSTS size_t i=0; for(;i+8*(U)<=n;i+=8*(U)){ STAGES(F) } if(i<n) exp53_spine_v128_formula6_compfastC(out+i,in+i,n-i); }

DEF_SP(exp53_spinepoly_mid_u4,4,F4,SP_MID_COMP)
DEF_SP(exp53_spinepoly_mid_u5,5,F5,SP_MID_COMP)
DEF_SP(exp53_spinepoly_mid_u6,6,F6,SP_MID_COMP)
DEF_SP(exp53_spinepoly_plain_u6,6,F6,SP_MID_PLAIN)
