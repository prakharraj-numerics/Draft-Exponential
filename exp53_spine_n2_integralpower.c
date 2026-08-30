/* Faithful n=2 power-of-integral EXP spine.

   e^r - 1 = r Q(r)^2,
   Q(r) = sum_{k>=0} c_k(2) r^k/(2k+1),
   e^z sqrt(z/(e^z-1)) = sum c_k(2) z^k.

   Degree-4 faithful truncation on |r| <= ln(2)/256:
     Q4(r)=1+r/4+5r^2/96+r^3/128+79r^4/92160.

   IMPORTANT: the square reconstruction is deliberately preserved.  This file does
   not expand 1+r Q4(r)^2 into an ordinary exp polynomial.
*/
#include "exp53_spine_v128_formula6_finalinfra.c"

#define N2_Q1 0x1.0000000000000p-2
#define N2_Q2 0x1.aaaaaaaaaaaabp-5
#define N2_Q3 0x1.0000000000000p-7
#define N2_Q4 0x1.c16c16c16c16cp-11

#define N2_CONSTS \
 const __m512d nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2), \
               nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4);

#define N2_H1(L) __m512d nh##L=_mm512_fmadd_pd(nq4,r##L,nq3);
#define N2_H2(L) nh##L=_mm512_fmadd_pd(nh##L,r##L,nq2);
#define N2_H3(L) nh##L=_mm512_fmadd_pd(nh##L,r##L,nq1);
#define N2_H4(L) nh##L=_mm512_fmadd_pd(nh##L,r##L,one);
#define N2_SQ(L) __m512d ns##L=_mm512_mul_pd(nh##L,nh##L);
#define N2_ER(L) __m512d neh##L=_mm512_fmadd_pd(r##L,ns##L,one);
#define N2_ERLOW(L) __m512d nel##L=_mm512_fmadd_pd(r##L,ns##L,_mm512_sub_pd(one,neh##L));
#define N2_PH(L) __m512d nph##L=_mm512_mul_pd(neh##L,th##L);
#define N2_PERR(L) __m512d npl##L=_mm512_fmadd_pd(neh##L,th##L,_mm512_sub_pd(_mm512_setzero_pd(),nph##L));
#define N2_PERRL(L) npl##L=_mm512_fmadd_pd(nel##L,th##L,npl##L);
#define N2_YPLAIN(L) __m512d ny##L=_mm512_mul_pd(nph##L,sc##L);
#define N2_YCOMP(L) __m512d ny##L=_mm512_mul_pd(_mm512_add_pd(nph##L,npl##L),sc##L);
#define N2_STORE(L) _mm512_storeu_pd(out+i+8*(L),ny##L);

/* Gather immediately after k is available; hide it under reduction + Horner. */
#define N2_FAST_STAGES(F) F(S_LOAD) F(S_BIAS) F(S_K) F(E_QJTAB) F(E_SCALE) F(S_RH) F(S_RM) F(S_RL) F(N2_H1) F(N2_H2) F(N2_H3) F(N2_H4) F(N2_SQ) F(N2_ER) F(N2_PH) F(N2_YPLAIN) F(N2_STORE)
#define N2_PC_STAGES(F)   F(S_LOAD) F(S_BIAS) F(S_K) F(E_QJTAB) F(E_SCALE) F(S_RH) F(S_RM) F(S_RL) F(N2_H1) F(N2_H2) F(N2_H3) F(N2_H4) F(N2_SQ) F(N2_ER) F(N2_PH) F(N2_PERR) F(N2_YCOMP) F(N2_STORE)
#define N2_RC_STAGES(F)   F(S_LOAD) F(S_BIAS) F(S_K) F(E_QJTAB) F(E_SCALE) F(S_RH) F(S_RM) F(S_RL) F(N2_H1) F(N2_H2) F(N2_H3) F(N2_H4) F(N2_SQ) F(N2_ER) F(N2_ERLOW) F(N2_PH) F(N2_PERR) F(N2_PERRL) F(N2_YCOMP) F(N2_STORE)

/* Single-vector masked tail, same n=2 spine. */
static __attribute__((target("avx512f,avx512dq,fma"),always_inline)) inline
void n2_tail(double *restrict out,const double *restrict in,size_t n,int mode)
{
    COMMON_CONSTS N2_CONSTS
    while(n){
        unsigned c=n>=8?8:(unsigned)n; __mmask8 km=(__mmask8)((1u<<c)-1u);
        __m512d x0=_mm512_maskz_loadu_pd(km,in);
        __m512d bz0=_mm512_fmadd_pd(x0,inv,magic);
        __m512d k0=_mm512_sub_pd(bz0,magic);
        __m512i kn0=_mm512_sub_epi64(_mm512_castpd_si512(bz0),mb);
        __m512i jj0=_mm512_and_epi64(kn0,mask), q0=_mm512_srai_epi64(kn0,7);
        __m512d th0=_mm512_i64gather_pd(jj0,TAB128,8), sc0=exp2_from_q(q0);
        __m512d r0=_mm512_fnmadd_pd(k0,hi,x0);
        r0=_mm512_fnmadd_pd(k0,mi,r0); r0=_mm512_fnmadd_pd(k0,lo,r0);
        __m512d h0=_mm512_fmadd_pd(nq4,r0,nq3);
        h0=_mm512_fmadd_pd(h0,r0,nq2); h0=_mm512_fmadd_pd(h0,r0,nq1); h0=_mm512_fmadd_pd(h0,r0,one);
        __m512d s0=_mm512_mul_pd(h0,h0), eh0=_mm512_fmadd_pd(r0,s0,one);
        __m512d ph0=_mm512_mul_pd(eh0,th0), yy;
        if(mode==0) yy=_mm512_mul_pd(ph0,sc0);
        else {
            __m512d pl0=_mm512_fmadd_pd(eh0,th0,_mm512_sub_pd(_mm512_setzero_pd(),ph0));
            if(mode==2){
                __m512d el0=_mm512_fmadd_pd(r0,s0,_mm512_sub_pd(one,eh0));
                pl0=_mm512_fmadd_pd(el0,th0,pl0);
            }
            yy=_mm512_mul_pd(_mm512_add_pd(ph0,pl0),sc0);
        }
        _mm512_mask_storeu_pd(out,km,yy);
        in+=c; out+=c; n-=c;
    }
}

#define DEF_N2(NAME,U,F,STAGES,MODE) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void NAME(double *restrict out,const double *restrict in,size_t n){ \
    COMMON_CONSTS N2_CONSTS size_t i=0; \
    for(;i+8*(U)<=n;i+=8*(U)){ STAGES(F) } \
    if(i<n) n2_tail(out+i,in+i,n-i,MODE); \
}

DEF_N2(exp53_n2_fast_u2,2,F2,N2_FAST_STAGES,0)
DEF_N2(exp53_n2_fast_u4,4,F4,N2_FAST_STAGES,0)
DEF_N2(exp53_n2_fast_u6,6,F6,N2_FAST_STAGES,0)
DEF_N2(exp53_n2_pc_u2,2,F2,N2_PC_STAGES,1)
DEF_N2(exp53_n2_pc_u4,4,F4,N2_PC_STAGES,1)
DEF_N2(exp53_n2_pc_u6,6,F6,N2_PC_STAGES,1)
DEF_N2(exp53_n2_rc_u4,4,F4,N2_RC_STAGES,2)
DEF_N2(exp53_n2_rc_u6,6,F6,N2_RC_STAGES,2)
