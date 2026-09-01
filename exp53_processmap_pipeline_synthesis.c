/* EXP53 process-map synthesis experiment.
   IMPORTANT: every candidate is derived from the frozen VM-style kernel.
   Math, constants, operation order inside each lane, early-gather policy,
   final reconstruction, and frozen tail are unchanged.  Only the number of
   independent 8-lane vectors kept in flight is varied to test whether the
   process-map gather/FMA balance prefers a pipeline width other than u4.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#define DEFINE_PM_WIDTH(FN,U) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void FN(double *restrict out,const double *restrict in,size_t n){ \
    const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI), \
      mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO), \
      magic=_mm512_set1_pd(N2F_MAGIC), one=_mm512_set1_pd(1.0), \
      nq1=_mm512_set1_pd(N2F_Q1), nq2=_mm512_set1_pd(N2F_Q2), \
      nq3=_mm512_set1_pd(N2F_Q3), nq4=_mm512_set1_pd(N2F_Q4); \
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), mask=_mm512_set1_epi64(127); \
    size_t i=0; \
    for(;i+(size_t)(8*(U))<=n;i+=(size_t)(8*(U))){ \
      __m512d x[U],biased[U],k[U],r[U],h[U],s[U],er[U],el[U],scale[U],ph[U],y[U]; \
      __m512i kn[U],j[U],q[U],tb[U],sb[U]; \
      for(int L=0;L<(U);L++) x[L]=_mm512_loadu_pd(in+i+8*L); \
      for(int L=0;L<(U);L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic); \
      for(int L=0;L<(U);L++){ \
        k[L]=_mm512_sub_pd(biased[L],magic); \
        kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); \
        j[L]=_mm512_and_epi64(kn[L],mask); \
        q[L]=_mm512_srai_epi64(kn[L],7); \
      } \
      /* Preserve the frozen kernel's key latency-hiding decision: all gathers launch early. */ \
      for(int L=0;L<(U);L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
      for(int L=0;L<(U);L++){ \
        r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); \
        r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); \
        r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); \
      } \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
      /* These are exactly the process-map winning machine operations: vmulpd + vfmadd. */ \
      for(int L=0;L<(U);L++) s[L]=_mm512_mul_pd(h[L],h[L]); \
      for(int L=0;L<(U);L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one); \
      for(int L=0;L<(U);L++) el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L])); \
      for(int L=0;L<(U);L++){ \
        sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); \
        scale[L]=_mm512_castsi512_pd(sb[L]); \
      } \
      for(int L=0;L<(U);L++){ \
        ph[L]=_mm512_mul_pd(er[L],scale[L]); \
        y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]); \
        _mm512_storeu_pd(out+i+8*L,y[L]); \
      } \
    } \
    if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i); \
}

DEFINE_PM_WIDTH(exp53_pm_u3,3)
DEFINE_PM_WIDTH(exp53_pm_u4,4)
DEFINE_PM_WIDTH(exp53_pm_u5,5)
DEFINE_PM_WIDTH(exp53_pm_u6,6)

/* A second u4 schedule tests whether issuing reduction immediately after each
   gather is better than the frozen all-gathers-first schedule.  Arithmetic is
   unchanged; only independent-operation ordering changes. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_pm_u4_stagger(double *restrict out,const double *restrict in,size_t n){
    const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI),
      mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO),
      magic=_mm512_set1_pd(N2F_MAGIC), one=_mm512_set1_pd(1.0),
      nq1=_mm512_set1_pd(N2F_Q1), nq2=_mm512_set1_pd(N2F_Q2),
      nq3=_mm512_set1_pd(N2F_Q3), nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+32<=n;i+=32){
      __m512d x[4],biased[4],k[4],r[4],h[4],s[4],er[4],el[4],scale[4],ph[4],y[4];
      __m512i kn[4],j[4],q[4],tb[4],sb[4];
      for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
      for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
      for(int L=0;L<4;L++){
        k[L]=_mm512_sub_pd(biased[L],magic);
        kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);
        j[L]=_mm512_and_epi64(kn[L],mask);
        q[L]=_mm512_srai_epi64(kn[L],7);
      }
      for(int L=0;L<4;L++){
        tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
        r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
        r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);
        r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);
      }
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);
      for(int L=0;L<4;L++) s[L]=_mm512_mul_pd(h[L],h[L]);
      for(int L=0;L<4;L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one);
      for(int L=0;L<4;L++) el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));
      for(int L=0;L<4;L++){
        sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
        scale[L]=_mm512_castsi512_pd(sb[L]);
      }
      for(int L=0;L<4;L++){
        ph[L]=_mm512_mul_pd(er[L],scale[L]);
        y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]);
        _mm512_storeu_pd(out+i+8*L,y[L]);
      }
    }
    if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i);
}
