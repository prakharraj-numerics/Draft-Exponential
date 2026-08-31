/* Compiler-native software-pipelining attack for the faithful n=2 EXP53 spine.
   Frozen baseline is included read-only and is not modified.

   We duplicate the exact 32-input VM-style body and vary ONLY loop metadata:
     - default/no hint
     - clang software-pipeline enable
     - software-pipeline enable with explicit II=1,2,4
     - outer-loop interleave count 2 and 4

   All arithmetic order, TAB128 anchors, reduction, ER-low correction and
   fused exponent scaling remain identical to exp53_n2_vmstyle_u4_0381_frozen.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#define N2SWP_NONE
#define N2SWP_ENABLE _Pragma("clang loop pipeline(enable)")
#define N2SWP_II1    _Pragma("clang loop pipeline(enable)") _Pragma("clang loop pipeline_initiation_interval(1)")
#define N2SWP_II2    _Pragma("clang loop pipeline(enable)") _Pragma("clang loop pipeline_initiation_interval(2)")
#define N2SWP_II4    _Pragma("clang loop pipeline(enable)") _Pragma("clang loop pipeline_initiation_interval(4)")
#define N2SWP_INT2   _Pragma("clang loop interleave_count(2)")
#define N2SWP_INT4   _Pragma("clang loop interleave_count(4)")

#define DEFINE_N2_SWP(NAME,HINT) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void NAME(double *restrict out,const double *restrict in,size_t n) \
{ \
    const __m512d inv=_mm512_set1_pd(N2F_INV128), \
                  hi=_mm512_set1_pd(N2F_L128_HI), \
                  mi=_mm512_set1_pd(N2F_L128_MI), \
                  lo=_mm512_set1_pd(N2F_L128_LO), \
                  magic=_mm512_set1_pd(N2F_MAGIC), \
                  one=_mm512_set1_pd(1.0), \
                  nq1=_mm512_set1_pd(N2F_Q1), \
                  nq2=_mm512_set1_pd(N2F_Q2), \
                  nq3=_mm512_set1_pd(N2F_Q3), \
                  nq4=_mm512_set1_pd(N2F_Q4); \
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), \
                  mask=_mm512_set1_epi64(127); \
    size_t i=0; \
    HINT \
    for(;i+32<=n;i+=32){ \
        __m512d x[4],biased[4],k[4],r[4],h[4],s[4],er[4],el[4],scale[4],ph[4],y[4]; \
        __m512i kn[4],j[4],q[4],tb[4],sb[4]; \
        for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+8*L); \
        for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic); \
        for(int L=0;L<4;L++){ \
            k[L]=_mm512_sub_pd(biased[L],magic); \
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); \
            j[L]=_mm512_and_epi64(kn[L],mask); \
            q[L]=_mm512_srai_epi64(kn[L],7); \
        } \
        for(int L=0;L<4;L++) \
            tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
        for(int L=0;L<4;L++){ \
            r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); \
            r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); \
            r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); \
        } \
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
        for(int L=0;L<4;L++) s[L]=_mm512_mul_pd(h[L],h[L]); \
        for(int L=0;L<4;L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one); \
        for(int L=0;L<4;L++) \
            el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L])); \
        for(int L=0;L<4;L++){ \
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); \
            scale[L]=_mm512_castsi512_pd(sb[L]); \
        } \
        for(int L=0;L<4;L++){ \
            ph[L]=_mm512_mul_pd(er[L],scale[L]); \
            y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]); \
            _mm512_storeu_pd(out+i+8*L,y[L]); \
        } \
    } \
    if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i); \
}

DEFINE_N2_SWP(exp53_n2_native_default,N2SWP_NONE)
DEFINE_N2_SWP(exp53_n2_native_swp_enable,N2SWP_ENABLE)
DEFINE_N2_SWP(exp53_n2_native_swp_ii1,N2SWP_II1)
DEFINE_N2_SWP(exp53_n2_native_swp_ii2,N2SWP_II2)
DEFINE_N2_SWP(exp53_n2_native_swp_ii4,N2SWP_II4)
DEFINE_N2_SWP(exp53_n2_native_interleave2,N2SWP_INT2)
DEFINE_N2_SWP(exp53_n2_native_interleave4,N2SWP_INT4)
