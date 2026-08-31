/* Critical-path wave scheduling attack for faithful n=2 EXP53.
   Frozen baseline is included read-only.

   Design goal: expose 6- and 8-vector independent Horner chains while keeping
   live state compressed to (r,scale) before Q4. This is not the earlier
   PIPE3/PIPE4 conveyor: the steady-state mathematical DAG is deliberately
   widened at its FP-critical section.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#define DEFINE_CRIT_WAVE(NAME,NV,STEP) \
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
    for(;i+(STEP)<=n;i+=(STEP)){ \
        __m512d r[NV],scale[NV],h[NV],s[NV]; \
        /* Phase 1: compress every vector to the two values needed later. */ \
        _Pragma("clang loop unroll(full)") \
        for(int L=0;L<(NV);L++){ \
            __m512d x=_mm512_loadu_pd(in+i+8*(size_t)L); \
            __m512d biased=_mm512_fmadd_pd(x,inv,magic); \
            __m512d k=_mm512_sub_pd(biased,magic); \
            __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb); \
            __m512i j=_mm512_and_epi64(kn,mask); \
            __m512i q=_mm512_srai_epi64(kn,7); \
            __m512i tb=_mm512_i64gather_epi64(j,(const long long*)N2_FROZEN_TAB128,8); \
            __m512d rr=_mm512_fnmadd_pd(k,hi,x); \
            rr=_mm512_fnmadd_pd(k,mi,rr); \
            rr=_mm512_fnmadd_pd(k,lo,rr); \
            r[L]=rr; \
            __m512i sb=_mm512_add_epi64(tb,_mm512_slli_epi64(q,52)); \
            scale[L]=_mm512_castsi512_pd(sb); \
        } \
        /* Phase 2: critical FP wave. NV independent Horner chains. */ \
        _Pragma("clang loop unroll(full)") \
        for(int L=0;L<(NV);L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
        _Pragma("clang loop unroll(full)") \
        for(int L=0;L<(NV);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
        _Pragma("clang loop unroll(full)") \
        for(int L=0;L<(NV);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
        _Pragma("clang loop unroll(full)") \
        for(int L=0;L<(NV);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
        _Pragma("clang loop unroll(full)") \
        for(int L=0;L<(NV);L++) s[L]=_mm512_mul_pd(h[L],h[L]); \
        /* Drain vectors. Keeping ER-low local shortens live ranges and lets the
           register allocator release each vector immediately after its store. */ \
        _Pragma("clang loop unroll(full)") \
        for(int L=0;L<(NV);L++){ \
            __m512d er=_mm512_fmadd_pd(r[L],s[L],one); \
            __m512d el=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er)); \
            __m512d ph=_mm512_mul_pd(er,scale[L]); \
            __m512d y=_mm512_fmadd_pd(el,scale[L],ph); \
            _mm512_storeu_pd(out+i+8*(size_t)L,y); \
        } \
    } \
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i); \
}

DEFINE_CRIT_WAVE(exp53_n2_critical_u6_48,6,48)
DEFINE_CRIT_WAVE(exp53_n2_critical_u8_64,8,64)
