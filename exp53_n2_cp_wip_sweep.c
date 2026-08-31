/* Critical-path WIP sweep after CP64 exposed one ZMM stack spill.
   Tests 5, 6, and 7 simultaneous AVX-512 vectors (40/48/56 doubles).
   Frozen baseline and CP64 are included read-only.
*/
#include "exp53_n2_cp64_attack.c"

#define DEFINE_CP_WIP(NAME,K) \
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
    for(; i+8*(K)<=n; i+=8*(K)){ \
        __m512d x[K],biased[K],k[K],r[K],scale[K],h[K]; \
        __m512i kn[K],j[K],q[K],tb[K],sb[K]; \
        for(int L=0;L<K;L++) x[L]=_mm512_loadu_pd(in+i+8*L); \
        for(int L=0;L<K;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic); \
        for(int L=0;L<K;L++){ \
            k[L]=_mm512_sub_pd(biased[L],magic); \
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); \
            j[L]=_mm512_and_epi64(kn[L],mask); \
            q[L]=_mm512_srai_epi64(kn[L],7); \
        } \
        for(int L=0;L<K;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
        for(int L=0;L<K;L++) r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); \
        for(int L=0;L<K;L++) r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); \
        for(int L=0;L<K;L++) r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); \
        for(int L=0;L<K;L++){ \
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); \
            scale[L]=_mm512_castsi512_pd(sb[L]); \
        } \
        for(int L=0;L<K;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
        for(int L=0;L<K;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
        for(int L=0;L<K;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
        for(int L=0;L<K;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
        for(int G=0;G<K;G+=4){ \
            int M=(K-G<4)?(K-G):4; \
            __m512d s[4],er[4],el[4],ph[4],y[4]; \
            for(int t=0;t<M;t++) s[t]=_mm512_mul_pd(h[G+t],h[G+t]); \
            for(int t=0;t<M;t++) er[t]=_mm512_fmadd_pd(r[G+t],s[t],one); \
            for(int t=0;t<M;t++) el[t]=_mm512_fmadd_pd(r[G+t],s[t],_mm512_sub_pd(one,er[t])); \
            for(int t=0;t<M;t++) ph[t]=_mm512_mul_pd(er[t],scale[G+t]); \
            for(int t=0;t<M;t++){ \
                y[t]=_mm512_fmadd_pd(el[t],scale[G+t],ph[t]); \
                _mm512_storeu_pd(out+i+8*(G+t),y[t]); \
            } \
        } \
    } \
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i); \
}

DEFINE_CP_WIP(exp53_n2_cp40_attack,5)
DEFINE_CP_WIP(exp53_n2_cp48_attack,6)
DEFINE_CP_WIP(exp53_n2_cp56_attack,7)
