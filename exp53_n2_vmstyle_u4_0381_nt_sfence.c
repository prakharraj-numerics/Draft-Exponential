/* NON-TEMPORAL STORE CACHE ATTACK — candidate only. DO NOT modify frozen files.

   Mathematical body and operation order are copied from
   exp53_n2_vmstyle_u4_0381_frozen. The only intended hot-path change is:
     temporal _mm512_storeu_pd -> non-temporal _mm512_stream_pd
   plus one final _mm_sfence() for function-return ordering semantics.

   Production-safe alignment handling:
   - if out is not 64-byte aligned, a short temporal prefix is computed by the
     frozen function until the remaining output pointer is 64-byte aligned;
   - full 32-value blocks then use streaming stores;
   - one SFENCE is issued after the NT block sequence;
   - the final remainder uses the frozen temporal implementation.
*/
#include <stdint.h>
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_vmstyle_u4_0381_nt_sfence(double *restrict out,
                                        const double *restrict in,
                                        size_t n)
{
    if (!n) return;

    uintptr_t a=(uintptr_t)out;
    if ((a & 7u) != 0u) {
        exp53_n2_vmstyle_u4_0381_frozen(out,in,n);
        return;
    }

    size_t prefix=0;
    const size_t mis=(size_t)(a & 63u);
    if (mis) {
        prefix=(64u-mis)/sizeof(double);
        if (prefix>n) prefix=n;
        exp53_n2_vmstyle_u4_0381_frozen(out,in,prefix);
        out+=prefix; in+=prefix; n-=prefix;
        if (!n) return;
    }

    const __m512d inv=_mm512_set1_pd(N2F_INV128),
                  hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI),
                  lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC),
                  one=_mm512_set1_pd(1.0),
                  nq1=_mm512_set1_pd(N2F_Q1),
                  nq2=_mm512_set1_pd(N2F_Q2),
                  nq3=_mm512_set1_pd(N2F_Q3),
                  nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),
                  mask=_mm512_set1_epi64(127);

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

        for(int L=0;L<4;L++)
            tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);

        for(int L=0;L<4;L++){
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
        for(int L=0;L<4;L++)
            el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));

        for(int L=0;L<4;L++){
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
            scale[L]=_mm512_castsi512_pd(sb[L]);
        }

        for(int L=0;L<4;L++){
            ph[L]=_mm512_mul_pd(er[L],scale[L]);
            y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]);
            _mm512_stream_pd(out+i+8*L,y[L]);
        }
    }

    if (i) _mm_sfence();
    if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i);
}
