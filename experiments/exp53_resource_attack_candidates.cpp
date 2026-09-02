/* EXPERIMENT ONLY — resource-efficiency attacks on frozen EXP53.
   Production sources are included read-only for constants, baseline and tail.
   Mathematical reduction/Q4/ER-low spine is unchanged in G/S/FULL.
   FACTORED changes only TAB128 lookup representation and is accuracy-gated. */
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define restrict __restrict
extern "C" {
#include "../production/exp53_n2_vmstyle_u4_0381_frozen.c"
}
#undef restrict

enum class AttackMode { GatherUndef, Scalef, Full, Factored };

template<AttackMode M>
__attribute__((target("avx512f,avx512dq,fma"),noinline))
static void exp53_attack_impl(double *__restrict out,
                              const double *__restrict in,
                              size_t n) {
    const __m512d inv=_mm512_set1_pd(N2F_INV128),
                  hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI),
                  lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC),
                  one=_mm512_set1_pd(1.0),
                  nq1=_mm512_set1_pd(N2F_Q1),
                  nq2=_mm512_set1_pd(N2F_Q2),
                  nq3=_mm512_set1_pd(N2F_Q3),
                  nq4=_mm512_set1_pd(N2F_Q4),
                  inv128=_mm512_set1_pd(0x1p-7);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),
                  mask=_mm512_set1_epi64(127);

    __m512i factor_lo0{}, factor_lo1{}, factor_hi{};
    if constexpr (M == AttackMode::Factored) {
        factor_lo0 = _mm512_castpd_si512(_mm512_setr_pd(
            N2_FROZEN_TAB128[0],N2_FROZEN_TAB128[1],N2_FROZEN_TAB128[2],N2_FROZEN_TAB128[3],
            N2_FROZEN_TAB128[4],N2_FROZEN_TAB128[5],N2_FROZEN_TAB128[6],N2_FROZEN_TAB128[7]));
        factor_lo1 = _mm512_castpd_si512(_mm512_setr_pd(
            N2_FROZEN_TAB128[8],N2_FROZEN_TAB128[9],N2_FROZEN_TAB128[10],N2_FROZEN_TAB128[11],
            N2_FROZEN_TAB128[12],N2_FROZEN_TAB128[13],N2_FROZEN_TAB128[14],N2_FROZEN_TAB128[15]));
        /* Row a=6 is biased +1 ULP.  Offline exhaustive 128-entry analysis
           reduces factor-product table error from worst 2 ULP to worst 1 ULP. */
        factor_hi = _mm512_castpd_si512(_mm512_setr_pd(
            N2_FROZEN_TAB128[0],N2_FROZEN_TAB128[16],N2_FROZEN_TAB128[32],N2_FROZEN_TAB128[48],
            N2_FROZEN_TAB128[64],N2_FROZEN_TAB128[80],0x1.ae89f995ad3aep+0,N2_FROZEN_TAB128[112]));
    }

    size_t i=0;
    auto block = [&](size_t at) __attribute__((always_inline)) {
        __m512d x[4],biased[4],k[4],r[4],h[4],s[4],er[4],el[4],scale[4],ph[4],y[4];
        __m512i kn[4],j[4],q[4],tb[4];
        for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+at+8*L);
        for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
        for(int L=0;L<4;L++){
            k[L]=_mm512_sub_pd(biased[L],magic);
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);
            j[L]=_mm512_and_epi64(kn[L],mask);
            if constexpr (M == AttackMode::GatherUndef)
                q[L]=_mm512_srai_epi64(kn[L],7);
        }

        if constexpr (M != AttackMode::Factored) {
            for(int L=0;L<4;L++) {
                if constexpr (M == AttackMode::GatherUndef || M == AttackMode::Full) {
                    tb[L]=_mm512_mask_i64gather_epi64(_mm512_undefined_epi32(),(__mmask8)0xff,
                                                      j[L],(const long long*)N2_FROZEN_TAB128,8);
                } else {
                    tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
                }
            }
        }

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
        for(int L=0;L<4;L++) el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));

        for(int L=0;L<4;L++){
            if constexpr (M == AttackMode::GatherUndef) {
                const __m512i sb=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
                scale[L]=_mm512_castsi512_pd(sb);
            } else if constexpr (M == AttackMode::Scalef || M == AttackMode::Full) {
                const __m512d qf=_mm512_mul_pd(k[L],inv128);
                scale[L]=_mm512_scalef_pd(_mm512_castsi512_pd(tb[L]),qf);
            } else {
                const __m512i high_idx=_mm512_srli_epi64(j[L],4);
                const __m512i vlo=_mm512_permutex2var_epi64(factor_lo0,j[L],factor_lo1);
                const __m512i vhi=_mm512_permutexvar_epi64(high_idx,factor_hi);
                const __m512d tab=_mm512_mul_pd(_mm512_castsi512_pd(vlo),_mm512_castsi512_pd(vhi));
                const __m512d qf=_mm512_mul_pd(k[L],inv128);
                scale[L]=_mm512_scalef_pd(tab,qf);
            }
        }

        for(int L=0;L<4;L++){
            ph[L]=_mm512_mul_pd(er[L],scale[L]);
            y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]);
            _mm512_storeu_pd(out+at+8*L,y[L]);
        }
    };

    if constexpr (M == AttackMode::Full || M == AttackMode::Factored) {
        const size_t limit=n & ~(size_t)31;
        for(;i<limit;i+=32) block(i);
    } else {
        for(;i+32<=n;i+=32) block(i);
    }

    if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i);
}

extern "C" void exp53_attack_gather(double* out,const double* in,size_t n) {
    exp53_attack_impl<AttackMode::GatherUndef>(out,in,n);
}
extern "C" void exp53_attack_scalef(double* out,const double* in,size_t n) {
    exp53_attack_impl<AttackMode::Scalef>(out,in,n);
}
extern "C" void exp53_attack_full(double* out,const double* in,size_t n) {
    exp53_attack_impl<AttackMode::Full>(out,in,n);
}
extern "C" void exp53_attack_factored(double* out,const double* in,size_t n) {
    exp53_attack_impl<AttackMode::Factored>(out,in,n);
}
