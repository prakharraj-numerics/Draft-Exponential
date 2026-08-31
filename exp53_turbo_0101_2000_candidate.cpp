// Experimental Turbo SIMD backend for the frozen EXP53 mathematical machinery.
// Pinned Turbo snapshot: 9b212dedfaa4d4e08a854776e1ddaca7d746444c
// No turbo::simd::exp or replacement transcendental math is used.
// Operation order mirrors exp53_n2_vmstyle_u4_0381_frozen.c.
#define restrict __restrict__
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
#undef restrict

#include "turbo/simd/simd.h"

namespace ts = turbo::simd;

__attribute__((noinline))
void exp53_turbo_0101_2000_candidate(double* out, const double* in, size_t n) {
    using D = ts::batch<double>;
    using I = ts::batch<int64_t>;
    static_assert(D::size == 8, "Turbo must select 8-lane AVX-512 FP64 on this benchmark");
    static_assert(I::size == 8, "Turbo must select 8-lane AVX-512 int64 on this benchmark");

    const D inv(N2F_INV128), hi(N2F_L128_HI), mi(N2F_L128_MI), lo(N2F_L128_LO),
            magic(N2F_MAGIC), one(1.0), nq1(N2F_Q1), nq2(N2F_Q2), nq3(N2F_Q3), nq4(N2F_Q4);
    const I mb((int64_t)N2F_MAGIC_BITS), mask((int64_t)127);
    const int64_t* tab_bits = reinterpret_cast<const int64_t*>(N2_FROZEN_TAB128);

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        D x[4], biased[4], k[4], r[4], h[4], s[4], er[4], el[4], scale[4], ph[4], y[4];
        I kn[4], j[4], q[4], tb[4], sb[4];

        for (int L = 0; L < 4; ++L) x[L] = D::load_unaligned(in + i + 8 * L);
        for (int L = 0; L < 4; ++L) biased[L] = ts::fma(x[L], inv, magic);

        for (int L = 0; L < 4; ++L) {
            k[L] = biased[L] - magic;
            kn[L] = ts::bitwise_cast<int64_t>(biased[L]) - mb;
            j[L] = kn[L] & mask;
            q[L] = kn[L] >> 7;
        }

        // Preserve the VM-style early-gather schedule.
        for (int L = 0; L < 4; ++L) tb[L] = I::gather(tab_bits, j[L]);

        for (int L = 0; L < 4; ++L) {
            r[L] = ts::fnma(k[L], hi, x[L]);
            r[L] = ts::fnma(k[L], mi, r[L]);
            r[L] = ts::fnma(k[L], lo, r[L]);
        }

        for (int L = 0; L < 4; ++L) h[L] = ts::fma(nq4, r[L], nq3);
        for (int L = 0; L < 4; ++L) h[L] = ts::fma(h[L], r[L], nq2);
        for (int L = 0; L < 4; ++L) h[L] = ts::fma(h[L], r[L], nq1);
        for (int L = 0; L < 4; ++L) h[L] = ts::fma(h[L], r[L], one);

        for (int L = 0; L < 4; ++L) s[L] = h[L] * h[L];
        for (int L = 0; L < 4; ++L) er[L] = ts::fma(r[L], s[L], one);
        for (int L = 0; L < 4; ++L) el[L] = ts::fma(r[L], s[L], one - er[L]);

        for (int L = 0; L < 4; ++L) {
            sb[L] = tb[L] + (q[L] << 52);
            scale[L] = ts::bitwise_cast<double>(sb[L]);
        }

        for (int L = 0; L < 4; ++L) {
            ph[L] = er[L] * scale[L];
            y[L] = ts::fma(el[L], scale[L], ph[L]);
            y[L].store_unaligned(out + i + 8 * L);
        }
    }

    if (i < n) exp53_n2_fused_u4_038_frozen(out + i, in + i, n - i);
}
