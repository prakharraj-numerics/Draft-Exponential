// Experimental Highway SIMD backend for the frozen EXP53 mathematical machinery.
// No hwy::Exp or replacement math is used. Operation order mirrors the frozen AVX-512 kernel.
#define HWY_COMPILE_ONLY_STATIC
#include "hwy/highway.h"
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

namespace hn = hwy::HWY_NAMESPACE;

__attribute__((noinline))
void exp53_highway_0101_2000_candidate(double* out, const double* in, size_t n) {
  const hn::FixedTag<double, 8> d;
  const hn::RebindToSigned<decltype(d)> di;

  const auto inv   = hn::Set(d, N2F_INV128);
  const auto hi    = hn::Set(d, N2F_L128_HI);
  const auto mi    = hn::Set(d, N2F_L128_MI);
  const auto lo    = hn::Set(d, N2F_L128_LO);
  const auto magic = hn::Set(d, N2F_MAGIC);
  const auto one   = hn::Set(d, 1.0);
  const auto nq1   = hn::Set(d, N2F_Q1);
  const auto nq2   = hn::Set(d, N2F_Q2);
  const auto nq3   = hn::Set(d, N2F_Q3);
  const auto nq4   = hn::Set(d, N2F_Q4);
  const auto mb    = hn::Set(di, static_cast<int64_t>(N2F_MAGIC_BITS));
  const auto mask  = hn::Set(di, 127);
  const auto* tab_bits = reinterpret_cast<const int64_t*>(N2_FROZEN_TAB128);

  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    hn::Vec<decltype(d)> x[4], biased[4], k[4], r[4], h[4], s[4], er[4], el[4], scale[4], ph[4], y[4];
    hn::Vec<decltype(di)> kn[4], j[4], q[4], tb[4], sb[4];

    for (int L = 0; L < 4; ++L) x[L] = hn::LoadU(d, in + i + 8 * L);
    for (int L = 0; L < 4; ++L) biased[L] = hn::MulAdd(x[L], inv, magic);

    for (int L = 0; L < 4; ++L) {
      k[L] = hn::Sub(biased[L], magic);
      kn[L] = hn::Sub(hn::BitCast(di, biased[L]), mb);
      j[L] = hn::And(kn[L], mask);
      q[L] = hn::ShiftRight<7>(kn[L]);
    }

    for (int L = 0; L < 4; ++L) tb[L] = hn::GatherIndex(di, tab_bits, j[L]);

    for (int L = 0; L < 4; ++L) {
      r[L] = hn::NegMulAdd(k[L], hi, x[L]);
      r[L] = hn::NegMulAdd(k[L], mi, r[L]);
      r[L] = hn::NegMulAdd(k[L], lo, r[L]);
    }

    for (int L = 0; L < 4; ++L) h[L] = hn::MulAdd(nq4, r[L], nq3);
    for (int L = 0; L < 4; ++L) h[L] = hn::MulAdd(h[L], r[L], nq2);
    for (int L = 0; L < 4; ++L) h[L] = hn::MulAdd(h[L], r[L], nq1);
    for (int L = 0; L < 4; ++L) h[L] = hn::MulAdd(h[L], r[L], one);

    for (int L = 0; L < 4; ++L) s[L] = hn::Mul(h[L], h[L]);
    for (int L = 0; L < 4; ++L) er[L] = hn::MulAdd(r[L], s[L], one);
    for (int L = 0; L < 4; ++L) el[L] = hn::MulAdd(r[L], s[L], hn::Sub(one, er[L]));

    for (int L = 0; L < 4; ++L) {
      sb[L] = hn::Add(tb[L], hn::ShiftLeft<52>(q[L]));
      scale[L] = hn::BitCast(d, sb[L]);
    }

    for (int L = 0; L < 4; ++L) {
      ph[L] = hn::Mul(er[L], scale[L]);
      y[L] = hn::MulAdd(el[L], scale[L], ph[L]);
      hn::StoreU(y[L], d, out + i + 8 * L);
    }
  }

  if (i < n) exp53_n2_fused_u4_038_frozen(out + i, in + i, n - i);
}
