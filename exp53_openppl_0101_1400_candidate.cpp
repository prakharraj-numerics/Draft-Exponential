// Experimental OpenPPL/PPLNN x86 primitive integration for EXP53, n=101..1400.
// Production untouched. PPLNN's x86 backend delegates CPU primitives to ppl.kernel.cpu;
// this candidate uses that layer's own OpenMP parallel macros/scheduling while preserving
// the frozen EXP53 AVX-512 arithmetic exactly in the vector kernel.
#include "ppl/kernel/x86/common/macros.h"
#include "exp53_highway_sync_1600_3000_constants_frozen.hpp"
#include <immintrin.h>
#include <omp.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace c = exp53_hwy_const_frozen;

static inline __m512d exp53_vec8_openppl(__m512d x) {
    const __m512d inv   = _mm512_set1_pd(c::INV128);
    const __m512d hi    = _mm512_set1_pd(c::L128_HI);
    const __m512d mi    = _mm512_set1_pd(c::L128_MI);
    const __m512d lo    = _mm512_set1_pd(c::L128_LO);
    const __m512d magic = _mm512_set1_pd(c::MAGIC);
    const __m512d one   = _mm512_set1_pd(1.0);
    const __m512d q1    = _mm512_set1_pd(c::Q1);
    const __m512d q2    = _mm512_set1_pd(c::Q2);
    const __m512d q3    = _mm512_set1_pd(c::Q3);
    const __m512d q4    = _mm512_set1_pd(c::Q4);

    __m512d biased = _mm512_fmadd_pd(x, inv, magic);
    __m512d k      = _mm512_sub_pd(biased, magic);
    __m512i kn     = _mm512_sub_epi64(_mm512_castpd_si512(biased), _mm512_set1_epi64((long long)c::MAGIC_BITS));
    __m512i j      = _mm512_and_si512(kn, _mm512_set1_epi64(127));
    __m512i q      = _mm512_srai_epi64(kn, 7);
    __m512i tb     = _mm512_i64gather_epi64(j, reinterpret_cast<const void*>(c::TAB128), 8);

    __m512d r = _mm512_fnmadd_pd(k, hi, x);
    r = _mm512_fnmadd_pd(k, mi, r);
    r = _mm512_fnmadd_pd(k, lo, r);

    __m512d h = _mm512_fmadd_pd(q4, r, q3);
    h = _mm512_fmadd_pd(h, r, q2);
    h = _mm512_fmadd_pd(h, r, q1);
    h = _mm512_fmadd_pd(h, r, one);
    __m512d s  = _mm512_mul_pd(h, h);
    __m512d er = _mm512_fmadd_pd(r, s, one);
    __m512d el = _mm512_fmadd_pd(r, s, _mm512_sub_pd(one, er));

    __m512i sb = _mm512_add_epi64(tb, _mm512_slli_epi64(q, 52));
    __m512d scale = _mm512_castsi512_pd(sb);
    __m512d ph = _mm512_mul_pd(er, scale);
    return _mm512_fmadd_pd(el, scale, ph);
}

static inline void exp53_range_openppl(double* out, const double* in, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512d x0 = _mm512_loadu_pd(in + i + 0);
        __m512d x1 = _mm512_loadu_pd(in + i + 8);
        __m512d x2 = _mm512_loadu_pd(in + i + 16);
        __m512d x3 = _mm512_loadu_pd(in + i + 24);
        _mm512_storeu_pd(out + i + 0,  exp53_vec8_openppl(x0));
        _mm512_storeu_pd(out + i + 8,  exp53_vec8_openppl(x1));
        _mm512_storeu_pd(out + i + 16, exp53_vec8_openppl(x2));
        _mm512_storeu_pd(out + i + 24, exp53_vec8_openppl(x3));
    }
    for (; i + 8 <= n; i += 8) {
        _mm512_storeu_pd(out + i, exp53_vec8_openppl(_mm512_loadu_pd(in + i)));
    }
    if (i < n) {
        alignas(64) double ti[8] = {0,0,0,0,0,0,0,0};
        alignas(64) double to[8];
        const size_t rem = n - i;
        std::memcpy(ti, in + i, rem * sizeof(double));
        _mm512_store_pd(to, exp53_vec8_openppl(_mm512_load_pd(ti)));
        std::memcpy(out + i, to, rem * sizeof(double));
    }
}

// Canonical ppl.kernel.cpu shape: PPL's own PRAGMA_OMP_PARALLEL_FOR over 32-value tiles.
extern "C" void exp53_openppl_canonical_0101_1400(double* out, const double* in, size_t n) {
    const size_t body = (n / 32) * 32;
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < (int64_t)body; i += 32) {
        exp53_range_openppl(out + i, in + i, 32);
    }
    if (body < n) exp53_range_openppl(out + body, in + body, n - body);
}

// Tuned two-thread static split using the same OpenMP runtime PPL relies on.
extern "C" void exp53_openppl_split_0101_1400(double* out, const double* in, size_t n, int helper_pct) {
    size_t helper = (n * (size_t)helper_pct) / 100;
    helper = (helper / 32) * 32;
    size_t split = n - helper;
    split = (split / 32) * 32;
    if (split == 0 || split >= n) {
        exp53_range_openppl(out, in, n);
        return;
    }
    #pragma omp parallel num_threads(2)
    {
        const int tid = omp_get_thread_num();
        if (tid == 0) exp53_range_openppl(out, in, split);
        else exp53_range_openppl(out + split, in + split, n - split);
    }
}
