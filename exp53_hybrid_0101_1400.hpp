#pragma once

/* EXP53 hybrid execution path for n=101..1400.

   Derived from the exact-Xeon fine-process map (run 33525805664):
     - XNNPACK pthreadpool dependency owns the public 2-way dispatch/barrier.
     - Native AVX-512 owns index generation and the fused register pipeline.
     - The Turbo square winner is the same AVX-512 vmulpd operation emitted
       here directly, avoiding a Turbo<->native register/materialization boundary.
     - The Highway final-FMA winner is the same AVX-512 vfmadd operation emitted
       here directly, avoiding a native<->Highway boundary.
     - custom2's useful split rule is retained: exactly two 32-element-aligned
       partitions, with no queueing/work stealing policy at this layer.

   Weak/tied micro-process winners are intentionally not switched independently;
   doing so would add representation boundaries larger than their measured gain.

   Mathematical machinery, constants, Horner order, ER/EL repair and scale-bit
   reconstruction are unchanged from the frozen EXP53 implementation.
*/

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <pthreadpool.h>
#include "exp53_highway_sync_1600_3000_constants_frozen.hpp"

namespace exp53_hybrid_0101_1400_ns {
namespace c = exp53_hwy_const_frozen;

static inline void vec8(double* out, const double* in) {
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
    const __m512i mb    = _mm512_set1_epi64((long long)c::MAGIC_BITS);
    const __m512i mask  = _mm512_set1_epi64(127);
    const auto* tab = reinterpret_cast<const long long*>(c::TAB128);

    const __m512d x = _mm512_loadu_pd(in);
    const __m512d biased = _mm512_fmadd_pd(x, inv, magic);
    const __m512d k = _mm512_sub_pd(biased, magic);

    // Native winner: index generation.
    const __m512i kn = _mm512_sub_epi64(_mm512_castpd_si512(biased), mb);
    const __m512i j  = _mm512_and_epi64(kn, mask);
    const __m512i q  = _mm512_srai_epi64(kn, 7);
    const __m512i tb = _mm512_i64gather_epi64(j, tab, 8);

    __m512d r = _mm512_fnmadd_pd(k, hi, x);
    r = _mm512_fnmadd_pd(k, mi, r);
    r = _mm512_fnmadd_pd(k, lo, r);

    __m512d h = _mm512_fmadd_pd(q4, r, q3);
    h = _mm512_fmadd_pd(h, r, q2);
    h = _mm512_fmadd_pd(h, r, q1);
    h = _mm512_fmadd_pd(h, r, one);

    // Turbo's process-map winner collapses to vmulpd in the fused kernel.
    const __m512d s = _mm512_mul_pd(h, h);
    const __m512d er = _mm512_fmadd_pd(r, s, one);
    const __m512d el = _mm512_fmadd_pd(r, s, _mm512_sub_pd(one, er));

    const __m512i sb = _mm512_add_epi64(tb, _mm512_slli_epi64(q, 52));
    const __m512d scale = _mm512_castsi512_pd(sb);
    const __m512d ph = _mm512_mul_pd(er, scale);

    // Highway's final-FMA winner collapses to vfmadd in the fused kernel.
    const __m512d y = _mm512_fmadd_pd(el, scale, ph);
    _mm512_storeu_pd(out, y);
}

static inline void kernel(double* out, const double* in, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        vec8(out + i,      in + i);
        vec8(out + i + 8,  in + i + 8);
        vec8(out + i + 16, in + i + 16);
        vec8(out + i + 24, in + i + 24);
    }
    for (; i + 8 <= n; i += 8) vec8(out + i, in + i);
    if (i < n) {
        alignas(64) double ti[8] = {0,0,0,0,0,0,0,0};
        alignas(64) double to[8];
        const size_t rem = n - i;
        std::memcpy(ti, in + i, rem * sizeof(double));
        vec8(to, ti);
        std::memcpy(out + i, to, rem * sizeof(double));
    }
}

struct TwoTaskCtx {
    double* out;
    const double* in;
    size_t split;
    size_t n;
};

static inline void two_task(void* p, size_t index) {
    auto* ctx = static_cast<TwoTaskCtx*>(p);
    const size_t begin = index == 0 ? 0 : ctx->split;
    const size_t end   = index == 0 ? ctx->split : ctx->n;
    if (end > begin) kernel(ctx->out + begin, ctx->in + begin, end - begin);
}

} // namespace exp53_hybrid_0101_1400_ns

class Exp53Hybrid0101_1400 {
public:
    explicit Exp53Hybrid0101_1400(size_t threads = 2)
        : pool_(pthreadpool_create(threads > 1 ? 2 : 1)) {}

    ~Exp53Hybrid0101_1400() {
        if (pool_) pthreadpool_destroy(pool_);
    }

    Exp53Hybrid0101_1400(const Exp53Hybrid0101_1400&) = delete;
    Exp53Hybrid0101_1400& operator=(const Exp53Hybrid0101_1400&) = delete;

    void run(double* out, const double* in, size_t n, long workers = 2) {
        using namespace exp53_hybrid_0101_1400_ns;
        if (!n) return;
        if (!pool_ || workers <= 1 || n < 64 || pthreadpool_get_threads_count(pool_) < 2) {
            kernel(out, in, n);
            return;
        }

        // custom2-derived split arithmetic: exactly two 32-value-aligned tasks.
        size_t split = ((n / 2) / 32) * 32;
        if (split < 32) split = 32;
        if (split >= n) {
            kernel(out, in, n);
            return;
        }

        TwoTaskCtx ctx{out, in, split, n};
        pthreadpool_parallelize_1d(pool_, two_task, &ctx, 2, 0);
    }

private:
    pthreadpool_t pool_;
};
