#pragma once

#include <algorithm>
#include <cstddef>
#include <ff/parallel_for.hpp>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

typedef void (*exp53_batch_fn)(double*, const double*, size_t);

/* Persistent FastFlow wrapper for repeated batch calls.

   Design goals:
   - worker threads are created once and reused;
   - spin-wait + spin barrier are enabled for repeated very-short calls;
   - static mode gives each active worker one contiguous 32-element-aligned
     chunk, minimizing scheduler traffic and preserving the frozen SIMD path;
   - dynamic mode is retained as an explicit comparison for uneven batches;
   - all calls are blocking: return means every worker has completed.
*/
class Exp53FastFlowExecutor {
public:
    Exp53FastFlowExecutor(long max_workers, exp53_batch_fn fn)
        : pf_(max_workers, true, true), max_workers_(std::max(1L, max_workers)), fn_(fn) {}

    void run_static(double *out, const double *in, size_t n, long workers) {
        if (!n) return;
        long active = std::min(std::max(1L, workers), max_workers_);
        const size_t full32 = n / 32;
        if (active <= 1 || full32 < 2) {
            fn_(out, in, n);
            return;
        }
        active = std::min<long>(active, (long)full32);
        if (active <= 1) {
            fn_(out, in, n);
            return;
        }

        /* Exactly one balanced contiguous range per active worker. Every range
           except the last is a multiple of 32 elements; the last owns n%32. */
        pf_.parallel_for_static(0, active, 1, 0,
            [&](const long w) {
                const size_t b0 = (full32 * (size_t)w) / (size_t)active;
                const size_t b1 = (full32 * (size_t)(w + 1)) / (size_t)active;
                const size_t lo = 32 * b0;
                size_t hi = 32 * b1;
                if (w == active - 1) hi = n;
                fn_(out + lo, in + lo, hi - lo);
            }, active);
    }

    void run_dynamic(double *out, const double *in, size_t n,
                     long workers, size_t block_elems) {
        if (!n) return;
        long active = std::min(std::max(1L, workers), max_workers_);
        block_elems = std::max<size_t>(32, (block_elems / 32) * 32);
        const long blocks = (long)((n + block_elems - 1) / block_elems);
        active = std::min(active, std::max(1L, blocks));
        if (active <= 1) {
            fn_(out, in, n);
            return;
        }

        /* Dynamic FastFlow scheduler. Each work unit is 32-aligned except the
           last remainder block; outputs are disjoint, so no worker locks are
           required. */
        pf_.parallel_for(0, blocks, 1, 1,
            [&](const long b) {
                const size_t lo = (size_t)b * block_elems;
                const size_t hi = std::min(n, lo + block_elems);
                fn_(out + lo, in + lo, hi - lo);
            }, active);
    }

private:
    ff::ParallelFor pf_;
    long max_workers_;
    exp53_batch_fn fn_;
};
