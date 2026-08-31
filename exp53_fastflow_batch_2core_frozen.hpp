#pragma once

/* FROZEN BATCH SURVIVOR — synchronized FastFlow + faithful n=2 AVX-512 EXP.

   DO NOT MODIFY THIS FILE.

   Mathematical kernel:
     exp53_n2_vmstyle_u4_0381_frozen

   FastFlow runtime tested/pinned in validation workflow:
     fastflow/fastflow @ d476f66ab924d8d122f54b4b90aee00ef979aea8

   Validation workflow run:
     33399802068

   Exact CPU:
     Intel(R) Xeon(R) 6973P-C
     GitHub runner exposed 2 physical cores / 4 logical CPUs.

   Winning execution policy on this runner:
     - persistent FastFlow worker pool
     - spin-wait workers + spin barrier
     - one worker per physical core (2 workers)
     - balanced contiguous static regions aligned to 32 elements
     - dynamic large-block mode retained for tuned very-large-batch use
     - worker creation/destruction outside measured calls
     - dispatch + completion synchronization included in measured calls

   Correctness on 200,000 mixed-domain inputs:
     static bitdiff vs frozen serial = 0
     dynamic bitdiff vs frozen serial = 0
     maxULP = 1
     gt1 = 0

   Clean shard 2 median ns/input, OURS serial -> OURS+FastFlow effective:
     n=10079:   0.394928549 -> 0.359561484   (1.098x)
     n=12288:   0.391135317 -> 0.334285857   (1.170x)
     n=65536:   0.384712594 -> 0.258516656   (1.488x)
     n=262144:  0.527478536 -> 0.254524740   (2.072x)
     n=1000000: 0.541838750 -> 0.274047000   (1.977x)

   Fair FastFlow-wrapped VML_HA at n=1,000,000 on shard 2:
     OURS+FastFlow = 0.274047000 ns/input
     VML_HA+FastFlow = 0.271398500 ns/input

   Tiny batches remain serial; exact production crossover between 253 and
   10,079 was not frozen because it was not measured in the validation sweep.
*/

#include <algorithm>
#include <cstddef>
#include <ff/parallel_for.hpp>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

typedef void (*exp53_frozen_batch_fn)(double*, const double*, size_t);

class Exp53FastFlowFrozenExecutor {
public:
    Exp53FastFlowFrozenExecutor(long max_workers, exp53_frozen_batch_fn fn)
        : pf_(max_workers, true, true),
          max_workers_(std::max(1L, max_workers)), fn_(fn) {}

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
    exp53_frozen_batch_fn fn_;
};
