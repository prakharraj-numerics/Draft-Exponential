#pragma once

/* EXP53 production batch policy.

   Default/common case:
     temporal FastFlow over the immutable frozen kernel.

   Explicit rare streaming/write-once case:
     the same FastFlow schedule over the NT-store kernel.

   IMPORTANT:
   - No n-based automatic NT threshold is used.
   - Caller semantics decide the store policy.
   - Frozen survivor files are not modified.
   - One persistent FastFlow pool is shared by both policies so selecting the
     rare NT mode does not leave a second spin-wait worker pool resident.
*/

#include <algorithm>
#include <cstddef>
#include <ff/parallel_for.hpp>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_nt_sfence(double*, const double*, size_t);

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : pf_(max_workers, true, true),
          max_workers_(std::max(1L, max_workers)) {}

    /* Common/default API: temporal stores. */
    void run(double *out, const double *in, size_t n, long workers = 2) {
        run_with(exp53_n2_vmstyle_u4_0381_frozen, out, in, n, workers);
    }

    /* Explicit rare API: output will not be consumed from cache soon. */
    void run_streaming_write_once(double *out, const double *in, size_t n,
                                  long workers = 2) {
        run_with(exp53_n2_vmstyle_u4_0381_nt_sfence, out, in, n, workers);
    }

    /* Policy API for callers that want one entry point. */
    void run(double *out, const double *in, size_t n,
             Exp53OutputPolicy policy, long workers = 2) {
        if (policy == Exp53OutputPolicy::StreamingWriteOnce)
            run_streaming_write_once(out, in, n, workers);
        else
            run(out, in, n, workers);
    }

private:
    using fn_t = void (*)(double*, const double*, size_t);

    /* This is intentionally the same balanced static 32-element-aligned
       partition used by exp53_fastflow_batch_2core_frozen.hpp. */
    void run_with(fn_t fn, double *out, const double *in, size_t n,
                  long workers) {
        if (!n) return;
        long active = std::min(std::max(1L, workers), max_workers_);
        const size_t full32 = n / 32;
        if (active <= 1 || full32 < 2) {
            fn(out, in, n);
            return;
        }
        active = std::min<long>(active, (long)full32);
        if (active <= 1) {
            fn(out, in, n);
            return;
        }

        pf_.parallel_for_static(0, active, 1, 0,
            [&](const long w) {
                const size_t b0 = (full32 * (size_t)w) / (size_t)active;
                const size_t b1 = (full32 * (size_t)(w + 1)) / (size_t)active;
                const size_t lo = 32 * b0;
                size_t hi = 32 * b1;
                if (w == active - 1) hi = n;
                fn(out + lo, in + lo, hi - lo);
            }, active);
    }

    ff::ParallelFor pf_;
    long max_workers_;
};
