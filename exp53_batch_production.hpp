#pragma once

/* EXP53 production batch policy — FINAL FROZEN custom 2-core survivor.

   Default/common case:
     custom permanent 2-core dispatcher over the immutable temporal kernel.

   Explicit rare streaming/write-once case:
     the same custom dispatcher over the NT-store kernel.

   IMPORTANT:
   - No n-based automatic NT threshold is used.
   - Caller semantics decide the store policy.
   - Active production no longer depends on FastFlow.
   - The immutable implementation is in
       exp53_batch_custom_2core_nt_frozen.hpp
   - workers<=1 remains a serial escape hatch for API compatibility.
*/

#include <algorithm>
#include <cstddef>
#include "exp53_batch_custom_2core_nt_frozen.hpp"

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    /* Common/default API: temporal stores. */
    void run(double *out, const double *in, size_t n, long workers = 2) {
        if (max_workers_ <= 1 || workers <= 1)
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        else
            frozen_.run(out, in, n);
    }

    /* Explicit rare API: output will not be consumed from cache soon. */
    void run_streaming_write_once(double *out, const double *in, size_t n,
                                  long workers = 2) {
        if (max_workers_ <= 1 || workers <= 1)
            exp53_n2_vmstyle_u4_0381_nt_sfence(out, in, n);
        else
            frozen_.run_streaming_write_once(out, in, n);
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
    long max_workers_;
    Exp53CustomPermanent2CoreFrozen frozen_;
};
