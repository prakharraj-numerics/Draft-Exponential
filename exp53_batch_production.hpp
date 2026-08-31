#pragma once

/* EXP53 production batch policy — FINAL frozen custom 2-core survivor.

   Size dispatch:
     n <= 3000:
       frozen serial temporal kernel only.
       No custom 2-core dispatch and no NT stores, even if the caller requests
       StreamingWriteOnce.

     n > 3000, default/common case:
       custom permanent 2-core dispatcher over the immutable temporal kernel.

     n > 3000, explicit rare streaming/write-once case:
       the same custom dispatcher over the NT-store kernel.

   IMPORTANT:
   - The custom runtime is constructed lazily on the first n>3000 parallel
     call.  A workload that stays at n<=3000 never creates the helper thread.
   - The 3000 cutoff selects serial vs custom dispatch; it does NOT auto-select
     NT above the cutoff.  For n > 3000 caller semantics still decide the store
     policy.
   - Active production does not depend on FastFlow.
   - The immutable custom implementation is in
       exp53_batch_custom_2core_nt_frozen.hpp
   - workers<=1 remains a serial escape hatch for API compatibility.
   - Frozen survivor files are not modified.
*/

#include <algorithm>
#include <cstddef>
#include <memory>
#include "exp53_batch_custom_2core_nt_frozen.hpp"

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    static constexpr size_t kSerialTemporalMaxN = 3000;

    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    /* Common/default API: serial temporal through 3K, custom temporal above. */
    void run(double *out, const double *in, size_t n, long workers = 2) {
        if (n <= kSerialTemporalMaxN || max_workers_ <= 1 || workers <= 1) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else {
            custom().run(out, in, n);
        }
    }

    /* Explicit rare API: still forced to serial temporal through 3K. */
    void run_streaming_write_once(double *out, const double *in, size_t n,
                                  long workers = 2) {
        if (n <= kSerialTemporalMaxN) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else if (max_workers_ <= 1 || workers <= 1) {
            exp53_n2_vmstyle_u4_0381_nt_sfence(out, in, n);
        } else {
            custom().run_streaming_write_once(out, in, n);
        }
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
    Exp53CustomPermanent2CoreFrozen& custom() {
        if (!frozen_)
            frozen_ = std::make_unique<Exp53CustomPermanent2CoreFrozen>();
        return *frozen_;
    }

    long max_workers_;
    std::unique_ptr<Exp53CustomPermanent2CoreFrozen> frozen_;
};
