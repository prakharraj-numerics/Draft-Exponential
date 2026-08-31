#pragma once

/* EXP53 production batch policy — frozen small-u2z + serial + custom 2-core.

   Size dispatch, default/common temporal case:
     n <= 100:
       frozen two-ZMM small-batch kernel.

     100 < n <= 3000:
       frozen serial temporal VM-style kernel.

     n > 3000:
       custom permanent 2-core dispatcher over the immutable temporal kernel.

   Explicit rare streaming/write-once case:
     n <= 100:
       frozen u2z temporal path (tiny batches do not use NT stores).

     100 < n <= 3000:
       frozen serial temporal path (small/medium batches do not use NT stores).

     n > 3000:
       custom dispatcher over the immutable NT-store kernel, unless workers<=1,
       in which case the immutable serial NT-store kernel is used.

   IMPORTANT:
   - The custom runtime is constructed lazily on the first n>3000 parallel
     call. A workload that stays at n<=3000 never creates the helper thread.
   - The 3000 cutoff selects serial vs custom dispatch; it does NOT auto-select
     NT above the cutoff. Caller semantics still decide the store policy.
   - The n<=100 cutoff is backed by exact-Xeon run 33422301867, where the
     frozen-u2z candidate was the winning own-kernel path at n=50 and n=100 in
     both unit and mid domains; Intel regained the lead from n=150 onward.
   - Active production does not depend on FastFlow, OpenMP, oneTBB, or Taskflow.
   - Immutable survivor files are not modified.
*/

#include <algorithm>
#include <cstddef>
#include <memory>
#include "exp53_batch_custom_2core_nt_frozen.hpp"

extern "C" void exp53_small_u2z_0100_frozen(double*, const double*, size_t);

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    static constexpr size_t kSmallU2ZMaxN = 100;
    static constexpr size_t kSerialTemporalMaxN = 3000;

    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    /* Common/default API: u2z <=100, serial temporal through 3K, custom above. */
    void run(double *out, const double *in, size_t n, long workers = 2) {
        if (n <= kSmallU2ZMaxN) {
            exp53_small_u2z_0100_frozen(out, in, n);
        } else if (n <= kSerialTemporalMaxN || max_workers_ <= 1 || workers <= 1) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else {
            custom().run(out, in, n);
        }
    }

    /* Explicit rare API: NT only above 3K; tiny/small ranges stay temporal. */
    void run_streaming_write_once(double *out, const double *in, size_t n,
                                  long workers = 2) {
        if (n <= kSmallU2ZMaxN) {
            exp53_small_u2z_0100_frozen(out, in, n);
        } else if (n <= kSerialTemporalMaxN) {
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
