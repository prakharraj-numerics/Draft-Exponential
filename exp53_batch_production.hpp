#pragma once

/* EXP53 production batch policy — frozen VCL+u2z + serial + specialized/custom 2-core.

   Size dispatch, default/common temporal case:
     n <= 100:
       frozen VCL+u2z small-batch kernel.

     100 < n < 2500:
       frozen serial temporal VM-style kernel.

     2500 <= n <= 3000, workers > 1:
       frozen range-specialized permanent 2-core U2Z path.

     n > 3000, workers > 1:
       frozen custom permanent 2-core dispatcher.

   Explicit rare streaming/write-once case:
     n <= 100:
       frozen VCL+u2z temporal path (tiny batches do not use NT stores).

     100 < n < 2500:
       frozen serial temporal path.

     2500 <= n <= 3000, workers > 1:
       frozen range-specialized 2-core U2Z temporal path.

     n > 3000:
       custom dispatcher over the immutable NT-store kernel, unless workers<=1,
       in which case the immutable serial NT-store kernel is used.

   IMPORTANT:
   - The range-specialized runtime is constructed lazily on the first
     2500..3000 parallel call; the general custom runtime remains independently
     lazy for n>3000.
   - The 2500..3000 specialized band is the frozen Conservative geometry from
     exact-Xeon sweep run 33426419558: helper gets trailing floor16(46% of n),
     caller gets the head, using immutable U2Z math on CPU0/CPU2.
   - NT remains an explicit caller semantic. The <=3000 paths remain temporal,
     exactly as before; no size automatically selects NT stores.
   - The n<=100 cutoff is backed by exact-Xeon run 33424808629, where frozen
     VCL+u2z beat Intel VML_HA at n=50 and n=100 in both unit and mid domains;
     Intel regained the lead from n=150 onward.
   - VCL is required only for building the frozen n<=100 implementation.
   - Active production does not depend on FastFlow, OpenMP, oneTBB, or Taskflow.
   - Immutable survivor files are not modified.
*/

#include <algorithm>
#include <cstddef>
#include <memory>
#include "exp53_batch_custom_2core_nt_frozen.hpp"
#include "exp53_batch_range2core_2500_3000_frozen.hpp"

extern "C" void exp53_vcl_u2z_0100_frozen(double*, const double*, size_t);

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    static constexpr size_t kSmallVCLU2ZMaxN = 100;
    static constexpr size_t kRange2CoreMinN = 2500;
    static constexpr size_t kRange2CoreMaxN = 3000;

    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    /* Common/default API: VCL+u2z <=100, serial <2500,
       specialized 2-core 2500..3000, general custom above 3000. */
    void run(double *out, const double *in, size_t n, long workers = 2) {
        if (n <= kSmallVCLU2ZMaxN) {
            exp53_vcl_u2z_0100_frozen(out, in, n);
        } else if (n >= kRange2CoreMinN && n <= kRange2CoreMaxN &&
                   max_workers_ > 1 && workers > 1) {
            range2core().run(out, in, n);
        } else if (n <= kRange2CoreMaxN || max_workers_ <= 1 || workers <= 1) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else {
            custom().run(out, in, n);
        }
    }

    /* Explicit rare API: <=3000 remains temporal; NT is only used above 3K. */
    void run_streaming_write_once(double *out, const double *in, size_t n,
                                  long workers = 2) {
        if (n <= kSmallVCLU2ZMaxN) {
            exp53_vcl_u2z_0100_frozen(out, in, n);
        } else if (n >= kRange2CoreMinN && n <= kRange2CoreMaxN &&
                   max_workers_ > 1 && workers > 1) {
            range2core().run(out, in, n);
        } else if (n <= kRange2CoreMaxN) {
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
    Exp53Range2Core2500_3000Frozen& range2core() {
        if (!range2core_)
            range2core_ = std::make_unique<Exp53Range2Core2500_3000Frozen>();
        return *range2core_;
    }

    Exp53CustomPermanent2CoreFrozen& custom() {
        if (!frozen_)
            frozen_ = std::make_unique<Exp53CustomPermanent2CoreFrozen>();
        return *frozen_;
    }

    long max_workers_;
    std::unique_ptr<Exp53Range2Core2500_3000Frozen> range2core_;
    std::unique_ptr<Exp53CustomPermanent2CoreFrozen> frozen_;
};
