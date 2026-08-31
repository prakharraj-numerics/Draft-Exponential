#pragma once

/* EXP53 production batch policy — frozen VCL+u2z + serial + custom 2-core.

   Size dispatch, default/common temporal case:
     n <= 100:
       frozen VCL+u2z small-batch kernel.

     100 < n <= 3000:
       frozen serial temporal VM-style kernel.

     n > 3000, workers > 1:
       frozen custom permanent 2-core dispatcher.

   Explicit rare streaming/write-once case:
     n <= 100:
       frozen VCL+u2z temporal path (tiny batches do not use NT stores).

     100 < n <= 3000:
       frozen serial temporal path.

     n > 3000:
       custom dispatcher over the immutable NT-store kernel, unless workers<=1,
       in which case the immutable serial NT-store kernel is used.

   IMPORTANT:
   - The attempted 2500..3000 range2core U2Z promotion from exact-Xeon speed
     sweep run 33426419558 was rejected by exact-Xeon correctness smoke
     run 33427533268: bitwise mismatch vs frozen serial at n=2500.
     Its frozen file is retained only as failed-candidate evidence and is NOT
     active production.
   - NT remains an explicit caller semantic; no size automatically selects NT.
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

extern "C" void exp53_vcl_u2z_0100_frozen(double*, const double*, size_t);

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    static constexpr size_t kSmallVCLU2ZMaxN = 100;
    static constexpr size_t kSerialTemporalMaxN = 3000;

    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    void run(double *out, const double *in, size_t n, long workers = 2) {
        if (n <= kSmallVCLU2ZMaxN) {
            exp53_vcl_u2z_0100_frozen(out, in, n);
        } else if (n <= kSerialTemporalMaxN || max_workers_ <= 1 || workers <= 1) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else {
            custom().run(out, in, n);
        }
    }

    void run_streaming_write_once(double *out, const double *in, size_t n,
                                  long workers = 2) {
        if (n <= kSmallVCLU2ZMaxN) {
            exp53_vcl_u2z_0100_frozen(out, in, n);
        } else if (n <= kSerialTemporalMaxN) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else if (max_workers_ <= 1 || workers <= 1) {
            exp53_n2_vmstyle_u4_0381_nt_sfence(out, in, n);
        } else {
            custom().run_streaming_write_once(out, in, n);
        }
    }

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
