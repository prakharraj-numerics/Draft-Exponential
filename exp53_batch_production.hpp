#pragma once

/* EXP53 production batch policy — frozen VCL+u2z + serial + 2K..3K store-seq + custom2.

   Size dispatch, default/common temporal case:
     n <= 100:
       frozen VCL+u2z small-batch kernel.

     100 < n < 2000:
       frozen serial temporal VM-style kernel.

     2000 <= n <= 3000, workers > 1:
       frozen store-sequence permanent 2-core path validated on exact Xeon 6973P-C.

     n > 3000, workers > 1:
       existing frozen custom permanent 2-core dispatcher.

   Explicit rare streaming/write-once case — intentionally unchanged:
     n <= 100:
       frozen VCL+u2z temporal path (tiny batches do not use NT stores).

     100 < n <= 3000:
       frozen serial temporal path.

     n > 3000:
       existing custom dispatcher over the immutable NT-store kernel, unless workers<=1,
       in which case the immutable serial NT-store kernel is used.

   IMPORTANT:
   - The active 2000..3000 default temporal promotion is frozen in
     exp53_custom2_storeseq_2000_3000_frozen.hpp and was selected from exact-Xeon
     microarchitecture run 33437986300. Its frozen helper-share schedule is
     34% for 2000..2149, 38% for 2150..2849, and 41% for 2850..3000.
   - n<=100 and n>3000 arrangements are deliberately untouched by this promotion.
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
#include "exp53_custom2_storeseq_2000_3000_frozen.hpp"

extern "C" void exp53_vcl_u2z_0100_frozen(double*, const double*, size_t);

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    static constexpr size_t kSmallVCLU2ZMaxN = 100;
    static constexpr size_t kStoreSeqMinN = 2000;
    static constexpr size_t kStoreSeqMaxN = 3000;
    static constexpr size_t kSerialTemporalMaxN = 3000;

    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    void run(double *out, const double *in, size_t n, long workers = 2) {
        if (n <= kSmallVCLU2ZMaxN) {
            exp53_vcl_u2z_0100_frozen(out, in, n);
        } else if (max_workers_ <= 1 || workers <= 1 || n < kStoreSeqMinN) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else if (n <= kStoreSeqMaxN) {
            storeseq_2k3k().run(out, in, n);
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
    Exp53Custom2StoreSeq2000_3000Frozen& storeseq_2k3k() {
        if (!storeseq_2k3k_)
            storeseq_2k3k_ = std::make_unique<Exp53Custom2StoreSeq2000_3000Frozen>();
        return *storeseq_2k3k_;
    }

    Exp53CustomPermanent2CoreFrozen& custom() {
        if (!frozen_)
            frozen_ = std::make_unique<Exp53CustomPermanent2CoreFrozen>();
        return *frozen_;
    }

    long max_workers_;
    std::unique_ptr<Exp53Custom2StoreSeq2000_3000Frozen> storeseq_2k3k_;
    std::unique_ptr<Exp53CustomPermanent2CoreFrozen> frozen_;
};
