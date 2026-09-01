#pragma once

/* EXP53 production batch policy — frozen VCL+u2z + hybrid + Highway sync + custom2.

   Size dispatch, default/common temporal case:
     n <= 100:
       frozen VCL+u2z small-batch kernel.

     100 < n <= 1400:
       exact-Xeon process-map-derived hybrid:
         * XNNPACK pthreadpool 2-way dispatch/barrier
         * native AVX-512 index generation / fused register pipeline
         * Turbo-square and Highway-final-FMA winning operations emitted directly
           as their equivalent AVX-512 instructions to avoid library boundaries
         * custom2-derived 32-element-aligned two-way split arithmetic

     1400 < n <= 3000, workers > 1:
       frozen Google Highway synchronized permanent 2-core path validated on
       exact Intel Xeon 6973P-C.

     n > 3000, workers > 1:
       existing frozen custom permanent 2-core dispatcher.

   Explicit rare streaming/write-once case — intentionally unchanged:
     n <= 100:
       frozen VCL+u2z temporal path.

     100 < n <= 3000:
       frozen serial temporal path.

     n > 3000:
       existing custom dispatcher over the immutable NT-store kernel, unless
       workers<=1, in which case the immutable serial NT-store kernel is used.

   IMPORTANT:
   - The active 101..1400 default temporal path is isolated in
     exp53_hybrid_0101_1400.hpp.
   - The active >1400..3000 default temporal promotion remains frozen in
     exp53_highway_sync_1400_3000_frozen.hpp; the frozen file itself is unchanged.
   - The hybrid preserves the EXP53 mathematical machinery and constants.
   - Process-map evidence: exact Xeon 6973P-C run 33525805664.
   - The prior 1600..3000 frozen Highway file remains in the repository as a
     rollback checkpoint but is no longer active in default production routing.
   - The prior 2000..3000 StoreSeq frozen file also remains as rollback evidence.
   - n<=100 and n>3000 arrangements are deliberately untouched.
   - StreamingWriteOnce semantics are deliberately untouched.
   - VCL is required only for building the frozen n<=100 implementation.
   - Highway 1.4.0 is required for the frozen >1400..3000 implementation.
   - pthreadpool from the cached XNNPACK dependency is required for 101..1400.
*/

#include <algorithm>
#include <cstddef>
#include <memory>
#include "exp53_batch_custom_2core_nt_frozen.hpp"
#include "exp53_hybrid_0101_1400.hpp"
#include "exp53_highway_sync_1400_3000_frozen.hpp"

extern "C" void exp53_vcl_u2z_0100_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_nt_sfence(double*, const double*, size_t);

enum class Exp53OutputPolicy {
    ReuseOrConsumeSoon = 0,
    StreamingWriteOnce = 1
};

class Exp53BatchProductionExecutor {
public:
    static constexpr size_t kSmallVCLU2ZMaxN = 100;
    static constexpr size_t kHybridMaxN = 1400;
    static constexpr size_t kHighwaySyncMinN = 1401;
    static constexpr size_t kHighwaySyncMaxN = 3000;
    static constexpr size_t kSerialTemporalMaxN = 3000;

    explicit Exp53BatchProductionExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    void run(double *out, const double *in, size_t n, long workers = 2) {
        if (n <= kSmallVCLU2ZMaxN) {
            exp53_vcl_u2z_0100_frozen(out, in, n);
        } else if (max_workers_ <= 1 || workers <= 1) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        } else if (n <= kHybridMaxN) {
            hybrid_0101_1400().run(out, in, n, workers);
        } else if (n <= kHighwaySyncMaxN) {
            highway_1400_3000().run(out, in, n);
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
    Exp53Hybrid0101_1400& hybrid_0101_1400() {
        if (!hybrid_0101_1400_)
            hybrid_0101_1400_ = std::make_unique<Exp53Hybrid0101_1400>(2);
        return *hybrid_0101_1400_;
    }

    Exp53HighwaySync1400_3000Frozen& highway_1400_3000() {
        if (!highway_1400_3000_)
            highway_1400_3000_ = std::make_unique<Exp53HighwaySync1400_3000Frozen>();
        return *highway_1400_3000_;
    }

    Exp53CustomPermanent2CoreFrozen& custom() {
        if (!frozen_)
            frozen_ = std::make_unique<Exp53CustomPermanent2CoreFrozen>();
        return *frozen_;
    }

    long max_workers_;
    std::unique_ptr<Exp53Hybrid0101_1400> hybrid_0101_1400_;
    std::unique_ptr<Exp53HighwaySync1400_3000Frozen> highway_1400_3000_;
    std::unique_ptr<Exp53CustomPermanent2CoreFrozen> frozen_;
};
