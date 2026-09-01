#pragma once

/* EXP53 FROZEN PRODUCTION SURVIVOR — custom permanent 2-core + rare NT mode.

   Frozen after the completed 50 .. 8M boundary sweep on Intel Xeon 6973P-C.
   Do not modify this file. Future experiments must use new candidate files.

   Mathematical kernels:
     default/common: exp53_n2_vmstyle_u4_0381_frozen
     rare streaming/write-once: exp53_n2_vmstyle_u4_0381_nt_sfence

   Execution model on the validated 2-physical-core Xeon runner:
     - caller thread owns the first 32-element-aligned half
     - one permanent helper thread owns the second half
     - caller pinned to CPU0, helper pinned to CPU2
       (validated topology: CPU0/1 = physical core0, CPU2/3 = physical core1)
     - one release/acquire generation handoff starts the helper
     - one release/acquire completion counter ends the call
     - no queue, scheduler, work stealing, or task allocation

   Store policy:
     - run(): temporal stores, default/common case
     - run_streaming_write_once(): NT stores, explicit rare case only
     - no automatic n-based NT threshold

   Validation evidence before freeze:
     - final boundary workflow run 33412987407
     - exact Xeon 6973P-C shards 6 and 54
     - both output policies bit-identical to frozen serial
     - custom runtime beat or tied merged FastFlow over the completed 50..8M sweep;
       no tested size gave FastFlow a replicated performance advantage

   This frozen implementation intentionally preserves the exact validated
   candidate behavior, including the serial fallback for fewer than two full
   32-element blocks.
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <algorithm>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_nt_sfence(double*, const double*, size_t);

class Exp53CustomPermanent2CoreFrozen {
public:
    using fn_t = void (*)(double*, const double*, size_t);

    Exp53CustomPermanent2CoreFrozen()
        : generation_(0), completed_(0), stop_(false),
          out_(nullptr), in_(nullptr), n2_(0), fn_(nullptr) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53CustomPermanent2CoreFrozen() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53CustomPermanent2CoreFrozen(const Exp53CustomPermanent2CoreFrozen&) = delete;
    Exp53CustomPermanent2CoreFrozen& operator=(const Exp53CustomPermanent2CoreFrozen&) = delete;

    void run(double *out, const double *in, size_t n) {
        run_with(exp53_n2_vmstyle_u4_0381_frozen, out, in, n);
    }

    void run_streaming_write_once(double *out, const double *in, size_t n) {
        run_with(exp53_n2_vmstyle_u4_0381_nt_sfence, out, in, n);
    }

private:
    static void pin_current_thread(int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    void helper_loop() {
        pin_current_thread(2);
        helper_ready_.store(true, std::memory_order_release);
        uint64_t seen = generation_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t g;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) {
                _mm_pause();
            }
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;

            fn_t fn = fn_;
            double *out = out_;
            const double *in = in_;
            const size_t n = n2_;
            fn(out, in, n);
            completed_.store(g, std::memory_order_release);
        }
    }

    void run_with(fn_t fn, double *out, const double *in, size_t n) {
        if (!n) return;
        const size_t full32 = n / 32;
        if (full32 < 2) {
            fn(out, in, n);
            return;
        }

        const size_t split_blocks = full32 / 2;
        const size_t split = split_blocks * 32;
        if (split == 0 || split >= n) {
            fn(out, in, n);
            return;
        }

        out_ = out + split;
        in_ = in + split;
        n2_ = n - split;
        fn_ = fn;

        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        fn(out, in, split);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_{false};
    std::atomic<bool> stop_;

    double *out_;
    const double *in_;
    size_t n2_;
    fn_t fn_;
};
