#pragma once

/* EXPERIMENTAL ONLY — absolute-aggressive 100..3000 two-core geometry search.
   Production/frozen files are untouched.

   Design:
   - caller CPU0 + permanent helper CPU2
   - immutable bitwise-safe frozen serial EXP53 worker on both cores
   - no queue/task scheduler/allocation/work stealing
   - caller supplies helper share per call, allowing dense per-size tuning
   - split alignment selectable: 16 or 32 values
   - one release/acquire generation handoff + one completion wait

   The benchmark must correctness-gate every geometry against the immutable
   frozen serial result before reporting it as eligible.
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

class Exp53AbsoluteAggressive2CoreCandidate {
public:
    Exp53AbsoluteAggressive2CoreCandidate()
        : generation_(0), completed_(0), stop_(false), helper_ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53AbsoluteAggressive2CoreCandidate() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53AbsoluteAggressive2CoreCandidate(const Exp53AbsoluteAggressive2CoreCandidate&) = delete;
    Exp53AbsoluteAggressive2CoreCandidate& operator=(const Exp53AbsoluteAggressive2CoreCandidate&) = delete;

    void run_share(double *out, const double *in, size_t n,
                   unsigned helper_percent, size_t align_values) {
        if (!n) return;
        if (helper_percent == 0 || helper_percent >= 100) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
            return;
        }
        if (align_values != 16 && align_values != 32) align_values = 32;

        size_t h = (n * (size_t)helper_percent) / 100;
        h = (h / align_values) * align_values;
        if (h < align_values || h >= n) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
            return;
        }

        const size_t split = n - h;
        out2_ = out + split;
        in2_ = in + split;
        n2_ = h;

        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        exp53_n2_vmstyle_u4_0381_frozen(out, in, split);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
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
            while ((g = generation_.load(std::memory_order_acquire)) == seen) _mm_pause();
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;
            double *o = out2_;
            const double *i = in2_;
            const size_t n = n2_;
            exp53_n2_vmstyle_u4_0381_frozen(o, i, n);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> stop_;
    alignas(64) std::atomic<bool> helper_ready_;
    double *out2_;
    const double *in2_;
    size_t n2_;
};
