#pragma once

/* EXP53 frozen range-specialized permanent two-core path for 2500..3000.

   Frozen from the Conservative policy of
   exp53_range2core_0101_2999_candidates.hpp after exact-Xeon sweep
   run 33426419558 (Intel Xeon 6973P-C, ICX 2026.1.1).

   For this frozen production band the tested Conservative geometry is:
     - caller pinned to CPU0
     - permanent helper pinned to CPU2
     - U2Z mathematical worker on both cores
     - helper receives trailing floor16(46% of n)
     - caller receives the head plus the only possible scalar tail
     - one release/acquire generation handoff and one completion wait
     - no queue, scheduler, allocation, task graph, or work stealing

   Mathematical worker remains the immutable exp53_small_u2z_0100_frozen.
   Production input domain remains [-100,100].

   DO NOT MODIFY after freeze; make a new candidate/frozen file for changes.
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_small_u2z_0100_frozen(double*, const double*, size_t);

class Exp53Range2Core2500_3000Frozen {
public:
    Exp53Range2Core2500_3000Frozen()
        : generation_(0), completed_(0), stop_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53Range2Core2500_3000Frozen() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53Range2Core2500_3000Frozen(const Exp53Range2Core2500_3000Frozen&) = delete;
    Exp53Range2Core2500_3000Frozen& operator=(const Exp53Range2Core2500_3000Frozen&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (!n) return;

        const size_t h = a16(n * 46 / 100);
        if (h < 16 || h >= n) {
            exp53_small_u2z_0100_frozen(out, in, n);
            return;
        }

        const size_t split = n - h;
        out2_ = out + split;
        in2_ = in + split;
        n2_ = h;

        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        exp53_small_u2z_0100_frozen(out, in, split);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
    }

private:
    static size_t a16(size_t x) { return (x / 16) * 16; }

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
            exp53_small_u2z_0100_frozen(o, i, n);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_{false};
    std::atomic<bool> stop_;
    double *out2_;
    const double *in2_;
    size_t n2_;
};
