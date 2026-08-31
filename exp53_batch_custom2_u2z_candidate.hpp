#pragma once

/* EXP53 experimental custom2 + U2Z candidate.
   Does not modify frozen production/custom files.

   Same permanent two-core handoff as the frozen custom2 survivor, but:
     - mathematical worker function is exp53_small_u2z_0100_frozen
     - split granularity is 16 values (one U2Z loop), not 32
     - serial fallback only when fewer than two full 16-value blocks exist

   Intended solely for 50..3000 crossover benchmarking.
*/
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_small_u2z_0100_frozen(double*, const double*, size_t);

class Exp53Custom2U2ZCandidate {
public:
    using fn_t = void (*)(double*, const double*, size_t);

    Exp53Custom2U2ZCandidate()
        : generation_(0), completed_(0), stop_(false),
          out_(nullptr), in_(nullptr), n2_(0) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53Custom2U2ZCandidate() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53Custom2U2ZCandidate(const Exp53Custom2U2ZCandidate&) = delete;
    Exp53Custom2U2ZCandidate& operator=(const Exp53Custom2U2ZCandidate&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (!n) return;
        const size_t full16 = n / 16;
        if (full16 < 2) {
            exp53_small_u2z_0100_frozen(out, in, n);
            return;
        }
        const size_t split_blocks = full16 / 2;
        const size_t split = split_blocks * 16;
        if (!split || split >= n) {
            exp53_small_u2z_0100_frozen(out, in, n);
            return;
        }

        out_ = out + split;
        in_ = in + split;
        n2_ = n - split;
        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        exp53_small_u2z_0100_frozen(out, in, split);
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
            double *out = out_;
            const double *in = in_;
            const size_t n = n2_;
            exp53_small_u2z_0100_frozen(out, in, n);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_{false};
    std::atomic<bool> stop_;
    double *out_;
    const double *in_;
    size_t n2_;
};
