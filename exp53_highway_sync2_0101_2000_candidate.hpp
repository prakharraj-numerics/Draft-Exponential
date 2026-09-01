#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

void exp53_highway_0101_2000_candidate(double*, const double*, size_t);

/* Experimental only: permanent 2-physical-core executor for the exact Highway
   bulk implementation. Caller is CPU0, helper is CPU2 on the validated
   Xeon 6973P-C topology. One release/acquire generation handoff starts helper;
   one release/acquire completion publication joins each call. No queues,
   allocations, task objects, or per-call thread creation. */
class Exp53HighwaySync2Candidate {
public:
    Exp53HighwaySync2Candidate()
        : generation_(0), completed_(0), stop_(false), helper_ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53HighwaySync2Candidate() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53HighwaySync2Candidate(const Exp53HighwaySync2Candidate&) = delete;
    Exp53HighwaySync2Candidate& operator=(const Exp53HighwaySync2Candidate&) = delete;

    void run(double* out, const double* in, size_t n, unsigned helper_percent) {
        if (!n) return;
        const size_t full32 = n / 32;
        if (full32 < 2 || helper_percent == 0) {
            exp53_highway_0101_2000_candidate(out, in, n);
            return;
        }

        size_t helper_blocks = (full32 * helper_percent + 50) / 100;
        if (helper_blocks < 1) helper_blocks = 1;
        if (helper_blocks >= full32) helper_blocks = full32 - 1;
        const size_t caller_blocks = full32 - helper_blocks;
        const size_t split = caller_blocks * 32;

        out2_ = out + split;
        in2_ = in + split;
        n2_ = n - split;  // helper owns remainder too; Highway exact tail handles it.

        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        exp53_highway_0101_2000_candidate(out, in, split);
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

            double* out = out2_;
            const double* in = in2_;
            const size_t n = n2_;
            exp53_highway_0101_2000_candidate(out, in, n);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> stop_;
    alignas(64) std::atomic<bool> helper_ready_;

    double* out2_;
    const double* in2_;
    size_t n2_;
};
