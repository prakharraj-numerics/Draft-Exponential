#pragma once

/* EXPERIMENTAL ONLY — custom2 architecture with conservative size-dependent
   helper participation for 100..3000.

   Critical correctness rule: the caller/helper split itself is always a
   multiple of 32 values. Both sides execute the immutable frozen u4 worker,
   preserving the frozen worker's 32-value block decomposition. Production
   and all frozen files remain untouched.
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

class Exp53Custom2ConservativeSafeCandidate {
public:
    Exp53Custom2ConservativeSafeCandidate()
        : generation_(0), completed_(0), stop_(false), helper_ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53Custom2ConservativeSafeCandidate() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53Custom2ConservativeSafeCandidate(const Exp53Custom2ConservativeSafeCandidate&) = delete;
    Exp53Custom2ConservativeSafeCandidate& operator=(const Exp53Custom2ConservativeSafeCandidate&) = delete;

    static unsigned conservative_share(size_t n) {
        if (n < 900)  return 0;
        if (n < 1300) return 25;
        if (n < 1800) return 34;
        if (n < 2300) return 41;
        return 46;
    }

    void run(double *out, const double *in, size_t n) {
        const unsigned hp = conservative_share(n);
        if (!n || hp == 0) {
            if (n) exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
            return;
        }

        /* Choose caller split first and force it to a 32-value boundary.
           Helper receives the trailing remainder. This is the key difference
           from the failed aggressive experiment, which aligned helper length. */
        const size_t desired_h = (n * (size_t)hp) / 100;
        size_t split = n - desired_h;
        split = ((split + 31u) / 32u) * 32u;
        if (split < 32 || split >= n || n - split < 32) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
            return;
        }

        out2_ = out + split;
        in2_ = in + split;
        n2_ = n - split;

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
            exp53_n2_vmstyle_u4_0381_frozen(out2_, in2_, n2_);
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
