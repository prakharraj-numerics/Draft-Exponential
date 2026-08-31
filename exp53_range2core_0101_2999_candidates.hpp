#pragma once

/* Experimental EXP53 101..2999 range-specialized two-core micro-batchers.
   Frozen/production files are untouched.

   Design goals:
   - exactly two physical cores: caller CPU0, permanent helper CPU2
   - one release/acquire handoff and one completion per parallel call
   - no queue, allocation, scheduler, task graph, or work stealing
   - U2Z worker, 16-value alignment
   - deliberately asymmetric helper share at lower n to hide wake/handoff latency
   - serial escape for sizes where prior evidence says 2-core dispatch is toxic

   Three schedules are exposed so one Xeon sweep can determine the best geometry.
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_small_u2z_0100_frozen(double*, const double*, size_t);

class Exp53Range2CoreBase {
public:
    enum class Policy { Conservative, Balanced, Aggressive };

    explicit Exp53Range2CoreBase(Policy p)
        : policy_(p), generation_(0), completed_(0), stop_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53Range2CoreBase() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53Range2CoreBase(const Exp53Range2CoreBase&) = delete;
    Exp53Range2CoreBase& operator=(const Exp53Range2CoreBase&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (!n) return;
        const size_t h = helper_count(n);
        if (h < 16 || h >= n) {
            exp53_small_u2z_0100_frozen(out, in, n);
            return;
        }

        // Give the helper the trailing aligned block(s); caller owns the head.
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

    size_t helper_count(size_t n) const {
        // Keep one global tail only: helper sizes are 16-aligned, caller gets remainder.
        if (policy_ == Policy::Conservative) {
            if (n < 900) return 0;
            if (n < 1300) return a16(n * 25 / 100);
            if (n < 1800) return a16(n * 34 / 100);
            if (n < 2300) return a16(n * 41 / 100);
            return a16(n * 46 / 100);
        }
        if (policy_ == Policy::Balanced) {
            if (n < 650) return 0;
            if (n < 1000) return a16(n * 22 / 100);
            if (n < 1400) return a16(n * 32 / 100);
            if (n < 1900) return a16(n * 40 / 100);
            if (n < 2400) return a16(n * 45 / 100);
            return a16(n * 48 / 100);
        }
        // Aggressive: starts parallelism earlier and converges to near-even sooner.
        if (n < 450) return 0;
        if (n < 750) return a16(n * 20 / 100);
        if (n < 1100) return a16(n * 30 / 100);
        if (n < 1500) return a16(n * 38 / 100);
        if (n < 2000) return a16(n * 44 / 100);
        return a16(n * 49 / 100);
    }

    static void pin_current_thread(int cpu) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
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

    Policy policy_;
    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_{false};
    std::atomic<bool> stop_;
    double *out2_;
    const double *in2_;
    size_t n2_;
};

class Exp53Range2CoreConservative : public Exp53Range2CoreBase {
public: Exp53Range2CoreConservative() : Exp53Range2CoreBase(Policy::Conservative) {}
};
class Exp53Range2CoreBalanced : public Exp53Range2CoreBase {
public: Exp53Range2CoreBalanced() : Exp53Range2CoreBase(Policy::Balanced) {}
};
class Exp53Range2CoreAggressive : public Exp53Range2CoreBase {
public: Exp53Range2CoreAggressive() : Exp53Range2CoreBase(Policy::Aggressive) {}
};
