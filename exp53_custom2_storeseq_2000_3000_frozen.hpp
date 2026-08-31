#pragma once

/* EXP53 FROZEN 2000..3000 temporal batch path.

   Validated on exact Intel Xeon 6973P-C in workflow run 33437986300.
   This file is production-frozen. Do not modify in place.

   Scope:
     - default/common temporal output policy only
     - n in [2000,3000]
     - workers > 1

   Out of scope and intentionally untouched:
     - n <= 100 frozen VCL+u2z path
     - 101..1999 frozen serial temporal path
     - n > 3000 frozen custom2/NT arrangements
     - explicit StreamingWriteOnce policy for n <= 3000

   Mechanism:
     - one permanent helper pinned to CPU2; caller pinned to CPU0
     - caller is sole generation writer
     - generation publication is atomic store(release), not fetch_add
     - helper completion is store(release); caller waits acquire
     - frozen temporal EXP53 kernel on both partitions
     - 32-value split alignment
     - helper-share schedule frozen from exact-Xeon validation:
         2000..2149 -> 34%
         2150..2849 -> 38%
         2850..3000 -> 41%
*/

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

class Exp53Custom2StoreSeq2000_3000Frozen {
public:
    Exp53Custom2StoreSeq2000_3000Frozen()
        : generation_(0), completed_(0), stop_(false), helper_ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0), local_generation_(0) {
        pin(0);
        helper_ = std::thread([this] { helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53Custom2StoreSeq2000_3000Frozen() {
        stop_.store(true, std::memory_order_relaxed);
        const uint64_t g = ++local_generation_;
        generation_.store(g, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53Custom2StoreSeq2000_3000Frozen(const Exp53Custom2StoreSeq2000_3000Frozen&) = delete;
    Exp53Custom2StoreSeq2000_3000Frozen& operator=(const Exp53Custom2StoreSeq2000_3000Frozen&) = delete;

    static unsigned helper_share(size_t n) {
        if (n < 2150) return 34;
        if (n < 2850) return 38;
        return 41;
    }

    void run(double* out, const double* in, size_t n) {
        const unsigned hp = helper_share(n);
        size_t helper_n = n * static_cast<size_t>(hp) / 100;
        size_t split = n - helper_n;
        split = ((split + 31) / 32) * 32;

        if (split < 32 || split >= n || n - split < 32) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
            return;
        }

        out2_ = out + split;
        in2_ = in + split;
        n2_ = n - split;

        const uint64_t g = ++local_generation_;
        generation_.store(g, std::memory_order_release);

        exp53_n2_vmstyle_u4_0381_frozen(out, in, split);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
    }

private:
    static void pin(int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    void helper_loop() {
        pin(2);
        helper_ready_.store(true, std::memory_order_release);
        uint64_t seen = generation_.load(std::memory_order_relaxed);

        for (;;) {
            uint64_t g;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                _mm_pause();
            }
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
    alignas(64) double* out2_;
    const double* in2_;
    size_t n2_;
    uint64_t local_generation_;
};
