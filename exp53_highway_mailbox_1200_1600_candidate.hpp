#pragma once

/* EXP53 experimental only: low-overhead Highway sync candidate for n=1200..1600.
   Production untouched.

   Fixes applied from the 1200..1600 audit:
   - custom2-style generation store/release (no fetch_add on hot path)
   - one permanent helper pinned CPU2, caller pinned CPU0
   - exact 32-element split; helper gets only full 32-value blocks
   - caller owns the sole remainder/tail
   - precomputed size-bucket split schedule: no percentage multiply/divide in hot path
   - deliberately slightly-underloaded helper so caller completion tends to hide helper finish

   Math is reused verbatim from the frozen Highway implementation.
*/

#include "exp53_highway_sync_1600_3000_frozen.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <immintrin.h>

class Exp53HighwayMailbox1200_1600Candidate {
public:
    Exp53HighwayMailbox1200_1600Candidate()
        : generation_(0), completed_(0), stop_(false), ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0), local_generation_(0) {
        exp53_hwy_frozen_ns::pin(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53HighwayMailbox1200_1600Candidate() {
        stop_.store(true, std::memory_order_relaxed);
        const uint64_t g = ++local_generation_;
        generation_.store(g, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }

    Exp53HighwayMailbox1200_1600Candidate(const Exp53HighwayMailbox1200_1600Candidate&) = delete;
    Exp53HighwayMailbox1200_1600Candidate& operator=(const Exp53HighwayMailbox1200_1600Candidate&) = delete;

    // Caller split, always a multiple of 32. Helper owns [split,n) but only
    // full 32-blocks; any final remainder is kept on caller side by choosing
    // helper_n=floor(target/32)*32 and split=n-helper_n.
    static inline size_t helper_blocks_for(size_t n) {
        // Tuned starting schedule from earlier exact-Xeon best-share data.
        // 1200..1311 ~28-30%, 1312..1439 ~30%, 1440..1535 ~31-32%,
        // 1536..1600 ~33%. Rounded down to full blocks to underload helper.
        if (n < 1312) return 11; // 352 values
        if (n < 1440) return 13; // 416
        if (n < 1536) return 14; // 448
        return 16;               // 512
    }

    void run(double* out, const double* in, size_t n) {
        if (n < 1200 || n > 1600) {
            exp53_hwy_frozen_ns::kernel(out, in, n);
            return;
        }

        size_t helper_n = helper_blocks_for(n) * 32;
        if (helper_n >= n || helper_n < 32) {
            exp53_hwy_frozen_ns::kernel(out, in, n);
            return;
        }

        // Critical: make helper region a pure full-block region. Caller gets
        // prefix plus the only remainder, eliminating duplicate tail/setup.
        const size_t rem = n & 31u;
        size_t split = n - helper_n;
        if ((split & 31u) != rem) split += ((rem + 32u - (split & 31u)) & 31u);
        if (split >= n || n - split < 32 || ((n - split) & 31u)) {
            helper_n = ((n * 30u) / 100u) & ~size_t(31);
            split = n - helper_n;
        }

        out2_ = out + split;
        in2_ = in + split;
        n2_ = n - split;

        const uint64_t g = ++local_generation_;
        generation_.store(g, std::memory_order_release);

        // Caller owns all irregular work; helper sees only full 32-blocks.
        exp53_hwy_frozen_ns::kernel(out, in, split);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
    }

private:
    void helper_loop() {
        exp53_hwy_frozen_ns::pin(2);
        ready_.store(true, std::memory_order_release);
        uint64_t seen = generation_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t g;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) {
                if (stop_.load(std::memory_order_relaxed)) return;
                _mm_pause();
            }
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;
            exp53_hwy_frozen_ns::kernel(out2_, in2_, n2_);
            completed_.store(g, std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> stop_;
    alignas(64) std::atomic<bool> ready_;
    alignas(64) double* out2_;
    const double* in2_;
    size_t n2_;
    uint64_t local_generation_;
};
