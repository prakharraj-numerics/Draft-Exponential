#pragma once
/* Isolated executor candidate for compute-efficiency audit.
   production/ is untouched. Routing is identical to production except:
   - serial temporal 101..1399 uses the faithful masked-tail candidate;
   - custom2 helper uses the masked-tail candidate only when its second chunk
     has a nonzero remainder modulo 32;
   - Highway 1400..3000 and all full-32 chunks remain production code. */
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
#include "production/exp53_highway_sync_1400_3000_frozen.hpp"

extern "C" void exp53_vcl_u2z_0100_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_masktail_candidate(double*, const double*, size_t);

class Exp53TailEffCustom2Candidate {
public:
    using fn_t = void (*)(double*, const double*, size_t);
    Exp53TailEffCustom2Candidate()
        : generation_(0), completed_(0), stop_(false),
          out_(nullptr), in_(nullptr), n2_(0), fn_(nullptr) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }
    ~Exp53TailEffCustom2Candidate() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }
    Exp53TailEffCustom2Candidate(const Exp53TailEffCustom2Candidate&) = delete;
    Exp53TailEffCustom2Candidate& operator=(const Exp53TailEffCustom2Candidate&) = delete;

    void run(double *out, const double *in, size_t n) {
        if (!n) return;
        const size_t full32 = n / 32;
        if (full32 < 2) {
            exp53_n2_vmstyle_u4_0381_masktail_candidate(out, in, n);
            return;
        }
        const size_t split_blocks = full32 / 2;
        const size_t split = split_blocks * 32;
        if (split == 0 || split >= n) {
            exp53_n2_vmstyle_u4_0381_masktail_candidate(out, in, n);
            return;
        }

        out_ = out + split;
        in_ = in + split;
        n2_ = n - split;
        fn_ = (n2_ & 31u) ? exp53_n2_vmstyle_u4_0381_masktail_candidate
                           : exp53_n2_vmstyle_u4_0381_frozen;

        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        /* split is always 32-aligned, so caller stays byte-for-byte production math. */
        exp53_n2_vmstyle_u4_0381_frozen(out, in, split);
        while (completed_.load(std::memory_order_acquire) != g) _mm_pause();
    }

private:
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
            fn_t fn = fn_;
            fn(out_, in_, n2_);
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
    fn_t fn_;
};

class Exp53TailEffExecutorCandidate {
public:
    explicit Exp53TailEffExecutorCandidate(long max_workers=2)
        : max_workers_(max_workers > 1 ? 2L : 1L) {}

    void run(double *out, const double *in, size_t n, long workers=2) {
        if (n <= 100) {
            exp53_vcl_u2z_0100_frozen(out, in, n);
        } else if (max_workers_ <= 1 || workers <= 1 || n < 1400) {
            exp53_n2_vmstyle_u4_0381_masktail_candidate(out, in, n);
        } else if (n <= 3000) {
            highway().run(out, in, n);
        } else {
            custom().run(out, in, n);
        }
    }

private:
    Exp53HighwaySync1400_3000Frozen& highway() {
        if (!highway_) highway_ = std::make_unique<Exp53HighwaySync1400_3000Frozen>();
        return *highway_;
    }
    Exp53TailEffCustom2Candidate& custom() {
        if (!custom_) custom_ = std::make_unique<Exp53TailEffCustom2Candidate>();
        return *custom_;
    }
    long max_workers_;
    std::unique_ptr<Exp53HighwaySync1400_3000Frozen> highway_;
    std::unique_ptr<Exp53TailEffCustom2Candidate> custom_;
};
