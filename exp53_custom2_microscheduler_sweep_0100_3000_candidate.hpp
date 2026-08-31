#pragma once

/* EXPERIMENTAL ONLY. 100..3000 custom2 microscheduler candidate.
   Frozen production/kernel files are untouched.
   Permanent helper on CPU2, caller CPU0, cache-line isolated generation/completion.
   The benchmark supplies the helper share; caller split is aligned to 32 values. */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

class Exp53Custom2MicroschedulerSweepCandidate {
public:
    Exp53Custom2MicroschedulerSweepCandidate()
        : generation_(0), completed_(0), stop_(false), helper_ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }
    ~Exp53Custom2MicroschedulerSweepCandidate() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }
    Exp53Custom2MicroschedulerSweepCandidate(const Exp53Custom2MicroschedulerSweepCandidate&) = delete;
    Exp53Custom2MicroschedulerSweepCandidate& operator=(const Exp53Custom2MicroschedulerSweepCandidate&) = delete;

    void run_share(double *out, const double *in, size_t n, unsigned helper_pct) {
        if (!n) return;
        if (helper_pct == 0) { exp53_n2_vmstyle_u4_0381_frozen(out,in,n); return; }
        const size_t desired_h=(n*(size_t)helper_pct)/100;
        size_t split=n-desired_h;
        split=((split+31u)/32u)*32u;
        if (split<32 || split>=n || n-split<32) { exp53_n2_vmstyle_u4_0381_frozen(out,in,n); return; }
        out2_=out+split; in2_=in+split; n2_=n-split;
        const uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;
        exp53_n2_vmstyle_u4_0381_frozen(out,in,split);
        while (completed_.load(std::memory_order_acquire)!=g) _mm_pause();
    }
private:
    static void pin_current_thread(int cpu) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu,&set);
        (void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
    }
    void helper_loop() {
        pin_current_thread(2);
        helper_ready_.store(true,std::memory_order_release);
        uint64_t seen=generation_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t g;
            while ((g=generation_.load(std::memory_order_acquire))==seen) _mm_pause();
            seen=g;
            if (stop_.load(std::memory_order_relaxed)) return;
            exp53_n2_vmstyle_u4_0381_frozen(out2_,in2_,n2_);
            completed_.store(g,std::memory_order_release);
        }
    }
    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> stop_;
    alignas(64) std::atomic<bool> helper_ready_;
    double *out2_; const double *in2_; size_t n2_;
};
