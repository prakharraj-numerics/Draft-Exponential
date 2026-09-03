#pragma once

/* EXP53 EXPERIMENT ONLY.
   Same frozen EXP53 math and routing as production. The only experiment is
   helper lifecycle: the selected helper keeps the original spin handoff;
   constructed inactive helpers block on a condition variable.

   Synchronization invariant (v3):
     - payload + generation are published BEFORE waking a parked helper;
     - the worker's `seen` generation is never reset on wake;
     - park -> wake transitions are serialized with the same mutex used by
       condition_variable::wait, preventing a lost notification between the
       predicate check and the thread actually sleeping;
     - an already-active helper stays on the original mutex-free spin path.
*/

#define HWY_COMPILE_ONLY_STATIC
#include "hwy/highway.h"
#include "production/exp53_highway_sync_1600_3000_constants_frozen.hpp"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

extern "C" void exp53_vcl_u2z_0100_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_nt_sfence(double*, const double*, size_t);

namespace exp53_park_hwy_ns {
namespace hn = hwy::HWY_NAMESPACE;
namespace c = exp53_hwy_const_frozen;

static HWY_INLINE void vec8(double* out, const double* in) {
    const hn::FixedTag<double, 8> d;
    const hn::RebindToSigned<decltype(d)> di;
    const auto inv=hn::Set(d,c::INV128), hi=hn::Set(d,c::L128_HI), mi=hn::Set(d,c::L128_MI),
               lo=hn::Set(d,c::L128_LO), magic=hn::Set(d,c::MAGIC), one=hn::Set(d,1.0),
               q1=hn::Set(d,c::Q1), q2=hn::Set(d,c::Q2), q3=hn::Set(d,c::Q3), q4=hn::Set(d,c::Q4);
    const auto mb=hn::Set(di,(int64_t)c::MAGIC_BITS), mask=hn::Set(di,127);
    const auto* tab=reinterpret_cast<const int64_t*>(c::TAB128);

    auto x=hn::LoadU(d,in);
    auto biased=hn::MulAdd(x,inv,magic);
    auto k=hn::Sub(biased,magic);
    auto kn=hn::Sub(hn::BitCast(di,biased),mb);
    auto j=hn::And(kn,mask);
    auto q=hn::ShiftRight<7>(kn);
    auto tb=hn::GatherIndex(di,tab,j);

    auto r=hn::NegMulAdd(k,hi,x);
    r=hn::NegMulAdd(k,mi,r);
    r=hn::NegMulAdd(k,lo,r);

    auto h=hn::MulAdd(q4,r,q3);
    h=hn::MulAdd(h,r,q2);
    h=hn::MulAdd(h,r,q1);
    h=hn::MulAdd(h,r,one);
    auto s=hn::Mul(h,h);
    auto er=hn::MulAdd(r,s,one);
    auto el=hn::MulAdd(r,s,hn::Sub(one,er));

    auto sb=hn::Add(tb,hn::ShiftLeft<52>(q));
    auto scale=hn::BitCast(d,sb);
    auto ph=hn::Mul(er,scale);
    auto y=hn::MulAdd(el,scale,ph);
    hn::StoreU(y,d,out);
}

static inline void kernel(double* out, const double* in, size_t n) {
    size_t i=0;
    for(; i+32<=n; i+=32) {
        vec8(out+i, in+i);
        vec8(out+i+8, in+i+8);
        vec8(out+i+16, in+i+16);
        vec8(out+i+24, in+i+24);
    }
    for(; i+8<=n; i+=8) vec8(out+i,in+i);
    if(i<n) {
        alignas(64) double ti[8]={0,0,0,0,0,0,0,0};
        alignas(64) double to[8];
        const size_t rem=n-i;
        std::memcpy(ti,in+i,rem*sizeof(double));
        vec8(to,ti);
        std::memcpy(out+i,to,rem*sizeof(double));
    }
}

static inline void pin(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu,&set);
    (void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
}
} // namespace exp53_park_hwy_ns

class Exp53HighwaySyncParkCandidate {
public:
    Exp53HighwaySyncParkCandidate()
        : generation_(0), completed_(0), stop_(false), active_(false), helper_ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        exp53_park_hwy_ns::pin(0);
        helper_=std::thread([this]{ helper_loop(); });
        while(!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53HighwaySyncParkCandidate() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_.store(true,std::memory_order_release);
            active_.store(false,std::memory_order_release);
        }
        cv_.notify_one();
        if(helper_.joinable()) helper_.join();
    }

    Exp53HighwaySyncParkCandidate(const Exp53HighwaySyncParkCandidate&) = delete;
    Exp53HighwaySyncParkCandidate& operator=(const Exp53HighwaySyncParkCandidate&) = delete;

    static unsigned helper_share(size_t n) {
        if(n < 1440) return 30;
        if(n < 1500) return 32;
        if(n < 1536) return 31;
        if(n < 1600) return 33;
        if(n < 2000) return 35;
        if(n < 2850) return 40;
        return 42;
    }

    void park() { active_.store(false,std::memory_order_release); }
    bool active() const { return active_.load(std::memory_order_acquire); }

    void run(double* out,const double* in,size_t n) {
        if(n < 1400 || n > 3000) {
            exp53_park_hwy_ns::kernel(out,in,n);
            return;
        }
        size_t helper=(n*(size_t)helper_share(n))/100;
        helper=(helper/32)*32;
        if(helper<32 || helper>=n) {
            exp53_park_hwy_ns::kernel(out,in,n);
            return;
        }
        size_t split=n-helper;
        split=(split/32)*32;
        if(split<32 || split>=n) {
            exp53_park_hwy_ns::kernel(out,in,n);
            return;
        }

        out2_=out+split;
        in2_=in+split;
        n2_=n-split;
        const uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;
        activate_published();
        exp53_park_hwy_ns::kernel(out,in,split);
        while(completed_.load(std::memory_order_acquire)!=g) _mm_pause();
    }

private:
    void activate_published() {
        if(active_.load(std::memory_order_acquire)) return;
        bool notify=false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if(!active_.load(std::memory_order_relaxed)) {
                active_.store(true,std::memory_order_release);
                notify=true;
            }
        }
        if(notify) cv_.notify_one();
    }

    void helper_loop() {
        exp53_park_hwy_ns::pin(2);
        uint64_t seen=generation_.load(std::memory_order_relaxed);
        helper_ready_.store(true,std::memory_order_release);
        for(;;) {
            if(stop_.load(std::memory_order_acquire)) return;
            if(!active_.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk,[this,&seen]{
                    return stop_.load(std::memory_order_acquire) ||
                           (active_.load(std::memory_order_acquire) &&
                            generation_.load(std::memory_order_acquire)!=seen);
                });
                if(stop_.load(std::memory_order_acquire)) return;
            }

            uint64_t g=seen;
            while(active_.load(std::memory_order_acquire) &&
                  (g=generation_.load(std::memory_order_acquire))==seen) {
                if(stop_.load(std::memory_order_acquire)) return;
                _mm_pause();
            }
            if(stop_.load(std::memory_order_acquire)) return;
            if(!active_.load(std::memory_order_acquire)) continue;
            if(g==seen) continue;
            seen=g;
            exp53_park_hwy_ns::kernel(out2_,in2_,n2_);
            completed_.store(g,std::memory_order_release);
        }
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> stop_;
    alignas(64) std::atomic<bool> active_;
    alignas(64) std::atomic<bool> helper_ready_;
    std::mutex mu_;
    std::condition_variable cv_;
    alignas(64) double* out2_;
    const double* in2_;
    size_t n2_;
};

class Exp53Custom2ParkCandidate {
public:
    using fn_t = void (*)(double*, const double*, size_t);

    Exp53Custom2ParkCandidate()
        : generation_(0), completed_(0), stop_(false), active_(false), helper_ready_(false),
          out_(nullptr), in_(nullptr), n2_(0), fn_(nullptr) {
        pin_current_thread(0);
        helper_=std::thread([this]{ helper_loop(); });
        while(!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53Custom2ParkCandidate() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_.store(true,std::memory_order_release);
            active_.store(false,std::memory_order_release);
        }
        cv_.notify_one();
        if(helper_.joinable()) helper_.join();
    }

    Exp53Custom2ParkCandidate(const Exp53Custom2ParkCandidate&) = delete;
    Exp53Custom2ParkCandidate& operator=(const Exp53Custom2ParkCandidate&) = delete;

    void park() { active_.store(false,std::memory_order_release); }
    bool active() const { return active_.load(std::memory_order_acquire); }

    void run(double* out,const double* in,size_t n) {
        run_with(exp53_n2_vmstyle_u4_0381_frozen,out,in,n);
    }
    void run_streaming_write_once(double* out,const double* in,size_t n) {
        run_with(exp53_n2_vmstyle_u4_0381_nt_sfence,out,in,n);
    }

private:
    static void pin_current_thread(int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu,&set);
        (void)pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
    }

    void activate_published() {
        if(active_.load(std::memory_order_acquire)) return;
        bool notify=false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if(!active_.load(std::memory_order_relaxed)) {
                active_.store(true,std::memory_order_release);
                notify=true;
            }
        }
        if(notify) cv_.notify_one();
    }

    void helper_loop() {
        pin_current_thread(2);
        uint64_t seen=generation_.load(std::memory_order_relaxed);
        helper_ready_.store(true,std::memory_order_release);
        for(;;) {
            if(stop_.load(std::memory_order_acquire)) return;
            if(!active_.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk,[this,&seen]{
                    return stop_.load(std::memory_order_acquire) ||
                           (active_.load(std::memory_order_acquire) &&
                            generation_.load(std::memory_order_acquire)!=seen);
                });
                if(stop_.load(std::memory_order_acquire)) return;
            }

            uint64_t g=seen;
            while(active_.load(std::memory_order_acquire) &&
                  (g=generation_.load(std::memory_order_acquire))==seen) {
                if(stop_.load(std::memory_order_acquire)) return;
                _mm_pause();
            }
            if(stop_.load(std::memory_order_acquire)) return;
            if(!active_.load(std::memory_order_acquire)) continue;
            if(g==seen) continue;
            seen=g;
            fn_t fn=fn_;
            double* out=out_;
            const double* in=in_;
            const size_t n=n2_;
            fn(out,in,n);
            completed_.store(g,std::memory_order_release);
        }
    }

    void run_with(fn_t fn,double* out,const double* in,size_t n) {
        if(!n) return;
        const size_t full32=n/32;
        if(full32<2) { fn(out,in,n); return; }
        const size_t split_blocks=full32/2;
        const size_t split=split_blocks*32;
        if(split==0 || split>=n) { fn(out,in,n); return; }

        out_=out+split;
        in_=in+split;
        n2_=n-split;
        fn_=fn;
        const uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;
        activate_published();
        fn(out,in,split);
        while(completed_.load(std::memory_order_acquire)!=g) _mm_pause();
    }

    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> stop_;
    alignas(64) std::atomic<bool> active_;
    alignas(64) std::atomic<bool> helper_ready_;
    std::mutex mu_;
    std::condition_variable cv_;
    double* out_;
    const double* in_;
    size_t n2_;
    fn_t fn_;
};

class Exp53ParkInactiveHelpersCandidate {
public:
    static constexpr size_t kSmallVCLU2ZMaxN=100;
    static constexpr size_t kHighwaySyncMinN=1400;
    static constexpr size_t kHighwaySyncMaxN=3000;
    static constexpr size_t kSerialTemporalMaxN=3000;
    static constexpr size_t kRareTemporalNoNTMinN=15000;
    static constexpr size_t kRareTemporalNoNTMaxN=65000;

    explicit Exp53ParkInactiveHelpersCandidate(long max_workers=2)
        : max_workers_(max_workers>1?2L:1L) {}

    void run(double* out,const double* in,size_t n,long workers=2) {
        if(n<=kSmallVCLU2ZMaxN) {
            park_all();
            exp53_vcl_u2z_0100_frozen(out,in,n);
        } else if(max_workers_<=1 || workers<=1 || n<kHighwaySyncMinN) {
            park_all();
            exp53_n2_vmstyle_u4_0381_frozen(out,in,n);
        } else if(n<=kHighwaySyncMaxN) {
            park_custom();
            highway().run(out,in,n);
        } else {
            park_highway();
            custom().run(out,in,n);
        }
    }

    void run_streaming_write_once(double* out,const double* in,size_t n,long workers=2) {
        if(n<=kSmallVCLU2ZMaxN) {
            park_all();
            exp53_vcl_u2z_0100_frozen(out,in,n);
        } else if(n<=kSerialTemporalMaxN) {
            park_all();
            exp53_n2_vmstyle_u4_0381_frozen(out,in,n);
        } else if(max_workers_<=1 || workers<=1) {
            park_all();
            exp53_n2_vmstyle_u4_0381_nt_sfence(out,in,n);
        } else if(n>=kRareTemporalNoNTMinN && n<=kRareTemporalNoNTMaxN) {
            park_highway();
            custom().run(out,in,n);
        } else {
            park_highway();
            custom().run_streaming_write_once(out,in,n);
        }
    }

    void prime_both(double* out,const double* in) {
        run(out,in,2000,2);
        run(out,in,5000,2);
    }

    bool highway_exists() const { return (bool)highway_; }
    bool custom_exists() const { return (bool)custom_; }
    bool highway_active() const { return highway_ && highway_->active(); }
    bool custom_active() const { return custom_ && custom_->active(); }

private:
    void park_highway() { if(highway_) highway_->park(); }
    void park_custom() { if(custom_) custom_->park(); }
    void park_all() { park_highway(); park_custom(); }

    Exp53HighwaySyncParkCandidate& highway() {
        if(!highway_) highway_=std::make_unique<Exp53HighwaySyncParkCandidate>();
        return *highway_;
    }
    Exp53Custom2ParkCandidate& custom() {
        if(!custom_) custom_=std::make_unique<Exp53Custom2ParkCandidate>();
        return *custom_;
    }

    long max_workers_;
    std::unique_ptr<Exp53HighwaySyncParkCandidate> highway_;
    std::unique_ptr<Exp53Custom2ParkCandidate> custom_;
};
