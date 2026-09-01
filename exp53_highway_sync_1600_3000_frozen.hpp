#pragma once

/* EXP53 FROZEN Highway synchronized temporal batch path, n=1600..3000.

   Frozen from exact-Xeon validation runs:
     33496774491 : Highway fixed-tail sync 101..2000, Xeon 6973P-C
     33497347768 : Highway vs frozen StoreSeq 2000..3000, Xeon 6973P-C

   Mathematical machinery is unchanged:
     Q4(r)=1+r/4+5r^2/96+r^3/128+79r^4/92160
     e^r = 1 + r Q4(r)^2
   Same reduction constants, TAB128, Horner order, square and ER-low repair.
   No hwy::Exp and no replacement exp implementation is used.

   The historical frozen scalar/libm remainder is NOT used here. The remainder
   is evaluated by the same EXP53 mathematical machinery in 8-lane Highway
   vectors (with padded inactive lanes for the final partial vector).

   Validation gates in the cited runs:
     maxULP vs std::exp = 1
     gt1 = 0
     bulk_bitdiff vs frozen VM-style kernel = 0

   Execution:
     - caller pinned CPU0, permanent helper pinned CPU2
     - temporal stores
     - 32-element aligned partition
     - Highway 1.4.0 static AVX-512 backend
     - helper share frozen by size:
         1600..1999 -> 35%
         2000..2849 -> 40%
         2850..3000 -> 42%

   IMPORTANT:
     - This file is production-frozen. Do not modify in place.
     - n<=100 and n>3000 production arrangements are untouched.
     - StreamingWriteOnce routing is intentionally untouched.
*/

#define HWY_COMPILE_ONLY_STATIC
#include "hwy/highway.h"
#include "exp53_highway_sync_1600_3000_constants_frozen.hpp"
#include <atomic>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

namespace exp53_hwy_frozen_ns {
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
        vec8(out+i,    in+i);
        vec8(out+i+8,  in+i+8);
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

} // namespace exp53_hwy_frozen_ns

class Exp53HighwaySync1600_3000Frozen {
public:
    Exp53HighwaySync1600_3000Frozen()
        : generation_(0), completed_(0), stop_(false), helper_ready_(false),
          out2_(nullptr), in2_(nullptr), n2_(0) {
        exp53_hwy_frozen_ns::pin(0);
        helper_=std::thread([this]{ helper_loop(); });
        while(!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }

    ~Exp53HighwaySync1600_3000Frozen() {
        stop_.store(true,std::memory_order_relaxed);
        generation_.fetch_add(1,std::memory_order_release);
        if(helper_.joinable()) helper_.join();
    }

    Exp53HighwaySync1600_3000Frozen(const Exp53HighwaySync1600_3000Frozen&) = delete;
    Exp53HighwaySync1600_3000Frozen& operator=(const Exp53HighwaySync1600_3000Frozen&) = delete;

    static unsigned helper_share(size_t n) {
        if(n < 2000) return 35;
        if(n < 2850) return 40;
        return 42;
    }

    void run(double* out,const double* in,size_t n) {
        if(n < 1600 || n > 3000) {
            exp53_hwy_frozen_ns::kernel(out,in,n);
            return;
        }
        size_t helper=(n*(size_t)helper_share(n))/100;
        helper=(helper/32)*32;
        if(helper<32 || helper>=n) {
            exp53_hwy_frozen_ns::kernel(out,in,n);
            return;
        }
        size_t split=n-helper;
        split=(split/32)*32;
        if(split<32 || split>=n) {
            exp53_hwy_frozen_ns::kernel(out,in,n);
            return;
        }

        out2_=out+split;
        in2_=in+split;
        n2_=n-split;
        const uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;
        exp53_hwy_frozen_ns::kernel(out,in,split);
        while(completed_.load(std::memory_order_acquire)!=g) _mm_pause();
    }

private:
    void helper_loop() {
        exp53_hwy_frozen_ns::pin(2);
        helper_ready_.store(true,std::memory_order_release);
        uint64_t seen=generation_.load(std::memory_order_relaxed);
        for(;;) {
            uint64_t g;
            while((g=generation_.load(std::memory_order_acquire))==seen) _mm_pause();
            seen=g;
            if(stop_.load(std::memory_order_relaxed)) return;
            exp53_hwy_frozen_ns::kernel(out2_,in2_,n2_);
            completed_.store(g,std::memory_order_release);
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
};
