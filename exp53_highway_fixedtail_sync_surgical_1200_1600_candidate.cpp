// Experimental surgical patch on old synchronized Highway executor for n=1200..1600.
// Production untouched. Same math, same partition geometry, same tail ownership.
#define HWY_COMPILE_ONLY_STATIC
#include "hwy/highway.h"
#include <atomic>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

#define restrict __restrict__
extern "C" {
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
}
#undef restrict

namespace hn = hwy::HWY_NAMESPACE;

static HWY_INLINE void exp53_hwy_vec8_surg(double* out, const double* in) {
  const hn::FixedTag<double, 8> d;
  const hn::RebindToSigned<decltype(d)> di;
  const auto inv=hn::Set(d,N2F_INV128), hi=hn::Set(d,N2F_L128_HI), mi=hn::Set(d,N2F_L128_MI),
             lo=hn::Set(d,N2F_L128_LO), magic=hn::Set(d,N2F_MAGIC), one=hn::Set(d,1.0),
             q1=hn::Set(d,N2F_Q1), q2=hn::Set(d,N2F_Q2), q3=hn::Set(d,N2F_Q3), q4=hn::Set(d,N2F_Q4);
  const auto mb=hn::Set(di,(int64_t)N2F_MAGIC_BITS), mask=hn::Set(di,127);
  const auto* tab=reinterpret_cast<const int64_t*>(N2_FROZEN_TAB128);
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

extern "C" __attribute__((noinline))
void exp53_highway_fixedtail_surg_1200_1600(double* out,const double* in,size_t n) {
  size_t i=0;
  for(; i+32<=n; i+=32) {
    exp53_hwy_vec8_surg(out+i,    in+i);
    exp53_hwy_vec8_surg(out+i+8,  in+i+8);
    exp53_hwy_vec8_surg(out+i+16, in+i+16);
    exp53_hwy_vec8_surg(out+i+24, in+i+24);
  }
  for(; i+8<=n; i+=8) exp53_hwy_vec8_surg(out+i,in+i);
  if(i<n) {
    alignas(64) double ti[8]={0,0,0,0,0,0,0,0};
    alignas(64) double to[8];
    const size_t rem=n-i;
    std::memcpy(ti,in+i,rem*sizeof(double));
    exp53_hwy_vec8_surg(to,ti);
    std::memcpy(out+i,to,rem*sizeof(double));
  }
}

class Exp53HighwayFixedTail2CoreSurgical {
public:
  Exp53HighwayFixedTail2CoreSurgical():gen_(0),done_(0),stop_(false),out_(nullptr),in_(nullptr),n_(0),local_gen_(0) {
    pin(0);
    helper_=std::thread([this]{ loop(); });
    while(!ready_.load(std::memory_order_acquire)) _mm_pause();
  }
  ~Exp53HighwayFixedTail2CoreSurgical(){
    stop_.store(true,std::memory_order_relaxed);
    const uint64_t g=++local_gen_;
    gen_.store(g,std::memory_order_release);
    helper_.join();
  }
  void run(double*out,const double*in,size_t n,int helper_pct) {
    if(n<128 || helper_pct<=0){ exp53_highway_fixedtail_surg_1200_1600(out,in,n); return; }
    // Preserve old executor geometry exactly.
    size_t helper=(n*(size_t)helper_pct)/100;
    helper=(helper/32)*32;
    if(helper<32 || helper>=n){ exp53_highway_fixedtail_surg_1200_1600(out,in,n); return; }
    size_t split=n-helper;
    split=(split/32)*32;
    if(split<32 || split>=n){ exp53_highway_fixedtail_surg_1200_1600(out,in,n); return; }
    out_=out+split; in_=in+split; n_=n-split;
    // Surgical change only: store/release generation instead of fetch_add.
    const uint64_t g=++local_gen_;
    gen_.store(g,std::memory_order_release);
    exp53_highway_fixedtail_surg_1200_1600(out,in,split);
    while(done_.load(std::memory_order_acquire)!=g) _mm_pause();
  }
private:
  static void pin(int c){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(c,&s); (void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s); }
  void loop(){
    pin(2); ready_.store(true,std::memory_order_release); uint64_t seen=gen_.load(std::memory_order_relaxed);
    for(;;){
      uint64_t g;
      while((g=gen_.load(std::memory_order_acquire))==seen)_mm_pause();
      seen=g;
      if(stop_.load(std::memory_order_relaxed))return;
      exp53_highway_fixedtail_surg_1200_1600(out_,in_,n_);
      done_.store(g,std::memory_order_release);
    }
  }
  std::thread helper_;
  alignas(64) std::atomic<uint64_t> gen_,done_;
  alignas(64) std::atomic<bool> ready_{false};
  std::atomic<bool> stop_;
  double*out_; const double*in_; size_t n_;
  uint64_t local_gen_;
};

extern "C" {
void* exp53_hwy2_surg_create(){ return new Exp53HighwayFixedTail2CoreSurgical(); }
void exp53_hwy2_surg_destroy(void*p){ delete static_cast<Exp53HighwayFixedTail2CoreSurgical*>(p); }
void exp53_hwy2_surg_run(void*p,double*out,const double*in,size_t n,int pct){ static_cast<Exp53HighwayFixedTail2CoreSurgical*>(p)->run(out,in,n,pct); }
}
