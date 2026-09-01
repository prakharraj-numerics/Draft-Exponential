// Experimental surgical patch on old synchronized Highway executor for n=1200..1600.
// Production untouched. Same math, same partition geometry, same tail ownership.
#include <atomic>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

// Reuse the exact old synchronized Highway mathematical kernel from the original
// candidate translation unit. This avoids duplicating frozen C symbols and keeps
// the experiment surgical: executor handoff only.
extern "C" void exp53_highway_fixedtail_0101_2000_candidate(double*,const double*,size_t);

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
    if(n<128 || helper_pct<=0){ exp53_highway_fixedtail_0101_2000_candidate(out,in,n); return; }
    // Preserve old executor geometry exactly.
    size_t helper=(n*(size_t)helper_pct)/100;
    helper=(helper/32)*32;
    if(helper<32 || helper>=n){ exp53_highway_fixedtail_0101_2000_candidate(out,in,n); return; }
    size_t split=n-helper;
    split=(split/32)*32;
    if(split<32 || split>=n){ exp53_highway_fixedtail_0101_2000_candidate(out,in,n); return; }
    out_=out+split; in_=in+split; n_=n-split;
    // Surgical change only: store/release generation instead of fetch_add.
    const uint64_t g=++local_gen_;
    gen_.store(g,std::memory_order_release);
    exp53_highway_fixedtail_0101_2000_candidate(out,in,split);
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
      exp53_highway_fixedtail_0101_2000_candidate(out_,in_,n_);
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
