#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
void exp53_highway_u2z_0100_2000_candidate(double*,const double*,size_t);
class Exp53HighwayU2ZSync2Candidate{
public:
  Exp53HighwayU2ZSync2Candidate():generation_(0),completed_(0),stop_(false),ready_(false),out2_(nullptr),in2_(nullptr),n2_(0){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
  ~Exp53HighwayU2ZSync2Candidate(){stop_.store(true,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);if(helper_.joinable())helper_.join();}
  void run(double*out,const double*in,size_t n,unsigned hp){if(!n)return;size_t full=n/16;if(full<2||hp==0){exp53_highway_u2z_0100_2000_candidate(out,in,n);return;}size_t hb=(full*hp+50)/100;if(hb<1)hb=1;if(hb>=full)hb=full-1;size_t split=(full-hb)*16;out2_=out+split;in2_=in+split;n2_=n-split;uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;exp53_highway_u2z_0100_2000_candidate(out,in,split);while(completed_.load(std::memory_order_acquire)!=g)_mm_pause();}
private:
  static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);pthread_setaffinity_np(pthread_self(),sizeof(s),&s);} 
  void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t g;while((g=generation_.load(std::memory_order_acquire))==seen)_mm_pause();seen=g;if(stop_.load(std::memory_order_relaxed))return;exp53_highway_u2z_0100_2000_candidate(out2_,in2_,n2_);completed_.store(g,std::memory_order_release);}}
  std::thread helper_;alignas(64) std::atomic<uint64_t> generation_,completed_;alignas(64) std::atomic<bool> stop_,ready_;double*out2_;const double*in2_;size_t n2_;
};
