#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include "exp53_batch_production.hpp"
#include "exp53_batch_custom_2core_candidate.hpp"

static volatile double sink_value=0.0;
static void* xalloc(size_t b){void*p=nullptr;if(posix_memalign(&p,64,b)!=0||!p){std::perror("posix_memalign");std::exit(2);}return p;}
static double now_ns(){timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (double)t.tv_sec*1e9+t.tv_nsec;}
static uint64_t bits(double x){uint64_t u;std::memcpy(&u,&x,8);return u;}
static uint64_t rs=0x9e3779b97f4a7c15ULL; static uint64_t ru(){uint64_t x=rs;x^=x>>12;x^=x<<25;x^=x>>27;rs=x;return x*2685821657736338717ULL;}
static double rd(){return -100.0+200.0*(double)(ru()>>11)*(1.0/9007199254740992.0);}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static int calls_for(size_t n){uint64_t c=16000000ULL/n;if(c<4)c=4;if(c>64)c=64;return(int)c;}
static size_t stream_slots(size_t n){const size_t target=128ULL*1024ULL*1024ULL,one=n*8;size_t s=(target+one-1)/one;if(s<8)s=8;if(s>256)s=256;return s;}
class Ring{public:Ring(size_t s,size_t n):s_(s),n_(n){p_=(double*)xalloc(s_*n_*8);std::memset(p_,0,s_*n_*8);}~Ring(){std::free(p_);}double*at(size_t i){return p_+(i%s_)*n_;}size_t slots()const{return s_;}private:double*p_;size_t s_,n_;};
template<class E>static uint64_t accuracy(E&ex,bool nt){const size_t n=200000;double*in=(double*)xalloc(n*8),*r=(double*)xalloc(n*8),*o=(double*)xalloc(n*8);for(size_t i=0;i<n;i++)in[i]=rd();exp53_n2_vmstyle_u4_0381_frozen(r,in,n);if(nt)ex.run_streaming_write_once(o,in,n);else ex.run(o,in,n);uint64_t d=0;for(size_t i=0;i<n;i++)d+=bits(r[i])!=bits(o[i]);std::free(in);std::free(r);std::free(o);return d;}
template<class E>static double hot(E&ex,double*out,const double*in,size_t n,int c){double a=now_ns();for(int r=0;r<c;r++)ex.run(out,in,n);double b=now_ns();sink_value+=out[(size_t)c%n];return(b-a)/((double)c*n);}
template<class E>static double stream(E&ex,Ring&ring,const double*in,size_t n,int c,int ph){double a=now_ns();for(int r=0;r<c;r++)ex.run_streaming_write_once(ring.at((size_t)r+ph),in,n);double b=now_ns();sink_value+=ring.at(ph)[ph%n];return(b-a)/((double)c*n);}
template<class E>static int run(const char*name,E&ex){printf("STACK=%s default=temporal rare=streaming_nt\n",name);printf("ACCURACY stack=%s default_bitdiff=%llu streaming_bitdiff=%llu\n",name,(unsigned long long)accuracy(ex,false),(unsigned long long)accuracy(ex,true));const size_t ns[]={3000000,4000000,5000000,8000000};for(size_t n:ns){rs=0x9e3779b97f4a7c15ULL^n;double*in=(double*)xalloc(n*8),*out=(double*)xalloc(n*8);for(size_t i=0;i<n;i++)in[i]=rd();Ring ring(stream_slots(n),n);int c=calls_for(n);for(int w=0;w<32;w++)ex.run(out,in,n);for(int w=0;w<16;w++)ex.run_streaming_write_once(ring.at(w),in,n);std::vector<double>h,s;for(int q=0;q<25;q++)h.push_back(hot(ex,out,in,n,c));for(int q=0;q<25;q++)s.push_back(stream(ex,ring,in,n,c,q*7));printf("METHOD stack=%s n=%zu calls=%d slots=%zu hot_ns=%.9f stream_nt_ns=%.9f\n",name,n,c,ring.slots(),median(h),median(s));std::free(out);std::free(in);}return 0;}
int main(int argc,char**argv){if(argc!=2)return 2;std::string m=argv[1];if(m=="custom"){Exp53CustomPermanent2CoreCandidate ex;return run("custom2",ex);}if(m=="ff"){Exp53BatchProductionExecutor ex(2);return run("merged_ff",ex);}return 2;}
