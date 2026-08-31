#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "exp53_batch_production.hpp"
#include "exp53_batch_custom_2core_candidate.hpp"

static volatile double sink_value = 0.0;

static void *xalloc(size_t bytes) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        std::perror("posix_memalign");
        std::exit(2);
    }
    return p;
}

static double now_ns() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static uint64_t bits(double x) {
    uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t ru() {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 2685821657736338717ULL;
}
static double rd() {
    return -100.0 + 200.0 * (double)(ru() >> 11) * (1.0 / 9007199254740992.0);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static int calls_for(size_t n) {
    uint64_t c = 12000000ULL / n;
    if (c < 192) c = 192;
    if (c > 4096) c = 4096;
    return (int)c;
}

static size_t stream_slots(size_t n) {
    const size_t target = 128ULL * 1024ULL * 1024ULL;
    const size_t one = n * sizeof(double);
    size_t s = (target + one - 1) / one;
    if (s < 8) s = 8;
    if (s > 2048) s = 2048;
    return s;
}

class OutputRing {
public:
    OutputRing(size_t slots, size_t n) : slots_(slots), n_(n) {
        base_ = (double*)xalloc(slots_ * n_ * sizeof(double));
        std::memset(base_, 0, slots_ * n_ * sizeof(double));
    }
    ~OutputRing() { std::free(base_); }
    double *at(size_t s) { return base_ + (s % slots_) * n_; }
    size_t slots() const { return slots_; }
private:
    double *base_;
    size_t slots_, n_;
};

template<class Exec>
static uint64_t accuracy_default(Exec &ex) {
    const size_t n = 200000;
    double *in=(double*)xalloc(n*8), *ref=(double*)xalloc(n*8), *out=(double*)xalloc(n*8);
    for (size_t i=0;i<n;i++) in[i]=rd();
    exp53_n2_vmstyle_u4_0381_frozen(ref,in,n);
    ex.run(out,in,n);
    uint64_t d=0; for(size_t i=0;i<n;i++) d += bits(ref[i]) != bits(out[i]);
    std::free(in); std::free(ref); std::free(out); return d;
}

template<class Exec>
static uint64_t accuracy_stream(Exec &ex) {
    const size_t n = 200000;
    double *in=(double*)xalloc(n*8), *ref=(double*)xalloc(n*8), *out=(double*)xalloc(n*8);
    for (size_t i=0;i<n;i++) in[i]=rd();
    exp53_n2_vmstyle_u4_0381_frozen(ref,in,n);
    ex.run_streaming_write_once(out,in,n);
    uint64_t d=0; for(size_t i=0;i<n;i++) d += bits(ref[i]) != bits(out[i]);
    std::free(in); std::free(ref); std::free(out); return d;
}

template<class Exec>
static double measure_hot(Exec &ex,double *out,const double *in,size_t n,int calls) {
    const double a=now_ns();
    for(int r=0;r<calls;r++) ex.run(out,in,n);
    const double b=now_ns();
    sink_value += out[(size_t)calls % n];
    return (b-a)/((double)calls*(double)n);
}

template<class Exec>
static double measure_stream(Exec &ex,OutputRing &ring,const double *in,size_t n,int calls,int phase) {
    const double a=now_ns();
    for(int r=0;r<calls;r++) ex.run_streaming_write_once(ring.at((size_t)r+(size_t)phase),in,n);
    const double b=now_ns();
    sink_value += ring.at((size_t)phase)[((size_t)phase*157u)%n];
    return (b-a)/((double)calls*(double)n);
}

template<class Exec>
static int run_method(const char *name, Exec &ex) {
    std::printf("STACK=%s default=temporal rare=streaming_nt\n", name);
    std::printf("ACCURACY stack=%s default_bitdiff=%llu streaming_bitdiff=%llu\n",
        name,
        (unsigned long long)accuracy_default(ex),
        (unsigned long long)accuracy_stream(ex));

    const size_t ns[] = {5000,8000,15000,25000,35000,50000,65000};
    for(size_t n:ns) {
        rng_state = 0x9e3779b97f4a7c15ULL ^ (uint64_t)n;
        double *in=(double*)xalloc(n*8), *hot=(double*)xalloc(n*8);
        for(size_t i=0;i<n;i++) in[i]=rd();
        const size_t slots=stream_slots(n);
        OutputRing ring(slots,n);
        const int calls=calls_for(n);

        for(int w=0;w<64;w++) ex.run(hot,in,n);
        for(int w=0;w<16;w++) ex.run_streaming_write_once(ring.at((size_t)w),in,n);

        std::vector<double> h,s;
        h.reserve(25); s.reserve(25);
        for(int q=0;q<25;q++) h.push_back(measure_hot(ex,hot,in,n,calls));
        for(int q=0;q<25;q++) s.push_back(measure_stream(ex,ring,in,n,calls,q*7));

        std::printf("METHOD stack=%s n=%zu calls=%d slots=%zu hot_ns=%.9f stream_nt_ns=%.9f\n",
            name,n,calls,slots,median(h),median(s));
        std::free(hot); std::free(in);
    }
    return 0;
}

int main(int argc,char **argv) {
    if(argc!=2) { std::fprintf(stderr,"usage: %s custom|ff\n",argv[0]); return 2; }
    std::string m=argv[1];
    if(m=="custom") {
        Exp53CustomPermanent2CoreCandidate ex;
        return run_method("custom2",ex);
    }
    if(m=="ff") {
        Exp53BatchProductionExecutor ex(2);
        return run_method("merged_ff",ex);
    }
    return 2;
}
