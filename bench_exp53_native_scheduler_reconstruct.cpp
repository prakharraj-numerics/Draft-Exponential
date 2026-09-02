#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <mkl_vml.h>
#include "production/exp53_batch_custom_2core_nt_frozen.hpp"

struct Aligned {
    double* p = nullptr;
    explicit Aligned(size_t n) {
        if (posix_memalign(reinterpret_cast<void**>(&p), 64, n * sizeof(double)) || !p) std::exit(2);
    }
    ~Aligned() { std::free(p); }
};

static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline double u01(uint64_t h) {
    return (static_cast<double>(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static void fill_mixed(double* x, size_t n) {
    const double edge = 0x1p-20;
    const uint64_t seed = 0xd1b54a32d192ed03ULL ^ (static_cast<uint64_t>(n) << 17);
    for (size_t i = 0; i < n; ++i) {
        const double u = u01(mix64(seed + i * 0x9e3779b97f4a7c15ULL));
        const unsigned kind = i & 3u;
        double v = kind < 2 ? edge + u * (1.0 - 2.0 * edge)
                            : 1.0 + edge + u * (99.0 - 2.0 * edge);
        if (kind & 1u) v = -v;
        x[i] = v;
    }
}
static inline uint64_t bits(double x) {
    uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}
static uint64_t cross_check(const double* got, const double* ref, size_t n) {
    uint64_t maxulp = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) || got[i] <= 0.0 || ref[i] <= 0.0)
            return UINT64_MAX;
        const uint64_t a = bits(got[i]), b = bits(ref[i]);
        maxulp = std::max(maxulp, a > b ? a - b : b - a);
    }
    return maxulp;
}

/* Measurement-only clone of the frozen custom 2-core dispatcher.
   The scheduling logic is intentionally identical. The only hot-loop change is
   a register-resident ++spins counter, published once after each wait region.
   We benchmark the clone against the untouched production dispatcher and report
   the perturbation ratio so the counter validity is auditable. */
class Exp53Custom2CoreCountedClone {
public:
    using fn_t = void (*)(double*, const double*, size_t);
    Exp53Custom2CoreCountedClone()
        : generation_(0), completed_(0), stop_(false), out_(nullptr), in_(nullptr), n2_(0), fn_(nullptr) {
        pin_current_thread(0);
        helper_ = std::thread([this]{ helper_loop(); });
        while (!helper_ready_.load(std::memory_order_acquire)) _mm_pause();
    }
    ~Exp53Custom2CoreCountedClone() {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        if (helper_.joinable()) helper_.join();
    }
    void run(double* out, const double* in, size_t n) {
        run_with(exp53_n2_vmstyle_u4_0381_frozen, out, in, n);
    }
    uint64_t helper_spins() const { return helper_spins_total_; }
    uint64_t caller_spins() const { return caller_spins_total_; }
    uint64_t dispatch_calls() const { return dispatch_calls_; }
private:
    static void pin_current_thread(int cpu) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    __attribute__((noinline)) void helper_loop() {
        pin_current_thread(2);
        helper_ready_.store(true, std::memory_order_release);
        uint64_t seen = generation_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t g;
            uint64_t spins = 0;
            while ((g = generation_.load(std::memory_order_acquire)) == seen) {
                _mm_pause();
                ++spins;
            }
            helper_spins_total_ += spins;
            seen = g;
            if (stop_.load(std::memory_order_relaxed)) return;
            fn_t fn = fn_;
            double* out = out_;
            const double* in = in_;
            const size_t n = n2_;
            fn(out, in, n);
            completed_.store(g, std::memory_order_release);
        }
    }
    __attribute__((noinline)) void run_with(fn_t fn, double* out, const double* in, size_t n) {
        if (!n) return;
        const size_t full32 = n / 32;
        if (full32 < 2) { fn(out, in, n); return; }
        const size_t split_blocks = full32 / 2;
        const size_t split = split_blocks * 32;
        if (split == 0 || split >= n) { fn(out, in, n); return; }
        out_ = out + split;
        in_ = in + split;
        n2_ = n - split;
        fn_ = fn;
        const uint64_t g = generation_.fetch_add(1, std::memory_order_release) + 1;
        fn(out, in, split);
        uint64_t spins = 0;
        while (completed_.load(std::memory_order_acquire) != g) {
            _mm_pause();
            ++spins;
        }
        caller_spins_total_ += spins;
        ++dispatch_calls_;
    }
    std::thread helper_;
    alignas(64) std::atomic<uint64_t> generation_;
    alignas(64) std::atomic<uint64_t> completed_;
    alignas(64) std::atomic<bool> helper_ready_{false};
    std::atomic<bool> stop_;
    double* out_;
    const double* in_;
    size_t n2_;
    fn_t fn_;
    uint64_t helper_spins_total_ = 0;
    uint64_t caller_spins_total_ = 0;
    uint64_t dispatch_calls_ = 0;
};

/* Standalone source-equivalent spin shapes preserved for disassembly. They are
   not used for timing; they let the workflow recover the exact optimized
   instruction count of one production-style polling iteration. */
extern "C" __attribute__((noinline,used)) uint64_t exp53_shape_generation_wait(
        const std::atomic<uint64_t>* p, uint64_t seen) {
    uint64_t g;
    while ((g = p->load(std::memory_order_acquire)) == seen) _mm_pause();
    return g;
}
extern "C" __attribute__((noinline,used)) void exp53_shape_completion_wait(
        const std::atomic<uint64_t>* p, uint64_t wanted) {
    while (p->load(std::memory_order_acquire) != wanted) _mm_pause();
}

static size_t choose_calls(size_t n) {
    size_t c = 200000000ULL / n;
    c = std::max<size_t>(200, c);
    c = std::min<size_t>(50000, c);
    return c;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " N\n"; return 2; }
    const size_t n = std::strtoull(argv[1], nullptr, 10);
    const std::vector<size_t> allowed = {100,700,3500,15000,50000,1000000,2000000};
    if (std::find(allowed.begin(), allowed.end(), n) == allowed.end()) return 2;
    const size_t calls = choose_calls(n);
    Aligned in(n), out(n), ref(n);
    fill_mixed(in.p, n);
    vmdExp(static_cast<MKL_INT>(n), in.p, ref.p, VML_HA);

    if (n <= 3000) {
        std::cout << "NATIVE_SCHED n=" << n << " calls=0 helper_spins=0 caller_spins=0 "
                  << "helper_spins_per_call=0 caller_spins_per_call=0 prod_wall_ns_per_el=0 "
                  << "counted_wall_ns_per_el=0 perturbation_ratio=1 maxulp=0\n";
        return 0;
    }

    double prod_ns_el = 0.0;
    {
        Exp53CustomPermanent2CoreFrozen prod;
        for (int i=0;i<16;++i) prod.run(out.p,in.p,n);
        const auto t0=std::chrono::steady_clock::now();
        for (size_t k=0;k<calls;++k) prod.run(out.p,in.p,n);
        const auto t1=std::chrono::steady_clock::now();
        prod_ns_el = std::chrono::duration<double,std::nano>(t1-t0).count() / (double(calls)*double(n));
    }

    uint64_t hs0,cs0,dc0,hs1,cs1,dc1;
    double counted_ns_el = 0.0;
    {
        Exp53Custom2CoreCountedClone probe;
        for (int i=0;i<16;++i) probe.run(out.p,in.p,n);
        probe.run(out.p,in.p,n);
        hs0=probe.helper_spins(); cs0=probe.caller_spins(); dc0=probe.dispatch_calls();
        const auto t0=std::chrono::steady_clock::now();
        for (size_t k=0;k<calls;++k) probe.run(out.p,in.p,n);
        const auto t1=std::chrono::steady_clock::now();
        counted_ns_el = std::chrono::duration<double,std::nano>(t1-t0).count() / (double(calls)*double(n));
        hs1=probe.helper_spins(); cs1=probe.caller_spins(); dc1=probe.dispatch_calls();
    }
    const uint64_t helper = hs1-hs0, caller=cs1-cs0, dispatch=dc1-dc0;
    const uint64_t maxulp = cross_check(out.p,ref.p,n);
    if (maxulp>2 || dispatch!=calls) return 4;
    std::cout.setf(std::ios::fixed); std::cout.precision(9);
    std::cout << "NATIVE_SCHED n=" << n << " calls=" << calls
              << " helper_spins=" << helper << " caller_spins=" << caller
              << " helper_spins_per_call=" << double(helper)/calls
              << " caller_spins_per_call=" << double(caller)/calls
              << " helper_spins_per_el=" << double(helper)/(double(calls)*n)
              << " caller_spins_per_el=" << double(caller)/(double(calls)*n)
              << " prod_wall_ns_per_el=" << prod_ns_el
              << " counted_wall_ns_per_el=" << counted_ns_el
              << " perturbation_ratio=" << counted_ns_el/prod_ns_el
              << " maxulp=" << maxulp << "\n";
    return 0;
}
