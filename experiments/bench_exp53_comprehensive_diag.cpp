#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <vector>
#include <mkl_vml.h>
#include "production/exp53_batch_production.hpp"

struct Aligned {
    double* p = nullptr;
    explicit Aligned(size_t n) {
        if (posix_memalign(reinterpret_cast<void**>(&p), 64, n * sizeof(double)) || !p) std::exit(2);
    }
    ~Aligned() { std::free(p); }
};

static inline uint64_t mix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static inline double u01(uint64_t h) {
    return (static_cast<double>(h >> 11) + 0.5) * 0x1p-53;
}

static void fill_domain(double* x, size_t n, int domain) {
    const double e = 0x1p-20;
    const uint64_t seed = UINT64_C(0xd1b54a32d192ed03)
                        ^ (static_cast<uint64_t>(n) << 17)
                        ^ (static_cast<uint64_t>(domain) << 57);
    for (size_t i = 0; i < n; ++i) {
        const double u = u01(mix64(seed + i * UINT64_C(0x9e3779b97f4a7c15)));
        double v = domain == 0 ? e + u * (1.0 - 2.0 * e)
                               : 1.0 + e + u * (99.0 - 2.0 * e);
        x[i] = (i & 1u) ? -v : v;
    }
}

static inline uint64_t bits(double x) {
    uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}

static uint64_t maxulp_vs(const double* got, const double* ref, size_t n) {
    uint64_t m = 0;
    for (size_t i = 0; i < n; ++i) {
        const double g = got[i], r = ref[i];
        if (!std::isfinite(g) || !std::isfinite(r) || g <= 0.0 || r <= 0.0) return UINT64_MAX;
        const uint64_t a = bits(g), b = bits(r);
        m = std::max(m, a > b ? a - b : b - a);
    }
    return m;
}

static double cpu_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)) std::exit(5);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static int thread_count() {
    std::ifstream f("/proc/self/status");
    std::string s;
    while (std::getline(f, s)) {
        if (s.rfind("Threads:", 0) == 0) return std::atoi(s.c_str()+8);
    }
    return -1;
}

static bool allowed_n(size_t n) {
    static const size_t a[] = {50,100,700,1200,1400,1600,2500,3000,3500,4500,5000,8000,
                               15000,20000,25000,30000,50000,1000000,2000000};
    return std::find(std::begin(a), std::end(a), n) != std::end(a);
}

static size_t native_calls(size_t n) {
    size_t c = 20000000ULL / n;
    if (c < 2) c = 2;
    if (c > 200000) c = 200000;
    return c;
}

static void prime_both_helpers(Exp53BatchProductionExecutor& ex, int domain) {
    Aligned in(5000), out(5000);
    fill_domain(in.p, 5000, domain);
    // 2000 creates the frozen Highway 1400..3000 helper; 5000 creates custom2.
    for (int k=0;k<3;++k) ex.run(out.p, in.p, 2000, 2);
    for (int k=0;k<3;++k) ex.run(out.p, in.p, 5000, 2);
}

extern "C" __attribute__((noinline,used)) void exp53_diag_profile_start() {
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline,used)) void exp53_diag_profile_stop() {
    asm volatile("" ::: "memory");
}

static int run_native(const std::string& stack, size_t n, int domain) {
    Aligned in(n), out(n), ref(n);
    fill_domain(in.p, n, domain);
    vmdExp(static_cast<MKL_INT>(n), in.p, ref.p, VML_HA);

    std::unique_ptr<Exp53BatchProductionExecutor> ex;
    if (stack != "intel") ex = std::make_unique<Exp53BatchProductionExecutor>(2);
    if (stack == "prod_warm") prime_both_helpers(*ex, domain);

    auto invoke = [&] {
        if (stack == "intel") vmdExp(static_cast<MKL_INT>(n), in.p, out.p, VML_HA);
        else ex->run(out.p, in.p, n, 2);
    };

    invoke();
    const uint64_t maxulp = stack == "intel" ? 0 : maxulp_vs(out.p, ref.p, n);
    if (maxulp > 2) {
        std::cerr << "accuracy failure maxulp_vs_intel=" << maxulp << "\n";
        return 4;
    }
    for (int i=0;i<6;++i) invoke();

    const size_t calls = native_calls(n);
    std::vector<double> wall, cpu;
    wall.reserve(5); cpu.reserve(5);
    volatile double sink = 0.0;
    for (int s=0;s<5;++s) {
        const double c0 = cpu_ns();
        const auto w0 = std::chrono::steady_clock::now();
        for (size_t k=0;k<calls;++k) invoke();
        const auto w1 = std::chrono::steady_clock::now();
        const double c1 = cpu_ns();
        wall.push_back(std::chrono::duration<double,std::nano>(w1-w0).count()/(calls*static_cast<double>(n)));
        cpu.push_back((c1-c0)/(calls*static_cast<double>(n)));
        sink += out.p[(static_cast<size_t>(s)*104729ULL)%n] * 0x1p-1022;
    }
    rusage ru{}; getrusage(RUSAGE_SELF,&ru);
    const double w=median(wall), c=median(cpu);
    std::cout << std::fixed << std::setprecision(9)
              << "NATIVE stack=" << stack << " domain=" << (domain?"mid":"unit")
              << " n=" << n << " calls=" << calls
              << " wall_ns_el=" << w << " cpu_ns_el=" << c
              << " effective_cores=" << c/w
              << " threads=" << thread_count()
              << " maxrss_kib=" << ru.ru_maxrss
              << " maxulp_vs_intel=" << maxulp << "\n";
    if (sink==123.0) std::cerr << sink;
    return 0;
}

static int run_sde(const std::string& stack, size_t n, int domain, size_t calls) {
    Aligned in(n), out(n), ref(n);
    out.p[0] = 0.0;
    fill_domain(in.p, n, domain);
    vmdExp(static_cast<MKL_INT>(n), in.p, ref.p, VML_HA);

    std::unique_ptr<Exp53BatchProductionExecutor> ex;
    if (stack == "prod_fresh" || stack == "prod_warm") {
        ex = std::make_unique<Exp53BatchProductionExecutor>(2);
        if (stack == "prod_warm") prime_both_helpers(*ex, domain);
        for (int i=0;i<3;++i) ex->run(out.p,in.p,n,2);
        const uint64_t u=maxulp_vs(out.p,ref.p,n);
        if (u>2) return 4;
    } else if (stack == "intel") {
        for (int i=0;i<3;++i) vmdExp(static_cast<MKL_INT>(n),in.p,out.p,VML_HA);
    }

    exp53_diag_profile_start();
    if (stack == "prod_fresh" || stack == "prod_warm") {
        for (size_t k=0;k<calls;++k) ex->run(out.p,in.p,n,2);
    } else if (stack == "intel") {
        for (size_t k=0;k<calls;++k) vmdExp(static_cast<MKL_INT>(n),in.p,out.p,VML_HA);
    } else {
        for (size_t k=0;k<calls;++k) asm volatile("" : : "r"(k) : "memory");
    }
    exp53_diag_profile_stop();

    volatile double sink = stack == "noop" ? 0.0 : out.p[(n*17ULL+calls)%n];
    std::cout << "SDE_RUN stack=" << stack << " domain=" << (domain?"mid":"unit")
              << " n=" << n << " calls=" << calls
              << " elements=" << static_cast<unsigned long long>(n*calls)
              << " threads=" << thread_count() << " sink=" << sink << "\n";
    return 0;
}

int main(int argc,char**argv) {
    mkl_set_num_threads_local(1);
    if (argc < 5) {
        std::cerr << "usage: native|sde prod_fresh|prod_warm|intel|noop N unit|mid [calls]\n";
        return 2;
    }
    const std::string mode=argv[1], stack=argv[2], dom=argv[4];
    const size_t n=std::strtoull(argv[3],nullptr,10);
    if (!allowed_n(n) || (dom!="unit"&&dom!="mid")) return 2;
    const int domain=dom=="mid";
    if (mode=="native") {
        if (stack!="prod_fresh"&&stack!="prod_warm"&&stack!="intel") return 2;
        return run_native(stack,n,domain);
    }
    if (mode=="sde") {
        if (argc!=6 || (stack!="prod_fresh"&&stack!="prod_warm"&&stack!="intel"&&stack!="noop")) return 2;
        const size_t calls=std::strtoull(argv[5],nullptr,10);
        if (!calls) return 2;
        return run_sde(stack,n,domain,calls);
    }
    return 2;
}
