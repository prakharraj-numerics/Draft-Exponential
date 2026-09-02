#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <vector>
#include <mkl_vml.h>
#include "exp53_resource_aware_dispatch.hpp"

struct Aligned {
    double* p = nullptr;
    explicit Aligned(size_t n) {
        if (posix_memalign(reinterpret_cast<void**>(&p), 64, n * sizeof(double)) || !p) std::exit(2);
    }
    ~Aligned() { std::free(p); }
};

static inline uint64_t mix(uint64_t x) {
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
        const double u = u01(mix(seed + i * 0x9e3779b97f4a7c15ULL));
        const unsigned kind = i & 3u;
        double v = kind < 2 ? edge + u * (1.0 - 2.0 * edge)
                            : 1.0 + edge + u * (99.0 - 2.0 * edge);
        if (kind & 1u) v = -v;
        x[i] = v;
    }
}

static inline uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
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

static double cpu_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)) std::exit(5);
    return static_cast<double>(t.tv_sec) * 1e9 + static_cast<double>(t.tv_nsec);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " balanced|intel N\n";
        return 2;
    }
    const std::string stack = argv[1];
    if (stack != "balanced" && stack != "intel") return 2;
    const size_t n = std::strtoull(argv[2], nullptr, 10);
    const std::vector<size_t> allowed = {100,700,3500,15000,50000,1000000,2000000};
    if (std::find(allowed.begin(), allowed.end(), n) == allowed.end()) return 2;

    Aligned in(n), out(n), ref(n);
    fill_mixed(in.p, n);
    vmdExp(static_cast<MKL_INT>(n), in.p, ref.p, VML_HA);

    Exp53ResourceAwareExecutor executor(2);
    auto invoke = [&] {
        if (stack == "balanced") executor.run(out.p, in.p, n, Exp53ResourcePolicy::Balanced);
        else vmdExp(static_cast<MKL_INT>(n), in.p, out.p, VML_HA);
    };

    invoke();
    const uint64_t maxulp = cross_check(out.p, ref.p, n);
    if (maxulp > 2) {
        std::cerr << "correctness failure maxulp=" << maxulp << "\n";
        return 4;
    }
    for (int i = 0; i < 8; ++i) invoke();

    size_t calls = 50000000ULL / n;
    if (calls < 1) calls = 1;
    if (calls > 500000) calls = 500000;

    std::vector<double> wall, cpu;
    wall.reserve(7);
    cpu.reserve(7);
    volatile double sink = 0.0;
    for (int sample = 0; sample < 7; ++sample) {
        const double c0 = cpu_ns();
        const auto w0 = std::chrono::steady_clock::now();
        for (size_t k = 0; k < calls; ++k) invoke();
        const auto w1 = std::chrono::steady_clock::now();
        const double c1 = cpu_ns();
        wall.push_back(std::chrono::duration<double,std::nano>(w1-w0).count() / (calls * static_cast<double>(n)));
        cpu.push_back((c1-c0) / (calls * static_cast<double>(n)));
        sink += out.p[(sample * 104729ULL) % n] * 0x1p-1022;
    }

    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const double wall_med = median(wall);
    const double cpu_med = median(cpu);
    std::cout << std::fixed << std::setprecision(9)
              << "RESULT stack=" << stack << " n=" << n << " calls=" << calls
              << " wall_ns_per_element=" << wall_med
              << " cpu_ns_per_element=" << cpu_med
              << " effective_cores=" << (cpu_med / wall_med)
              << " maxrss_kib=" << usage.ru_maxrss
              << " maxulp=" << maxulp << "\n";
    if (sink == 123.0) std::cerr << sink;
    return 0;
}
