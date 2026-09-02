#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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
    uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}
static uint64_t cross_check(const double* got, const double* ref, size_t n) {
    uint64_t maxulp = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) || got[i] <= 0.0 || ref[i] <= 0.0) return UINT64_MAX;
        const uint64_t a = bits(got[i]), b = bits(ref[i]);
        maxulp = std::max(maxulp, a > b ? a-b : b-a);
    }
    return maxulp;
}
static uint64_t read_u64(const std::string& path) {
    std::ifstream f(path);
    uint64_t x = 0;
    if (!(f >> x)) { std::cerr << "cannot read " << path << "\n"; std::exit(6); }
    return x;
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); return v[v.size()/2];
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: bench ours|intel N energy_uj_path max_energy_range_uj_path\n";
        return 2;
    }
    const std::string stack = argv[1];
    const size_t n = std::strtoull(argv[2], nullptr, 10);
    const std::string energy_path = argv[3];
    const std::string max_path = argv[4];
    const std::vector<size_t> allowed = {100,700,3500,15000,50000,1000000,2000000};
    if ((stack!="ours" && stack!="intel") || std::find(allowed.begin(),allowed.end(),n)==allowed.end()) return 2;

    const uint64_t max_range = read_u64(max_path);
    Aligned in(n), out(n), ref(n);
    fill_mixed(in.p,n);
    vmdExp(static_cast<MKL_INT>(n), in.p, ref.p, VML_HA);
    Exp53BatchProductionExecutor executor(2);
    auto invoke = [&] {
        if (stack=="ours") executor.run(out.p,in.p,n,2);
        else vmdExp(static_cast<MKL_INT>(n),in.p,out.p,VML_HA);
    };

    invoke();
    const uint64_t maxulp = cross_check(out.p,ref.p,n);
    if (maxulp>2) return 4;
    for (int i=0;i<8;++i) invoke();

    size_t calls = 50000000ULL/n;
    if (calls<1) calls=1;
    if (calls>500000) calls=500000;
    const double elems = calls * static_cast<double>(n);

    std::vector<double> joules_per_element, wall_ns_per_element;
    volatile double sink=0.0;
    for (int sample=0; sample<7; ++sample) {
        const uint64_t e0=read_u64(energy_path);
        const auto w0=std::chrono::steady_clock::now();
        for (size_t k=0;k<calls;++k) invoke();
        const auto w1=std::chrono::steady_clock::now();
        const uint64_t e1=read_u64(energy_path);
        const uint64_t duj = e1>=e0 ? e1-e0 : (max_range-e0)+e1;
        joules_per_element.push_back((static_cast<double>(duj)*1e-6)/elems);
        wall_ns_per_element.push_back(std::chrono::duration<double,std::nano>(w1-w0).count()/elems);
        sink += out.p[(sample*104729ULL)%n] * 0x1p-1022;
    }

    std::cout << std::fixed << std::setprecision(12)
              << "ENERGY stack=" << stack << " n=" << n << " calls=" << calls
              << " wall_ns_per_element=" << median(wall_ns_per_element)
              << " joules_per_element=" << median(joules_per_element)
              << " nanojoules_per_element=" << median(joules_per_element)*1e9
              << " maxulp=" << maxulp << "\n";
    if (sink==123.0) std::cerr << sink;
    return 0;
}
