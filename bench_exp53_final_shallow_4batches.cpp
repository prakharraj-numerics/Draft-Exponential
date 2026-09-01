#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include "production/exp53_batch_production.hpp"

struct AlignedDoubles {
    double *p = nullptr;
    explicit AlignedDoubles(size_t n) {
        if (posix_memalign((void**)&p, 64, n * sizeof(double)) != 0 || !p) {
            std::fprintf(stderr, "allocation failed for %zu doubles\n", n);
            std::exit(2);
        }
    }
    ~AlignedDoubles() { std::free(p); }
    AlignedDoubles(const AlignedDoubles&) = delete;
    AlignedDoubles& operator=(const AlignedDoubles&) = delete;
};

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double unit01(uint64_t h) {
    return ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

enum CaseId { UNIT_POS=0, UNIT_NEG=1, MID_POS=2, MID_NEG=3 };

static const char *case_name(int c) {
    static const char *names[4] = {"unit_pos","unit_neg","mid_pos","mid_neg"};
    return names[c];
}

static void fill_case(double *x, size_t n, int c) {
    const double e = 0x1p-20;
    const uint64_t seed = 0xd1b54a32d192ed03ULL ^ ((uint64_t)n << 17) ^ ((uint64_t)c << 57);
    for (size_t i = 0; i < n; ++i) {
        const double u = unit01(splitmix64(seed + i * 0x9e3779b97f4a7c15ULL));
        double v;
        if (c == UNIT_POS || c == UNIT_NEG) v = e + u * (1.0 - 2.0 * e);
        else v = 1.0 + e + u * (99.0 - 2.0 * e);
        if (c == UNIT_NEG || c == MID_NEG) v = -v;
        x[i] = v;
    }
}

static inline uint64_t bits_of(double x) {
    uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}

struct CrossCheck { uint64_t maxulp = 0; size_t gt2 = 0; };

static CrossCheck compare_to_intel(const double *got, const double *ref, size_t n) {
    CrossCheck a;
    for (size_t i = 0; i < n; ++i) {
        const double g = got[i], r = ref[i];
        uint64_t ulp = UINT64_MAX;
        if (std::isfinite(g) && std::isfinite(r) && g > 0.0 && r > 0.0) {
            const uint64_t gb = bits_of(g), rb = bits_of(r);
            ulp = gb > rb ? gb-rb : rb-gb;
        } else if (g == r) ulp = 0;
        if (ulp > a.maxulp) a.maxulp = ulp;
        if (ulp > 2) ++a.gt2;
    }
    return a;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static size_t calls_for(size_t n) {
    // Same four random arrays only; repeated calls are timing repeats, not new inputs.
    constexpr size_t TARGET_VALUES = 4000000ULL;
    size_t c = TARGET_VALUES / n;
    if (c < 1) c = 1;
    if (c > 100000) c = 100000;
    return c;
}

int main(int argc, char **argv) {
    if (argc != 2 || (std::string(argv[1]) != "ours" && std::string(argv[1]) != "intel")) {
        std::fprintf(stderr, "usage: %s ours|intel\n", argv[0]);
        return 2;
    }
    const std::string stack = argv[1];
    const bool ours = stack == "ours";
    const std::vector<size_t> sizes = {50,750,2500,4000,10000,20000,60000,200000,2000000,4000000};

    Exp53BatchProductionExecutor *ex = ours ? new Exp53BatchProductionExecutor(2) : nullptr;
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "STACK=" << stack << " FOUR_RANDOM_BATCHES_PER_SIZE=1\n";

    for (size_t n : sizes) {
        for (int c = 0; c < 4; ++c) {
            AlignedDoubles in(n), out(n), ref(n), check(n);
            fill_case(in.p, n, c);
            vmdExp((MKL_INT)n, in.p, ref.p, VML_HA);

            CrossCheck cc{};
            if (ours) {
                ex->run(check.p, in.p, n, 2);
                cc = compare_to_intel(check.p, ref.p, n);
            }

            auto invoke = [&]() {
                if (ours) ex->run(out.p, in.p, n, 2);
                else vmdExp((MKL_INT)n, in.p, out.p, VML_HA);
            };

            for (int w=0; w<3; ++w) invoke();
            const size_t calls = calls_for(n);
            std::vector<double> samples;
            samples.reserve(5);
            for (int s=0; s<5; ++s) {
                auto t0 = std::chrono::steady_clock::now();
                for (size_t k=0; k<calls; ++k) invoke();
                auto t1 = std::chrono::steady_clock::now();
                const double ns = std::chrono::duration<double,std::nano>(t1-t0).count();
                samples.push_back(ns / ((double)calls * (double)n));
            }
            const double med = median(samples);
            std::cout << "RESULT stack=" << stack
                      << " n=" << n
                      << " case=" << case_name(c)
                      << " calls=" << calls
                      << " ns=" << med
                      << " maxulp=" << (ours ? cc.maxulp : 0)
                      << " gt2=" << (ours ? cc.gt2 : 0)
                      << "\n";
        }
    }
    delete ex;
    return 0;
}
