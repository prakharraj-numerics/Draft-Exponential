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
#include "production/exp53_batch_custom_2core_nt_frozen.hpp"

struct AlignedDoubles {
    double* p = nullptr;
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

static double unit01(uint64_t h) {
    return ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static void fill_inputs(double* x, size_t n, int domain) {
    const uint64_t seed0 = 0x243f6a8885a308d3ULL ^ ((uint64_t)n << 19) ^ ((uint64_t)domain << 59);
    for (size_t i = 0; i < n; ++i) {
        const double u = unit01(splitmix64(seed0 + i * 0x9e3779b97f4a7c15ULL));
        const double e = 0x1p-20;
        double m;
        if (domain == 0) m = e + u * (1.0 - 2.0 * e);
        else m = 1.0 + e + u * (99.0 - 2.0 * e);
        x[i] = (i & 1) ? -m : m;
    }
}

static uint64_t bits_of(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

struct Accuracy {
    uint64_t maxulp = 0;
    size_t gt2 = 0;
};

static Accuracy compare_ulp(const double* got, const double* ref, size_t n) {
    Accuracy a;
    for (size_t i = 0; i < n; ++i) {
        const double g = got[i], r = ref[i];
        uint64_t ulp = UINT64_MAX;
        if (std::isfinite(g) && std::isfinite(r) && g > 0.0 && r > 0.0) {
            const uint64_t gb = bits_of(g), rb = bits_of(r);
            ulp = gb > rb ? gb - rb : rb - gb;
        } else if (g == r) {
            ulp = 0;
        }
        if (ulp > a.maxulp) a.maxulp = ulp;
        if (ulp > 2) ++a.gt2;
    }
    return a;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static size_t call_count(size_t n) {
    /* Exact geometry used by the old MEDIUM streaming benchmark: about
       6 million output values per timing sample. This stays well below the
       256 MiB ring capacity at 15K..65K, so a sample does not wrap and turn
       into a cache-reuse benchmark. */
    constexpr size_t TARGET_VALUES = 6000000ULL;
    size_t c = TARGET_VALUES / n;
    if (c < 2) c = 2;
    if (c > 100000) c = 100000;
    return c;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s temporal|nt|intel\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "temporal" && mode != "nt" && mode != "intel") return 2;

    Exp53CustomPermanent2CoreFrozen* custom = nullptr;
    if (mode != "intel") custom = new Exp53CustomPermanent2CoreFrozen();

    constexpr size_t RING_BYTES = 256ULL * 1024ULL * 1024ULL;
    const std::vector<size_t> sizes = {
        15000, 20000, 25000, 30000, 35000, 40000,
        45000, 50000, 55000, 60000, 65000
    };

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "MODE=" << mode << " OUTPUT_PATTERN=256MiB_rotating_true_write_once_ring TARGET_VALUES=6000000\n";

    for (int domain = 0; domain < 2; ++domain) {
        for (size_t n : sizes) {
            AlignedDoubles in(n), ref(n), check(n);
            fill_inputs(in.p, n, domain);
            vmdExp((MKL_INT)n, in.p, ref.p, VML_HA);

            Accuracy acc{};
            if (mode == "temporal") {
                custom->run(check.p, in.p, n);
                acc = compare_ulp(check.p, ref.p, n);
            } else if (mode == "nt") {
                custom->run_streaming_write_once(check.p, in.p, n);
                acc = compare_ulp(check.p, ref.p, n);
            }

            const size_t stride = (n + 7u) & ~size_t(7u);
            size_t slots = RING_BYTES / (stride * sizeof(double));
            if (slots < 2) slots = 2;
            AlignedDoubles out(stride * slots);

            auto invoke = [&](double* dst) {
                if (mode == "temporal") custom->run(dst, in.p, n);
                else if (mode == "nt") custom->run_streaming_write_once(dst, in.p, n);
                else vmdExp((MKL_INT)n, in.p, dst, VML_HA);
            };

            for (int w = 0; w < 3; ++w) invoke(out.p + ((size_t)w % slots) * stride);

            const size_t calls = call_count(n);
            if (calls >= slots) {
                std::fprintf(stderr, "invalid geometry: calls=%zu slots=%zu n=%zu\n", calls, slots, n);
                return 3;
            }

            std::vector<double> samples;
            samples.reserve(7);
            for (int s = 0; s < 7; ++s) {
                const auto t0 = std::chrono::steady_clock::now();
                for (size_t c = 0; c < calls; ++c) {
                    const size_t slot = (c + (size_t)s * calls) % slots;
                    invoke(out.p + slot * stride);
                }
                const auto t1 = std::chrono::steady_clock::now();
                const double ns = std::chrono::duration<double,std::nano>(t1 - t0).count();
                samples.push_back(ns / ((double)calls * (double)n));
            }

            const double med = median(samples);
            std::cout << "RESULT mode=" << mode
                      << " domain=" << (domain == 0 ? "unit" : "mid")
                      << " n=" << n
                      << " calls=" << calls
                      << " slots=" << slots
                      << " ns=" << med
                      << " maxulp=" << (mode == "intel" ? 0 : acc.maxulp)
                      << " gt2=" << (mode == "intel" ? 0 : acc.gt2)
                      << "\n";
        }
    }

    delete custom;
    return 0;
}
