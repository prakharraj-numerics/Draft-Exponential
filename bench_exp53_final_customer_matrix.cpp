#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include "exp53_batch_production.hpp"

struct AlignedDoubles {
    double *p = nullptr;
    size_t count = 0;
    explicit AlignedDoubles(size_t n) : count(n) {
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

static const char *domain_name(int d) {
    return d == 0 ? "unit" : (d == 1 ? "mid" : "extreme");
}

static void fill_inputs(double *x, size_t n, int domain) {
    const uint64_t seed0 = 0x6a09e667f3bcc909ULL ^ ((uint64_t)n << 17) ^ ((uint64_t)domain << 57);
    for (size_t i = 0; i < n; ++i) {
        double u = unit01(splitmix64(seed0 + i * 0x9e3779b97f4a7c15ULL));
        double m;
        if (domain == 0) {
            const double e = 0x1p-20;
            m = e + u * (1.0 - 2.0 * e);              // strictly 0 < |x| < 1
        } else if (domain == 1) {
            const double e = 0x1p-20;
            m = 1.0 + e + u * (99.0 - 2.0 * e);       // strictly 1 < |x| < 100
        } else {
            m = 1000.0 + 0x1p-10 + u * (4000.0 - 0x1p-9); // strictly 1000 < |x| < 5000
        }
        // Exact half positive / half negative for every requested (even) n,
        // interleaved so both physical workers see both signs.
        x[i] = (i & 1) ? -m : m;
    }
}

static uint64_t bits_of(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

struct Accuracy {
    size_t bad = 0;
    uint64_t maxulp = 0;
};

static Accuracy compare_exp_outputs(const double *got, const double *ref, size_t n) {
    Accuracy a;
    for (size_t i = 0; i < n; ++i) {
        double g = got[i], r = ref[i];
        bool ok = false;
        uint64_t ulp = 0;
        if (std::isnan(r)) {
            ok = std::isnan(g);
        } else if (std::isinf(r)) {
            ok = std::isinf(g) && !std::signbit(g);
        } else if (r == 0.0) {
            ok = (g == 0.0) && !std::signbit(g);
        } else if (std::isfinite(g) && g > 0.0) {
            uint64_t gb = bits_of(g), rb = bits_of(r);
            ulp = gb > rb ? gb - rb : rb - gb;
            ok = ulp <= 1;
        }
        if (!ok) ++a.bad;
        if (ulp > a.maxulp) a.maxulp = ulp;
    }
    return a;
}

static const std::vector<size_t>& sizes_for_phase(const std::string& phase) {
    static const std::vector<size_t> low     = {50,100,250,500,800,1000,2000,3000};
    static const std::vector<size_t> medium  = {5000,8000,15000,25000,35000,50000,65000};
    static const std::vector<size_t> high    = {200000,500000,1000000,1500000,2500000};
    static const std::vector<size_t> highest = {4000000,5000000,8000000};
    if (phase == "low") return low;
    if (phase == "medium") return medium;
    if (phase == "high") return high;
    return highest;
}

static size_t target_inputs(const std::string& phase) {
    if (phase == "low") return 4000000ULL;
    if (phase == "medium") return 6000000ULL;
    if (phase == "high") return 12000000ULL;
    return 16000000ULL;
}

static size_t call_count(size_t n, const std::string& phase) {
    size_t c = target_inputs(phase) / n;
    if (c < 2) c = 2;
    if (c > 100000) c = 100000;
    return c;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static void run_custom_once(Exp53BatchProductionExecutor& ex, bool stream,
                            double *out, const double *in, size_t n) {
    if (stream) ex.run_streaming_write_once(out, in, n);
    else ex.run(out, in, n);
}

static void run_intel_once(double *out, const double *in, size_t n) {
    vmdExp((MKL_INT)n, in, out, VML_HA);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s custom|intel hot|stream\n", argv[0]);
        return 2;
    }
    const std::string stack = argv[1];
    const std::string policy = argv[2];
    const bool custom = stack == "custom";
    const bool stream = policy == "stream";
    if ((!custom && stack != "intel") || (policy != "hot" && policy != "stream")) return 2;

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "STACK=" << stack << " POLICY=" << policy << "\n";

    // Construct the permanent 2-core executor only in custom processes.
    Exp53BatchProductionExecutor *pex = nullptr;
    if (custom) pex = new Exp53BatchProductionExecutor(2);

    const std::vector<std::string> phases = {"low","medium","high","highest"};
    constexpr size_t RING_BYTES = 256ULL * 1024ULL * 1024ULL;

    for (const auto& phase : phases) {
        for (int domain = 0; domain < 3; ++domain) {
            for (size_t n : sizes_for_phase(phase)) {
                AlignedDoubles in(n), ref(n), check(n);
                fill_inputs(in.p, n, domain);
                vmdExp((MKL_INT)n, in.p, ref.p, VML_HA);

                Accuracy acc{};
                if (custom) {
                    run_custom_once(*pex, stream, check.p, in.p, n);
                    acc = compare_exp_outputs(check.p, ref.p, n);
                }

                const size_t calls = call_count(n, phase);
                const size_t stride = (n + 7u) & ~size_t(7u); // each slot 64-byte aligned
                size_t slots = 1;
                if (stream) {
                    slots = RING_BYTES / (stride * sizeof(double));
                    if (slots < 2) slots = 2;
                }
                const size_t out_elems = stream ? stride * slots : stride;
                AlignedDoubles out(out_elems);

                auto invoke = [&](double *dst) {
                    if (custom) run_custom_once(*pex, stream, dst, in.p, n);
                    else run_intel_once(dst, in.p, n);
                };

                // Warm-up outside timing.
                for (int w = 0; w < 3; ++w) {
                    size_t s = stream ? (size_t)w % slots : 0;
                    invoke(out.p + s * stride);
                }

                std::vector<double> samples;
                samples.reserve(7);
                for (int s = 0; s < 7; ++s) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (size_t c = 0; c < calls; ++c) {
                        size_t slot = stream ? ((c + (size_t)s * calls) % slots) : 0;
                        invoke(out.p + slot * stride);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double ns = std::chrono::duration<double,std::nano>(t1-t0).count();
                    samples.push_back(ns / ((double)calls * (double)n));
                }
                double med = median(samples);
                const int valid = custom ? (acc.bad == 0 ? 1 : 0) : 1;
                std::cout << "RESULT stack=" << stack
                          << " policy=" << policy
                          << " phase=" << phase
                          << " domain=" << domain_name(domain)
                          << " n=" << n
                          << " calls=" << calls
                          << " slots=" << slots
                          << " ns=" << med
                          << " valid=" << valid
                          << " bad=" << (custom ? acc.bad : 0)
                          << " maxulp=" << (custom ? acc.maxulp : 0)
                          << "\n";
            }
        }
    }

    delete pex;
    return 0;
}
