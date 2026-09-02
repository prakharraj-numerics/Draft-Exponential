#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " ours|intel N CALLS\n";
        return 2;
    }
    const std::string stack = argv[1];
    const size_t n = std::strtoull(argv[2], nullptr, 10);
    const size_t calls = std::strtoull(argv[3], nullptr, 10);
    const std::vector<size_t> allowed = {100,700,3500,15000,50000,1000000,2000000};
    if ((stack != "ours" && stack != "intel") ||
        std::find(allowed.begin(), allowed.end(), n) == allowed.end()) return 2;

    Aligned in(n), out(n), ref(n);
    fill_mixed(in.p, n);
    vmdExp(static_cast<MKL_INT>(n), in.p, ref.p, VML_HA);

    Exp53BatchProductionExecutor executor(2);
    auto invoke = [&] {
        if (stack == "ours") executor.run(out.p, in.p, n, 2);
        else vmdExp(static_cast<MKL_INT>(n), in.p, out.p, VML_HA);
    };

    // One identical warm-up call is present in both K-call and zero-call SDE runs.
    // Differential subtraction removes this plus loader, dispatch, correctness, and shutdown work.
    invoke();
    const uint64_t maxulp = cross_check(out.p, ref.p, n);
    if (maxulp > 2) {
        std::cerr << "correctness failure maxulp=" << maxulp << "\n";
        return 4;
    }

    for (size_t k = 0; k < calls; ++k) invoke();

    volatile double sink = out.p[(n * 17ULL + calls) % n];
    std::cout << "SDE_RUN stack=" << stack << " n=" << n << " calls=" << calls
              << " elements=" << (static_cast<unsigned long long>(n) * calls)
              << " maxulp=" << maxulp << " sink=" << sink << "\n";
    return 0;
}
