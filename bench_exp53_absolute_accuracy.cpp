#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <mpfr.h>
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
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static inline uint64_t ulp_distance_positive(double a, double b) {
    const uint64_t ua = bits(a), ub = bits(b);
    return ua > ub ? ua - ub : ub - ua;
}

class MPFRReference {
public:
    MPFRReference() {
        mpfr_init2(x_, 64);
        mpfr_init2(lo256_, 256); mpfr_init2(hi256_, 256);
        mpfr_init2(lo1024_, 1024); mpfr_init2(hi1024_, 1024);
    }
    ~MPFRReference() {
        mpfr_clear(x_);
        mpfr_clear(lo256_); mpfr_clear(hi256_);
        mpfr_clear(lo1024_); mpfr_clear(hi1024_);
    }

    double exp_correctly_rounded(double x, bool& used1024) {
        mpfr_set_d(x_, x, MPFR_RNDN); // exact: binary64 input fits in 64-bit MPFR significand

        mpfr_exp(lo256_, x_, MPFR_RNDD);
        mpfr_exp(hi256_, x_, MPFR_RNDU);
        const double dlo = mpfr_get_d(lo256_, MPFR_RNDN);
        const double dhi = mpfr_get_d(hi256_, MPFR_RNDN);
        if (bits(dlo) == bits(dhi)) {
            used1024 = false;
            return dlo;
        }

        mpfr_exp(lo1024_, x_, MPFR_RNDD);
        mpfr_exp(hi1024_, x_, MPFR_RNDU);
        const double d2lo = mpfr_get_d(lo1024_, MPFR_RNDN);
        const double d2hi = mpfr_get_d(hi1024_, MPFR_RNDN);
        if (bits(d2lo) != bits(d2hi)) {
            std::cerr << "REFERENCE_AMBIGUOUS x=" << std::hexfloat << x << std::defaultfloat << "\n";
            std::exit(7);
        }
        used1024 = true;
        return d2lo;
    }

private:
    mpfr_t x_, lo256_, hi256_, lo1024_, hi1024_;
};

int main() {
    const std::vector<size_t> sizes = {100, 700, 3500, 15000, 50000, 1000000, 2000000};
    Exp53BatchProductionExecutor executor(2);
    MPFRReference refgen;

    uint64_t grand_total = 0, grand_exact = 0, grand_one = 0, grand_gt1 = 0;
    uint64_t grand_fallback1024 = 0, grand_maxulp = 0;

    for (size_t n : sizes) {
        Aligned in(n), out(n);
        fill_mixed(in.p, n);
        executor.run(out.p, in.p, n, 2);

        uint64_t exact = 0, one = 0, gt1 = 0, fallback1024 = 0, maxulp = 0;
        size_t worst_i = 0;
        double worst_x = 0.0, worst_got = 0.0, worst_ref = 0.0;

        for (size_t i = 0; i < n; ++i) {
            bool used1024 = false;
            const double ref = refgen.exp_correctly_rounded(in.p[i], used1024);
            fallback1024 += used1024 ? 1 : 0;
            const double got = out.p[i];
            if (!(got > 0.0) || !std::isfinite(got) || !(ref > 0.0) || !std::isfinite(ref)) {
                std::cerr << "NONFINITE_OR_NONPOSITIVE n=" << n << " i=" << i << "\n";
                return 8;
            }
            const uint64_t d = ulp_distance_positive(got, ref);
            if (d == 0) ++exact;
            else if (d == 1) ++one;
            else ++gt1;
            if (d > maxulp) {
                maxulp = d;
                worst_i = i; worst_x = in.p[i]; worst_got = got; worst_ref = ref;
            }
        }

        grand_total += n; grand_exact += exact; grand_one += one; grand_gt1 += gt1;
        grand_fallback1024 += fallback1024;
        grand_maxulp = std::max(grand_maxulp, maxulp);

        std::cout << "ABSACC n=" << n
                  << " total=" << n
                  << " exact=" << exact
                  << " one_ulp=" << one
                  << " gt1=" << gt1
                  << " maxulp=" << maxulp
                  << " ref1024=" << fallback1024
                  << " worst_i=" << worst_i
                  << std::hexfloat
                  << " worst_x=" << worst_x
                  << " worst_got=" << worst_got
                  << " worst_ref=" << worst_ref
                  << std::defaultfloat << "\n";
    }

    std::cout << "ABSACC_TOTAL total=" << grand_total
              << " exact=" << grand_exact
              << " one_ulp=" << grand_one
              << " gt1=" << grand_gt1
              << " maxulp=" << grand_maxulp
              << " ref1024=" << grand_fallback1024 << "\n";

    return grand_gt1 == 0 ? 0 : 9;
}
