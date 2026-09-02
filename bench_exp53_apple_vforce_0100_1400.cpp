#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstring>
#include <string>
#include <vector>

struct Aligned {
    double* p = nullptr;
    explicit Aligned(size_t n) {
        if (posix_memalign(reinterpret_cast<void**>(&p), 64, n * sizeof(double)) != 0 || !p) std::exit(2);
    }
    ~Aligned() { std::free(p); }
    Aligned(const Aligned&) = delete;
    Aligned& operator=(const Aligned&) = delete;
};

static inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static double unit01(uint64_t h) {
    return (static_cast<double>(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static void fill(double* x, size_t n, int domain) {
    const uint64_t seed = 0x243f6a8885a308d3ULL ^ (static_cast<uint64_t>(n) << 19)
                        ^ (static_cast<uint64_t>(domain) << 61);
    for (size_t i = 0; i < n; ++i) {
        const double u = unit01(mix(seed + i * 0x9e3779b97f4a7c15ULL));
        const double edge = 0x1p-20;
        const double mag = domain == 0 ? edge + u * (1.0 - 2.0 * edge)
                                       : 1.0 + edge + u * (99.0 - 2.0 * edge);
        x[i] = (i & 1) ? -mag : mag;
    }
}

static size_t calls_for(size_t n) {
    size_t calls = 5000000ULL / n;
    calls = std::max<size_t>(calls, 1000);
    calls = std::min<size_t>(calls, 50000);
    return calls;
}

static double median(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static uint64_t ordered_bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return (u & (1ULL << 63)) ? ~u + 1 : u | (1ULL << 63);
}

int main() {
    std::cout << std::fixed << std::setprecision(9);
    volatile double sink = 0.0;
    size_t checked = 0;
    uint64_t max_ulp = 0;

    for (int domain = 0; domain < 2; ++domain) {
        for (size_t n = 100; n <= 1400; n += 50) {
            Aligned in(n), out(n);
            fill(in.p, n, domain);
            const int count = static_cast<int>(n);
            auto run = [&] { vvexp(out.p, in.p, &count); };

            run();
            for (size_t i = 0; i < n; ++i) {
                const double ref = std::exp(in.p[i]);
                if (!std::isfinite(out.p[i]) || !std::isfinite(ref)) {
                    std::cerr << "nonfinite domain=" << domain << " n=" << n << " i=" << i << "\n";
                    return 3;
                }
                const double rel = std::abs(out.p[i] - ref) / ref;
                if (rel > 8.0 * std::numeric_limits<double>::epsilon()) {
                    std::cerr << "accuracy domain=" << domain << " n=" << n << " i=" << i
                              << " got=" << std::setprecision(17) << out.p[i]
                              << " ref=" << ref << " rel=" << rel << "\n";
                    return 4;
                }
                const uint64_t a = ordered_bits(out.p[i]);
                const uint64_t b = ordered_bits(ref);
                const uint64_t ulp = a > b ? a - b : b - a;
                max_ulp = std::max(max_ulp, ulp);
                ++checked;
            }

            for (int w = 0; w < 10; ++w) run();
            const size_t calls = calls_for(n);
            std::vector<double> samples;
            samples.reserve(9);
            for (int sample = 0; sample < 9; ++sample) {
                const auto begin = std::chrono::steady_clock::now();
                for (size_t k = 0; k < calls; ++k) run();
                const auto end = std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count()
                                  / (static_cast<double>(calls) * n));
            }
            sink += out.p[(n * 13u + static_cast<size_t>(domain)) % n] * 0x1p-1022;
            std::cout << "RESULT stack=apple_vforce domain=" << (domain ? "mid" : "unit")
                      << " n=" << n << " calls=" << calls
                      << " ns=" << median(samples) << "\n";
        }
    }

    std::cout << "CORRECTNESS checked=" << checked << " max_ulp_vs_system_exp=" << max_ulp << "\n";
    if (sink == 123.0) std::cerr << sink;
    return 0;
}
