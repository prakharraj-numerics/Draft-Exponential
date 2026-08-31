// Exhaustive low-batch crossover sweep: frozen EXP53 + oneTBB static 2-way vs Intel VML_HA.
// Sizes: n=50,100,...,3000. No custom2, no FastFlow, no NT stores.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/partitioner.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double *out,
                                                  const double *in,
                                                  size_t n);

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

static double unit01(uint64_t h) {
    return ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static const char *domain_name(int d) { return d == 0 ? "unit" : "mid"; }

static void fill_inputs(double *x, size_t n, int domain) {
    const uint64_t seed0 = 0x6a09e667f3bcc909ULL ^ ((uint64_t)n << 17) ^ ((uint64_t)domain << 57);
    for (size_t i = 0; i < n; ++i) {
        const double u = unit01(splitmix64(seed0 + i * 0x9e3779b97f4a7c15ULL));
        const double e = 0x1p-20;
        const double m = (domain == 0)
            ? (e + u * (1.0 - 2.0 * e))
            : (1.0 + e + u * (99.0 - 2.0 * e));
        x[i] = (i & 1) ? -m : m;
    }
}

static size_t call_count(size_t n) {
    size_t c = 4000000ULL / n;
    if (c < 2) c = 2;
    if (c > 100000) c = 100000;
    return c;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

class TbbExp53Executor {
    oneapi::tbb::global_control gc_;
    oneapi::tbb::task_arena arena_;
public:
    TbbExp53Executor()
      : gc_(oneapi::tbb::global_control::max_allowed_parallelism, 2),
        arena_(2) {
        arena_.initialize();
        // Force worker/team creation outside timed calls.
        arena_.execute([&] {
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<int>(0, 2, 1),
                [](const oneapi::tbb::blocked_range<int>& r) {
                    volatile int x = 0;
                    for (int i = r.begin(); i != r.end(); ++i) x += i;
                    (void)x;
                },
                oneapi::tbb::static_partitioner{});
        });
    }

    void run(double *out, const double *in, size_t n) {
        // Same sensible escape used by low-overhead two-core paths: if two full
        // 32-value blocks do not exist, do not pay a parallel scheduler at all.
        if (n < 64) {
            exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
            return;
        }

        // Split at a 32-value boundary nearest half. This guarantees two
        // contiguous independent chunks and preserves the frozen kernel exactly.
        size_t split = ((n / 2) / 32) * 32;
        if (split < 32) split = 32;
        if (n - split < 32) split = n - 32;

        arena_.execute([&] {
            // Range [0,2), grainsize 1 + static_partitioner => exactly the two
            // fixed logical chunks, without dynamic recursive load balancing.
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<int>(0, 2, 1),
                [&](const oneapi::tbb::blocked_range<int>& r) {
                    for (int k = r.begin(); k != r.end(); ++k) {
                        if (k == 0)
                            exp53_n2_vmstyle_u4_0381_frozen(out, in, split);
                        else
                            exp53_n2_vmstyle_u4_0381_frozen(out + split, in + split, n - split);
                    }
                },
                oneapi::tbb::static_partitioner{});
        });
    }
};

static void run_intel(double *out, const double *in, size_t n) {
    vmdExp((MKL_INT)n, in, out, VML_HA);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s tbb|intel\n", argv[0]);
        return 2;
    }
    const std::string stack = argv[1];
    if (stack != "tbb" && stack != "intel") return 2;
    const bool use_tbb = stack == "tbb";
    const bool reverse = std::getenv("SWEEP_REVERSE") && std::string(std::getenv("SWEEP_REVERSE")) == "1";

    std::vector<size_t> sizes;
    for (size_t n = 50; n <= 3000; n += 50) sizes.push_back(n);
    if (reverse) std::reverse(sizes.begin(), sizes.end());

    std::vector<int> domains = {0,1};
    if (reverse) std::reverse(domains.begin(), domains.end());

    // Created only in the TBB process lifetime. Intel comparator never coexists
    // with a live oneTBB arena/workers.
    TbbExp53Executor *exec = use_tbb ? new TbbExp53Executor() : nullptr;

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "STACK=" << stack << " REVERSE=" << (reverse ? 1 : 0) << "\n";

    volatile double sink = 0.0;
    for (int domain : domains) {
        for (size_t n : sizes) {
            AlignedDoubles in(n), out(n);
            fill_inputs(in.p, n, domain);
            const size_t calls = call_count(n);

            auto invoke = [&]() {
                if (use_tbb) exec->run(out.p, in.p, n);
                else run_intel(out.p, in.p, n);
            };

            for (int w = 0; w < 5; ++w) invoke();

            std::vector<double> samples;
            samples.reserve(7);
            for (int s = 0; s < 7; ++s) {
                const auto t0 = std::chrono::steady_clock::now();
                for (size_t c = 0; c < calls; ++c) invoke();
                const auto t1 = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double,std::nano>(t1-t0).count();
                samples.push_back(elapsed / ((double)calls * (double)n));
            }
            sink += out.p[(n * 13u + (size_t)domain) % n] * 0x1p-1022;
            const double med = median(samples);
            std::cout << "RESULT stack=" << stack
                      << " domain=" << domain_name(domain)
                      << " n=" << n
                      << " calls=" << calls
                      << " ns=" << med
                      << "\n";
        }
    }
    delete exec;
    if (sink == 123456789.0) std::fprintf(stderr, "sink=%g\n", (double)sink);
    return 0;
}
