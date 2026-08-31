#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <mkl_vml.h>

#include "exp53_fastflow_batch.hpp"

static volatile double sink_value = 0.0;

static void *xalloc(size_t bytes) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        std::perror("posix_memalign");
        std::exit(2);
    }
    return p;
}

static uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}
static uint64_t ordered(double x) {
    const uint64_t u = bits(x);
    return (u >> 63) ? ~u : (u | 0x8000000000000000ULL);
}
static uint64_t ulpdiff(double a, double b) {
    const uint64_t x = ordered(a), y = ordered(b);
    return x > y ? x - y : y - x;
}

static uint64_t rng_state = 0x2545f4914f6cdd1dULL;
static uint64_t ru(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 2685821657736338717ULL;
}
static double rd(void) {
    const double z = (double)(ru() >> 11) * (1.0 / 9007199254740992.0);
    return -100.0 + 200.0 * z;
}

static void vha(double *out, const double *in, size_t n) {
    vmdExp((MKL_INT)n, in, out, VML_HA);
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec * 1.0e9 + (double)t.tv_nsec;
}

struct Stats {
    double best_ns_per_input;
    double median_ns_per_input;
    double median_call_ns;
};

template <class F>
static Stats bench(F &&fn, double *out, const double *in, size_t n) {
    for (int w = 0; w < 8; ++w) fn(out, in, n);

    int calls = (int)(4000000ULL / n);
    if (calls < 4) calls = 4;
    if (calls > 5000) calls = 5000;

    std::vector<double> v;
    v.reserve(7);
    for (int q = 0; q < 7; ++q) {
        const double a = now_ns();
        for (int r = 0; r < calls; ++r) fn(out, in, n);
        const double b = now_ns();
        const double per_input = (b - a) / ((double)calls * (double)n);
        v.push_back(per_input);
        sink_value += out[((size_t)q * 157u) % n];
    }
    std::sort(v.begin(), v.end());
    Stats s;
    s.best_ns_per_input = v.front();
    s.median_ns_per_input = v[v.size() / 2];
    s.median_call_ns = s.median_ns_per_input * (double)n;
    return s;
}

struct Best {
    double med = std::numeric_limits<double>::infinity();
    double best = std::numeric_limits<double>::infinity();
    std::string cfg = "none";
    void consider(const Stats &s, const std::string &c) {
        if (s.median_ns_per_input < med) {
            med = s.median_ns_per_input;
            best = s.best_ns_per_input;
            cfg = c;
        }
    }
};

static std::vector<long> worker_sweep(unsigned hw) {
    const long cap = std::max(1L, std::min<long>((long)hw, 32L));
    std::vector<long> w{1};
    for (long x = 2; x <= cap; x *= 2) w.push_back(x);
    if (w.back() != cap) w.push_back(cap);
    return w;
}

static void accuracy_screen(long workers) {
    const size_t n = 200000;
    double *in = (double*)xalloc(n * sizeof(double));
    double *serial = (double*)xalloc(n * sizeof(double));
    double *st = (double*)xalloc(n * sizeof(double));
    double *dy = (double*)xalloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) in[i] = rd();

    exp53_n2_vmstyle_u4_0381_frozen(serial, in, n);
    {
        Exp53FastFlowExecutor ex(workers, exp53_n2_vmstyle_u4_0381_frozen);
        ex.run_static(st, in, n, workers);
        ex.run_dynamic(dy, in, n, workers, 512);
    }

    uint64_t bd_st = 0, bd_dy = 0, mx = 0, gt1 = 0;
    for (size_t i = 0; i < n; ++i) {
        if (bits(st[i]) != bits(serial[i])) ++bd_st;
        if (bits(dy[i]) != bits(serial[i])) ++bd_dy;
        const double ref = (double)std::exp((long double)in[i]);
        const uint64_t d = ulpdiff(st[i], ref);
        mx = std::max(mx, d);
        if (d > 1) ++gt1;
    }
    std::printf("FASTFLOW_ACCURACY workers=%ld static_bitdiff=%llu dynamic_bitdiff=%llu maxULP=%llu gt1=%llu\n",
        workers,
        (unsigned long long)bd_st,
        (unsigned long long)bd_dy,
        (unsigned long long)mx,
        (unsigned long long)gt1);

    std::free(in); std::free(serial); std::free(st); std::free(dy);
}

int main(void) {
    const size_t ns[] = {100, 253, 10079, 12288, 65536, 262144, 1000000};
    constexpr size_t NN = sizeof(ns) / sizeof(ns[0]);
    const size_t maxn = ns[NN - 1];

    double *in = (double*)xalloc(maxn * sizeof(double));
    double *out = (double*)xalloc(maxn * sizeof(double));
    for (size_t i = 0; i < maxn; ++i) in[i] = rd();

    const unsigned hw_raw = std::thread::hardware_concurrency();
    const unsigned hw = hw_raw ? hw_raw : 1;
    const std::vector<long> workers = worker_sweep(hw);
    std::printf("FASTFLOW_ENV hw_concurrency=%u worker_cap=%ld\n", hw, workers.back());

    Stats serial_ours[NN], serial_vml[NN];
    Best best_ours[NN], best_vml[NN];

    /* Serial baselines are measured before any spinning FastFlow pool exists. */
    for (size_t z = 0; z < NN; ++z) {
        const size_t n = ns[z];
        serial_ours[z] = bench([](double *o, const double *x, size_t m) {
            exp53_n2_vmstyle_u4_0381_frozen(o, x, m);
        }, out, in, n);
        serial_vml[z] = bench([](double *o, const double *x, size_t m) {
            vha(o, x, m);
        }, out, in, n);
        std::printf("FASTFLOW_SERIAL n=%zu ours_med=%.9f ours_best=%.9f vml_med=%.9f vml_best=%.9f\n",
            n,
            serial_ours[z].median_ns_per_input, serial_ours[z].best_ns_per_input,
            serial_vml[z].median_ns_per_input, serial_vml[z].best_ns_per_input);
    }

    const size_t dyn_blocks[] = {128, 512, 2048, 8192};

    /* OURS: one persistent pool per worker-count. Construction/destruction is
       outside timed regions; dispatch + completion synchronization stays inside. */
    for (long w : workers) {
        if (w <= 1) continue;
        Exp53FastFlowExecutor ex(w, exp53_n2_vmstyle_u4_0381_frozen);
        for (size_t z = 0; z < NN; ++z) {
            const size_t n = ns[z];
            Stats s = bench([&](double *o, const double *x, size_t m) {
                ex.run_static(o, x, m, w);
            }, out, in, n);
            const std::string cfg = "static-w" + std::to_string(w);
            best_ours[z].consider(s, cfg);
            std::printf("FASTFLOW_RUN impl=ours mode=static workers=%ld n=%zu med=%.9f best=%.9f call_ns=%.3f\n",
                w, n, s.median_ns_per_input, s.best_ns_per_input, s.median_call_ns);

            for (size_t g : dyn_blocks) {
                if ((n + g - 1) / g < 2) continue;
                Stats d = bench([&](double *o, const double *x, size_t m) {
                    ex.run_dynamic(o, x, m, w, g);
                }, out, in, n);
                const std::string dcfg = "dynamic-w" + std::to_string(w) + "-b" + std::to_string(g);
                best_ours[z].consider(d, dcfg);
                std::printf("FASTFLOW_RUN impl=ours mode=dynamic workers=%ld block=%zu n=%zu med=%.9f best=%.9f call_ns=%.3f\n",
                    w, g, n, d.median_ns_per_input, d.best_ns_per_input, d.median_call_ns);
            }
        }
    }

    /* Fair control: identical FastFlow scheduling around sequential VML_HA chunks. */
    for (long w : workers) {
        if (w <= 1) continue;
        Exp53FastFlowExecutor ex(w, vha);
        for (size_t z = 0; z < NN; ++z) {
            const size_t n = ns[z];
            Stats s = bench([&](double *o, const double *x, size_t m) {
                ex.run_static(o, x, m, w);
            }, out, in, n);
            const std::string cfg = "static-w" + std::to_string(w);
            best_vml[z].consider(s, cfg);
            std::printf("FASTFLOW_RUN impl=vml mode=static workers=%ld n=%zu med=%.9f best=%.9f call_ns=%.3f\n",
                w, n, s.median_ns_per_input, s.best_ns_per_input, s.median_call_ns);

            for (size_t g : dyn_blocks) {
                if ((n + g - 1) / g < 2) continue;
                Stats d = bench([&](double *o, const double *x, size_t m) {
                    ex.run_dynamic(o, x, m, w, g);
                }, out, in, n);
                const std::string dcfg = "dynamic-w" + std::to_string(w) + "-b" + std::to_string(g);
                best_vml[z].consider(d, dcfg);
                std::printf("FASTFLOW_RUN impl=vml mode=dynamic workers=%ld block=%zu n=%zu med=%.9f best=%.9f call_ns=%.3f\n",
                    w, g, n, d.median_ns_per_input, d.best_ns_per_input, d.median_call_ns);
            }
        }
    }

    accuracy_screen(workers.back());

    for (size_t z = 0; z < NN; ++z) {
        const double ours_effective = std::min(serial_ours[z].median_ns_per_input, best_ours[z].med);
        const double vml_effective = std::min(serial_vml[z].median_ns_per_input, best_vml[z].med);
        const char *ours_kind = (serial_ours[z].median_ns_per_input <= best_ours[z].med) ? "serial" : best_ours[z].cfg.c_str();
        const char *vml_kind = (serial_vml[z].median_ns_per_input <= best_vml[z].med) ? "serial" : best_vml[z].cfg.c_str();
        std::printf("FASTFLOW_BEST n=%zu ours_parallel_med=%.9f ours_cfg=%s vml_parallel_med=%.9f vml_cfg=%s ours_effective_med=%.9f ours_effective_cfg=%s vml_effective_med=%.9f vml_effective_cfg=%s serial_to_ours_speedup=%.6f vml_effective_over_ours=%.6f\n",
            ns[z], best_ours[z].med, best_ours[z].cfg.c_str(), best_vml[z].med, best_vml[z].cfg.c_str(),
            ours_effective, ours_kind, vml_effective, vml_kind,
            serial_ours[z].median_ns_per_input / ours_effective,
            vml_effective / ours_effective);
    }

    std::free(in); std::free(out);
    return sink_value == 1234567.0;
}
