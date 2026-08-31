#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>
#include <mkl_vml.h>

#include "exp53_batch_production.hpp"

static volatile double sink_value = 0.0;

static void *xalloc(size_t bytes) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        std::perror("posix_memalign");
        std::exit(2);
    }
    return p;
}

static double now_ns() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t ru() {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 2685821657736338717ULL;
}
static double rd() {
    return -100.0 + 200.0 * (double)(ru() >> 11) * (1.0 / 9007199254740992.0);
}

static void vha(double *out, const double *in, size_t n) {
    vmdExp((MKL_INT)n, in, out, VML_HA);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static int calls_for(size_t n) {
    uint64_t c = 10000000ULL / n;
    if (c < 128) c = 128;
    if (c > 4096) c = 4096;
    return (int)c;
}

static double measure_ours(Exp53BatchProductionExecutor &ex,
                           double *out, const double *in,
                           size_t n, int calls) {
    const double a = now_ns();
    for (int r = 0; r < calls; ++r) ex.run(out, in, n);
    const double b = now_ns();
    sink_value += out[(size_t)calls % n];
    return (b - a) / ((double)calls * (double)n);
}

static double measure_vml(double *out, const double *in,
                          size_t n, int calls) {
    const double a = now_ns();
    for (int r = 0; r < calls; ++r) vha(out, in, n);
    const double b = now_ns();
    sink_value += out[(size_t)calls % n];
    return (b - a) / ((double)calls * (double)n);
}

static int run_ours() {
    const size_t ns[] = {5000, 8000, 15000, 25000, 35000, 50000, 65000};
    // The persistent FastFlow pool exists only in this OURS process.
    Exp53BatchProductionExecutor ex(2);
    for (size_t n : ns) {
        double *in = (double*)xalloc(n * sizeof(double));
        double *out = (double*)xalloc(n * sizeof(double));
        for (size_t i = 0; i < n; ++i) in[i] = rd();
        for (int w = 0; w < 64; ++w) ex.run(out, in, n);
        const int calls = calls_for(n);
        std::vector<double> t;
        t.reserve(25);
        for (int q = 0; q < 25; ++q) t.push_back(measure_ours(ex, out, in, n, calls));
        std::printf("METHOD ours n=%zu calls=%d ns=%.9f\n", n, calls, median(t));
        std::free(in);
        std::free(out);
    }
    return 0;
}

static int run_vml() {
    const size_t ns[] = {5000, 8000, 15000, 25000, 35000, 50000, 65000};
    // IMPORTANT: no Exp53BatchProductionExecutor is constructed in this process,
    // so Intel is timed with zero FastFlow worker/spin threads alive.
    for (size_t n : ns) {
        double *in = (double*)xalloc(n * sizeof(double));
        double *out = (double*)xalloc(n * sizeof(double));
        for (size_t i = 0; i < n; ++i) in[i] = rd();
        for (int w = 0; w < 64; ++w) vha(out, in, n);
        const int calls = calls_for(n);
        std::vector<double> t;
        t.reserve(25);
        for (int q = 0; q < 25; ++q) t.push_back(measure_vml(out, in, n, calls));
        std::printf("METHOD vml n=%zu calls=%d ns=%.9f\n", n, calls, median(t));
        std::free(in);
        std::free(out);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s ours|vml\n", argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "ours") == 0) return run_ours();
    if (std::strcmp(argv[1], "vml") == 0) return run_vml();
    std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
    return 2;
}
