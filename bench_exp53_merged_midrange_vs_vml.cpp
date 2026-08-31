#include <algorithm>
#include <array>
#include <cmath>
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

int main() {
    const size_t ns[] = {5000, 8000, 15000, 25000, 35000, 50000, 65000};
    Exp53BatchProductionExecutor ex(2);

    std::printf("MERGED_POLICY default=temporal_fastflow workers=2 intel=plain_VML_HA\n");

    for (size_t n : ns) {
        double *in = (double*)xalloc(n * sizeof(double));
        double *ours = (double*)xalloc(n * sizeof(double));
        double *vml = (double*)xalloc(n * sizeof(double));
        for (size_t i = 0; i < n; ++i) in[i] = rd();

        // Warm both methods and fully instantiate the persistent FastFlow pool.
        for (int w = 0; w < 64; ++w) {
            ex.run(ours, in, n);
            vha(vml, in, n);
        }

        const int calls = calls_for(n);
        std::vector<double> to, tv;
        to.reserve(25);
        tv.reserve(25);

        // Alternate order to reduce drift/order bias.
        for (int q = 0; q < 25; ++q) {
            if ((q & 1) == 0) {
                to.push_back(measure_ours(ex, ours, in, n, calls));
                tv.push_back(measure_vml(vml, in, n, calls));
            } else {
                tv.push_back(measure_vml(vml, in, n, calls));
                to.push_back(measure_ours(ex, ours, in, n, calls));
            }
        }

        const double mo = median(to);
        const double mv = median(tv);
        std::printf("MERGED_VS_VML n=%zu calls=%d ours_ff=%.9f intel_vml_ha=%.9f intel_over_ours=%.6f ours_adv_pct=%.3f\n",
                    n, calls, mo, mv, mv / mo, (mv / mo - 1.0) * 100.0);

        std::free(in);
        std::free(ours);
        std::free(vml);
    }

    return sink_value == 1234567.0;
}
