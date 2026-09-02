#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <mkl_vml.h>
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
    uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}
static uint64_t cross_check(const double* got, const double* ref, size_t n) {
    uint64_t maxulp = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) || got[i] <= 0.0 || ref[i] <= 0.0) return UINT64_MAX;
        const uint64_t a = bits(got[i]), b = bits(ref[i]);
        maxulp = std::max(maxulp, a > b ? a - b : b - a);
    }
    return maxulp;
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); return v[v.size()/2];
}

static int perf_open(uint32_t type, uint64_t config) {
    perf_event_attr pe{};
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = 0;             // events run continuously; we use read deltas around only the measured loop
    pe.inherit = 1;              // helper thread is created after counters, so its work is included
    pe.inherit_stat = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return static_cast<int>(syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0));
}
struct Counter {
    const char* name;
    int fd = -1;
    Counter(const char* n, uint32_t type, uint64_t config): name(n), fd(perf_open(type, config)) {}
    ~Counter() { if (fd >= 0) close(fd); }
    bool ok() const { return fd >= 0; }
    uint64_t read_now() const {
        uint64_t v = 0;
        if (fd < 0 || ::read(fd, &v, sizeof(v)) != sizeof(v)) return 0;
        return v;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " ours|intel N\n";
        return 2;
    }
    const std::string stack = argv[1];
    if (stack != "ours" && stack != "intel") return 2;
    const size_t n = std::strtoull(argv[2], nullptr, 10);
    const std::vector<size_t> allowed = {100,700,3500,15000,50000,1000000,2000000};
    if (std::find(allowed.begin(), allowed.end(), n) == allowed.end()) return 2;

    Aligned in(n), out(n), ref(n);
    fill_mixed(in.p, n);
    vmdExp(static_cast<MKL_INT>(n), in.p, ref.p, VML_HA);

    // Generic architectural PMU events: actual hardware counters, not CPU-time proxies.
    Counter cycles("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    Counter instructions("instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    Counter cache_refs("cache_references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    Counter cache_misses("cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    Counter branches("branches", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    Counter branch_misses("branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    Counter stalled_front("stalled_frontend_cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND);
    Counter stalled_back("stalled_backend_cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND);

    // Intel FP_ARITH_INST_RETIRED.512B_PACKED_DOUBLE (event 0xC7, umask 0x40) on supported Xeons.
    // Reported only if the PMU accepts the raw event; otherwise unavailable.
    Counter fp512("fp512_packed_double", PERF_TYPE_RAW, 0x40c7);

    std::unique_ptr<Exp53BatchProductionExecutor> executor;
    if (stack == "ours") executor = std::make_unique<Exp53BatchProductionExecutor>(2);
    auto invoke = [&] {
        if (stack == "ours") executor->run(out.p, in.p, n, 2);
        else vmdExp(static_cast<MKL_INT>(n), in.p, out.p, VML_HA);
    };

    invoke();
    const uint64_t maxulp = cross_check(out.p, ref.p, n);
    if (maxulp > 2) {
        std::cerr << "correctness failure maxulp=" << maxulp << "\n";
        return 4;
    }
    for (int i = 0; i < 8; ++i) invoke();

    size_t calls = 20000000ULL / n;
    if (calls < 1) calls = 1;
    if (calls > 200000) calls = 200000;
    const double elems = static_cast<double>(calls) * static_cast<double>(n);

    std::cout << "COUNTER_AVAIL stack=" << stack << " n=" << n
              << " cycles=" << cycles.ok()
              << " instructions=" << instructions.ok()
              << " cache_references=" << cache_refs.ok()
              << " cache_misses=" << cache_misses.ok()
              << " branches=" << branches.ok()
              << " branch_misses=" << branch_misses.ok()
              << " stalled_frontend=" << stalled_front.ok()
              << " stalled_backend=" << stalled_back.ok()
              << " fp512=" << fp512.ok() << "\n";

    std::vector<double> wall_v, cyc_v, ins_v, cref_v, cmiss_v, br_v, bmiss_v, sf_v, sb_v, fp512_v, ipc_v;
    wall_v.reserve(5); cyc_v.reserve(5); ins_v.reserve(5); cref_v.reserve(5); cmiss_v.reserve(5);
    br_v.reserve(5); bmiss_v.reserve(5); sf_v.reserve(5); sb_v.reserve(5); fp512_v.reserve(5); ipc_v.reserve(5);

    volatile double sink = 0.0;
    for (int sample = 0; sample < 5; ++sample) {
        const uint64_t c0=cycles.read_now(), i0=instructions.read_now(), cr0=cache_refs.read_now(), cm0=cache_misses.read_now();
        const uint64_t b0=branches.read_now(), bm0=branch_misses.read_now(), sf0=stalled_front.read_now(), sb0=stalled_back.read_now(), f0=fp512.read_now();
        const auto w0 = std::chrono::steady_clock::now();
        for (size_t k=0; k<calls; ++k) invoke();
        const auto w1 = std::chrono::steady_clock::now();
        const uint64_t c1=cycles.read_now(), i1=instructions.read_now(), cr1=cache_refs.read_now(), cm1=cache_misses.read_now();
        const uint64_t b1=branches.read_now(), bm1=branch_misses.read_now(), sf1=stalled_front.read_now(), sb1=stalled_back.read_now(), f1=fp512.read_now();

        wall_v.push_back(std::chrono::duration<double,std::nano>(w1-w0).count()/elems);
        if (cycles.ok()) cyc_v.push_back((c1-c0)/elems);
        if (instructions.ok()) ins_v.push_back((i1-i0)/elems);
        if (cache_refs.ok()) cref_v.push_back((cr1-cr0)/elems);
        if (cache_misses.ok()) cmiss_v.push_back((cm1-cm0)/elems);
        if (branches.ok()) br_v.push_back((b1-b0)/elems);
        if (branch_misses.ok()) bmiss_v.push_back((bm1-bm0)/elems);
        if (stalled_front.ok()) sf_v.push_back((sf1-sf0)/elems);
        if (stalled_back.ok()) sb_v.push_back((sb1-sb0)/elems);
        if (fp512.ok()) fp512_v.push_back((f1-f0)/elems);
        if (cycles.ok() && instructions.ok() && c1>c0) ipc_v.push_back(static_cast<double>(i1-i0)/static_cast<double>(c1-c0));
        sink += out.p[(sample*104729ULL)%n] * 0x1p-1022;
    }

    auto val = [](const std::vector<double>& v)->double { return v.empty() ? -1.0 : median(v); };
    const double cyc=val(cyc_v), ins=val(ins_v), cref=val(cref_v), cmiss=val(cmiss_v), br=val(br_v), bmiss=val(bmiss_v);
    const double sf=val(sf_v), sb=val(sb_v), fpd=val(fp512_v), ipc=val(ipc_v);
    const double cache_miss_rate = (cref>0.0 && cmiss>=0.0) ? cmiss/cref : -1.0;
    const double branch_miss_rate = (br>0.0 && bmiss>=0.0) ? bmiss/br : -1.0;
    const double fp512_lanes = fpd>=0.0 ? 8.0*fpd : -1.0;

    std::cout << std::fixed << std::setprecision(9)
              << "HWRESULT stack=" << stack << " n=" << n << " calls=" << calls
              << " wall_ns_per_element=" << median(wall_v)
              << " cycles_per_element=" << cyc
              << " instructions_per_element=" << ins
              << " ipc=" << ipc
              << " cache_references_per_element=" << cref
              << " cache_misses_per_element=" << cmiss
              << " cache_miss_rate=" << cache_miss_rate
              << " branches_per_element=" << br
              << " branch_misses_per_element=" << bmiss
              << " branch_miss_rate=" << branch_miss_rate
              << " stalled_frontend_cycles_per_element=" << sf
              << " stalled_backend_cycles_per_element=" << sb
              << " fp512_inst_per_element=" << fpd
              << " fp512_double_lanes_per_element=" << fp512_lanes
              << " memory_bytes_per_element=-1"
              << " joules_per_element=-1"
              << " maxulp=" << maxulp << "\n";
    if (sink == 123.0) std::cerr << sink;
    return 0;
}
