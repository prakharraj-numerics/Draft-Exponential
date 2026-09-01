#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
void exp53_highway_nt_0100_2000_candidate(double*, const double*, size_t);
#include "exp53_highway_nt_sync2_0100_2000_candidate.hpp"

struct Buffer {
  double* p;
  explicit Buffer(size_t n) { if (posix_memalign((void**)&p, 64, n * sizeof(double))) std::abort(); }
  ~Buffer() { free(p); }
};

static uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static double unit01(uint64_t h) {
  return ((double)(h >> 11) + 0.5) / 9007199254740992.0;
}

static void fill_inputs(double* x, size_t n, int domain) {
  const uint64_t seed = 0x243f6a8885a308d3ULL ^ ((uint64_t)n << 19) ^ ((uint64_t)domain << 61);
  for (size_t i = 0; i < n; ++i) {
    const double q = unit01(mix64(seed + i * 0x9e3779b97f4a7c15ULL));
    const double a = domain ? 1.000001 + q * 98.999998 : 0x1p-20 + q * (1.0 - 0x1p-19);
    x[i] = (i & 1) ? -a : a;
  }
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static size_t calls_for(size_t n) {
  size_t c = 220000 / n;
  return std::max<size_t>(64, std::min<size_t>(2200, c));
}

template <class F>
static double time_ns_per_element(F f, size_t n, size_t calls) {
  for (int i = 0; i < 12; ++i) f();
  std::vector<double> samples;
  for (int rep = 0; rep < 11; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (size_t z = 0; z < calls; ++z) f();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / calls / n);
  }
  return median(samples);
}

int main() {
  constexpr size_t max_n = 2000;
  Buffer in(max_n), ref(max_n), out(max_n);
  const size_t sizes[] = {100,125,150,200,250,300,400,500,600,750,900,1000,1100,1250,1400,1500,1600,1700,1750,1800,1850,1900,1950,2000};
  const unsigned shares[] = {20,25,30,35,40,45,50};
  Exp53HighwayNTSync2Candidate sync2;

  std::cout << std::fixed << std::setprecision(9)
            << "HIGHWAY_NT_SYNC2 exact_bitwise=required cpus=0+2 sfence=per_kernel\n";

  for (int domain = 0; domain < 2; ++domain) {
    const char* name = domain ? "mid" : "unit";
    for (size_t n : sizes) {
      fill_inputs(in.p, n, domain);
      exp53_n2_vmstyle_u4_0381_frozen(ref.p, in.p, n);
      const size_t calls = calls_for(n);

      double best = 1e300;
      unsigned best_share = 0;
      for (unsigned share : shares) {
        sync2.run(out.p, in.p, n, share);
        if (std::memcmp(out.p, ref.p, n * sizeof(double)) != 0) {
          std::cout << "FAIL domain=" << name << " n=" << n << " share=" << share << " bitdiff=1\n";
          return 9;
        }
        const double t = time_ns_per_element([&] { sync2.run(out.p, in.p, n, share); }, n, calls);
        if (t < best) { best = t; best_share = share; }
      }

      const double serial_nt = time_ns_per_element([&] { exp53_highway_nt_0100_2000_candidate(out.p, in.p, n); }, n, calls);
      const double frozen = time_ns_per_element([&] { exp53_n2_vmstyle_u4_0381_frozen(out.p, in.p, n); }, n, calls);
      const double intel = time_ns_per_element([&] { vmdExp((MKL_INT)n, in.p, out.p, VML_HA); }, n, calls);

      std::cout << "FINAL domain=" << name << " n=" << n
                << " best_share=" << best_share
                << " sync_nt_ns=" << best
                << " serial_nt_ns=" << serial_nt
                << " frozen_ns=" << frozen
                << " intel_ns=" << intel
                << " intel_over_sync_nt=" << intel / best
                << " frozen_over_sync_nt=" << frozen / best
                << " bitdiff=0\n";
    }
  }
  return 0;
}
