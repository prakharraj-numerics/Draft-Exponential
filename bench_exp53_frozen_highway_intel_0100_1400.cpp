#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>

void exp53_highway_0101_2000_candidate(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

struct Aligned {
  double* p = nullptr;
  explicit Aligned(size_t n) {
    if (posix_memalign((void**)&p, 64, n * sizeof(double)) || !p) std::exit(2);
  }
  ~Aligned() { std::free(p); }
};

static inline uint64_t sm(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
static double u01(uint64_t h) {
  return ((double)(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static void fill(double* x, size_t n, int d) {
  uint64_t s = 0x6a09e667f3bcc909ULL ^ ((uint64_t)n << 17) ^ ((uint64_t)d << 57);
  for (size_t i = 0; i < n; ++i) {
    double u = u01(sm(s + i * 0x9e3779b97f4a7c15ULL));
    double e = 0x1p-20;
    double m = d == 0 ? (e + u * (1.0 - 2 * e)) : (1.0 + e + u * (99.0 - 2 * e));
    x[i] = (i & 1) ? -m : m;
  }
}
static size_t calls(size_t n) {
  size_t c = 6000000ULL / n;
  if (c < 100) c = 100;
  if (c > 100000) c = 100000;
  return c;
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static uint64_t bits(double x) {
  uint64_t u;
  std::memcpy(&u, &x, 8);
  return u;
}
static size_t bitdiff(const double* a, const double* b, size_t n) {
  size_t d = 0;
  for (size_t i = 0; i < n; ++i) d += bits(a[i]) != bits(b[i]);
  return d;
}

template <class F>
static double time_one(F&& f, size_t n, size_t c) {
  for (int w = 0; w < 8; ++w) f();
  std::vector<double> samples;
  samples.reserve(9);
  for (int s = 0; s < 9; ++s) {
    auto a = std::chrono::steady_clock::now();
    for (size_t k = 0; k < c; ++k) f();
    auto b = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::nano>(b - a).count() / ((double)c * n));
  }
  return median(samples);
}

int main() {
  constexpr size_t MAXN = 1400;
  Aligned in(MAXN), out(MAXN), ref(MAXN);
  std::vector<size_t> ns;
  ns.push_back(100);
  for (size_t n = 150; n <= 1400; n += 50) ns.push_back(n);
  volatile double sink = 0;
  size_t total_bitdiff = 0;
  int wins_h = 0, wins_f = 0, wins_i = 0;
  std::vector<double> f_over_h, i_over_h, i_over_f;

  std::cout << std::fixed << std::setprecision(9)
            << "EXP53_FROZEN_HIGHWAY_INTEL sizes=100,150..1400 highway=static_avx512 math=frozen intel=VML_HA\n";

  for (int d = 0; d < 2; ++d) {
    const char* dn = d ? "mid" : "unit";
    for (size_t n : ns) {
      fill(in.p, n, d);
      exp53_n2_vmstyle_u4_0381_frozen(ref.p, in.p, n);
      exp53_highway_0101_2000_candidate(out.p, in.p, n);
      size_t bd = bitdiff(out.p, ref.p, n);
      total_bitdiff += bd;
      std::cout << "CHECK domain=" << dn << " n=" << n << " bitdiff=" << bd << "\n";

      size_t c = calls(n);
      auto hf = [&] { exp53_highway_0101_2000_candidate(out.p, in.p, n); };
      auto ff = [&] { exp53_n2_vmstyle_u4_0381_frozen(out.p, in.p, n); };
      auto ii = [&] { vmdExp((MKL_INT)n, in.p, out.p, VML_HA); };

      // Rotate first-measured stack by cell to reduce systematic order/frequency bias.
      double ht = 0, ft = 0, it = 0;
      int rot = (int)((n / 50 + d) % 3);
      if (rot == 0) {
        ht = time_one(hf, n, c); ft = time_one(ff, n, c); it = time_one(ii, n, c);
      } else if (rot == 1) {
        ft = time_one(ff, n, c); it = time_one(ii, n, c); ht = time_one(hf, n, c);
      } else {
        it = time_one(ii, n, c); ht = time_one(hf, n, c); ft = time_one(ff, n, c);
      }

      double best = std::min(ht, std::min(ft, it));
      const double eps = best * 0.002; // 0.2% tie band for cell winner count.
      if (ht <= best + eps) ++wins_h;
      else if (ft <= best + eps) ++wins_f;
      else ++wins_i;
      f_over_h.push_back(ft / ht);
      i_over_h.push_back(it / ht);
      i_over_f.push_back(it / ft);
      sink += out.p[(n * 13u + d) % n] * 0x1p-1022;

      std::cout << "RESULT domain=" << dn << " n=" << n << " calls=" << c
                << " highway_ns=" << ht << " frozen_ns=" << ft << " intel_ns=" << it
                << " frozen_over_highway=" << ft / ht
                << " intel_over_highway=" << it / ht
                << " intel_over_frozen=" << it / ft
                << " bitdiff=" << bd << "\n";
    }
  }

  std::cout << "CORRECT bitdiff_max=" << total_bitdiff << "\n";
  std::cout << "WINS highway=" << wins_h << " frozen=" << wins_f << " intel=" << wins_i << "\n";
  std::cout << "SUMMARY median_frozen_over_highway=" << median(f_over_h)
            << " median_intel_over_highway=" << median(i_over_h)
            << " median_intel_over_frozen=" << median(i_over_f) << "\n";
  if (sink == 123.0) std::cerr << sink;
  return total_bitdiff == 0 ? 0 : 9;
}
