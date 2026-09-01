#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>

extern "C" void* exp53_xnnpack_create(size_t threads);
extern "C" void exp53_xnnpack_destroy(void* pool);
extern "C" void exp53_xnnpack_run_mode(void* pool, double* out, const double* in, size_t n, size_t tile, int mode);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double* out, const double* in, size_t n);

struct Buffer {
  double* p;
  explicit Buffer(size_t n) {
    if (posix_memalign(reinterpret_cast<void**>(&p), 64, n * sizeof(double)) != 0) std::abort();
  }
  ~Buffer() { free(p); }
};

static uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static double unit01(uint64_t h) {
  return (static_cast<double>(h >> 11) + 0.5) / 9007199254740992.0;
}

static void fill_inputs(double* x, size_t n, bool mid) {
  const uint64_t seed = 0x6a09e667f3bcc909ULL ^ (static_cast<uint64_t>(n) << 20) ^ (static_cast<uint64_t>(mid) << 60);
  for (size_t i = 0; i < n; ++i) {
    const double q = unit01(mix64(seed + i * 0x9e3779b97f4a7c15ULL));
    const double a = mid ? 1.000001 + q * 98.999998 : 0x1p-20 + q * (1.0 - 0x1p-19);
    x[i] = (i & 1) ? -a : a;
  }
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static size_t calls_for(size_t n) {
  const size_t c = 260000 / n;
  return std::max<size_t>(96, std::min<size_t>(2200, c));
}

template<class F>
static double time_ns_per_input(F f, size_t n, size_t calls) {
  for (int i = 0; i < 16; ++i) f();
  std::vector<double> samples;
  for (int r = 0; r < 15; ++r) {
    const auto a = std::chrono::steady_clock::now();
    for (size_t c = 0; c < calls; ++c) f();
    const auto b = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::nano>(b - a).count() / calls / n);
  }
  return median(samples);
}

int main() {
  const size_t sizes[] = {101,125,150,160,192,200,224,250,256,300,320,384,400,448,500,512,600,640,750,768,896,900,960,1000,1024,1100,1152,1200,1250,1280,1344,1399,1400};
  const size_t tiles[] = {32,64,96,128,160,192,256,320,384,512};

  Buffer in(1408), frozen(1408), out(1408), intel(1408);
  void* pool1 = exp53_xnnpack_create(1);
  void* pool2 = exp53_xnnpack_create(2);
  if (pool1 == nullptr || pool2 == nullptr) return 3;

  std::cout << std::fixed << std::setprecision(9)
            << "XNNPACK_INTERNAL_FP64 fastpath_sweep frozen_math\n";

  for (int domain = 0; domain < 2; ++domain) {
    const bool mid = domain != 0;
    const char* domain_name = mid ? "mid" : "unit";
    for (size_t n : sizes) {
      fill_inputs(in.p, n, mid);
      exp53_n2_vmstyle_u4_0381_frozen(frozen.p, in.p, n);
      const size_t calls = calls_for(n);
      double best = 1.0e99;
      size_t best_tile = 0;
      int best_threads = 0;
      int best_mode = -1;
      size_t max_bitdiff = 0;

      for (int threads = 1; threads <= 2; ++threads) {
        void* pool = threads == 1 ? pool1 : pool2;
        for (int mode = 0; mode < 4; ++mode) {
          const size_t tile_count = mode >= 2 ? 1 : sizeof(tiles) / sizeof(tiles[0]);
          for (size_t ti = 0; ti < tile_count; ++ti) {
            const size_t tile = mode >= 2 ? 256 : tiles[ti];
            exp53_xnnpack_run_mode(pool, out.p, in.p, n, tile, mode);
            size_t bitdiff = 0;
            for (size_t i = 0; i < n; ++i) {
              if (std::memcmp(out.p + i, frozen.p + i, sizeof(double)) != 0) ++bitdiff;
            }
            max_bitdiff = std::max(max_bitdiff, bitdiff);
            if (bitdiff != 0) continue;
            const double t = time_ns_per_input([&]{ exp53_xnnpack_run_mode(pool, out.p, in.p, n, tile, mode); }, n, calls);
            if (t < best) {
              best = t;
              best_tile = tile;
              best_threads = threads;
              best_mode = mode;
            }
          }
        }
      }

      const double frozen_ns = time_ns_per_input([&]{ exp53_n2_vmstyle_u4_0381_frozen(frozen.p, in.p, n); }, n, calls);
      const double intel_ns = time_ns_per_input([&]{ vmdExp(static_cast<MKL_INT>(n), in.p, intel.p, VML_HA); }, n, calls);
      std::cout << "FINAL domain=" << domain_name
                << " n=" << n
                << " xnn_ns=" << best
                << " threads=" << best_threads
                << " mode=" << best_mode
                << " tile=" << best_tile
                << " frozen_ns=" << frozen_ns
                << " intel_ns=" << intel_ns
                << " intel_over_xnn=" << intel_ns / best
                << " frozen_over_xnn=" << frozen_ns / best
                << " bitdiff=" << max_bitdiff
                << "\n";
    }
  }

  exp53_xnnpack_destroy(pool1);
  exp53_xnnpack_destroy(pool2);
  return 0;
}
