// Experimental XNNPACK execution-layer integration for frozen EXP53, n=101..1400.
// XNNPACK supplies initialization + its pthreadpool dependency. The math kernel is the frozen EXP53 implementation.
#include <cstddef>
#include <cstdint>
#include <xnnpack.h>
#include <pthreadpool.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

struct Exp53XnnCtx { double* out; const double* in; };

static void exp53_xnn_task(void* p, size_t start, size_t size) {
  auto* c = static_cast<Exp53XnnCtx*>(p);
  exp53_n2_vmstyle_u4_0381_frozen(c->out + start, c->in + start, size);
}

extern "C" void* exp53_xnnpack_create(size_t threads) {
  if (xnn_initialize(nullptr) != xnn_status_success) return nullptr;
  return reinterpret_cast<void*>(pthreadpool_create(threads));
}

extern "C" void exp53_xnnpack_destroy(void* p) {
  if (p) pthreadpool_destroy(reinterpret_cast<pthreadpool_t>(p));
}

extern "C" void exp53_xnnpack_run(void* p, double* out, const double* in, size_t n, size_t tile) {
  auto pool = reinterpret_cast<pthreadpool_t>(p);
  if (!pool || n == 0) return;
  if (tile < 32) tile = 32;
  tile = (tile / 32) * 32;
  const size_t body = (n / 32) * 32;
  Exp53XnnCtx ctx{out, in};
  if (body) {
    pthreadpool_parallelize_1d_tile_1d(pool, exp53_xnn_task, &ctx, body, tile, 0);
  }
  if (body < n) {
    exp53_n2_vmstyle_u4_0381_frozen(out + body, in + body, n - body);
  }
}
