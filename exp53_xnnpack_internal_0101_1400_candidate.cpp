// Experimental XNNPACK execution-layer integration for frozen EXP53, n=101..1400.
// XNNPACK supplies initialization + its pthreadpool dependency. The math kernel is the frozen EXP53 implementation.
#include <cstddef>
#include <cstdint>
#include <xnnpack.h>
#include <pthreadpool.h>

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

struct Exp53XnnCtx {
  double* out;
  const double* in;
};

struct Exp53XnnTwoCtx {
  double* out;
  const double* in;
  size_t split;
  size_t body;
};

static void exp53_xnn_task(void* p, size_t start, size_t size) {
  auto* c = static_cast<Exp53XnnCtx*>(p);
  exp53_n2_vmstyle_u4_0381_frozen(c->out + start, c->in + start, size);
}

static void exp53_xnn_two_task(void* p, size_t index) {
  auto* c = static_cast<Exp53XnnTwoCtx*>(p);
  const size_t start = index == 0 ? 0 : c->split;
  const size_t end = index == 0 ? c->split : c->body;
  if (end > start) {
    exp53_n2_vmstyle_u4_0381_frozen(c->out + start, c->in + start, end - start);
  }
}

extern "C" void* exp53_xnnpack_create(size_t threads) {
  if (xnn_initialize(nullptr) != xnn_status_success) return nullptr;
  return reinterpret_cast<void*>(pthreadpool_create(threads));
}

extern "C" void exp53_xnnpack_destroy(void* p) {
  if (p) pthreadpool_destroy(reinterpret_cast<pthreadpool_t>(p));
}

// mode 0: existing tiled dispatch
// mode 1: tiled dispatch with denormals disabled by pthreadpool
// mode 2: exactly two aligned tasks
// mode 3: exactly two aligned tasks with denormals disabled
extern "C" void exp53_xnnpack_run_mode(
    void* p, double* out, const double* in, size_t n, size_t tile, int mode) {
  auto pool = reinterpret_cast<pthreadpool_t>(p);
  if (!pool || n == 0) return;

  if (tile < 32) tile = 32;
  tile = (tile / 32) * 32;
  const size_t body = (n / 32) * 32;
  const uint32_t flags = (mode & 1) ? PTHREADPOOL_FLAG_DISABLE_DENORMALS : 0;

  if (body) {
    if (mode >= 2 && pthreadpool_get_threads_count(pool) >= 2 && body >= 64) {
      size_t split = ((body / 2) / 32) * 32;
      if (split == 0) split = 32;
      if (split >= body) split = body - 32;
      Exp53XnnTwoCtx ctx{out, in, split, body};
      pthreadpool_parallelize_1d(pool, exp53_xnn_two_task, &ctx, 2, flags);
    } else {
      Exp53XnnCtx ctx{out, in};
      pthreadpool_parallelize_1d_tile_1d(pool, exp53_xnn_task, &ctx, body, tile, flags);
    }
  }

  if (body < n) {
    exp53_n2_vmstyle_u4_0381_frozen(out + body, in + body, n - body);
  }
}

extern "C" void exp53_xnnpack_run(
    void* p, double* out, const double* in, size_t n, size_t tile) {
  exp53_xnnpack_run_mode(p, out, in, n, tile, 0);
}
