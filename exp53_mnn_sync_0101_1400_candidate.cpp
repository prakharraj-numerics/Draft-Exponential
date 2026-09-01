// Experimental Alibaba MNN helper-only scheduling integration for EXP53, n=101..1400.
// Production untouched. MNN supplies only the persistent helper dispatch;
// every transcendental arithmetic operation is the frozen EXP53 formula.
#include "backend/cpu/ThreadPool.hpp"
#include "exp53_highway_sync_1600_3000_constants_frozen.hpp"
#include <immintrin.h>
#include <pthread.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace c = exp53_hwy_const_frozen;

static inline __m512d exp53_vec8_mnn(__m512d x) {
    const __m512d inv   = _mm512_set1_pd(c::INV128);
    const __m512d hi    = _mm512_set1_pd(c::L128_HI);
    const __m512d mi    = _mm512_set1_pd(c::L128_MI);
    const __m512d lo    = _mm512_set1_pd(c::L128_LO);
    const __m512d magic = _mm512_set1_pd(c::MAGIC);
    const __m512d one   = _mm512_set1_pd(1.0);
    const __m512d q1    = _mm512_set1_pd(c::Q1);
    const __m512d q2    = _mm512_set1_pd(c::Q2);
    const __m512d q3    = _mm512_set1_pd(c::Q3);
    const __m512d q4    = _mm512_set1_pd(c::Q4);
    __m512d biased = _mm512_fmadd_pd(x, inv, magic);
    __m512d k      = _mm512_sub_pd(biased, magic);
    __m512i kn     = _mm512_sub_epi64(_mm512_castpd_si512(biased), _mm512_set1_epi64((long long)c::MAGIC_BITS));
    __m512i j      = _mm512_and_si512(kn, _mm512_set1_epi64(127));
    __m512i q      = _mm512_srai_epi64(kn, 7);
    __m512i tb     = _mm512_i64gather_epi64(j, reinterpret_cast<const void*>(c::TAB128), 8);
    __m512d r = _mm512_fnmadd_pd(k, hi, x);
    r = _mm512_fnmadd_pd(k, mi, r);
    r = _mm512_fnmadd_pd(k, lo, r);
    __m512d h = _mm512_fmadd_pd(q4, r, q3);
    h = _mm512_fmadd_pd(h, r, q2);
    h = _mm512_fmadd_pd(h, r, q1);
    h = _mm512_fmadd_pd(h, r, one);
    __m512d s  = _mm512_mul_pd(h, h);
    __m512d er = _mm512_fmadd_pd(r, s, one);
    __m512d el = _mm512_fmadd_pd(r, s, _mm512_sub_pd(one, er));
    __m512i sb = _mm512_add_epi64(tb, _mm512_slli_epi64(q, 52));
    __m512d scale = _mm512_castsi512_pd(sb);
    __m512d ph = _mm512_mul_pd(er, scale);
    return _mm512_fmadd_pd(el, scale, ph);
}

static inline void exp53_range_mnn(double* out, const double* in, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512d x0 = _mm512_loadu_pd(in + i + 0);
        __m512d x1 = _mm512_loadu_pd(in + i + 8);
        __m512d x2 = _mm512_loadu_pd(in + i + 16);
        __m512d x3 = _mm512_loadu_pd(in + i + 24);
        _mm512_storeu_pd(out + i + 0,  exp53_vec8_mnn(x0));
        _mm512_storeu_pd(out + i + 8,  exp53_vec8_mnn(x1));
        _mm512_storeu_pd(out + i + 16, exp53_vec8_mnn(x2));
        _mm512_storeu_pd(out + i + 24, exp53_vec8_mnn(x3));
    }
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(out + i, exp53_vec8_mnn(_mm512_loadu_pd(in + i)));
    if (i < n) {
        alignas(64) double ti[8] = {0,0,0,0,0,0,0,0};
        alignas(64) double to[8];
        const size_t rem = n - i;
        std::memcpy(ti, in + i, rem * sizeof(double));
        _mm512_store_pd(to, exp53_vec8_mnn(_mm512_load_pd(ti)));
        std::memcpy(out + i, to, rem * sizeof(double));
    }
}

static inline void pin_cpu2_once() {
    thread_local bool pinned = false;
    if (pinned) return;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(2, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    pinned = true;
}

class Exp53MNN2CoreCandidate {
public:
    Exp53MNN2CoreCandidate() {
        MNN::ThreadPool::init(2, 0x5UL, pool_);
        work_index_ = pool_->acquireWorkIndex();
        task_.second = 2;
        task_.first = [this](int tid) {
            if (tid != 1) return;
            pin_cpu2_once();
            if (n_ > split_) exp53_range_mnn(out_ + split_, in_ + split_, n_ - split_);
        };
        pool_->active();
    }
    ~Exp53MNN2CoreCandidate() {
        pool_->deactive();
        pool_->releaseWorkIndex(work_index_);
    }
    void run(double* out, const double* in, size_t n, int helper_pct) {
        out_ = out; in_ = in; n_ = n;
        size_t helper = (n * (size_t)helper_pct) / 100;
        helper = (helper / 32) * 32;
        size_t split = n - helper;
        split_ = (split / 32) * 32;
        pool_->enqueueHelperOnly(&task_, work_index_);
        if (split_ > 0) exp53_range_mnn(out_, in_, split_);
        pool_->waitHelperOnly(work_index_);
    }
private:
    MNN::ThreadPool* pool_ = nullptr;
    int work_index_ = -1;
    MNN::ThreadPool::TASK task_;
    double* out_ = nullptr;
    const double* in_ = nullptr;
    size_t n_ = 0;
    size_t split_ = 0;
};

extern "C" void exp53_mnn_sync_0101_1400(double* out, const double* in, size_t n, int helper_pct) {
    static Exp53MNN2CoreCandidate exec;
    exec.run(out, in, n, helper_pct);
}
