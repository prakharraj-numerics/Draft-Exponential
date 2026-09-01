// Experimental fully integrated Turbo SIMD backend for EXP53, n=101..1400.
// Production untouched. Uses frozen EXP53 math/constants; no turbo exp().
// Difference from old Turbo attempt: no scalar/libm tail; final partial vector
// is evaluated by the same Turbo SIMD formula using padded inactive lanes.
#include "turbo/simd/simd.h"
#include "exp53_highway_sync_1600_3000_constants_frozen.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ts = turbo::simd;
namespace c = exp53_hwy_const_frozen;

using D = ts::batch<double>;
using I = ts::batch<int64_t>;
static_assert(D::size == 8, "Turbo must select 8-lane AVX-512 FP64");
static_assert(I::size == 8, "Turbo must select 8-lane AVX-512 int64");

static inline D vec8(const D& x) {
    const D inv(c::INV128), hi(c::L128_HI), mi(c::L128_MI), lo(c::L128_LO),
            magic(c::MAGIC), one(1.0), q1(c::Q1), q2(c::Q2), q3(c::Q3), q4(c::Q4);
    const I mb((int64_t)c::MAGIC_BITS), mask((int64_t)127);
    const int64_t* tab = reinterpret_cast<const int64_t*>(c::TAB128);
    D biased = ts::fma(x, inv, magic);
    D k = biased - magic;
    I kn = ts::bitwise_cast<int64_t>(biased) - mb;
    I j = kn & mask;
    I q = kn >> 7;
    I tb = I::gather(tab, j);
    D r = ts::fnma(k, hi, x);
    r = ts::fnma(k, mi, r);
    r = ts::fnma(k, lo, r);
    D h = ts::fma(q4, r, q3);
    h = ts::fma(h, r, q2);
    h = ts::fma(h, r, q1);
    h = ts::fma(h, r, one);
    D s = h * h;
    D er = ts::fma(r, s, one);
    D el = ts::fma(r, s, one - er);
    I sb = tb + (q << 52);
    D scale = ts::bitwise_cast<double>(sb);
    D ph = er * scale;
    return ts::fma(el, scale, ph);
}

template<int U>
static inline void run_u(double* out, const double* in, size_t n) {
    size_t i=0;
    constexpr size_t W = 8u * U;
    for(; i+W<=n; i+=W) {
        D x[U];
        for(int l=0;l<U;l++) x[l]=D::load_unaligned(in+i+8*l);
        for(int l=0;l<U;l++) vec8(x[l]).store_unaligned(out+i+8*l);
    }
    for(; i+8<=n; i+=8) vec8(D::load_unaligned(in+i)).store_unaligned(out+i);
    if(i<n) {
        alignas(64) double ti[8]={0,0,0,0,0,0,0,0};
        alignas(64) double to[8];
        const size_t rem=n-i;
        std::memcpy(ti,in+i,rem*sizeof(double));
        vec8(D::load_unaligned(ti)).store_unaligned(to);
        std::memcpy(out+i,to,rem*sizeof(double));
    }
}

extern "C" {
void exp53_turbo_full_u2_0101_1400(double*o,const double*i,size_t n){run_u<2>(o,i,n);} 
void exp53_turbo_full_u4_0101_1400(double*o,const double*i,size_t n){run_u<4>(o,i,n);} 
void exp53_turbo_full_u6_0101_1400(double*o,const double*i,size_t n){run_u<6>(o,i,n);} 
void exp53_turbo_full_u8_0101_1400(double*o,const double*i,size_t n){run_u<8>(o,i,n);} 
}
