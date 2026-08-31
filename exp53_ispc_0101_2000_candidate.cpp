// Experimental ISPC wrapper for frozen EXP53 machinery.
#define restrict __restrict__
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
#undef restrict
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "exp53_ispc_generated.h"

__attribute__((noinline))
void exp53_ispc_0101_2000_candidate(double* out, const double* in, size_t n) {
    const size_t bulk = n & ~(size_t)31;
    if (bulk) {
        ispc::exp53_ispc_kernel(out, const_cast<double*>(in), (int)bulk,
                                reinterpret_cast<uint64_t*>(const_cast<double*>(N2_FROZEN_TAB128)));
    }
    if (bulk < n) exp53_n2_fused_u4_038_frozen(out + bulk, in + bulk, n - bulk);
}

void exp53_ispc_debug8_wrapper(const double* in, uint64_t* dbg) {
    ispc::exp53_ispc_debug8(const_cast<double*>(in),
                            reinterpret_cast<uint64_t*>(const_cast<double*>(N2_FROZEN_TAB128)),
                            dbg);
}

void exp53_frozen_debug8_wrapper(const double* in, uint64_t* dbg) {
    const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI),
      mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO), magic=_mm512_set1_pd(N2F_MAGIC),
      one=_mm512_set1_pd(1.0), q1=_mm512_set1_pd(N2F_Q1), q2=_mm512_set1_pd(N2F_Q2),
      q3=_mm512_set1_pd(N2F_Q3), q4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), mask=_mm512_set1_epi64(127);
    __m512d x=_mm512_loadu_pd(in), biased=_mm512_fmadd_pd(x,inv,magic), k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb), j=_mm512_and_epi64(kn,mask), q=_mm512_srai_epi64(kn,7);
    __m512i tb=_mm512_i64gather_epi64(j,(const long long*)N2_FROZEN_TAB128,8);
    __m512d r1=_mm512_fnmadd_pd(k,hi,x), r2=_mm512_fnmadd_pd(k,mi,r1), r=_mm512_fnmadd_pd(k,lo,r2);
    __m512i sb=_mm512_add_epi64(tb,_mm512_slli_epi64(q,52)); __m512d scale=_mm512_castsi512_pd(sb);
    __m512d h1=_mm512_fmadd_pd(q4,r,q3), h2=_mm512_fmadd_pd(h1,r,q2), h3=_mm512_fmadd_pd(h2,r,q1), h=_mm512_fmadd_pd(h3,r,one);
    __m512d s=_mm512_mul_pd(h,h), er=_mm512_fmadd_pd(r,s,one), el=_mm512_fmadd_pd(r,s,_mm512_sub_pd(one,er));
    __m512d ph=_mm512_mul_pd(er,scale), y=_mm512_fmadd_pd(el,scale,ph);
    __m512i vals[20]={_mm512_castpd_si512(biased),_mm512_castpd_si512(k),kn,j,q,tb,
      _mm512_castpd_si512(r1),_mm512_castpd_si512(r2),_mm512_castpd_si512(r),sb,_mm512_castpd_si512(scale),
      _mm512_castpd_si512(h1),_mm512_castpd_si512(h2),_mm512_castpd_si512(h3),_mm512_castpd_si512(h),
      _mm512_castpd_si512(s),_mm512_castpd_si512(er),_mm512_castpd_si512(el),_mm512_castpd_si512(ph),_mm512_castpd_si512(y)};
    for(int st=0;st<20;st++) _mm512_storeu_si512((void*)(dbg+st*8),vals[st]);
}
