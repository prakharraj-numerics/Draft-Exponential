#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

/* EXP53 EXPERIMENT ONLY. Production/frozen candidates untouched.
   Same (m,d)=(32,8) formula. Compared with the first accurate residual-ULP
   implementation, this uses a linearized final-error decision estimator:
     E <- 2 E + rho_k
   across the four residual squarings and final square.  Thus the recovered
   local roundoff residuals are weighted approximately 16,8,4,2,1 at the
   output, without carrying the expensive nonlinear error state.  The final
   decision threshold is 0.5 output ULP.  Only the final encoding can move,
   by at most one representable double.
*/

namespace exp53_m32d8_residual_ulp_fast {

static inline __m512d q32d8(__m512d x) {
    const __m512d a8 = _mm512_set1_pd(0x1.a01a01a01a01ap-56);
    const __m512d a7 = _mm512_set1_pd(0x1.a01a01a01a01ap-48);
    const __m512d a6 = _mm512_set1_pd(0x1.6c16c16c16c17p-40);
    const __m512d a5 = _mm512_set1_pd(0x1.1111111111111p-32);
    const __m512d a4 = _mm512_set1_pd(0x1.5555555555555p-25);
    const __m512d a3 = _mm512_set1_pd(0x1.5555555555555p-18);
    const __m512d a2 = _mm512_set1_pd(0x1.0000000000000p-11);
    const __m512d inv32 = _mm512_set1_pd(0x1.0000000000000p-5);
    __m512d t=a8;
    t=_mm512_fmadd_pd(t,x,a7); t=_mm512_fmadd_pd(t,x,a6);
    t=_mm512_fmadd_pd(t,x,a5); t=_mm512_fmadd_pd(t,x,a4);
    t=_mm512_fmadd_pd(t,x,a3); t=_mm512_fmadd_pd(t,x,a2);
    const __m512d x2=_mm512_mul_pd(x,x);
    return _mm512_fmadd_pd(x2,t,_mm512_mul_pd(x,inv32));
}

static inline __m512d eval(__m512d x) {
    const __m512d one=_mm512_set1_pd(1.0), two=_mm512_set1_pd(2.0);
    const __m512i one_i=_mm512_set1_epi64(1);
    __m512d r=q32d8(x), e=_mm512_setzero_pd();

    // Linearized error decision state.  E recurrence automatically produces
    // the 16,8,4,2,1 output weights after the final update.
    for(int k=0;k<4;k++) {
        const __m512d tr=_mm512_add_pd(r,r);
        const __m512d rn=_mm512_fmadd_pd(r,r,tr);
        const __m512d rho=_mm512_fmadd_pd(r,r,_mm512_sub_pd(tr,rn));
        e=_mm512_fmadd_pd(two,e,rho);
        r=rn;
    }

    // Final square, recovering both rounding pieces.
    const __m512d t=_mm512_fmadd_pd(two,r,one);
    const __m512d d1=_mm512_fmadd_pd(two,r,_mm512_sub_pd(one,t));
    __m512d y=_mm512_fmadd_pd(r,r,t);
    const __m512d d2=_mm512_fmadd_pd(r,r,_mm512_sub_pd(t,y));
    e=_mm512_fmadd_pd(two,e,_mm512_add_pd(d1,d2));

    // threshold = 0.5 ULP => compare 2*E with one neighboring spacing.
    const __m512d e2=_mm512_add_pd(e,e);
    const __m512i yi=_mm512_castpd_si512(y);
    const __m512d yup=_mm512_castsi512_pd(_mm512_add_epi64(yi,one_i));
    const __m512d ydn=_mm512_castsi512_pd(_mm512_sub_epi64(yi,one_i));
    const __m512d up=_mm512_sub_pd(yup,y);
    const __m512d dn=_mm512_sub_pd(y,ydn);
    const __mmask8 mu=_mm512_cmp_pd_mask(e2,up,_CMP_GT_OQ);
    const __mmask8 md=_mm512_cmp_pd_mask(e2,_mm512_sub_pd(_mm512_setzero_pd(),dn),_CMP_LT_OQ);
    __m512i yo=yi;
    yo=_mm512_mask_add_epi64(yo,mu,yo,one_i);
    yo=_mm512_mask_sub_epi64(yo,md,yo,one_i);
    return _mm512_castsi512_pd(yo);
}

__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void kernel(double* __restrict out,const double* __restrict in,size_t n) {
    size_t i=0;
    // Eight independent ZMM vectors per trip to expose substantially more ILP
    // than the first accurate candidate's four-vector loop.
    for(;i+64<=n;i+=64) {
        __m512d x0=_mm512_loadu_pd(in+i+0),  x1=_mm512_loadu_pd(in+i+8);
        __m512d x2=_mm512_loadu_pd(in+i+16), x3=_mm512_loadu_pd(in+i+24);
        __m512d x4=_mm512_loadu_pd(in+i+32), x5=_mm512_loadu_pd(in+i+40);
        __m512d x6=_mm512_loadu_pd(in+i+48), x7=_mm512_loadu_pd(in+i+56);
        __m512d y0=eval(x0), y1=eval(x1), y2=eval(x2), y3=eval(x3);
        __m512d y4=eval(x4), y5=eval(x5), y6=eval(x6), y7=eval(x7);
        _mm512_storeu_pd(out+i+0,y0);   _mm512_storeu_pd(out+i+8,y1);
        _mm512_storeu_pd(out+i+16,y2);  _mm512_storeu_pd(out+i+24,y3);
        _mm512_storeu_pd(out+i+32,y4);  _mm512_storeu_pd(out+i+40,y5);
        _mm512_storeu_pd(out+i+48,y6);  _mm512_storeu_pd(out+i+56,y7);
    }
    for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i);_mm512_storeu_pd(out+i,eval(x));}
    if(i<n){unsigned rem=(unsigned)(n-i);__mmask8 m=(__mmask8)((1u<<rem)-1u);__m512d x=_mm512_maskz_loadu_pd(m,in+i);_mm512_mask_storeu_pd(out+i,m,eval(x));}
}

} // namespace exp53_m32d8_residual_ulp_fast
