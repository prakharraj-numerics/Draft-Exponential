#include <immintrin.h>
#include <mkl_vml.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

/* Experimental API-level VML integration.
   VML has no custom-kernel callback, so this keeps our reduction/table logic
   and runs the polynomial/reconstruction as whole-array VML arithmetic passes.
   This is intentionally NOT production and is expected to differ bitwise
   because VML exposes Mul/Add/Sqr but no fused FMA primitive. */

alignas(64) static double R[2000], H[2000], S[2000], ER[2000], EL[2000];
alignas(64) static double SCALE[2000], T0[2000], T1[2000];
alignas(64) static double ONE[2000], Q1A[2000], Q2A[2000], Q3A[2000], Q4A[2000];
static bool INIT=false;

static void init_consts(){
    if(INIT) return;
    for(int i=0;i<2000;i++){
        ONE[i]=1.0; Q1A[i]=N2F_Q1; Q2A[i]=N2F_Q2; Q3A[i]=N2F_Q3; Q4A[i]=N2F_Q4;
    }
    INIT=true;
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
static void reduce_scale(double* r,double* scale,const double* in,size_t n){
    const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+8<=n;i+=8){
        __m512d x=_mm512_loadu_pd(in+i);
        __m512d biased=_mm512_fmadd_pd(x,inv,magic);
        __m512d k=_mm512_sub_pd(biased,magic);
        __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
        __m512i j=_mm512_and_epi64(kn,mask);
        __m512i q=_mm512_srai_epi64(kn,7);
        __m512i tb=_mm512_i64gather_epi64(j,(const long long*)N2_FROZEN_TAB128,8);
        __m512d rv=_mm512_fnmadd_pd(k,hi,x);
        rv=_mm512_fnmadd_pd(k,mi,rv);
        rv=_mm512_fnmadd_pd(k,lo,rv);
        __m512i sb=_mm512_add_epi64(tb,_mm512_slli_epi64(q,52));
        _mm512_storeu_pd(r+i,rv);
        _mm512_storeu_pd(scale+i,_mm512_castsi512_pd(sb));
    }
    for(;i<n;i++){
        double biased=std::fma(in[i],N2F_INV128,N2F_MAGIC);
        double k=biased-N2F_MAGIC;
        uint64_t b; std::memcpy(&b,&biased,8);
        int64_t kn=(int64_t)(b-N2F_MAGIC_BITS);
        int64_t j=kn&127, q=kn>>7;
        double rv=std::fma(-k,N2F_L128_HI,in[i]);
        rv=std::fma(-k,N2F_L128_MI,rv);
        rv=std::fma(-k,N2F_L128_LO,rv);
        uint64_t tb; std::memcpy(&tb,&N2_FROZEN_TAB128[j],8);
        uint64_t sb=tb+((uint64_t)q<<52);
        r[i]=rv; std::memcpy(scale+i,&sb,8);
    }
}

extern "C" void exp53_vml_pipeline_0100_2000_candidate(double* out,const double* in,size_t n){
    init_consts();
    reduce_scale(R,SCALE,in,n);
    MKL_INT N=(MKL_INT)n;
    const MKL_INT64 M=VML_HA;

    /* H = (((Q4*r + Q3)*r + Q2)*r + Q1)*r + 1 */
    vmdMul(N,Q4A,R,T0,M); vmdAdd(N,T0,Q3A,H,M);
    vmdMul(N,H,R,T0,M);   vmdAdd(N,T0,Q2A,H,M);
    vmdMul(N,H,R,T0,M);   vmdAdd(N,T0,Q1A,H,M);
    vmdMul(N,H,R,T0,M);   vmdAdd(N,T0,ONE,H,M);

    vmdMul(N,H,H,S,M);
    vmdMul(N,R,S,T0,M);   vmdAdd(N,T0,ONE,ER,M);
    vmdSub(N,ONE,ER,T1,M);
    vmdMul(N,R,S,T0,M);   vmdAdd(N,T0,T1,EL,M);
    vmdMul(N,ER,SCALE,T0,M);
    vmdMul(N,EL,SCALE,T1,M);
    vmdAdd(N,T1,T0,out,M);
}
