/* Experimental only: isolate vector-tail cost and wider ILP for EXP53 100..2000.
   Frozen arithmetic/constants/table; production files untouched. */
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#define EXP53_DECL_CONSTS \
    const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI), \
      mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO), \
      magic=_mm512_set1_pd(N2F_MAGIC), one=_mm512_set1_pd(1.0), \
      nq1=_mm512_set1_pd(N2F_Q1), nq2=_mm512_set1_pd(N2F_Q2), \
      nq3=_mm512_set1_pd(N2F_Q3), nq4=_mm512_set1_pd(N2F_Q4); \
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), mask=_mm512_set1_epi64(127)

#define EXP53_STAGE(NL, BASE) do { \
    __m512d x[NL],biased[NL],k[NL],r[NL],h[NL],s[NL],er[NL],el[NL],scale[NL],ph[NL],y[NL]; \
    __m512i kn[NL],j[NL],q[NL],tb[NL],sb[NL]; \
    for(int L=0;L<NL;L++) x[L]=_mm512_loadu_pd(in+(BASE)+8*L); \
    for(int L=0;L<NL;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic); \
    for(int L=0;L<NL;L++){ k[L]=_mm512_sub_pd(biased[L],magic); kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); j[L]=_mm512_and_epi64(kn[L],mask); q[L]=_mm512_srai_epi64(kn[L],7); } \
    for(int L=0;L<NL;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
    for(int L=0;L<NL;L++){ r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); } \
    for(int L=0;L<NL;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
    for(int L=0;L<NL;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
    for(int L=0;L<NL;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
    for(int L=0;L<NL;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
    for(int L=0;L<NL;L++) s[L]=_mm512_mul_pd(h[L],h[L]); \
    for(int L=0;L<NL;L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one); \
    for(int L=0;L<NL;L++) el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L])); \
    for(int L=0;L<NL;L++){ sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); scale[L]=_mm512_castsi512_pd(sb[L]); } \
    for(int L=0;L<NL;L++){ ph[L]=_mm512_mul_pd(er[L],scale[L]); y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]); _mm512_storeu_pd(out+(BASE)+8*L,y[L]); } \
} while(0)

static inline __attribute__((always_inline,target("avx512f,avx512dq,fma")))
void exp53_mask_tail(double *out,const double *in,size_t n){
    EXP53_DECL_CONSTS;
    size_t i=0;
    for(;i+8<=n;i+=8){
        EXP53_STAGE(1,i);
    }
    if(i<n){
        unsigned rem=(unsigned)(n-i); __mmask8 km=(__mmask8)((1u<<rem)-1u);
        __m512d x=_mm512_maskz_loadu_pd(km,in+i);
        __m512d biased=_mm512_fmadd_pd(x,inv,magic), k=_mm512_sub_pd(biased,magic);
        __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb), j=_mm512_and_epi64(kn,mask), q=_mm512_srai_epi64(kn,7);
        __m512i tb=_mm512_mask_i64gather_epi64(_mm512_setzero_si512(),km,j,(const long long*)N2_FROZEN_TAB128,8);
        __m512d r=_mm512_fnmadd_pd(k,hi,x); r=_mm512_fnmadd_pd(k,mi,r); r=_mm512_fnmadd_pd(k,lo,r);
        __m512d h=_mm512_fmadd_pd(nq4,r,nq3); h=_mm512_fmadd_pd(h,r,nq2); h=_mm512_fmadd_pd(h,r,nq1); h=_mm512_fmadd_pd(h,r,one);
        __m512d s=_mm512_mul_pd(h,h), er=_mm512_fmadd_pd(r,s,one), el=_mm512_fmadd_pd(r,s,_mm512_sub_pd(one,er));
        __m512i sb=_mm512_add_epi64(tb,_mm512_slli_epi64(q,52)); __m512d scale=_mm512_castsi512_pd(sb);
        __m512d ph=_mm512_mul_pd(er,scale), y=_mm512_fmadd_pd(el,scale,ph);
        _mm512_mask_storeu_pd(out+i,km,y);
    }
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_u4_vectail_candidate(double *restrict out,const double *restrict in,size_t n){
    EXP53_DECL_CONSTS; size_t i=0;
    for(;i+32<=n;i+=32) EXP53_STAGE(4,i);
    if(i<n) exp53_mask_tail(out+i,in+i,n-i);
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_u6_vectail_candidate(double *restrict out,const double *restrict in,size_t n){
    EXP53_DECL_CONSTS; size_t i=0;
    for(;i+48<=n;i+=48) EXP53_STAGE(6,i);
    if(i<n) exp53_mask_tail(out+i,in+i,n-i);
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_u8_vectail_candidate(double *restrict out,const double *restrict in,size_t n){
    EXP53_DECL_CONSTS; size_t i=0;
    for(;i+64<=n;i+=64) EXP53_STAGE(8,i);
    if(i<n) exp53_mask_tail(out+i,in+i,n-i);
}
