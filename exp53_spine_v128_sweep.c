/* Optimization sweep built directly on the frozen/correct v128 kernel. */
#include "exp53_spine_v128_frozen.c"

static inline __m512d block512(__m512d x,const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask){
    __m512d biased=_mm512_fmadd_pd(x,inv,magic);
    __m512d k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512d r=_mm512_fnmadd_pd(k,hi,x); r=_mm512_fnmadd_pd(k,mi,r); r=_mm512_fnmadd_pd(k,lo,r);
    __m512d er=spine_residual128(r);
    __m512i j=_mm512_and_epi64(kn,mask),q=_mm512_srai_epi64(kn,7);
    __m512d t=_mm512_i64gather_pd(j,TAB128,8);
    return _mm512_mul_pd(_mm512_mul_pd(er,t),exp2_from_q(q));
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_u2(double *restrict out,const double *restrict in,size_t n){
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127); size_t i=0;
 for(;i+16<=n;i+=16){__m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8);__m512d y0=block512(x0,inv,hi,mi,lo,magic,mb,mask),y1=block512(x1,inv,hi,mi,lo,magic,mb,mask);_mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);} for(;i<n;i+=8){size_t m=n-i;if(m>=8)_mm512_storeu_pd(out+i,block512(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));else{for(;i<n;i++)out[i]=exp(in[i]);break;}}
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_u4(double *restrict out,const double *restrict in,size_t n){
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127); size_t i=0;
 for(;i+32<=n;i+=32){__m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24);__m512d y0=block512(x0,inv,hi,mi,lo,magic,mb,mask),y1=block512(x1,inv,hi,mi,lo,magic,mb,mask),y2=block512(x2,inv,hi,mi,lo,magic,mb,mask),y3=block512(x3,inv,hi,mi,lo,magic,mb,mask);_mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);_mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);} for(;i<n;){size_t m=n-i;if(m>=8){_mm512_storeu_pd(out+i,block512(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));i+=8;}else{for(;i<n;i++)out[i]=exp(in[i]);}}
}

/* Register-resident 16x8 decomposition: j=8*a+b, 2^(j/128)=2^(a/16)*2^(b/128). */
static const double A16[16]={0x1.0000000000000p+0,0x1.0b5586cf9890fp+0,0x1.172b83c7d517bp+0,0x1.2387a6e756238p+0,0x1.306fe0a31b715p+0,0x1.3dea64c123422p+0,0x1.4bfdad5362a27p+0,0x1.5ab07dd485429p+0,0x1.6a09e667f3bcdp+0,0x1.7a11473eb0187p+0,0x1.8ace5422aa0dbp+0,0x1.9c49182a3f090p+0,0x1.ae89f995ad3adp+0,0x1.c199bdd85529cp+0,0x1.d5818dcfba487p+0,0x1.ea4afa2a490dap+0};
static const double B8[8]={0x1.0000000000000p+0,0x1.0163da9fb3335p+0,0x1.02c9a3e778061p+0,0x1.04315e86e7f85p+0,0x1.059b0d3158574p+0,0x1.0706b29ddf6dep+0,0x1.0874518759bc8p+0,0x1.09e3ecac6f383p+0};

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_decomp(double *restrict out,const double *restrict in,size_t n){
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127),m7=_mm512_set1_epi64(7); const __m512d btab=_mm512_loadu_pd(B8); const __m512d a0=_mm512_loadu_pd(A16),a1=_mm512_loadu_pd(A16+8); size_t i=0;
 for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i),biased=_mm512_fmadd_pd(x,inv,magic),k=_mm512_sub_pd(biased,magic);__m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb),j=_mm512_and_epi64(kn,mask),q=_mm512_srai_epi64(kn,7),b=_mm512_and_epi64(j,m7),a=_mm512_srli_epi64(j,3);__m512d r=_mm512_fnmadd_pd(k,hi,x);r=_mm512_fnmadd_pd(k,mi,r);r=_mm512_fnmadd_pd(k,lo,r);__m512d er=spine_residual128(r);__m512d tb=_mm512_permutexvar_pd(b,btab);__mmask8 upper=_mm512_cmp_epi64_mask(a,_mm512_set1_epi64(8),_MM_CMPINT_GE);__m512i al=_mm512_and_epi64(a,m7);__m512d ta0=_mm512_permutexvar_pd(al,a0),ta1=_mm512_permutexvar_pd(al,a1),ta=_mm512_mask_blend_pd(upper,ta0,ta1);__m512d t=_mm512_mul_pd(ta,tb);_mm512_storeu_pd(out+i,_mm512_mul_pd(_mm512_mul_pd(er,t),exp2_from_q(q)));}for(;i<n;i++)out[i]=exp(in[i]);
}

/* AVX2 control: same v128 algorithm, 4 lanes, gather from the same 1 KiB table. */
__attribute__((target("avx2,fma")))
void exp53_spine_v128_avx2(double *restrict out,const double *restrict in,size_t n){
 const __m256d inv=_mm256_set1_pd(INV128),hi=_mm256_set1_pd(L128_HI),mi=_mm256_set1_pd(L128_MI),lo=_mm256_set1_pd(L128_LO),magic=_mm256_set1_pd(MAGIC),one=_mm256_set1_pd(1.0),half=_mm256_set1_pd(.5),a1=_mm256_set1_pd(1.0/24.0),a2=_mm256_set1_pd(1.0/720.0);size_t i=0;
 for(;i+4<=n;i+=4){__m256d x=_mm256_loadu_pd(in+i),biased=_mm256_fmadd_pd(x,inv,magic),k=_mm256_sub_pd(biased,magic);int64_t kb[4];_mm256_storeu_si256((__m256i*)kb,_mm256_sub_epi64(_mm256_castpd_si256(biased),_mm256_set1_epi64x((long long)MAGIC_BITS)));__m256d r=_mm256_fnmadd_pd(k,hi,x);r=_mm256_fnmadd_pd(k,mi,r);r=_mm256_fnmadd_pd(k,lo,r);__m256d u=_mm256_mul_pd(r,half),z=_mm256_mul_pd(u,u),p1=_mm256_fmadd_pd(a2,z,a1),p=_mm256_fmadd_pd(p1,z,half),pp=_mm256_fmadd_pd(_mm256_set1_pd(2.0/720.0),z,a1);pp=_mm256_fmadd_pd(pp,z,p);__m256d C=_mm256_mul_pd(z,p),s=_mm256_mul_pd(_mm256_add_pd(u,u),pp),h=_mm256_add_pd(one,_mm256_add_pd(C,s)),er=_mm256_mul_pd(h,h);double ev[4];_mm256_storeu_pd(ev,er);for(int lane=0;lane<4;lane++){long kk=(long)kb[lane],j=kk&127,q=kk>>7;union{uint64_t u;double d;}sc={(uint64_t)(q+1023)<<52};out[i+lane]=ev[lane]*TAB128[j]*sc.d;}}
 for(;i<n;i++)out[i]=exp(in[i]);
}
