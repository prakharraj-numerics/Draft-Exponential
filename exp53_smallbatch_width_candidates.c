/* Experimental small-batch candidates. Does NOT modify frozen production code.
   Same mathematical spine/constants/table as exp53_n2_vmstyle_u4_0381_frozen.c.
   Candidates:
     - u2z: two ZMM vectors (16 values) per loop
     - u4y: four YMM vectors (16 values) per loop, AVX-512VL/DQ used for 64-bit q shift
   Tail falls back to the immutable frozen implementation.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_small_u2z(double *restrict out,const double *restrict in,size_t n){
    const __m512d inv=_mm512_set1_pd(N2F_INV128),hi=_mm512_set1_pd(N2F_L128_HI),mi=_mm512_set1_pd(N2F_L128_MI),lo=_mm512_set1_pd(N2F_L128_LO),magic=_mm512_set1_pd(N2F_MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2F_Q1),nq2=_mm512_set1_pd(N2F_Q2),nq3=_mm512_set1_pd(N2F_Q3),nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+16<=n;i+=16){
        __m512d x[2],biased[2],k[2],r[2],h[2],s[2],er[2],el[2],scale[2],ph[2],y[2];
        __m512i kn[2],j[2],q[2],tb[2],sb[2];
        for(int L=0;L<2;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
        for(int L=0;L<2;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
        for(int L=0;L<2;L++){ k[L]=_mm512_sub_pd(biased[L],magic); kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); j[L]=_mm512_and_epi64(kn[L],mask); q[L]=_mm512_srai_epi64(kn[L],7); }
        for(int L=0;L<2;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
        for(int L=0;L<2;L++){ r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); }
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);
        for(int L=0;L<2;L++) s[L]=_mm512_mul_pd(h[L],h[L]);
        for(int L=0;L<2;L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one);
        for(int L=0;L<2;L++) el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));
        for(int L=0;L<2;L++){ sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); scale[L]=_mm512_castsi512_pd(sb[L]); }
        for(int L=0;L<2;L++){ ph[L]=_mm512_mul_pd(er[L],scale[L]); y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]); _mm512_storeu_pd(out+i+8*L,y[L]); }
    }
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}

__attribute__((target("avx2,avx512vl,avx512dq,fma"),noinline))
void exp53_small_u4y(double *restrict out,const double *restrict in,size_t n){
    const __m256d inv=_mm256_set1_pd(N2F_INV128),hi=_mm256_set1_pd(N2F_L128_HI),mi=_mm256_set1_pd(N2F_L128_MI),lo=_mm256_set1_pd(N2F_L128_LO),magic=_mm256_set1_pd(N2F_MAGIC),one=_mm256_set1_pd(1.0),nq1=_mm256_set1_pd(N2F_Q1),nq2=_mm256_set1_pd(N2F_Q2),nq3=_mm256_set1_pd(N2F_Q3),nq4=_mm256_set1_pd(N2F_Q4);
    const __m256i mb=_mm256_set1_epi64x((long long)N2F_MAGIC_BITS),mask=_mm256_set1_epi64x(127);
    size_t i=0;
    for(;i+16<=n;i+=16){
        __m256d x[4],biased[4],k[4],r[4],h[4],s[4],er[4],el[4],scale[4],ph[4],y[4];
        __m256i kn[4],j[4],q[4],tb[4],sb[4];
        for(int L=0;L<4;L++) x[L]=_mm256_loadu_pd(in+i+4*L);
        for(int L=0;L<4;L++) biased[L]=_mm256_fmadd_pd(x[L],inv,magic);
        for(int L=0;L<4;L++){ k[L]=_mm256_sub_pd(biased[L],magic); kn[L]=_mm256_sub_epi64(_mm256_castpd_si256(biased[L]),mb); j[L]=_mm256_and_si256(kn[L],mask); q[L]=_mm256_srai_epi64(kn[L],7); }
        for(int L=0;L<4;L++) tb[L]=_mm256_i64gather_epi64((const long long*)N2_FROZEN_TAB128,j[L],8);
        for(int L=0;L<4;L++){ r[L]=_mm256_fnmadd_pd(k[L],hi,x[L]); r[L]=_mm256_fnmadd_pd(k[L],mi,r[L]); r[L]=_mm256_fnmadd_pd(k[L],lo,r[L]); }
        for(int L=0;L<4;L++) h[L]=_mm256_fmadd_pd(nq4,r[L],nq3);
        for(int L=0;L<4;L++) h[L]=_mm256_fmadd_pd(h[L],r[L],nq2);
        for(int L=0;L<4;L++) h[L]=_mm256_fmadd_pd(h[L],r[L],nq1);
        for(int L=0;L<4;L++) h[L]=_mm256_fmadd_pd(h[L],r[L],one);
        for(int L=0;L<4;L++) s[L]=_mm256_mul_pd(h[L],h[L]);
        for(int L=0;L<4;L++) er[L]=_mm256_fmadd_pd(r[L],s[L],one);
        for(int L=0;L<4;L++) el[L]=_mm256_fmadd_pd(r[L],s[L],_mm256_sub_pd(one,er[L]));
        for(int L=0;L<4;L++){ sb[L]=_mm256_add_epi64(tb[L],_mm256_slli_epi64(q[L],52)); scale[L]=_mm256_castsi256_pd(sb[L]); }
        for(int L=0;L<4;L++){ ph[L]=_mm256_mul_pd(er[L],scale[L]); y[L]=_mm256_fmadd_pd(el[L],scale[L],ph[L]); _mm256_storeu_pd(out+i+4*L,y[L]); }
    }
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}
