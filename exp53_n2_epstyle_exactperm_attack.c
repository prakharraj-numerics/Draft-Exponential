/* EP-STYLE ATTACK — preserve frozen n=2 math exactly.

   Structural inspiration only from oneMKL AVX-512 VML EP inspection:
     - four independent ZMM streams
     - no gather instructions in Intel EP hot kernel
     - permute-based table/reconstruction organization
     - compact branch-light hot path

   Numerical contract of our kernel is unchanged:
     same v128 reduction constants and order
     same Q4 coefficients and Horner order
     same s = Q4^2
     same er = 1 + r*s and ER-low correction
     same exact 128 TAB values and fused exponent-bit scaling

   The only experiment here is exact TAB128 lookup without vgather.
   The 128 exact table doubles are partitioned into 8 banks of 16 values.
   Each 16-value bank is selected with AVX-512 permutex2var from two 8-value
   vectors, then mask-selected by bank number. No table approximation or
   factorization is used, so the selected table bit-pattern is identical to
   N2_FROZEN_TAB128[j].
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

static inline __m512d ep_exact_tab16x8(__m512i j)
{
    const __m512i m15 = _mm512_set1_epi64(15);
    __m512i slot = _mm512_and_epi64(j, m15);
    __m512i bank = _mm512_srli_epi64(j, 4);
    __m512d out = _mm512_setzero_pd();

    for (int b=0; b<8; ++b) {
        const double *p = N2_FROZEN_TAB128 + 16*b;
        __m512d a = _mm512_loadu_pd(p);
        __m512d c = _mm512_loadu_pd(p+8);
        __m512d v = _mm512_permutex2var_pd(a, slot, c);
        __mmask8 k = _mm512_cmpeq_epi64_mask(bank, _mm512_set1_epi64(b));
        out = _mm512_mask_mov_pd(out, k, v);
    }
    return out;
}

/* Variant A: build exact permute table values immediately after j/q extraction,
   then run the untouched frozen arithmetic. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_epstyle_perm8_early(double *restrict out,
                                  const double *restrict in,
                                  size_t n)
{
    const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC), one=_mm512_set1_pd(1.0),
                  nq1=_mm512_set1_pd(N2F_Q1), nq2=_mm512_set1_pd(N2F_Q2),
                  nq3=_mm512_set1_pd(N2F_Q3), nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),
                  mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(; i+32<=n; i+=32){
        __m512d x[4],biased[4],k[4],tab[4],r[4],h[4],s[4],er[4],el[4],scale[4],ph[4],y[4];
        __m512i kn[4],j[4],q[4],tb[4],sb[4];
        for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
        for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
        for(int L=0;L<4;L++){
            k[L]=_mm512_sub_pd(biased[L],magic);
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);
            j[L]=_mm512_and_epi64(kn[L],mask);
            q[L]=_mm512_srai_epi64(kn[L],7);
        }
        for(int L=0;L<4;L++) tab[L]=ep_exact_tab16x8(j[L]);
        for(int L=0;L<4;L++){
            r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
            r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);
            r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);
        }
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);
        for(int L=0;L<4;L++){
            s[L]=_mm512_mul_pd(h[L],h[L]);
            er[L]=_mm512_fmadd_pd(r[L],s[L],one);
            el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));
            tb[L]=_mm512_castpd_si512(tab[L]);
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
            scale[L]=_mm512_castsi512_pd(sb[L]);
            ph[L]=_mm512_mul_pd(er[L],scale[L]);
            y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]);
            _mm512_storeu_pd(out+i+8*L,y[L]);
        }
    }
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}

/* Variant B: Intel-EP-like ordering: finish residual reduction first, then
   permute-table reconstruction, then short per-stream polynomial/final chain. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_epstyle_perm8_afterreduce(double *restrict out,
                                        const double *restrict in,
                                        size_t n)
{
    const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC), one=_mm512_set1_pd(1.0),
                  nq1=_mm512_set1_pd(N2F_Q1), nq2=_mm512_set1_pd(N2F_Q2),
                  nq3=_mm512_set1_pd(N2F_Q3), nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),
                  mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(; i+32<=n; i+=32){
        __m512d x[4],biased[4],k[4],r[4],tab[4];
        __m512i kn[4],j[4],q[4];
        for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
        for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
        for(int L=0;L<4;L++){
            k[L]=_mm512_sub_pd(biased[L],magic);
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);
            j[L]=_mm512_and_epi64(kn[L],mask);
            q[L]=_mm512_srai_epi64(kn[L],7);
            r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
            r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);
            r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);
        }
        for(int L=0;L<4;L++) tab[L]=ep_exact_tab16x8(j[L]);
        for(int L=0;L<4;L++){
            __m512d h=_mm512_fmadd_pd(nq4,r[L],nq3);
            h=_mm512_fmadd_pd(h,r[L],nq2);
            h=_mm512_fmadd_pd(h,r[L],nq1);
            h=_mm512_fmadd_pd(h,r[L],one);
            __m512d s=_mm512_mul_pd(h,h);
            __m512d er=_mm512_fmadd_pd(r[L],s,one);
            __m512d el=_mm512_fmadd_pd(r[L],s,_mm512_sub_pd(one,er));
            __m512i tb=_mm512_castpd_si512(tab[L]);
            __m512i sb=_mm512_add_epi64(tb,_mm512_slli_epi64(q[L],52));
            __m512d scale=_mm512_castsi512_pd(sb);
            __m512d ph=_mm512_mul_pd(er,scale);
            __m512d y=_mm512_fmadd_pd(el,scale,ph);
            _mm512_storeu_pd(out+i+8*L,y);
        }
    }
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}
