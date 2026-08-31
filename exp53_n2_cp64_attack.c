/* 64-value critical-path / compressed-state attack for faithful n=2 EXP53.
   Frozen baseline is included read-only and is not modified.

   Design:
   - 8 AVX-512 vectors (64 doubles) per outer iteration.
   - launch all 8 independent TAB128 gathers early.
   - compress each vector to only (r, scale) before Q4.
   - execute the 8 independent Q4 chains stage-wise to maximize ready FP work.
   - final ER-low/reconstruction is drained in two 4-vector groups to control
     register liveness while retaining independent work.

   Mathematical order within each lane is identical to the frozen VM-style path:
     same k/j/q extraction, same 3-term residual reduction,
     same Q4 Horner order, same square, same ER-low correction,
     same fused TAB128 + 2^q scale construction.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_cp64_attack(double *restrict out,
                          const double *restrict in,
                          size_t n)
{
    const __m512d inv=_mm512_set1_pd(N2F_INV128),
                  hi=_mm512_set1_pd(N2F_L128_HI),
                  mi=_mm512_set1_pd(N2F_L128_MI),
                  lo=_mm512_set1_pd(N2F_L128_LO),
                  magic=_mm512_set1_pd(N2F_MAGIC),
                  one=_mm512_set1_pd(1.0),
                  nq1=_mm512_set1_pd(N2F_Q1),
                  nq2=_mm512_set1_pd(N2F_Q2),
                  nq3=_mm512_set1_pd(N2F_Q3),
                  nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),
                  mask=_mm512_set1_epi64(127);

    size_t i=0;
    for(; i+64<=n; i+=64){
        __m512d x[8],biased[8],k[8],r[8],scale[8],h[8];
        __m512i kn[8],j[8],q[8],tb[8],sb[8];

        /* Feed/load all eight vectors. */
        for(int L=0;L<8;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
        for(int L=0;L<8;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);

        /* Independent index/reduction metadata. */
        for(int L=0;L<8;L++){
            k[L]=_mm512_sub_pd(biased[L],magic);
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);
            j[L]=_mm512_and_epi64(kn[L],mask);
            q[L]=_mm512_srai_epi64(kn[L],7);
        }

        /* Earliest possible launch of all 8 independent gathers. */
        for(int L=0;L<8;L++)
            tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);

        /* Critical arithmetic branch while gathers are outstanding. */
        for(int L=0;L<8;L++) r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
        for(int L=0;L<8;L++) r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);
        for(int L=0;L<8;L++) r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);

        /* Collapse lookup branch to scale; only (r,scale) remains live. */
        for(int L=0;L<8;L++){
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
            scale[L]=_mm512_castsi512_pd(sb[L]);
        }

        /* Eight independent Horner chains.  Stage-wise ordering supplies
           abundant ready work across the ~4-cycle FMA dependency latency. */
        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);

        /* Drain four vectors at a time to cap peak live state. */
        for(int G=0;G<8;G+=4){
            __m512d s[4],er[4],el[4],ph[4],y[4];
            for(int t=0;t<4;t++) s[t]=_mm512_mul_pd(h[G+t],h[G+t]);
            for(int t=0;t<4;t++) er[t]=_mm512_fmadd_pd(r[G+t],s[t],one);
            for(int t=0;t<4;t++)
                el[t]=_mm512_fmadd_pd(r[G+t],s[t],_mm512_sub_pd(one,er[t]));
            for(int t=0;t<4;t++) ph[t]=_mm512_mul_pd(er[t],scale[G+t]);
            for(int t=0;t<4;t++){
                y[t]=_mm512_fmadd_pd(el[t],scale[G+t],ph[t]);
                _mm512_storeu_pd(out+i+8*(G+t),y[t]);
            }
        }
    }

    if(i<n)
        exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}
