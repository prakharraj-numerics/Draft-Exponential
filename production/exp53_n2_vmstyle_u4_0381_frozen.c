/* NEW FROZEN BASELINE — VM-style scheduled, faithful n=2 ER-low EXP.

   Mathematical spine is IDENTICAL to exp53_n2_fused_u4_038_frozen.c:
       Q4(r)=1+r/4+5r^2/96+r^3/128+79r^4/92160
       e^r = 1 + r Q4(r)^2
   Same v128 reduction, same Q4 coefficients and Horner order, same square,
   same ER-low correction, same TAB128 values, same fused table/exponent scale.

   ONLY execution scheduling changed:
     - derive j/q immediately after biased-k extraction
     - launch the four independent TAB128 gathers as early as possible
     - execute residual reduction + Q4 + ER-low while gathers are in flight
     - consume gathered table values only at final scale construction

   Validation run: GitHub Actions 33373846995, Intel Xeon 6973P-C, ICX.
   Strict math-preservation screen (200,000 mixed-domain inputs):
     shard 15: maxULP=1, gt1=0, bitdiff_vs_old_frozen=0
     shard 25: maxULP=1, gt1=0, bitdiff_vs_old_frozen=0

   Clean shard 25 timing:
     n=12288: Intel VML 0.322086 ns/input
              old frozen 0.385437
              this kernel 0.381834
     n=65536: Intel VML 0.318677 ns/input
              old frozen 0.383669
              this kernel 0.381784

   Shard 15 was noisier and showed effectively neutral scheduling performance;
   user explicitly selected this bit-identical early-u4 checkpoint as baseline.

   Dependency note: this file includes the previous immutable self-contained
   frozen checkpoint solely to reuse its exact constants/TAB128 and its tail
   implementation. Do not modify exp53_n2_fused_u4_038_frozen.c.
*/
#include "exp53_n2_fused_u4_038_frozen.c"

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_vmstyle_u4_0381_frozen(double *restrict out,
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
    for(;i+32<=n;i+=32){
        __m512d x[4],biased[4],k[4],r[4],h[4],s[4],er[4],el[4],scale[4],ph[4],y[4];
        __m512i kn[4],j[4],q[4],tb[4],sb[4];

        for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
        for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);

        for(int L=0;L<4;L++){
            k[L]=_mm512_sub_pd(biased[L],magic);
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);
            j[L]=_mm512_and_epi64(kn[L],mask);
            q[L]=_mm512_srai_epi64(kn[L],7);
        }

        /* VM-style latency hiding: launch all independent gathers ASAP. */
        for(int L=0;L<4;L++)
            tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);

        for(int L=0;L<4;L++){
            r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
            r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);
            r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);
        }

        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
        for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);

        for(int L=0;L<4;L++) s[L]=_mm512_mul_pd(h[L],h[L]);
        for(int L=0;L<4;L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one);
        for(int L=0;L<4;L++)
            el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));

        for(int L=0;L<4;L++){
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
            scale[L]=_mm512_castsi512_pd(sb[L]);
        }

        for(int L=0;L<4;L++){
            ph[L]=_mm512_mul_pd(er[L],scale[L]);
            y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]);
            _mm512_storeu_pd(out+i+8*L,y[L]);
        }
    }

    if(i<n)
        exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i);
}
