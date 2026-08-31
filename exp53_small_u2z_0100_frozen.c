/* EXP53 FROZEN SMALL-BATCH U2Z SURVIVOR — n <= 100 production path.

   Frozen from exp53_smallbatch_width_candidates.c after exact-Xeon sweep
   run 33422301867 on Intel Xeon 6973P-C (six qualifying shards).

   Result used for promotion:
     - u2z (two ZMM vectors / 16 values per loop) was the winning own-kernel
       path at n=50 and n=100 in both unit and mid domains.
     - Intel HA regained the lead from n=150 onward, so this frozen kernel is
       intentionally dispatched only through n=100.

   Mathematical spine/constants/table are identical to
   exp53_n2_vmstyle_u4_0381_frozen.c. Only loop width/scheduling differs.
   Tail falls back to the immutable frozen VM-style kernel.

   Integration note:
     This production translation unit includes the already-frozen rare-NT
     implementation, which in turn includes the immutable VM-style baseline.
     Therefore compiling THIS file alone exports all three production C symbols:
       exp53_n2_vmstyle_u4_0381_frozen
       exp53_n2_vmstyle_u4_0381_nt_sfence
       exp53_small_u2z_0100_frozen
     Do not separately compile the included C files into the same executable.

   DO NOT MODIFY. Future experiments must use new candidate files.
*/
#include "exp53_n2_vmstyle_u4_0381_nt_sfence.c"

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_small_u2z_0100_frozen(double *restrict out,
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
    for(;i+16<=n;i+=16){
        __m512d x[2],biased[2],k[2],r[2],h[2],s[2],er[2],el[2],scale[2],ph[2],y[2];
        __m512i kn[2],j[2],q[2],tb[2],sb[2];
        for(int L=0;L<2;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
        for(int L=0;L<2;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
        for(int L=0;L<2;L++){
            k[L]=_mm512_sub_pd(biased[L],magic);
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);
            j[L]=_mm512_and_epi64(kn[L],mask);
            q[L]=_mm512_srai_epi64(kn[L],7);
        }
        for(int L=0;L<2;L++)
            tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
        for(int L=0;L<2;L++){
            r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
            r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);
            r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);
        }
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
        for(int L=0;L<2;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);
        for(int L=0;L<2;L++) s[L]=_mm512_mul_pd(h[L],h[L]);
        for(int L=0;L<2;L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one);
        for(int L=0;L<2;L++)
            el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));
        for(int L=0;L<2;L++){
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
            scale[L]=_mm512_castsi512_pd(sb[L]);
        }
        for(int L=0;L<2;L++){
            ph[L]=_mm512_mul_pd(er[L],scale[L]);
            y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]);
            _mm512_storeu_pd(out+i+8*L,y[L]);
        }
    }
    if(i<n)
        exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}
