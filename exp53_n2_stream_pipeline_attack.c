/* Experimental cross-vector software-pipelined faithful n=2 EXP.
   Frozen baseline is included and must remain untouched.

   Goal: treat each AVX-512 vector (8 doubles) as a conveyor-belt token and
   overlap different mathematical stages from consecutive vectors in one CPU
   thread.  The formula, reduction, TAB128 anchors and ER-low correction are
   identical to exp53_n2_vmstyle_u4_0381_frozen.

   Two schedules are tested:
     PIPE3: S0 reduce+gather | S1 Q4+ER-low | S2 scale+store
     PIPE4: S0 reduce+gather | S1 Q4 | S2 ER-low | S3 scale+store

   The steady-state source order deliberately starts the newest S0 first so
   its gather is launched before older independent arithmetic/finalization.
   Arbitrary n is accepted. Full 8-lane vectors use the rolling pipeline; the
   final 1..7 lanes use the same formula via a masked AVX-512 tail.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

typedef struct { __m512d r; __m512i q,tb; } n2p0;
typedef struct { __m512d er,el; __m512i q,tb; } n2p1;
typedef struct { __m512d r,h; __m512i q,tb; } n2p4_1;
typedef struct { __m512d er,el; __m512i q,tb; } n2p4_2;

#define N2P_COMMON_CONSTS \
    const __m512d inv=_mm512_set1_pd(N2F_INV128), \
                  hi=_mm512_set1_pd(N2F_L128_HI), \
                  mi=_mm512_set1_pd(N2F_L128_MI), \
                  lo=_mm512_set1_pd(N2F_L128_LO), \
                  magic=_mm512_set1_pd(N2F_MAGIC), \
                  one=_mm512_set1_pd(1.0), \
                  nq1=_mm512_set1_pd(N2F_Q1), \
                  nq2=_mm512_set1_pd(N2F_Q2), \
                  nq3=_mm512_set1_pd(N2F_Q3), \
                  nq4=_mm512_set1_pd(N2F_Q4); \
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), \
                  mask127=_mm512_set1_epi64(127)

#define N2P_S0_FULL(DST,PTR) do { \
    __m512d _x=_mm512_loadu_pd((PTR)); \
    __m512d _biased=_mm512_fmadd_pd(_x,inv,magic); \
    __m512d _k=_mm512_sub_pd(_biased,magic); \
    __m512i _kn=_mm512_sub_epi64(_mm512_castpd_si512(_biased),mb); \
    __m512i _j=_mm512_and_epi64(_kn,mask127); \
    (DST).q=_mm512_srai_epi64(_kn,7); \
    (DST).tb=_mm512_i64gather_epi64(_j,(const long long*)N2_FROZEN_TAB128,8); \
    __m512d _r=_mm512_fnmadd_pd(_k,hi,_x); \
    _r=_mm512_fnmadd_pd(_k,mi,_r); \
    (DST).r=_mm512_fnmadd_pd(_k,lo,_r); \
} while(0)

#define N2P_Q4(H,R) do { \
    (H)=_mm512_fmadd_pd(nq4,(R),nq3); \
    (H)=_mm512_fmadd_pd((H),(R),nq2); \
    (H)=_mm512_fmadd_pd((H),(R),nq1); \
    (H)=_mm512_fmadd_pd((H),(R),one); \
} while(0)

#define N2P_S1_PIPE3(DST,SRC) do { \
    __m512d _h; N2P_Q4(_h,(SRC).r); \
    __m512d _s=_mm512_mul_pd(_h,_h); \
    (DST).er=_mm512_fmadd_pd((SRC).r,_s,one); \
    (DST).el=_mm512_fmadd_pd((SRC).r,_s,_mm512_sub_pd(one,(DST).er)); \
    (DST).q=(SRC).q; (DST).tb=(SRC).tb; \
} while(0)

#define N2P_S1_PIPE4(DST,SRC) do { \
    (DST).r=(SRC).r; N2P_Q4((DST).h,(SRC).r); \
    (DST).q=(SRC).q; (DST).tb=(SRC).tb; \
} while(0)

#define N2P_S2_PIPE4(DST,SRC) do { \
    __m512d _s=_mm512_mul_pd((SRC).h,(SRC).h); \
    (DST).er=_mm512_fmadd_pd((SRC).r,_s,one); \
    (DST).el=_mm512_fmadd_pd((SRC).r,_s,_mm512_sub_pd(one,(DST).er)); \
    (DST).q=(SRC).q; (DST).tb=(SRC).tb; \
} while(0)

#define N2P_FINAL_FULL(SRC,PTR) do { \
    __m512i _sb=_mm512_add_epi64((SRC).tb,_mm512_slli_epi64((SRC).q,52)); \
    __m512d _scale=_mm512_castsi512_pd(_sb); \
    __m512d _ph=_mm512_mul_pd((SRC).er,_scale); \
    __m512d _y=_mm512_fmadd_pd((SRC).el,_scale,_ph); \
    _mm512_storeu_pd((PTR),_y); \
} while(0)

static __attribute__((target("avx512f,avx512dq,fma"),always_inline)) inline
void n2p_tail(double *out,const double *in,size_t rem)
{
    N2P_COMMON_CONSTS;
    __mmask8 km=(__mmask8)((1u<<rem)-1u);
    __m512d x=_mm512_maskz_loadu_pd(km,in);
    __m512d biased=_mm512_fmadd_pd(x,inv,magic);
    __m512d k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512i j=_mm512_and_epi64(kn,mask127);
    __m512i q=_mm512_srai_epi64(kn,7);
    __m512i tb=_mm512_i64gather_epi64(j,(const long long*)N2_FROZEN_TAB128,8);
    __m512d r=_mm512_fnmadd_pd(k,hi,x);
    r=_mm512_fnmadd_pd(k,mi,r);
    r=_mm512_fnmadd_pd(k,lo,r);
    __m512d h; N2P_Q4(h,r);
    __m512d s=_mm512_mul_pd(h,h);
    __m512d er=_mm512_fmadd_pd(r,s,one);
    __m512d el=_mm512_fmadd_pd(r,s,_mm512_sub_pd(one,er));
    __m512i sb=_mm512_add_epi64(tb,_mm512_slli_epi64(q,52));
    __m512d scale=_mm512_castsi512_pd(sb);
    __m512d ph=_mm512_mul_pd(er,scale);
    __m512d y=_mm512_fmadd_pd(el,scale,ph);
    _mm512_mask_storeu_pd(out,km,y);
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_pipe3_stream(double *restrict out,const double *restrict in,size_t n)
{
    N2P_COMMON_CONSTS;
    const size_t nv=n>>3, rem=n&7;
    if(nv==0){ if(rem) n2p_tail(out,in,rem); return; }

    n2p0 b,c;
    n2p1 a,an;
    N2P_S0_FULL(b,in);
    if(nv==1){ N2P_S1_PIPE3(a,b); N2P_FINAL_FULL(a,out); goto tail; }

    N2P_S1_PIPE3(a,b);
    N2P_S0_FULL(b,in+8);

    for(size_t v=2;v<nv;v++){
        /* Newest token first: launch gather before older independent work. */
        N2P_S0_FULL(c,in+8*v);
        N2P_S1_PIPE3(an,b);
        N2P_FINAL_FULL(a,out+8*(v-2));
        a=an; b=c;
    }
    N2P_FINAL_FULL(a,out+8*(nv-2));
    N2P_S1_PIPE3(a,b);
    N2P_FINAL_FULL(a,out+8*(nv-1));

tail:
    if(rem) n2p_tail(out+8*nv,in+8*nv,rem);
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_pipe4_stream(double *restrict out,const double *restrict in,size_t n)
{
    N2P_COMMON_CONSTS;
    const size_t nv=n>>3, rem=n&7;
    if(nv==0){ if(rem) n2p_tail(out,in,rem); return; }

    n2p0 c,d;
    n2p4_1 b,bn;
    n2p4_2 a,an;

    N2P_S0_FULL(c,in);
    if(nv==1){ N2P_S1_PIPE4(b,c); N2P_S2_PIPE4(a,b); N2P_FINAL_FULL(a,out); goto tail; }

    N2P_S1_PIPE4(b,c);
    N2P_S0_FULL(c,in+8);
    if(nv==2){
        N2P_S2_PIPE4(a,b); N2P_S1_PIPE4(b,c);
        N2P_FINAL_FULL(a,out); N2P_S2_PIPE4(a,b); N2P_FINAL_FULL(a,out+8);
        goto tail;
    }

    N2P_S2_PIPE4(a,b);
    N2P_S1_PIPE4(b,c);
    N2P_S0_FULL(c,in+16);

    for(size_t v=3;v<nv;v++){
        /* Four-stage conveyor: S0(new), S1(prev), S2(prev2), S3(prev3). */
        N2P_S0_FULL(d,in+8*v);
        N2P_S1_PIPE4(bn,c);
        N2P_S2_PIPE4(an,b);
        N2P_FINAL_FULL(a,out+8*(v-3));
        a=an; b=bn; c=d;
    }

    N2P_FINAL_FULL(a,out+8*(nv-3));
    N2P_S2_PIPE4(a,b);
    N2P_FINAL_FULL(a,out+8*(nv-2));
    N2P_S1_PIPE4(b,c);
    N2P_S2_PIPE4(a,b);
    N2P_FINAL_FULL(a,out+8*(nv-1));

tail:
    if(rem) n2p_tail(out+8*nv,in+8*nv,rem);
}
