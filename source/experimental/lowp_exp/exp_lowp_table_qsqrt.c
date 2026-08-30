#include <mpfr.h>
#include <gmp.h>
#include <flint/arf.h>
#include <flint/arb.h>
#include <flint/flint.h>
#include <flint/fmpq.h>
#include <flint/fmpz.h>
#include <flint/mpn_extras.h>
#include <flint/longlong.h>
#include <math.h>
#include <stddef.h>

/*
  Direct-residual table-reduced EXP, preserving the C/sqrt derivation.

  C(u) = z P(z), z=u^2, P(z)=1/2!+z/4!+...
  sqrt(C(C+2)) = u sqrt(2P+zP^2).

  Since P is the C-series polynomial,
      sqrt(2P+zP^2) = 2(P+zP')
  identically for the analytic positive branch.  Hence
      sqrt(C(C+2)) = u + u*z*R(z),
      R(z)=1/3! + z/5! + z^2/7! + ...
  This is used as a specialized near-one square-root approximation on the
  tiny table residual.  It removes P^2, zP^2, mpn_sqrtrem and the root multiply.

  2-limb fixed products use a small inlined high-product kernel to avoid the
  generic full 2x2 product; n>=3 continues to use FLINT mulhigh/sqrhigh.

  Commercial low-p scope: 30 and 50 decimal digits only.  Higher precisions
  belong to the separate high-precision EXP lane and are intentionally rejected.
*/

enum { QS_MAX_TERMS=32, QS_SLOTS=20 };

typedef struct {
    unsigned digits;
    slong target_bits, work_bits, n;
    int terms_p, terms_r;
    mp_limb_t *pcoeff, *rcoeff, *buf, *prod, *tab10;
    fmpz_t q;
} exp_lowp_qsqrt_ctx;

static inline mp_ptr B(exp_lowp_qsqrt_ctx*c,int i){return c->buf+(size_t)i*(size_t)(2*c->n+4);}
static inline mp_ptr PC(exp_lowp_qsqrt_ctx*c,int i){return c->pcoeff+(size_t)i*(size_t)c->n;}
static inline mp_ptr RC(exp_lowp_qsqrt_ctx*c,int i){return c->rcoeff+(size_t)i*(size_t)c->n;}
static inline mp_ptr T10(exp_lowp_qsqrt_ctx*c,int p1,int p2){return c->tab10+((size_t)p1*32u+(size_t)p2)*(size_t)c->n;}

static inline void mulhi2(mp_ptr r,mp_srcptr a,mp_srcptr b){
#if FLINT_BITS == 64 && defined(__SIZEOF_INT128__)
    __uint128_t p11=(__uint128_t)a[1]*b[1];
    __uint128_t p10=(__uint128_t)a[1]*b[0];
    __uint128_t p01=(__uint128_t)a[0]*b[1];
    ulong lo=(ulong)p11;
    ulong hi=(ulong)(p11>>64);
    __uint128_t s=(__uint128_t)lo+(ulong)(p10>>64)+(ulong)(p01>>64);
    r[0]=(ulong)s;
    r[1]=hi+(ulong)(s>>64);
#else
    mp_limb_t t[4];flint_mpn_mul_n(t,a,b,2);r[0]=t[2];r[1]=t[3];
#endif
}
static inline void sqrhi2(mp_ptr r,mp_srcptr a){
#if FLINT_BITS == 64 && defined(__SIZEOF_INT128__)
    __uint128_t p11=(__uint128_t)a[1]*a[1];
    __uint128_t p10=(__uint128_t)a[1]*a[0];
    ulong lo=(ulong)p11,hi=(ulong)(p11>>64),ch=(ulong)(p10>>64);
    __uint128_t s=(__uint128_t)lo+ch+ch;
    r[0]=(ulong)s;r[1]=hi+(ulong)(s>>64);
#else
    mp_limb_t t[4];flint_mpn_sqr(t,a,2);r[0]=t[2];r[1]=t[3];
#endif
}
static inline void mulhi(exp_lowp_qsqrt_ctx*c,mp_ptr r,mp_srcptr a,mp_srcptr b){
    if(c->n==2)mulhi2(r,a,b);
    else if(c->n>=3)(void)flint_mpn_mulhigh_n(r,a,b,c->n);
    else{flint_mpn_mul_n(c->prod,a,b,c->n);flint_mpn_copyi(r,c->prod+c->n,c->n);}
}
static inline void sqrhi(exp_lowp_qsqrt_ctx*c,mp_ptr r,mp_srcptr a){
    if(c->n==2)sqrhi2(r,a);
    else if(c->n>=3)(void)flint_mpn_sqrhigh(r,a,c->n);
    else{flint_mpn_sqr(c->prod,a,c->n);flint_mpn_copyi(r,c->prod+c->n,c->n);}
}

static int choose_p_terms(slong bits){
    const long double il2=1.442695040888963407359924681001892L;
    const long double log2z=-20.0L;
    for(int t=2;t<QS_MAX_TERMS;t++){
        int m=t+1;
        long double e=(long double)m*log2z-lgammal((long double)(2*m+1))*il2;
        if(e<-(long double)(bits+32))return t;
    }
    return QS_MAX_TERMS-1;
}
static int choose_r_terms(slong bits){
    const long double il2=1.442695040888963407359924681001892L;
    for(int t=1;t<QS_MAX_TERMS;t++){
        int k=2*t+3;
        long double e=-(long double)10*k-lgammal((long double)(k+1))*il2;
        if(e<-(long double)(bits+32))return t;
    }
    return QS_MAX_TERMS-1;
}

static void build_coeffs(exp_lowp_qsqrt_ctx*c){
    slong n=c->n;mp_ptr tmp=B(c,18);
    flint_mpn_zero(PC(c,0),n);PC(c,0)[n-1]=UWORD(1)<<(FLINT_BITS-1);
    for(int i=1;i<c->terms_p;i++){
        ulong d=(ulong)(2*i+1)*(ulong)(2*i+2);
        (void)mpn_divrem_1(tmp,0,PC(c,i-1),n,d);flint_mpn_copyi(PC(c,i),tmp,n);
    }
    flint_mpn_zero(RC(c,0),n);flint_mpn_zero(tmp,n);tmp[n-1]=UWORD(1)<<(FLINT_BITS-1);
    (void)mpn_divrem_1(RC(c,0),0,tmp,n,3);
    for(int i=1;i<c->terms_r;i++){
        ulong d=(ulong)(2*i+2)*(ulong)(2*i+3);
        (void)mpn_divrem_1(tmp,0,RC(c,i-1),n,d);flint_mpn_copyi(RC(c,i),tmp,n);
    }
}
static void build_tab10(exp_lowp_qsqrt_ctx*c){
    slong n=c->n;mp_ptr full=B(c,17),hi=B(c,18);
    for(int p1=0;p1<ARB_EXP_TAB21_NUM;p1++){
        nn_srcptr a=arb_exp_tab21[p1]+ARB_EXP_TAB2_LIMBS-n;
        for(int p2=0;p2<ARB_EXP_TAB22_NUM;p2++){
            nn_srcptr b=arb_exp_tab22[p2]+ARB_EXP_TAB2_LIMBS-n;
            flint_mpn_mul_n(full,a,b,n);flint_mpn_copyi(hi,full+n,n);
            (void)mpn_lshift(T10(c,p1,p2),hi,n,1);
        }
    }
}

exp_lowp_qsqrt_ctx*exp_lowp_qsqrt_create(unsigned digits,int extra_limb){
    if(digits!=30u&&digits!=50u)return NULL;
    exp_lowp_qsqrt_ctx*c=(exp_lowp_qsqrt_ctx*)flint_calloc(1,sizeof(*c));if(!c)return NULL;
    c->digits=digits;c->target_bits=(slong)ceill((long double)digits*3.321928094887362347870319429489390L);
    c->n=(c->target_bits+16+FLINT_BITS-1)/FLINT_BITS+extra_limb;if(c->n<2)c->n=2;
    if(c->n>ARB_EXP_TAB2_LIMBS){flint_free(c);return NULL;}c->work_bits=c->n*FLINT_BITS;
    c->terms_p=choose_p_terms(c->work_bits);c->terms_r=choose_r_terms(c->work_bits);
    c->pcoeff=(mp_limb_t*)flint_calloc((size_t)c->terms_p*(size_t)c->n,sizeof(mp_limb_t));
    c->rcoeff=(mp_limb_t*)flint_calloc((size_t)c->terms_r*(size_t)c->n,sizeof(mp_limb_t));
    c->buf=(mp_limb_t*)flint_calloc((size_t)QS_SLOTS*(size_t)(2*c->n+4),sizeof(mp_limb_t));
    c->prod=(mp_limb_t*)flint_malloc((size_t)(2*c->n+4)*sizeof(mp_limb_t));
    c->tab10=(mp_limb_t*)flint_calloc((size_t)ARB_EXP_TAB21_NUM*ARB_EXP_TAB22_NUM*(size_t)c->n,sizeof(mp_limb_t));fmpz_init(c->q);
    if(!c->pcoeff||!c->rcoeff||!c->buf||!c->prod||!c->tab10){return NULL;}
    build_coeffs(c);build_tab10(c);return c;
}
void exp_lowp_qsqrt_destroy(exp_lowp_qsqrt_ctx*c){if(!c)return;if(c->pcoeff)flint_free(c->pcoeff);if(c->rcoeff)flint_free(c->rcoeff);if(c->buf)flint_free(c->buf);if(c->prod)flint_free(c->prod);if(c->tab10)flint_free(c->tab10);fmpz_clear(c->q);flint_free(c);}
slong exp_lowp_qsqrt_work_bits(const exp_lowp_qsqrt_ctx*c){return c?c->work_bits:0;}
int exp_lowp_qsqrt_pterms(const exp_lowp_qsqrt_ctx*c){return c?c->terms_p:0;}
int exp_lowp_qsqrt_rterms(const exp_lowp_qsqrt_ctx*c){return c?c->terms_r:0;}
arf_ptr exp_lowp_qsqrt_alloc_elems(size_t n){return _arf_vec_init((slong)n);}void exp_lowp_qsqrt_free_elems(arf_ptr p,size_t n){if(p)_arf_vec_clear(p,(slong)n);}
int exp_lowp_qsqrt_set_ratio(exp_lowp_qsqrt_ctx*c,arf_ptr out,long A,unsigned long D){if(!c||!out||!D)return 1;fmpq_t q;fmpq_init(q);fmpq_set_si(q,A,D);int r=arf_set_fmpq(out,q,c->work_bits+64,ARF_RND_NEAR);fmpq_clear(q);return r;}

static void series_p(exp_lowp_qsqrt_ctx*c,mp_ptr p,mp_srcptr z){mp_ptr t=B(c,4);flint_mpn_copyi(p,PC(c,c->terms_p-1),c->n);for(int i=c->terms_p-2;i>=0;i--){mulhi(c,t,p,z);(void)mpn_add_n(p,t,PC(c,i),c->n);}}
static void series_r(exp_lowp_qsqrt_ctx*c,mp_ptr r,mp_srcptr z){mp_ptr t=B(c,5);flint_mpn_copyi(r,RC(c,c->terms_r-1),c->n);for(int i=c->terms_r-2;i>=0;i--){mulhi(c,t,r,z);(void)mpn_add_n(r,t,RC(c,i),c->n);}}

static void formula_core(exp_lowp_qsqrt_ctx*c,mp_ptr ehalf,mp_srcptr w){
    slong n=c->n;mp_ptr u=B(c,0),z=B(c,1),p=B(c,2),C=B(c,3),r=B(c,6),zr=B(c,7),uzr=B(c,8),s=B(c,9),hf=B(c,10),eh=B(c,11);
    flint_mpn_copyi(u,w,n);
    sqrhi(c,z,u);
    series_p(c,p,z);mulhi(c,C,z,p);
    series_r(c,r,z);mulhi(c,zr,z,r);mulhi(c,uzr,u,zr);
    flint_mpn_copyi(s,u,n);(void)mpn_add_n(s,s,uzr,n);
    flint_mpn_copyi(hf,C,n);hf[n]=1;hf[n]+=mpn_add_n(hf,hf,s,n);
    (void)mpn_rshift(eh,hf,n+1,1);flint_mpn_copyi(ehalf,eh,n);
}

int exp_lowp_qsqrt_eval(exp_lowp_qsqrt_ctx*c,arf_ptr out,const arf_t x){
    if(!c||!out||!x||arf_is_special(x))return 1;slong n=c->n;ulong err=0;mp_ptr w=B(c,14),ehalf=B(c,15),fin=B(c,16);
    if(!_arb_get_mpn_fixed_mod_log2(w,c->q,&err,x,n))return 2;slong q=fmpz_get_si(c->q);
    ulong p1=w[n-1]>>(FLINT_BITS-ARB_EXP_TAB21_BITS);w[n-1]-=p1<<(FLINT_BITS-ARB_EXP_TAB21_BITS);
    ulong p2=w[n-1]>>(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));w[n-1]-=p2<<(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));
    formula_core(c,ehalf,w);
    if(p1==0&&p2==0){flint_mpn_copyi(fin,ehalf,n);arf_set_mpn(out,fin,n,0);arf_mul_2exp_si(out,out,q+1-n*FLINT_BITS);}
    else{mulhi(c,fin,ehalf,T10(c,(int)p1,(int)p2));arf_set_mpn(out,fin,n,0);arf_mul_2exp_si(out,out,q+2-n*FLINT_BITS);}
    (void)err;return 0;
}
int exp_lowp_qsqrt_get_mpfr(mpfr_ptr out,const arf_t x){arf_get_mpfr(out,x,MPFR_RNDN);return 0;}
