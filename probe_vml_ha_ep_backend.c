#include <mkl.h>
#include <mkl_vml.h>
#include <stdio.h>
#include <stdlib.h>

int main(void){
    enum {N=65536};
    double *x=(double*)aligned_alloc(64,(size_t)N*sizeof(double));
    double *y=(double*)aligned_alloc(64,(size_t)N*sizeof(double));
    if(!x||!y) return 2;
    for(int i=0;i<N;i++) x[i] = -20.0 + 40.0*(double)i/(double)(N-1);
    mkl_set_num_threads_local(1);
    vmdExp(N,x,y,VML_HA);
    volatile double s=y[123];
    vmdExp(N,x,y,VML_EP);
    s += y[456];
    printf("sink=%.17g\n",(double)s);
    free(x); free(y);
    return 0;
}
