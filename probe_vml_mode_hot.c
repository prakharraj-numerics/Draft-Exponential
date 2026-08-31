#include <mkl.h>
#include <mkl_vml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char**argv){
    enum {N=65536, R=40000};
    int mode = (argc>1 && strcmp(argv[1],"EP")==0) ? VML_EP : VML_HA;
    double *x=(double*)aligned_alloc(64,(size_t)N*sizeof(double));
    double *y=(double*)aligned_alloc(64,(size_t)N*sizeof(double));
    if(!x||!y) return 2;
    for(int i=0;i<N;i++) x[i] = -20.0 + 40.0*(double)i/(double)(N-1);
    mkl_set_num_threads_local(1);
    for(int w=0;w<100;w++) vmdExp(N,x,y,mode);
    volatile double s=0;
    for(int r=0;r<R;r++){ vmdExp(N,x,y,mode); s+=y[(r*97u)&(N-1)]; }
    printf("mode=%s sink=%.17g\n", mode==VML_EP?"EP":"HA", (double)s);
    free(x); free(y); return 0;
}
