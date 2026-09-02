#include <stdint.h>
#include <stdio.h>

#define SSC(tag) __asm__ __volatile__("movl %0, %%ebx; .byte 0x64, 0x67, 0x90" :: "i"(tag) : "%ebx", "memory")

int main(void) {
    volatile double s = 0.0;
    SSC(0xFACE);
    for (volatile uint64_t i = 0; i < 100000ULL; ++i) s += (double)i * 0.1;
    SSC(0xDEAD);
    printf("probe=%f\n", (double)s);
    return 0;
}
