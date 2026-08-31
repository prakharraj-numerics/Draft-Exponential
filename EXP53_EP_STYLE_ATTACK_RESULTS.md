# EXP53 — oneMKL VML EP structural attack

## Intel AVX-512 VML inspection

Backend actually loaded for Xeon 6973P-C:
`libmkl_vml_avx512.so.3`

Exact exported double-exp kernels inspected:
- HA: `mkl_vml_kernel_dExp_Z0HAynn`
- EP: `mkl_vml_kernel_dExp_Z0EPnnn`

Static census of the same AVX-512 binary:

| path | size | instructions | gathers | permutes | FMA | mul | branches/calls | distinct ZMM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 0x138b | 1098 | 0 | 18 | 33 | 0 | 142 | 28 |
| EP | 0x0bb7 | 655 | 0 | 9 | 30 | 5 | 70 | 25 |

Structural observation only: EP is substantially more compact, has no gather, uses fewer permutes/branches/registers, and its hot vector sequence visibly operates on four independent ZMM streams. No proprietary constants or numerical formulas were copied.

## Exact EP-inspired attack

Run: `33378233719`
Valid Intel Xeon 6973P-C shard: `23`
Job: `99444449105`

Candidate source: `exp53_n2_epstyle_exactperm_attack.c`

Both candidates preserve the frozen n=2 numerical DAG and exact TAB128 values. TAB128 lookup was made gather-free by partitioning the exact 128 doubles into eight 16-value banks and using AVX-512 `permutex2var` plus masks.

### Static candidate census

| kernel | instructions | gathers | permutes | distinct ZMM |
|---|---:|---:|---:|---:|
| frozen | 181 | 4 | 0 | 30 |
| EP_PERM8_EARLY | 359 | 0 | 32 | 32 |
| EP_PERM8_AFTERREDUCE | 361 | 0 | 32 | 32 |

### Accuracy screen — 200,000 mixed-domain inputs

| kernel | max ULP | bit differences vs frozen |
|---|---:|---:|
| frozen | 1 | 0 |
| EP_PERM8_EARLY | 1 | 0 |
| EP_PERM8_AFTERREDUCE | 1 | 0 |
| Intel VML HA | 1 | n/a |
| Intel VML EP | 1,371,025 | n/a |

### Xeon timing, ns/input

| kernel | n=12,288 | n=65,536 |
|---|---:|---:|
| frozen | 0.384285590 | 0.385949483 |
| EP_PERM8_EARLY | 0.749617155 | 0.750459790 |
| EP_PERM8_AFTERREDUCE | 0.755461149 | 0.755186159 |
| Intel VML HA | 0.324152614 | 0.319917224 |
| Intel VML EP | 0.231413499 | 0.228595201 |

## Verdict

The first literal no-gather transplant is rejected. It is numerically perfect (bit-identical to frozen) but roughly 1.95x slower because exact 128-entry selection expands to 32 vector permutes per 32 inputs and pushes register use to all 32 ZMM registers.

The useful Intel EP lesson remains structural: a gather-free path only wins if its table/reconstruction representation itself is intrinsically compact. Replacing one gather with many exact selection permutes is not the answer. The current frozen baseline remains unchanged.
