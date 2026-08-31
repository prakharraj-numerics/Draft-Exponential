# EXP53 frozen kernel under oneMKL VML mode environments

Validation/benchmark run: GitHub Actions `33375439236`, shard 31
CPU: Intel Xeon 6973P-C
Compiler: Intel ICX, `-O3 -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -DNDEBUG`
Kernel: `exp53_n2_vmstyle_u4_0381_frozen`
Mode wrappers: `exp53_n2_vmstyle_u4_vml_modes.c`

The mathematical kernel is identical in all three variants. `vmlSetMode(VML_HA)`, `vmlSetMode(VML_LA)`, or `vmlSetMode(VML_EP)` is selected once before the benchmark block, outside timed kernel calls.

## Accuracy — 200,000 mixed-domain inputs

| Variant | max ULP |
|---|---:|
| OURS + VML_HA environment | 1 |
| OURS + VML_LA environment | 1 |
| OURS + VML_EP environment | 1 |

## Timing — ns/input

| Variant | n=12,288 | n=65,536 |
|---|---:|---:|
| OURS + VML_HA environment | 0.384133953 | 0.385450317 |
| OURS + VML_LA environment | 0.384258740 | 0.385348628 |
| OURS + VML_EP environment | 0.386272873 | 0.385115878 |

## Interpretation

The differences are benchmark noise, not an HA/LA/EP speed hierarchy for this kernel. oneMKL VML accuracy modes affect oneMKL VML math routines. This frozen kernel is hand-written AVX-512 math and does not call a VML exponential routine, so changing VML mode does not change its arithmetic DAG or approximation accuracy.
