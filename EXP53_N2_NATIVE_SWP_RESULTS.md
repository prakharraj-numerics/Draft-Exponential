# EXP53 faithful n=2 — compiler-native software pipelining verdict

## Scope

Frozen baseline `exp53_n2_vmstyle_u4_0381_frozen` was not modified.

Target CPU: Intel Xeon 6973P-C.
Compiler: Intel oneAPI ICX 2026.1.1 (`2026.1.1.20260724`).
Workflow run: `33390775093`.
Accepted exact-Xeon shard: 10.

All candidates preserve the same faithful n=2 arithmetic, TAB128 anchors, residual reduction, Q4 Horner order, ER-low correction, and fused exponent scaling.

## Compiler controls tested

ICX 2026.1.1 rejects upstream Clang's newer `#pragma clang loop pipeline(enable)` syntax. It accepts:

- global LLVM machine-pipeliner switch `-mllvm -enable-pipeliner=true`
- `#pragma clang loop pipeline(disable)`
- `#pragma clang loop pipeline_initiation_interval(N)` for N = 1, 2, 4, 8
- `#pragma clang loop interleave_count(2)` and `(4)`

Candidates tested:

- `NATIVE_DEFAULT`
- `SWP_DISABLE`
- `SWP_II1`
- `SWP_II2`
- `SWP_II4`
- `SWP_II8`
- `INTERLEAVE2`
- `INTERLEAVE4`

## Assembly verdict

For every candidate under pipeliner-on:

- 146 hot-function instructions
- 4 gather sites
- 40 FMA/FNMA instructions
- 30 distinct ZMM registers
- 0 vector stack spills

After removing objdump's object-file header, `NATIVE_DEFAULT`, all II variants, and both interleave variants have the same normalized instruction stream. Enabling/disabling the LLVM machine pipeliner also leaves the semantic instruction stream unchanged. The earlier workflow field `pipeliner_on_off_same=0` was a normalization artifact caused only by hashing `/tmp/native_on.o` versus `/tmp/native_off.o` in the objdump header.

`remarks_on.txt` is empty: ICX emitted no machine-pipeliner success/missed/analysis remark for this loop.

Therefore the compiler-native software-pipelining controls do **not** create a modulo-scheduled / cross-iteration pipeline for this intrinsic-heavy AVX-512 loop on this ICX/x86 target.

## Accuracy

On 200,000 mixed-domain inputs every candidate:

- max ULP = 1
- >1 ULP = 0
- bit differences vs frozen = 0

## Timing, exact Xeon shard 10

ns/input:

| Candidate | n=10,079 | n=12,288 | n=65,536 |
|---|---:|---:|---:|
| Frozen | 0.392161240 | 0.383709703 | 0.387823166 |
| Native default | 0.391696438 | 0.383524786 | 0.387551931 |
| SWP disable | 0.391748861 | 0.383413810 | 0.387347832 |
| SWP II1 | 0.392081443 | 0.384438938 | 0.389478836 |
| SWP II2 | 0.393657684 | 0.383613585 | 0.388531797 |
| SWP II4 | 0.392301581 | 0.381477121 | 0.384349841 |
| SWP II8 | 0.392004415 | 0.383861637 | 0.388277110 |
| Interleave2 | 0.393764514 | 0.384016051 | 0.387758400 |
| Interleave4 | 0.391958025 | 0.384451253 | 0.389314880 |
| Intel VML_HA | 0.321997807 | 0.321661754 | 0.328611720 |

The apparent II4 advantage at 12,288 and 65,536 is not attributable to a different generated schedule: II4 and the other candidate bodies are instruction-for-instruction identical. It is therefore benchmark/layout/frequency noise, not a software-pipelining win.

## Final verdict

Compiler-native LLVM/ICX software pipelining is exhausted for this current loop shape and produces no real transformation. Do not replace the frozen baseline with any native-SWP candidate.

If we want true assembly-plant overlap across successive vectors/blocks, the next meaningful route is a custom hand-scheduled pipeline (or a materially restructured loop that exposes a schedulable recurrence to the compiler), while preserving the faithful n=2 formula.
