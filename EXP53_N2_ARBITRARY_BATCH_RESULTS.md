# EXP53 n=2 arbitrary-batch / streaming-pipeline benchmark

Run: 33388774509
CPU: Intel(R) Xeon(R) 6973P-C
Compiler: Intel oneAPI ICX 2026.1.1 (2026.1.1.20260724)
Eligible benchmark shards: 13 and 15

## Static candidate check
`exp53_n2_u4_masktail`: 146 instructions, 2 gathers, 0 stack-vector moves in the isolated disassembly.

## Accuracy
Both `U4_MASKTAIL` and `DYNAMIC`:
- 200,000 mixed-domain inputs: maxULP=1, gt1=0, bitdiff_vs_frozen=0
- exhaustive short-length sweep n=1..1025: maxULP=1, gt1=0

## Clean-shard timing summary
Shard 15 is used as the cleaner large-array result because shard 13 showed a noisy 65,536-point region.

ns/input:

| n | FROZEN | PIPE3 | U4_MASKTAIL | DYNAMIC | VML_HA |
|---:|---:|---:|---:|---:|---:|
| 31 | 2.573395 | 0.468873 | 0.472705 | 0.469264 | 1.073856 |
| 32 | 0.398535 | 0.433037 | 0.412440 | 0.433647 | 1.010165 |
| 63 | 1.606867 | 0.440179 | 0.474966 | 0.495099 | 0.609935 |
| 64 | 0.409509 | 0.428747 | 0.401113 | 0.426352 | 0.610109 |
| 100 | 0.479986 | 0.450877 | 0.437370 | 0.451381 | 0.479000 |
| 253 | 0.674607 | 0.406633 | 0.418169 | 0.407768 | 0.375375 |
| 511 | 0.533778 | 0.400553 | 0.405751 | 0.400785 | 0.359716 |
| 512 | 0.392003 | 0.399495 | 0.404887 | 0.408498 | 0.362257 |
| 513 | 0.397892 | 0.405562 | 0.403774 | 0.407579 | 0.368449 |
| 10079 | 0.392798 | 0.399906 | 0.385788 | 0.386120 | 0.322484 |
| 12288 | 0.385922 | 0.401109 | 0.385827 | 0.386806 | 0.329687 |
| 65536 | 0.385089 | 0.399413 | 0.384765 | 0.387063 | 0.324608 |

Shard 13 cross-check:
- n=10079: FROZEN 0.392137, PIPE3 0.399370, U4_MASKTAIL 0.385420, DYNAMIC 0.384686, VML_HA 0.322224
- n=12288: FROZEN 0.385202, PIPE3 0.398982, U4_MASKTAIL 0.384364, DYNAMIC 0.385547, VML_HA 0.322164
- n=65536: FROZEN 0.400993, PIPE3 0.417618, U4_MASKTAIL 0.400819, DYNAMIC 0.401731, VML_HA 0.348956 (noisy region)

## Verdict
The cross-vector assembly-line idea is real and useful for irregular/small batches, especially where the old frozen remainder path falls into scalar work. `PIPE3` dramatically improves lengths like 31, 63, 253 and 511 versus the frozen arbitrary-size behavior. However, for long steady arrays, explicit one-vector rolling `PIPE3` is slower than the existing u4 frozen scheduler. The best production-shaped result from this attack is therefore hybrid: retain the proven 32-input u4 hot body and use faithful AVX-512 vector/masked-tail handling for the remainder. `U4_MASKTAIL` is bit-identical to frozen on the 200k screen, has no >1 ULP cases in the short sweep, and is essentially neutral/slightly faster on long arrays while greatly improving irregular tails.

Do not replace the frozen baseline yet; the dynamic threshold policy is not fully tuned. The next useful attack is threshold/dispatch tuning across arbitrary n, using PIPE3 only where it wins and U4_MASKTAIL otherwise.