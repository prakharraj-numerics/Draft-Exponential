# EXP53 FastFlow frozen checkpoint

Frozen survivor:
- `exp53_fastflow_batch_2core_frozen.hpp`
- commit `5bc33e014162b8599a61aa915341f1fb4fd5e010`

Underlying mathematical kernel remains immutable:
- `exp53_n2_vmstyle_u4_0381_frozen.c`

FastFlow validation dependency:
- `fastflow/fastflow`
- commit `d476f66ab924d8d122f54b4b90aee00ef979aea8`

Validation workflow run:
- `33399802068`

Exact benchmark CPU:
- Intel(R) Xeon(R) 6973P-C
- 2 physical cores exposed, 2 SMT threads/core

Frozen execution policy:
- persistent FastFlow workers
- spin-wait + spin barrier
- one worker per physical core on the validated runner
- 32-element-aligned balanced static chunks as the default winning schedule
- large-block dynamic mode retained as an explicitly tuned option
- no mathematical change to the faithful n=2 kernel

Correctness:
- 200,000 mixed-domain inputs
- static bitdiff vs frozen serial = 0
- dynamic bitdiff vs frozen serial = 0
- maxULP = 1
- >1 ULP = 0

Primary clean-shard median results, ns/input:

| n | serial OURS | frozen OURS+FastFlow | speedup |
|---:|---:|---:|---:|
| 10,079 | 0.394928549 | 0.359561484 | 1.098x |
| 12,288 | 0.391135317 | 0.334285857 | 1.170x |
| 65,536 | 0.384712594 | 0.258516656 | 1.488x |
| 262,144 | 0.527478536 | 0.254524740 | 2.072x |
| 1,000,000 | 0.541838750 | 0.274047000 | 1.977x |

Fair FastFlow-wrapped oneMKL VML_HA at n=1,000,000 on the same shard:
- OURS+FastFlow = 0.274047000 ns/input
- VML_HA+FastFlow = 0.271398500 ns/input

Status:

**Frozen. Do not modify `exp53_fastflow_batch_2core_frozen.hpp`.**

Future experiments must use new candidate files and benchmark against this checkpoint and the underlying single-thread frozen kernel.
