# EXP53 permanent 2-core candidate vs merged FastFlow

## Scope

Candidate A:
- immutable frozen EXP53 mathematical kernel
- purpose-built permanent 2-core dispatcher
- caller pinned to CPU 0 (physical core 0)
- one permanent helper pinned to CPU 2 (physical core 1)
- caller computes first 32-aligned half; helper computes second half
- generation-counter launch + completion-counter rendezvous
- no scheduler, queue, stealing, task allocation, or third control thread
- default temporal path
- explicit rare streaming/write-once NT path

Current production B:
- `exp53_batch_production.hpp`
- persistent FastFlow, 2 workers, validated unrestricted process topology
- default temporal path
- explicit rare streaming/write-once NT path

The two runtime stacks are always benchmarked in separate process lifetimes so their spin-wait workers cannot interfere with one another.

Frozen survivor files were not modified.

## Environment

Workflow run: `33411916112`

Benchmark/workflow commit: `9b124bae3c4bc5cb659ab9e90368aec920a5bd02`

Compiler: Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)

FastFlow pin: `d476f66ab924d8d122f54b4b90aee00ef979aea8`

Exact CPU on accepted shards:
- Intel(R) Xeon(R) 6973P-C
- 2 physical cores / 4 logical CPUs
- CPU0/1 = physical core0; CPU2/3 = physical core1

Accepted exact Xeon shards:
- shard 9
- shard 61

Each stack is run in five independent process lifetimes. Each reported process value is itself the median of 25 timed samples. The workflow reports the median across the five process values.

Correctness on both exact shards:
- custom default bitdiff vs frozen serial = 0
- custom streaming NT bitdiff vs frozen serial = 0
- merged FF default bitdiff vs frozen serial = 0
- merged FF streaming NT bitdiff vs frozen serial = 0

## Shard 9 — median ns/input

| n | custom default | merged FF default | FF/custom | custom NT stream | merged FF NT stream | FF/custom stream |
|---:|---:|---:|---:|---:|---:|---:|
| 5K | 0.268017417 | 0.433328000 | 1.616790x | 0.317295083 | 0.475083250 | 1.497292x |
| 8K | 0.234716167 | 0.339746667 | 1.447479x | 0.274389583 | 0.380175000 | 1.385530x |
| 15K | 0.225730583 | 0.281066333 | 1.245141x | 0.246840667 | 0.304289250 | 1.232735x |
| 25K | 0.212157167 | 0.256046250 | 1.206871x | 0.228806917 | 0.274948333 | 1.201661x |
| 35K | 0.209842272 | 0.249026399 | 1.186731x | 0.224588972 | 0.269887469 | 1.201695x |
| 50K | 0.204879917 | 0.240291917 | 1.172843x | 0.220445583 | 0.259677167 | 1.177965x |
| 65K | 0.203089824 | 0.236971955 | 1.166833x | 0.215262420 | 0.254626763 | 1.182867x |

## Shard 61 — median ns/input

| n | custom default | merged FF default | FF/custom | custom NT stream | merged FF NT stream | FF/custom stream |
|---:|---:|---:|---:|---:|---:|---:|
| 5K | 0.262831000 | 0.425883333 | 1.620369x | 0.312711167 | 0.456998917 | 1.461409x |
| 8K | 0.234396417 | 0.338846667 | 1.445614x | 0.271730667 | 0.374972333 | 1.379941x |
| 15K | 0.219796250 | 0.281529917 | 1.280868x | 0.239823833 | 0.295991333 | 1.234203x |
| 25K | 0.206908917 | 0.247905750 | 1.198140x | 0.225730500 | 0.273913583 | 1.213454x |
| 35K | 0.207130744 | 0.239436926 | 1.155970x | 0.220410944 | 0.264922640 | 1.201949x |
| 50K | 0.201677833 | 0.234734083 | 1.163906x | 0.218202250 | 0.256432000 | 1.175203x |
| 65K | 0.198621154 | 0.230778446 | 1.161903x | 0.208769551 | 0.250355849 | 1.199197x |

## Two-shard consolidated midpoint

| n | custom default | merged FF default | FF/custom | custom NT stream | merged FF NT stream | FF/custom stream |
|---:|---:|---:|---:|---:|---:|---:|
| 5K | 0.265424208 | 0.429605666 | 1.6186x | 0.315003125 | 0.466041084 | 1.4795x |
| 8K | 0.234556292 | 0.339296667 | 1.4465x | 0.273060125 | 0.377573667 | 1.3827x |
| 15K | 0.222763417 | 0.281298125 | 1.2628x | 0.243332250 | 0.300140291 | 1.2335x |
| 25K | 0.209533042 | 0.251976000 | 1.2026x | 0.227268708 | 0.274430958 | 1.2075x |
| 35K | 0.208486508 | 0.244231663 | 1.1715x | 0.222499958 | 0.267405054 | 1.2018x |
| 50K | 0.203278875 | 0.237513000 | 1.1684x | 0.219323916 | 0.258054583 | 1.1766x |
| 65K | 0.200855489 | 0.233875200 | 1.1644x | 0.212015986 | 0.252491306 | 1.1909x |

## Verdict

The custom permanent 2-core runtime wins on both exact Xeon shards at every requested size and in both output policies.

The improvement is largest in the target low-batch regime:
- 5K default: ~1.62x vs merged FastFlow
- 8K default: ~1.45x
- 15K default: ~1.26x

The advantage remains material even at 25K-65K (~1.16x-1.20x default), so the observed crossover back to FastFlow is beyond the tested 65K range, if it exists at all on this 2-core runner.

The rare streaming/write-once NT path also benefits from the custom dispatcher by roughly 1.18x-1.48x across the sweep.

This file records a candidate result only. The current merged production path and frozen survivors are unchanged.
