# EXP53 permanent 2-core candidate vs merged FastFlow — large batches

## Scope

Candidate A:
- immutable frozen EXP53 mathematical kernel
- purpose-built permanent 2-core dispatcher
- caller pinned to CPU0, helper pinned to CPU2 (one logical CPU per physical core)
- default temporal path
- explicit rare streaming/write-once NT path

Current production B:
- `exp53_batch_production.hpp`
- persistent FastFlow, 2 workers, validated unrestricted process topology
- default temporal path
- explicit rare streaming/write-once NT path

The two stacks are benchmarked in separate process lifetimes; their spin-wait workers are never alive together.

Frozen survivors and current merged production wrapper were not modified.

## Environment

Workflow run: `33412390970`

Benchmark/workflow commit: `f82d2f12b99fa3e73b3b353b44193405a38d95a8`

Compiler: Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)

FastFlow pin: `d476f66ab924d8d122f54b4b90aee00ef979aea8`

Exact CPU:
- Intel(R) Xeon(R) 6973P-C
- 2 physical cores / 4 logical CPUs
- CPU0/1 = physical core0; CPU2/3 = physical core1

Accepted exact Xeon shards:
- shard 18
- shard 20

Each stack is run in five independent process lifetimes. Each process value is the median of 25 timed samples; the reported shard result is the median of those five process values.

Correctness on accepted shards:
- custom default bitdiff vs frozen serial = 0
- custom streaming NT bitdiff vs frozen serial = 0
- merged FF default bitdiff vs frozen serial = 0
- merged FF streaming NT bitdiff vs frozen serial = 0

## Shard 18 — median ns/input

| n | custom default | merged FF default | FF/custom | custom NT stream | merged FF NT stream | FF/custom stream |
|---:|---:|---:|---:|---:|---:|---:|
| 200K | 0.200136167 | 0.235437000 | 1.176384x | 0.205211083 | 0.242548417 | 1.181946x |
| 500K | 0.273556167 | 0.276037750 | 1.009072x | 0.229725583 | 0.260720083 | 1.134920x |
| 1M | 0.274203750 | 0.276999917 | 1.010197x | 0.287686750 | 0.293660167 | 1.020764x |
| 1.5M | 0.274225667 | 0.276167750 | 1.007082x | 0.287413833 | 0.293397417 | 1.020819x |
| 2.5M | 0.275032600 | 0.275813200 | 1.002838x | 0.287727650 | 0.293880500 | 1.021384x |

## Shard 20 — median ns/input

| n | custom default | merged FF default | FF/custom | custom NT stream | merged FF NT stream | FF/custom stream |
|---:|---:|---:|---:|---:|---:|---:|
| 200K | 0.201909500 | 0.227248667 | 1.125498x | 0.201170083 | 0.235467917 | 1.170492x |
| 500K | 0.270756833 | 0.274050667 | 1.012165x | 0.225136083 | 0.245440167 | 1.090186x |
| 1M | 0.271404750 | 0.274059333 | 1.009781x | 0.284684917 | 0.292008667 | 1.025726x |
| 1.5M | 0.270899917 | 0.273208250 | 1.008521x | 0.284479833 | 0.290946500 | 1.022732x |
| 2.5M | 0.270753550 | 0.272693850 | 1.007166x | 0.284027500 | 0.290928600 | 1.024297x |

## Two-shard consolidated midpoint

| n | custom default | merged FF default | FF/custom | custom NT stream | merged FF NT stream | FF/custom stream |
|---:|---:|---:|---:|---:|---:|---:|
| 200K | 0.201023 | 0.231343 | 1.1508x | 0.203191 | 0.239008 | 1.1763x |
| 500K | 0.272157 | 0.275044 | 1.0106x | 0.227431 | 0.253080 | 1.1128x |
| 1M | 0.272804 | 0.275530 | 1.0100x | 0.286186 | 0.292834 | 1.0232x |
| 1.5M | 0.272563 | 0.274688 | 1.0078x | 0.285947 | 0.292172 | 1.0218x |
| 2.5M | 0.272893 | 0.274254 | 1.0050x | 0.285878 | 0.292405 | 1.0228x |

## Verdict

- At 200K, custom2 remains materially faster: about 15% on the default path and about 18% on the NT streaming path.
- At 500K, default temporal is essentially parity (~1.1% custom advantage), while custom NT still has about an 11% advantage.
- At 1M, 1.5M and 2.5M, default custom2 and merged FastFlow are effectively tied, with custom2 ahead by only about 0.5-1.0%.
- The large streaming/NT path retains a small but consistent custom2 advantage of about 2.2-2.3% at 1M-2.5M.
- FastFlow does not beat custom2 at any requested size on either accepted Xeon shard, but its default temporal path has effectively caught up by 500K.

This is a candidate comparison only. No production promotion is performed by this report.
