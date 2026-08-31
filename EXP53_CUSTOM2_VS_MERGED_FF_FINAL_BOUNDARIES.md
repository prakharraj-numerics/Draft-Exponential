# EXP53 custom permanent 2-core vs merged FastFlow — final boundary sweep

## Scope

Final requested comparison of the same two complete stacks:

A. Custom permanent 2-core candidate
- immutable frozen EXP53 mathematical kernel
- caller pinned CPU0, helper pinned CPU2 on validated runner
- default temporal path
- explicit rare streaming/write-once NT path

B. Current merged FastFlow production stack
- persistent 2-worker FastFlow
- validated unrestricted process topology
- default temporal path
- explicit rare streaming/write-once NT path

Custom and FastFlow are always benchmarked in separate process lifetimes so spin-wait workers cannot interfere.

Frozen survivors and current production wrapper were not modified.

## Environment

Workflow run: `33412987407`

Benchmark/workflow commit: `40bdaa06fd8f2d34a377fcba6ca9ebc16b785eee`

Compiler: Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)

FastFlow pin: `d476f66ab924d8d122f54b4b90aee00ef979aea8`

Accepted exact Xeon shards:
- shard 6
- shard 54

CPU on both:
- Intel(R) Xeon(R) 6973P-C
- 2 physical cores / 4 logical CPUs
- CPU0/1 = physical core0; CPU2/3 = physical core1

Each stack was run in five independent process lifetimes; each process value was the median of 25 timed samples. Accuracy on both stacks and both policies: bitdiff=0 versus frozen serial.

## LOW sweep — two-shard midpoint, ns/input

| n | custom default | merged FF default | FF/custom | custom NT | merged FF NT | FF/custom NT |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.305885 | 1.676069 | 1.2835x | 1.820394 | 2.427362 | 1.3334x |
| 100 | 4.277069 | 10.135996 | 2.3698x | 5.369364 | 11.990337 | 2.2331x |
| 250 | 1.884077 | 4.261915 | 2.2621x | 2.224125 | 5.175782 | 2.3271x |
| 500 | 0.960978 | 2.235551 | 2.3263x | 1.328404 | 2.694276 | 2.0282x |
| 750 | 0.682901 | 1.519412 | 2.2249x | 0.974720 | 1.893884 | 1.9430x |
| 1K | 0.557525 | 1.174654 | 2.1069x | 0.771895 | 1.402956 | 1.8175x |
| 3K | 0.336882 | 0.560762 | 1.6646x | 0.401988 | 0.629645 | 1.5663x |

Notes:
- At n=50 the custom candidate intentionally falls back to the frozen serial kernel because fewer than two full 32-element blocks exist.
- From 100 upward the purpose-built two-way dispatcher is used where the 32-block split permits it.
- Custom wins every low point on both exact Xeon shards and in both output policies.

## HIGH sweep — two-shard midpoint, ns/input

| n | custom default | merged FF default | FF/custom | custom NT | merged FF NT | FF/custom NT |
|---:|---:|---:|---:|---:|---:|---:|
| 3M | 0.283769 | 0.286758 | 1.0105x | 0.293903 | 0.301683 | 1.0265x |
| 4M | 0.283878 | 0.286735 | 1.0101x | 0.293423 | 0.301245 | 1.0267x |
| 5M | 0.286122 | 0.295460 | 1.0326x | 0.293884 | 0.301313 | 1.0253x |
| 8M | 0.374760 | 0.389847 | 1.0403x | 0.297255 | 0.300606 | 1.0113x |

High-end replication detail:
- 3M default: custom wins on both shards (~0.6% and ~1.5%).
- 4M default: custom wins on both shards (~0.3% and ~1.7%).
- 5M default: custom wins on both shard medians, but magnitude differs strongly (~0.2% vs ~6.0%), so treat as noisy rather than a stable 3.3% gain.
- 8M default: custom wins on both shard medians (~1.5% and ~6.5%), but both methods show substantially more temporal variability at 8M.
- 3M-5M NT: custom wins on both shards by roughly 2-3%.
- 8M NT: shard 6 slightly favors FastFlow (~0.6%); shard 54 favors custom (~2.8%). This point is parity/noise, not a reliable custom or FastFlow win.

## Final verdict across all tested ranges

Combining the earlier 5K-2.5M sweeps with this final low/high boundary sweep:

- 50-3K: custom2 is dramatically better than merged FastFlow.
- 5K-200K: custom2 remains materially better.
- around 500K and above on the default temporal path: FastFlow catches up to near parity, but no tested point gives it a replicated clean win.
- 3M-8M default: effectively near-parity/noisy at large sizes, with custom still ahead in both accepted shard medians.
- streaming NT: custom remains better through 5M; 8M is effectively parity/noise.

Therefore on this exact 2-physical-core Xeon 6973P-C environment, there is no measured batch-size region in the completed sweep where merged FastFlow has a reproducible performance advantage over the purpose-built custom permanent 2-core runtime.

This report records benchmark evidence only; it does not promote or merge the custom candidate into production.
