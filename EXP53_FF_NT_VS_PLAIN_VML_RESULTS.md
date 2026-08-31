# EXP53 — corrected side-by-side: FastFlow temporal vs FastFlow NT vs plain Intel VML_HA

## Verdict

The earlier pinned `taskset -c 0,2` rerun was not representative of the frozen FastFlow survivor. It changed FastFlow's native affinity/topology behavior and degraded OURS+FastFlow from the previously validated ~0.27 ns/input regime to ~0.49 ns/input around n=1M.

The frozen FastFlow validation workflow (`exp53-fastflow-batch.yml`, run `33399802068`) did **not** impose an external process-wide `taskset`. Therefore the corrected side-by-side rerun restores the original unrestricted FastFlow process topology.

Corrected workflow run: `33409159099`
Commit: `00152b63a0a671eea7e6ffd6c3b04cc57beb03a8`
Benchmark: `bench_exp53_ff_nt_vs_plain_vml.cpp`
FastFlow pinned: `d476f66ab924d8d122f54b4b90aee00ef979aea8`
Compiler: Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)
Comparator: plain `vmdExp(..., VML_HA)` linked with `-lmkl_sequential`; Intel is **not** wrapped in FastFlow.
CPU on both accepted artifacts: `Intel(R) Xeon(R) 6973P-C`, 2 physical cores / 4 logical CPUs exposed.
Accepted exact-Xeon shards: 21 and 31.
NT assembly: 4 `vmovntpd` stores in the 32-value body.
Accuracy: bitdiff temporal vs NT = 0, maxULP = 1, gt1 = 0 on both shards.

## Normal repeated-output / cache-hot benchmark

This is the benchmark regime comparable to the frozen FastFlow survivor: each method repeatedly writes its own same output buffer. Median ns/input.

### Shard 21

| n | OURS+FF temporal | OURS+FF+NT | plain Intel VML_HA | Intel / OURS+FF | Intel / FF+NT |
|---:|---:|---:|---:|---:|---:|
| 200,000 | **0.234704** | 0.236581 | 0.513728 | **2.189x** | 2.171x |
| 500,000 | 0.274091 | **0.250953** | 0.542337 | 1.979x | **2.161x** |
| 800,000 | **0.274650** | 0.294281 | 0.519784 | **1.893x** | 1.766x |
| 1,000,000 | **0.289106** | 0.297478 | 0.529038 | **1.830x** | 1.778x |
| 1,200,000 | **0.275356** | 0.296203 | 0.544066 | **1.976x** | 1.837x |
| 1,500,000 | **0.275752** | 0.296348 | 0.549707 | **1.993x** | 1.855x |

### Shard 31

| n | OURS+FF temporal | OURS+FF+NT | plain Intel VML_HA | Intel / OURS+FF | Intel / FF+NT |
|---:|---:|---:|---:|---:|---:|
| 200,000 | **0.232990** | 0.236424 | 0.558672 | **2.398x** | 2.363x |
| 500,000 | 0.277348 | **0.240060** | 0.558225 | 2.013x | **2.325x** |
| 800,000 | **0.276858** | 0.290716 | 0.531775 | **1.921x** | 1.829x |
| 1,000,000 | **0.284184** | 0.296318 | 0.537816 | **1.892x** | 1.815x |
| 1,200,000 | **0.278150** | 0.298422 | 0.556526 | **2.001x** | 1.865x |
| 1,500,000 | **0.277652** | 0.296611 | 0.559916 | **2.017x** | 1.888x |

### Normal-regime conclusion

The frozen OURS+FastFlow survivor is confirmed. Across 800K-1.5M it is about 1.83x-2.02x faster than plain sequential Intel VML_HA on the two accepted exact Xeon shards. At 200K it is 2.19x-2.40x faster. At 500K, NT happens to beat temporal in both shards, but this is not a monotonic size rule: temporal retakes the lead from 800K through 1.5M in the cache-hot regime.

At n=1,000,000 specifically:
- shard 21: OURS+FF 0.289106, FF+NT 0.297478, Intel 0.529038 ns/input.
- shard 31: OURS+FF 0.284184, FF+NT 0.296318, Intel 0.537816 ns/input.
Thus plain Intel / OURS+FF = 1.830x and 1.892x respectively.

## True streaming / long output reuse distance (~256 MiB rotating destinations)

Each method gets its own independent rotating destination ring, so methods cannot pre-warm each other's output cache lines. Median ns/input.

### Shard 21

| n | OURS+FF temporal | OURS+FF+NT | plain Intel VML_HA | Intel / OURS+FF | Intel / FF+NT | temporal / NT |
|---:|---:|---:|---:|---:|---:|---:|
| 200,000 | 0.450381 | **0.238401** | 0.931443 | 2.068x | **3.907x** | 1.889x |
| 500,000 | 0.485592 | **0.253307** | 0.924523 | 1.904x | **3.650x** | 1.917x |
| 800,000 | 0.468278 | **0.295760** | 0.911038 | 1.946x | **3.080x** | 1.583x |
| 1,000,000 | 0.477024 | **0.298575** | 0.935283 | 1.961x | **3.132x** | 1.598x |
| 1,200,000 | 0.482608 | **0.296207** | 0.927784 | 1.922x | **3.132x** | 1.629x |
| 1,500,000 | 0.476565 | **0.298116** | 0.914612 | 1.919x | **3.068x** | 1.599x |

### Shard 31

| n | OURS+FF temporal | OURS+FF+NT | plain Intel VML_HA | Intel / OURS+FF | Intel / FF+NT | temporal / NT |
|---:|---:|---:|---:|---:|---:|---:|
| 200,000 | 0.502684 | **0.239453** | 0.965126 | 1.920x | **4.031x** | 2.099x |
| 500,000 | 0.506004 | **0.242249** | 0.974158 | 1.925x | **4.021x** | 2.089x |
| 800,000 | 0.506463 | **0.293257** | 0.971961 | 1.919x | **3.314x** | 1.727x |
| 1,000,000 | 0.507267 | **0.297951** | 0.969733 | 1.912x | **3.255x** | 1.703x |
| 1,200,000 | 0.506486 | **0.298791** | 0.981098 | 1.937x | **3.284x** | 1.695x |
| 1,500,000 | 0.502844 | **0.297950** | 0.960858 | 1.911x | **3.225x** | 1.688x |

### Streaming conclusion

For genuinely cold/write-once destinations with long reuse distance, non-temporal stores are a large real win: FF+NT is ~1.58x-2.10x faster than temporal FF over these sizes, and ~3.07x-4.03x faster than plain sequential VML_HA.

## Final correction

1. The frozen FastFlow survivor remains valid and fast. The ~0.49 ns/input numbers from the externally pinned rerun were an affinity/topology artifact and must not replace the frozen checkpoint.
2. There is no universal `n >= X => NT` rule from these data alone. The decisive factor is destination cache reuse:
   - cache-hot/reused output: temporal FF is generally best from 800K-1.5M; 500K is an observed NT exception on both accepted shards.
   - cold/write-once/long-reuse output: FF+NT wins strongly already at 200K.
3. For the competitive finished-stack comparison requested by the user, Intel remains plain sequential VML_HA. It is not given FastFlow.
