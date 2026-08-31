# EXP53 NT midrange pinned benchmark

## Scope

Dedicated remeasurement requested for exact batch sizes 7K, 12K, 20K, 35K and 65K after suspicious non-monotonic 8K/32K rows in the broader cache sweep.

Frozen files were not modified.

Methods compared through the same persistent two-worker FastFlow pool:
- OURS + FastFlow + NT: `exp53_n2_vmstyle_u4_0381_nt_sfence`
- OURS + FastFlow temporal control: `exp53_n2_vmstyle_u4_0381_frozen`
- Intel oneMKL `vmdExp(..., VML_HA)` linked sequentially inside each worker

Workflow run: `33406031491`
Workflow head: `01c76e8f3cef7dea6268dfafe0dd7ce0dc3aef08`

Exact Xeon 6973P-C jobs used here:
- shard 4: job `99533724518`
- shard 34: job `99533725059`

The workflow pins the process to logical CPUs `0,2`, which are the two different physical cores exposed on these runners. FastFlow uses one persistent worker per physical core. Each method is sampled 15 times with rotating method order. Both same-output hot and 8-output rotating-ring modes are measured.

Assembly audit: 4 `vmovntpd` instructions emitted in the NT 32-value body.

Accuracy on both accepted jobs: bitdiff NT vs frozen = 0, maxULP = 1, gt1 = 0.

## Hot-output median ns/input

### Shard 4

| n | OURS+FF+NT | Intel VML_HA+FF | OURS temporal+FF | Intel / NT |
|---:|---:|---:|---:|---:|
| 7,000 | 0.661474 | 0.553423 | 0.638660 | 0.8367x |
| 12,000 | 0.538064 | 0.429540 | 0.511112 | 0.7983x |
| 20,000 | 0.534870 | 0.433436 | 0.532322 | 0.8104x |
| 35,000 | 0.498661 | 0.402611 | 0.499999 | 0.8074x |
| 65,000 | 0.480487 | 0.380204 | 0.483519 | 0.7913x |

### Shard 34

| n | OURS+FF+NT | Intel VML_HA+FF | OURS temporal+FF | Intel / NT |
|---:|---:|---:|---:|---:|
| 7,000 | 0.620671 | 0.515725 | 0.602935 | 0.8309x |
| 12,000 | 0.563367 | 0.464591 | 0.555825 | 0.8247x |
| 20,000 | 0.491868 | 0.398786 | 0.489009 | 0.8108x |
| 35,000 | 0.463891 | 0.375768 | 0.464321 | 0.8100x |
| 65,000 | 0.437388 | 0.347820 | 0.438509 | 0.7952x |

## Rotating-ring confirmation

Shard 4, NT / VML / temporal:
- 7K: 0.604771 / 0.501580 / 0.576922
- 12K: 0.585799 / 0.475625 / 0.573466
- 20K: 0.541890 / 0.436876 / 0.532983
- 35K: 0.512218 / 0.409246 / 0.510062
- 65K: 0.502573 / 0.396461 / 0.490397

Shard 34, NT / VML / temporal:
- 7K: 0.633288 / 0.528665 / 0.612574
- 12K: 0.546717 / 0.443494 / 0.528963
- 20K: 0.511383 / 0.411653 / 0.502439
- 35K: 0.484843 / 0.387246 / 0.481846
- 65K: 0.456609 / 0.362188 / 0.449050

## Verdict

The previous 8K/32K oscillation was largely a scheduling/placement artifact. With the process pinned to the two physical cores, the midrange trend is much smoother.

For 7K-65K, non-temporal stores do **not** rescue performance:
- VML_HA remains about 19-21% faster than OURS+FastFlow+NT in the hot-output measurements.
- NT is slightly slower than, or effectively tied with, the frozen temporal FastFlow path.

Therefore the NT path should remain a large-stream/write-once optimization only. Do not dispatch to NT in this 7K-65K midrange based on these results.
