# EXP53 final production vs Intel VML_HA — serial-through-3K / custom2 crossover matrix

## Production policy

- n <= 3000: frozen serial temporal only; no custom helper is constructed and no NT stores are used, even through the streaming API.
- n > 3000 default: frozen custom permanent 2-core temporal.
- n > 3000 explicit streaming/write-once: frozen custom permanent 2-core + NT.
- 1K < |x| < 5K extreme-input domain is deferred because the frozen vector scale construction was validated for [-100,100].

## Benchmark

- Run: `33416669809`
- Head: `a2b318f6dde03f9cce9df4d72bf936721b6854fa`
- Exact Xeon 6973P-C accepted shards: 13, 19, 27, 46
- Intel comparator: plain sequential `vmdExp(..., VML_HA)`; no custom runtime / FastFlow in Intel process.
- Domains: unit `0<|x|<1`; mid `1<|x|<100`; each exactly half positive / half negative, interleaved.
- Values below are median across the four accepted shard FINAL medians. `x = Intel / OURS`; x>1 means OURS is faster.
- Streaming uses an approximately 256 MiB rotating aligned output ring.

## LOW

| n | unit hot O/I ns | x | mid hot O/I ns | x | unit stream O/I ns | x | mid stream O/I ns | x |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.293/0.689 | 0.532x | 1.290/0.687 | 0.533x | 2.419/1.891 | 0.786x | 2.380/1.870 | 0.778x |
| 100 | 0.484/0.478 | 0.982x | 0.485/0.481 | 0.991x | 1.491/1.556 | 1.041x | 1.479/1.548 | 1.049x |
| 250 | 0.661/0.383 | 0.575x | 0.660/0.379 | 0.574x | 1.604/1.405 | 0.873x | 1.629/1.403 | 0.852x |
| 500 | 0.483/0.367 | 0.758x | 0.482/0.367 | 0.758x | 1.405/1.313 | 0.928x | 1.403/1.304 | 0.922x |
| 800 | 0.393/0.346 | 0.880x | 0.394/0.347 | 0.882x | 1.304/1.306 | 0.993x | 1.312/1.309 | 1.001x |
| 1,000 | 0.420/0.342 | 0.811x | 0.418/0.340 | 0.812x | 1.340/1.301 | 0.965x | 1.344/1.305 | 0.978x |
| 2,000 | 0.408/0.332 | 0.810x | 0.415/0.328 | 0.793x | 1.341/1.274 | 0.950x | 1.344/1.266 | 0.937x |
| 3,000 | 0.411/0.327 | 0.794x | 0.410/0.325 | 0.792x | 1.340/1.254 | 0.937x | 1.339/1.251 | 0.925x |

## MEDIUM + CROSSOVER

| n | unit hot O/I ns | x | mid hot O/I ns | x | unit stream O/I ns | x | mid stream O/I ns | x |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,500 | 0.303/0.327 | 1.080x | 0.306/0.328 | 1.073x | 1.797/1.181 | 0.650x | 1.831/1.096 | 0.603x |
| 4,000 | 0.280/0.328 | 1.171x | 0.279/0.328 | 1.173x | 1.730/1.210 | 0.699x | 1.710/1.217 | 0.707x |
| 4,500 | 0.283/0.327 | 1.155x | 0.281/0.326 | 1.162x | 1.801/1.178 | 0.651x | 1.772/1.187 | 0.665x |
| 5,000 | 0.269/0.326 | 1.215x | 0.267/0.326 | 1.225x | 1.716/1.179 | 0.683x | 1.710/1.194 | 0.688x |
| 8,000 | 0.240/0.324 | 1.352x | 0.235/0.323 | 1.376x | 1.657/1.167 | 0.694x | 1.659/1.185 | 0.714x |
| 15,000 | 0.223/0.322 | 1.444x | 0.221/0.321 | 1.454x | 1.589/1.192 | 0.747x | 1.591/1.184 | 0.740x |
| 25,000 | 0.210/0.321 | 1.534x | 0.208/0.321 | 1.539x | 1.583/1.165 | 0.741x | 1.570/1.183 | 0.750x |
| 35,000 | 0.211/0.320 | 1.517x | 0.207/0.321 | 1.550x | 1.561/1.176 | 0.750x | 1.571/1.191 | 0.756x |
| 50,000 | 0.204/0.320 | 1.567x | 0.203/0.326 | 1.614x | 1.568/1.168 | 0.745x | 1.565/1.173 | 0.749x |
| 65,000 | 0.203/0.320 | 1.579x | 0.204/0.322 | 1.582x | 1.616/1.160 | 0.714x | 1.621/1.173 | 0.719x |

## HIGH

| n | unit hot O/I ns | x | mid hot O/I ns | x | unit stream O/I ns | x | mid stream O/I ns | x |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200,000 | 0.199/0.507 | 2.498x | 0.203/0.504 | 2.487x | 0.204/0.867 | 4.206x | 0.206/0.849 | 4.167x |
| 500,000 | 0.272/0.541 | 1.991x | 0.271/0.541 | 1.993x | 0.231/0.894 | 3.881x | 0.241/0.895 | 3.729x |
| 1,000,000 | 0.272/0.540 | 1.983x | 0.273/0.542 | 1.986x | 0.287/0.888 | 3.089x | 0.288/0.875 | 3.008x |
| 1,500,000 | 0.272/0.540 | 1.985x | 0.273/0.542 | 1.982x | 0.287/0.876 | 3.055x | 0.288/0.895 | 3.114x |
| 2,500,000 | 0.272/0.540 | 1.988x | 0.272/0.540 | 1.988x | 0.496/0.867 | 1.748x | 0.514/0.877 | 1.708x |

## HIGHEST

| n | unit hot O/I ns | x | mid hot O/I ns | x | unit stream O/I ns | x | mid stream O/I ns | x |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,000,000 | 0.272/0.543 | 1.999x | 0.273/0.548 | 2.007x | 0.287/0.892 | 3.116x | 0.289/0.892 | 3.059x |
| 5,000,000 | 0.275/0.550 | 2.016x | 0.277/0.548 | 1.968x | 0.287/0.886 | 3.089x | 0.287/0.884 | 3.081x |
| 8,000,000 | 0.353/0.729 | 2.035x | 0.343/0.692 | 2.061x | 0.289/0.915 | 3.140x | 0.291/0.933 | 3.207x |

## Conclusions

- Hot/default temporal: Intel wins through 3K; custom2 wins already at 3.5K in both domains. Therefore the observed temporal crossover lies in `(3000,3500]`.
- At 3.5K, four-shard median speedup is about 1.080x for unit and 1.073x for mid; at 4K about 1.171x/1.173x; at 4.5K about 1.155x/1.162x.
- At 200K hot, OURS is about 2.50x faster; from 500K through 5M hot it is roughly 2x faster, with 8M more variable but still ~2x.
- Explicit streaming NT is not beneficial versus Intel at 3.5K-65K in this true-streaming ring. It becomes strongly favorable by 200K (~4.2x), remains ~3-4x across much of the large range, with an anomalous ~1.7x at 2.5M under this ring geometry.
- All accepted custom rows had zero cross-screen failures under the benchmark's <=2-ULP Intel cross-check; max observed difference was 2 ULP.
