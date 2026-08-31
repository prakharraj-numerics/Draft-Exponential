# EXP53 faithful n=2 spine under Intel SVML-high

Date: 2026-08-31
Workflow run: `33385508683`
CPU: Intel Xeon 6973P-C
Compiler: Intel oneAPI ICX 2026.1.1 (2026.1.1.20260724)

## What was tested

The mathematical kernel was not changed. The frozen survivor remained:

- `exp53_n2_vmstyle_u4_0381_frozen`
- faithful n=2 spine `e^r = 1 + r Q4(r)^2`
- same v128 reduction
- same degree-4 Q polynomial and Horner order
- same ER-low correction
- same TAB128 values and fused table/exponent reconstruction

Two builds of the same source were compared:

- BASE: `-O3 -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -DNDEBUG -qopenmp-simd`
- SVML-HIGH: BASE plus `-fimf-use-svml=true -fimf-precision=high`

A separate compiler-vectorized `exp()` loop was included only as a positive control to verify that SVML was genuinely selected. It is not part of our formula kernel.

## SVML activation check

The SVML-HIGH control binary contains and calls the AVX-512 high-accuracy SVML exponential path:

- `__svml_exp8_ha_z0`
- `__svml_exp8_ha_mask_z0`

The BASE control instead calls scalar `exp()`.

For the frozen n=2 function, the apparent static count difference (181 instructions BASE versus 175 SVML-HIGH) is only alignment/NOP/address-layout noise. After removing alignment NOPs and normalizing branch addresses, both versions contain the same 138 semantic instructions. The hot 32-input AVX-512 path is unchanged: same four gathers and same 40 FMA/FNMA instructions.

## Accuracy screen

200,000 mixed-domain inputs, `expl(long double)` reference screen:

| build | candidate | max ULP | >1 ULP |
|---|---|---:|---:|
| BASE | OURS_N2 | 1 | 0 |
| SVML-HIGH | OURS_N2 | 1 | 0 |
| SVML-HIGH | compiler SVML exp control | 1 | 0 |
| SVML-HIGH | Intel VML_HA | 1 | 0 |

The OURS_N2 outputs were bit-identical to the frozen reference on this screen.

## Timing — clean Xeon shard 6

ns/input, best-of-nine benchmark:

| candidate | n=12,288 BASE | n=12,288 SVML-HIGH | n=65,536 BASE | n=65,536 SVML-HIGH |
|---|---:|---:|---:|---:|
| OURS_N2 | 0.384605750 | 0.385099292 | 0.387447516 | 0.390169928 |
| compiler `exp()` control | 2.657208551 | 0.397811867 | 2.656178995 | 0.431691180 |
| Intel VML_HA | 0.321589880 | 0.321368652 | 0.320077336 | 0.335505096 |

Shard 23 independently reproduced the n=12,288 result:

- OURS_N2 BASE: `0.384882204 ns/input`
- OURS_N2 SVML-HIGH: `0.385123128 ns/input`
- compiler SVML-high `exp()` control: `0.396546122 ns/input`
- Intel VML_HA: `0.321723462 ns/input`

Its 65,536 block was noisier, so shard 6 is the cleaner large-array comparison.

## Verdict

Enabling Intel SVML-high does **not** accelerate the faithful n=2 formula kernel. The hot path contains no transcendental library call for SVML to replace, and the normalized generated instructions are the same. The tiny timing differences are noise/slight regression.

The direct SVML-high exponential control is real and fast, but on the clean Xeon shard it is still slower than our formula kernel at both tested batch sizes. Using that direct SVML `exp()` inside the kernel would replace our mathematical approximation and is therefore not a valid optimization of the formula spine.

Keep `exp53_n2_vmstyle_u4_0381_frozen` unchanged as the current baseline.
