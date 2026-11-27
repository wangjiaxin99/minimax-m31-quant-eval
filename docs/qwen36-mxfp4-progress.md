# Qwen3.6-35B-A3B MXFP4: prefill KL against accuracy

Status on 2026-11-27. The KL method is described in [prefill-kl.md](prefill-kl.md). The main KL metric is the token
mean (the QDA paper's): the sum of the per-position KL over all answer positions, divided by their number.

## Goal

The experiment asks whether the prefill KL can be trusted. It measures:
- its determinism and resolution;
- whether it orders quantized checkpoints the way their accuracy does.

## Checkpoints

| Name | Quantized modules | Algorithm | Calibration |
|---|---|---|---|
| MXFP4-RTN / LH / GPTQ / GPTAQ / Qronos | Routed experts only: 92.9% of text parameters (Quark `qwen3_5_moe` template default) | as named | Quark default: first 512 pileval documents, 512 tokens each |
| X1-RTN | Routed experts, full attention q/k/v/o (10 layers), Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj` (30 layers) | RTN | none |
| X1-GPTQ (pileval) | as X1-RTN | GPTQ | as above |
| X1-GPTQ (A-small) | as X1-RTN | GPTQ | pileval + 524,288 self-generated tokens ([data](../data/calibration/qwen36-selfgen-a-small/README.md)) |
| X1-GPTQ (paper calibration) | as X1-RTN | GPTQ | The QDA W4A16 recipe's own `mix2_4096.jsonl` (qda-recipe-inputs): first 1,024 rows truncated to 4,096 tokens, as QDA reads them, re-packed to 902 x 4,096 (3.69M tokens); no pileval |

- Format: MXFP4 W4A4 with E2M1 values and an E8M0 scale per 32, and dynamic activation quantization.
- Kept in BF16 in every checkpoint: router, shared expert and its gate, Gated DeltaNet `in_proj_a`/`in_proj_b` and
  `conv1d`, `lm_head`, embeddings.
- Quark d355c5d with PR 7002, which is now merged into main (byte-identical files), and `QUARK_FUSED_MOE_ALGO=1`.
- Quark's layer-wise calibration reuses the first row's rotary embeddings, so every calibration row must have one
  length.

## Accuracy setup

- Harness: evalscope with vLLM 0.34.0, the default AITER kernels and no patch. One single-GPU server per run.
- Sampling: model-card settings, generation limit 81,920 tokens, no fixed seed.
- Benchmarks: GPQA-Diamond (198 questions x 16), AIME26 (30 x 32), and IFBench (300 x 16, stopped after the
  routed-expert models).
- Intervals: per-question paired bootstrap, 4,000 draws.

## KL (8,192 positions, K = 2000, token mean)

| Model | KL | 95% interval | vs X1-RTN | Top-1 flip | Answer mean (old metric) |
|---|---|---|---|---|---|
| BF16 (self) | 0 | - | - | 0 | 0 |
| MXFP4-RTN | 0.02068 | [0.01977, 0.02158] | -76.9% | 3.96% | 0.02774 |
| MXFP4-LH | 0.02147 | [0.02056, 0.02240] | -76.0% | 4.02% | 0.02851 |
| MXFP4-GPTQ | 0.02200 | [0.02105, 0.02294] | -75.5% | 4.09% | 0.02931 |
| MXFP4-GPTAQ | 0.02250 | [0.02157, 0.02347] | -74.9% | 4.11% | 0.02934 |
| MXFP4-Qronos | 0.02390 | [0.02291, 0.02491] | -73.3% | 4.25% | 0.03134 |
| X1-GPTQ (paper calibration) | 0.06957 | [0.06696, 0.07231] | -22.4% | 7.29% | 0.08460 |
| X1-GPTQ (A-small) | 0.07164 | [0.06883, 0.07456] | -20.0% | 7.40% | 0.08433 |
| X1-GPTQ (pileval) | 0.08159 | [0.07852, 0.08479] | -8.9% | 7.92% | 0.09308 |
| X1-RTN | 0.08961 | [0.08621, 0.09308] | - | 8.33% | 0.10253 |

- Every checkpoint was scored twice: all 600 answers and all 1,709,308 positions were bitwise identical. BF16 scored
  against itself gives KL 0.
- With the token mean, all 36 pairs of the nine checkpoints are separated. The median 95% detection threshold over
  these pairs is 1.2%; within the routed-expert models it is 0.6%.
- Only the X1 order changes between the two metrics. The answer mean puts A-small 0.3% below the paper calibration,
  which is not separable. The token mean puts the paper calibration 3.0% below A-small, which is separable.

## Accuracy

| Model | GPQA x16 | vs BF16 (95% interval) | AIME26 x32 | vs BF16 (95% interval) | AIME26 cut at 81,920 |
|---|---|---|---|---|---|
| BF16 | 84.28 | - | 91.98 | - | 3.96% |
| MXFP4-RTN | 84.25 | -0.03 (-1.20, +1.14) | | | |
| MXFP4-LH | 83.62 | -0.66 (-1.83, +0.47) | | | |
| MXFP4-GPTQ | 83.33 | -0.95 (-2.27, +0.38) | | | |
| MXFP4-GPTAQ | 83.27 | -1.01 (-2.43, +0.41) | | | |
| MXFP4-Qronos | 83.08 | -1.20 (-2.59, +0.22) | | | |
| X1-GPTQ (paper calibration) | 82.35 | **-1.93 (-3.72, -0.16)** | 89.38 | **-2.60 (-5.52, -0.42)** | 2.81% |
| X1-GPTQ (A-small) | 81.25 | **-3.03 (-4.74, -1.36)** | 87.81 | **-4.17 (-8.54, -1.15)** | 5.73% |
| X1-GPTQ (pileval) | 79.39 | **-4.89 (-7.04, -2.75)** | 86.77 | **-5.21 (-9.58, -2.08)** | 4.79% |
| X1-RTN | 79.96 | **-4.32 (-6.34, -2.27)** | 84.38 | **-7.60 (-13.44, -2.92)** | 6.56% |

IFBench x16 (prompt-level loose):
- BF16: 64.23.
- Routed-expert MXFP4: 62.40-63.73. All five are 0.5-1.8 points below BF16, and only LH is separable.
- Routed-expert MXFP4 cut 3.8-4.8% of its answers at the limit, against 0.5% for BF16. The cut answers were almost all
  repetition loops.

## Agreement between KL and accuracy

- **Routed experts against BF16**: the order of -KL equals the GPQA order (Kendall tau 1.0), but no accuracy pair is
  separable.
- **The two levels**: KL rises from about 0.02 (routed experts) to 0.07-0.09 (X1). GPQA goes from no measurable loss
  to a loss of 2-5 points, and AIME26 to a loss of 2.6-7.6 points.
- **Within X1** (paired, per question):

  | A - B | GPQA | AIME26 | KL, A vs B |
  |---|---|---|---|
  | RTN - GPTQ (pileval) | +0.57, n.s. | -2.40, n.s. | +9.8% |
  | RTN - A-small | -1.29, n.s. | **-3.44** | +25.1% |
  | RTN - paper calibration | **-2.40** | **-5.00** | +28.8% |
  | GPTQ (pileval) - A-small | **-1.86** | -1.04, n.s. | +13.9% |
  | GPTQ (pileval) - paper calibration | **-2.97** | **-2.60** | +17.3% |
  | A-small - paper calibration | -1.10, n.s. | -1.56, n.s. | +3.0% |

  - All six separable accuracy differences point the same way as the KL: the lower-KL checkpoint is more accurate.
  - The KL order within X1 is paper calibration < A-small < GPTQ (pileval) < RTN. The AIME order is the same, and
    so is the GPQA order, except RTN and GPTQ (pileval). Those two differ by 0.6 points, inside the noise.

## Cost (one MI355)

| Step | Time |
|---|---|
| Quantize RTN | 1-2 min |
| Quantize GPTQ, routed experts or X1, pileval (262k positions) | 1.0 h |
| Quantize GPTQ, A-small (786k positions, 1,536 rows of 512) | 3.0 h |
| Quantize GPTQ, paper calibration (3.69M tokens, 902 rows of 4,096) | 3.5 h |
| Quantize GPTAQ / Qronos | 2.85 h each |
| KL reference | 14 min, once |
| KL score per candidate | 3.7-4.1 min |
| GPQA x16 / AIME26 x32 / IFBench x16 | about 11 / 13-15 / 19 h |

Calibration time grows with the number of rows more than with the number of tokens.
