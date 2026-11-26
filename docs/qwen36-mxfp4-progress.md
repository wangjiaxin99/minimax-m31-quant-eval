# Qwen3.6-35B-A3B MXFP4: prefill KL against accuracy, progress

Status on 2026-11-26 at 17:00 Beijing time. Numbers marked *running* are interim and will change.
The KL method is described in [prefill-kl.md](prefill-kl.md).

## Goal

The experiment asks whether the prefill KL can be trusted. It measures:
- its determinism and resolution;
- whether it orders quantized checkpoints the way their accuracy does.

It does not search recipes and does not study kernels.

## Checkpoints

| Name | Quantized modules | Algorithm | Calibration |
|---|---|---|---|
| MXFP4-RTN / LH / GPTQ / GPTAQ / Qronos | Routed experts only: 92.9% of text parameters (Quark `qwen3_5_moe` template default) | as named | Quark default: first 512 pileval documents, 512 tokens each |
| X1-RTN | Routed experts, full attention q/k/v/o (10 layers), Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj` (30 layers) | RTN | none |
| X1-GPTQ | as X1-RTN | GPTQ | as above |

- Format: MXFP4 W4A4 with E2M1 values and an E8M0 scale per 32, and dynamic activation quantization.
- Kept in BF16 in every checkpoint: router, shared expert and its gate, Gated DeltaNet `in_proj_a`/`in_proj_b` and
  `conv1d`, `lm_head`, embeddings.
- Quark d355c5d with PR 7002 (cold experts are no longer zeroed in GPTQ and GPTAQ). PR 7002 is now merged into
  Quark main, and the merged files are byte-identical to the ones used here.
- Two notes on Quark's pileval loader:
  - It left-pads every document to 512 tokens with EOS and passes no attention mask.
  - Of the 262,144 calibration positions, 182,722 are text and 79,422 (30%) are padding.

## Accuracy setup

- Harness: evalscope with vLLM 0.34.0, the default AITER kernels and no patch. One single-GPU server per model.
- Sampling: model-card settings, generation limit 81,920 tokens, no fixed seed.
- Benchmarks: GPQA-Diamond (198 questions x 16) and IFBench (300 prompts x 16, main metric prompt-level loose).
- Intervals: per-question paired bootstrap, 4,000 draws.

## KL (8,192 positions, K = 2000, answer mean)

| Model | KL | 95% interval | vs MXFP4-RTN | Top-1 flip | Two scores |
|---|---|---|---|---|---|
| BF16 (self) | 0 | - | - | 0 | identical |
| MXFP4-RTN | 0.02774 | [0.02662, 0.02879] | - | 3.96% | identical |
| MXFP4-LH | 0.02851 | [0.02727, 0.02975] | +2.8% | 4.02% | identical |
| MXFP4-GPTQ | 0.02931 | [0.02818, 0.03046] | +5.7% | 4.09% | identical |
| MXFP4-GPTAQ | 0.02934 | [0.02810, 0.03068] | +5.8% | 4.11% | identical |
| MXFP4-Qronos | 0.03134 | [0.03008, 0.03269] | +13.0% | 4.25% | identical |
| X1-RTN | 0.10253 | - | +270% | 8.33% | identical |
| X1-GPTQ | 0.09308 | - | +236% (-9.2% vs X1-RTN) | 7.92% | identical |

Notes:
- "Identical" means all 600 answers and all 1,709,308 positions are bitwise equal between the two runs.
- X1-RTN was exported twice: once through Quark file-to-file and once through `quark-cli`. Both gave the same KL at
  every position.
- Order among the routed-expert checkpoints: RTN < LH < GPTQ ~ GPTAQ < Qronos. Nine of the ten pairs are separated;
  GPTQ and GPTAQ differ by 0.09% and are not.
- None of the four algorithms lowers the KL below RTN on the routed experts. On X1, GPTQ lowers it by 9.2%.

## GPQA-Diamond

| Model | x16 accuracy | vs BF16 (95% interval) | KL |
|---|---|---|---|
| BF16 | 84.28 | - | 0 |
| MXFP4-RTN | 84.25 | -0.03 (-1.20, +1.14) | 0.0277 |
| MXFP4-LH | 83.62 | -0.66 (-1.86, +0.51) | 0.0285 |
| MXFP4-GPTQ | 83.33 | -0.95 (-2.24, +0.41) | 0.0293 |
| MXFP4-GPTAQ | 83.27 | -1.01 (-2.43, +0.47) | 0.0293 |
| MXFP4-Qronos | 83.08 | -1.20 (-2.65, +0.16) | 0.0313 |
| X1-RTN, x3, *running* (586/594) | 80.80 on 197 questions | **-3.91 (-6.72, -1.15)** | 0.1025 |

- For the routed-expert checkpoints, none of the 15 pairs is significant.
- In 96 of the 198 questions, every model is correct in all 16 samples. The benchmark separates about 2 points.
- X1-RTN is the first checkpoint with a clear drop. Against the routed-expert RTN, the difference is -3.85
  (-6.44, -1.26) on the same questions.

## IFBench (prompt-level loose)

| Model | x16 | vs MXFP4-RTN (95% interval) |
|---|---|---|
| MXFP4-RTN | 63.58 | - |
| MXFP4-LH | 62.40 | -1.19 (-2.35, -0.02) |
| MXFP4-GPTQ | 63.08 | -0.50 (-1.65, +0.69) |
| MXFP4-GPTAQ | 63.08 | -0.50 (-1.75, +0.77) |
| MXFP4-Qronos | 63.73 | +0.15 (-1.08, +1.40) |
| BF16, *running* (4514/4800) | - | - |

- On BF16's finished samples, all models except LH are 0.6-0.7 points below BF16, and their intervals include 0.
  LH is 1.9 points below.
- Two of the ten pairs just exclude 0. Both involve LH, and no multiple-comparison correction was applied.

## Agreement between KL and accuracy

| | GPQA x16 | IFBench x16 |
|---|---|---|
| Kendall tau with -KL (5 routed-expert checkpoints) | +1.0 | -0.3 |
| Significant accuracy pairs | 0 / 15 | 2 / 10 (borderline) |

- On the routed-expert checkpoints, the accuracy differences are inside the benchmark noise. The two benchmarks also
  disagree with each other: Qronos is last on GPQA and first on IFBench.
- These checkpoints can neither confirm nor reject the KL.
- X1 is the first setting where a large KL increase (0.028 to 0.103) comes with a significant accuracy drop
  (about -4 GPQA points).

## Cost (one MI355)

| Step | Time |
|---|---|
| Quantize RTN | 1-2 min |
| Quantize LH | 42 min |
| Quantize GPTQ (routed experts and X1) | 60-62 min |
| Quantize GPTAQ / Qronos | 2 h 51 min each |
| KL reference | 14 min, once |
| KL score per candidate | 3.7-4.1 min |
| GPQA x16 | 10-12 h (sharing the card with IFBench) |
| IFBench x16 | about 19 h (sharing the card with GPQA) |

## Running and next

- Running:
  - X1-RTN GPQA x3;
  - BF16 IFBench x16, last samples;
  - BF16 AIME26 x32, 445 of 960 samples, as the paper's math protocol.
- Next, under discussion:
  1. X1-GPTQ calibrated on BF16 self-generated answers, as in the QDA paper. There, this lowered the KL by 30% on this
     model at W4A16. The question is whether the KL can be lowered well beyond the 9% that GPTQ alone gives.
  2. If it can, the X1 checkpoints form a KL ladder. They would then get GPQA x16 and AIME, to test whether accuracy
     follows the KL.
- Stopped: the INT4 W4A16 control, at the user's request. Its KL was 0.0119 (RTN) and 0.0130 (GPTQ). INT4-GPTQ is
  the only checkpoint whose two KL scores were not bitwise identical: they differ by 0.47%, cause unknown.
