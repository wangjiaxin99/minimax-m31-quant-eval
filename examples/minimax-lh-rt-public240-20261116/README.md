# MiniMax LH and Rotation+GPTQ — Public 240-prompt Rerun

Completed: 2026-11-16T14:14:12+08:00 (Asia/Shanghai)

Reference: `amd/MiniMax-M3.1-Flash-Quark-MXFP8-1026`. Candidates: `amd/MiniMax-M3.1-Flash-Quark-MXFP4-LH-Test` (Local Hessian) and `amd/MiniMax-M3.1-Flash-Quark-MXFP4-RT-Test` (Rotation + GPTQ).

Evaluator commit: `b993028749ecfecd10589ba77fe908759b3e284f`. Dataset: the bundled public `data/mixed/quick240.jsonl`, SHA256 `b668e18c1d9004d44f1d14159d2a71879016a5e9b8508432f3fa1cc769078683`. Four categories, 60 prompts each. A new MXFP8 reference was captured in the same environment used for both candidates, followed by three LH rounds and three RT rounds on the same four GPUs in sequence.

## Three-round summary

| Model | Mean KL (nat) ↓ | Shared-prefix Top-1 ↑ | Exact 32-token ↑ | Total evaluation (min) |
|---|---:|---:|---:|---:|
| LH | 0.03843885 | 96.21% | 34.31% | 36.79 |
| RT | 0.04018061 | 95.93% | 33.89% | 40.98 |

RT mean KL is 4.53% higher than LH. RT Top-1 is 0.27 percentage points lower and exact 32-token agreement is 0.42 percentage points lower. These small overall differences do not show an RT advantage in this run. No ordinary MXFP4 candidate was rerun under this new protocol, so this comparison does not establish either algorithm’s improvement over ordinary MXFP4.

The overall similarity hides different category behavior. RT exact 32-token agreement is higher on math (26.67% vs 19.44%), non-Latin (23.33% vs 21.67%), and terminal prompts (66.67% vs 60.56%), but lower on long documents (18.89% vs 35.56%). The long-document group has 60 windows from 20 source documents; these windows are not 60 independent source tasks.

## Observed round-to-round range

| Model | KL range (nat) | Top-1 range (pp) | Exact 32-token range (pp) |
|---|---:|---:|---:|
| LH | 0.00184735 | 0.55 | 1.67 |
| RT | 0.00100723 | 0.20 | 1.67 |

Range = maximum minus minimum across three rounds. RT has a smaller observed KL and Top-1 range; both models have the same 1.67 pp exact-sequence range. Three rounds describe this run and are insufficient to establish a general stability advantage.

## Per-round results

| Model | Round | Mean bucketed KL (nat) ↓ | Shared-prefix Top-1 ↑ | Exact 32-token ↑ | Finite KL positions | Time (min) |
|---|---|---:|---:|---:|---:|---:|
| LH | R1 | 0.03953208 | 95.98% | 34.58% | 5062/7680 | 12.53 |
| LH | R2 | 0.03768473 | 96.52% | 35.00% | 5062/7680 | 12.13 |
| LH | R3 | 0.03809973 | 96.12% | 33.33% | 5062/7680 | 12.14 |
| RT | R1 | 0.04060851 | 95.89% | 34.17% | 5062/7680 | 13.93 |
| RT | R2 | 0.04033203 | 95.86% | 34.58% | 5062/7680 | 13.53 |
| RT | R3 | 0.03960128 | 96.05% | 32.92% | 5062/7680 | 13.51 |
| LH | Mean (3 rounds) | 0.03843885 | 96.21% | 34.31% | — | 12.26 |
| RT | Mean (3 rounds) | 0.04018061 | 95.93% | 33.89% | — | 13.66 |

KL is averaged over valid positions within each prompt, then across prompts and rounds. Top-1 is counted directly over 7,680 positions per round. Exact 32-token agreement compares independently generated 32-token sequences. Coverage exclusions do not remove positions from Top-1.

All six rounds have exactly the same finite KL position set: 5,062/7,680 positions (65.91%) per round. The other 2,618 positions have a near-deterministic reference whose returned probabilities cannot resolve the fixed partition. There are no additional candidate-only exclusions, no missing prompts, and no evaluation errors. All 240 prompts have at least one finite KL position. The denominators for Top-1 and exact sequences remain 7,680 and 240 per round.

## Results by category

| Model | Category | Mean bucketed KL (nat) ↓ | Shared-prefix Top-1 ↑ | Exact 32-token ↑ | Finite KL positions |
|---|---|---:|---:|---:|---:|
| LH | long_document | 0.03921121 | 96.09% | 35.56% | 4659/5760 |
| LH | math | 0.03417583 | 95.17% | 19.44% | 4203/5760 |
| LH | non_latin | 0.06285639 | 94.72% | 21.67% | 4284/5760 |
| LH | terminal_bench | 0.01751195 | 98.84% | 60.56% | 2040/5760 |
| RT | long_document | 0.03866092 | 95.03% | 18.89% | 4659/5760 |
| RT | math | 0.03555054 | 95.31% | 26.67% | 4203/5760 |
| RT | non_latin | 0.06734574 | 94.69% | 23.33% | 4284/5760 |
| RT | terminal_bench | 0.01916523 | 98.70% | 66.67% | 2040/5760 |

## Environment and interpretation

vLLM PR #65918, commit `3ad9a190c304e338d3f3f535c7d1f0caff915ff4`; AITER main plus PR #7565, commit `de948849c604f4d16e378acf90637c159ac81280`; PyTorch `2.13.0+rocm10.0.0`; image ID `sha256:1d3dac5bca7b1b80983ec0bf4675d1b9feff74a1d683838ade67d06d35c26877`. No new source patch was added for this rerun.

Both candidates use `VLLM_ROCM_USE_AITER=1`, `AITER_BYPASS_TUNE_CONFIG=1`, and `--moe-backend aiter`; logs confirm `AITER_MXFP4_MXFP4`. LH requires no rotation support in its checkpoint. RT uses the rotation tensors and fused AITER inter-GEMM path. Other server settings are matched: TP4, NVFP4 KV, max model length 524288, raw logprobs.

Generation: temperature 0, 32 tokens. All positions 0–31 are requested. Coarse KL uses fixed reference Top-20 token buckets and one tail bucket. Evaluator concurrency 2, timeout 120 seconds, time budget 1800 seconds per job.

These metrics measure fidelity to the newly captured MXFP8 reference, not task success. They are not directly comparable to the original 300-prompt historical results or to the earlier cache because both input selection and reference capture differ. Compare LH and RT only within the common protocol above.

[Round counts and timings](rounds.csv) · [Category counts](by_category.csv). Aggregate values are published here; raw per-prompt outputs and reference caches remain in local experiment logs.

## Timing

| Job | Start (UTC+08:00) | Completion (UTC+08:00) | Native job duration (min) |
|---|---|---|---:|
| LH R1 | 2026-11-16T12:41:23+08:00 | 2026-11-16T12:54:05+08:00 | 12.53 |
| LH R2 | 2026-11-16T12:54:05+08:00 | 2026-11-16T13:06:34+08:00 | 12.13 |
| LH R3 | 2026-11-16T13:06:34+08:00 | 2026-11-16T13:18:45+08:00 | 12.14 |
| RT R1 | 2026-11-16T13:32:23+08:00 | 2026-11-16T13:46:20+08:00 | 13.93 |
| RT R2 | 2026-11-16T13:46:21+08:00 | 2026-11-16T14:00:03+08:00 | 13.53 |
| RT R3 | 2026-11-16T14:00:03+08:00 | 2026-11-16T14:13:50+08:00 | 13.51 |

MXFP8 reference capture took **88.74 seconds (1.48 minutes)**. Candidate totals above exclude model loading, readiness checks and smoke tests. Start/completion timestamps include a small orchestration polling delay; native durations come from the evaluator summaries. All three workflow servers were stopped after completion.
