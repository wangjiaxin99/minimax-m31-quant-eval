# Long teacher-forced KL inside vLLM (`m31-eval prefill-kl`)

PR2 (`sparse_live`) scores 32 positions per prompt with one HTTP request per position, against a reference captured
during greedy decode. That does not scale to thousands of positions, and the reference (decode) and candidate
(prefill) take different numeric paths, which leaves a self-KL floor (about 0.012 on MiniMax MXFP8).

`prefill-kl` reads every answer token in one prefill, on both sides:

0. `render`: chat prompts -> prompt token ids with the model's chat template (no GPU).
1. `gen`: the reference model answers each prompt `--answers` times (model-card sampling, seed `17 + d`,
   up to `--max-tokens`), once; the answers are cached as token ids.
2. `ref`: the reference model reads prompt + answer in one prefill; a vLLM worker extension takes the hidden states at
   the prompt-logprob step and keeps, per answer token, the Top-K log-probabilities, their ids and the log tail mass.
3. `score`: a candidate reads the same token sequences; the extension computes, on the GPU, the KL for each requested
   K (K <= stored K), whether the top-1 differs and the target token's log-probability. Only these per-position
   scalars leave the worker.
4. `compare`: bitwise equality of two score runs.
5. `analyze`: offline report over several score runs and, optionally, per-question accuracy (no GPU, NumPy only).

Engine settings: one GPU, prefix caching off, one sequence per call, torch inductor `deterministic=True`. Each
step checks that the hook's target log-probability matches the engine's own prompt log-probability (tolerance 1e-3).

Kernels: `--moe-backend` and `--linear-backend` are vLLM's engine options for the MoE and the quantized dense GEMM.
For MXFP4 use `emulation` for both; `--linear-backend` matters only when dense layers (attention, Gated DeltaNet) are
quantized. Every manifest records the quant methods, the dense GEMM kernels actually bound (`kernels.linear_kernels`,
e.g. `EmulationMxfp4LinearKernel` or `AiterMxfp4LinearKernel`) and the `VLLM_*` environment. A step stops when an
`emulation` backend was requested and another kernel was bound.

The worker extension is adapted from QDA (`qda_kl/vllm_hooks.py`, https://github.com/humanfia/qda, Apache-2.0) and
needs vLLM's V2 GPU model runner (vLLM 0.34).

## Commands (inside the vLLM container)

```bash
export PYTHONPATH=$PWD/src HIP_VISIBLE_DEVICES=0 VLLM_ROCM_USE_AITER_MOE=0
python3 -m m31eval.prefill_kl.run render --src data/mixed/quick300.jsonl --model BF16 --out prompts.jsonl
python3 -m m31eval.prefill_kl.run gen   --model BF16 --moe-backend triton --prompts prompts.jsonl \
        --answers 2 --seed 17 --max-tokens 8192 --out answers.jsonl
python3 -m m31eval.prefill_kl.run ref   --model BF16 --moe-backend triton --answers answers.jsonl --topk 2000 --out reference
python3 -m m31eval.prefill_kl.run score --model CKPT --moe-backend emulation --linear-backend emulation \
        --answers answers.jsonl --reference reference --ks 20,128,2000 --out score_r1
python3 -m m31eval.prefill_kl.run compare score_r1 score_r2
# offline, any Python with NumPy (no GPU):
python3 -m m31eval.prefill_kl.run analyze --answers answers.jsonl --run rtn=score_rtn_r1 --run gptq=score_gptq_r1 \
        [--accuracy bf16=gpqa_bf16.jsonl --accuracy rtn=gpqa_rtn.jsonl ...] --out analysis/kl
```

`render` reads the fidelity dataset rows `{id, domain, messages, tools?}` and writes `{id, domain, n_prompt_tokens,
prompt_token_ids}`; `--chat-template-kwargs '{"enable_thinking": false}'` overrides template defaults.
`analyze --accuracy` reads per-sample rows `{"question": ..., "score": 0..1}`; a model given only accuracy (no run)
is the reference, with KL 0. It writes `<out>.json` (all numbers) and `<out>.md` (tables).

## Outputs

- `answers.jsonl`: `{key, id, domain, draw, seed, prompt_len, finish_reason, prompt_token_ids, answer_token_ids}`.
- `reference/seqs/NNNNN.npz`: `topv` f32 [n, K], `topi` i32 [n, K], `log_rest` f32 [n], `tgt` f32 [n].
- `score_*/seqs/NNNNN.npz`: `kl_<K>` f64 [n], `flip` bool [n], `tgt` f32 [n]; `manifest.json` holds a summary
  (answer-mean and token-mean KL per K, top-1 flip rate), the engine arguments, the vLLM version and the bound kernels.

Row j of every array belongs to answer token j, so any prefix length (32, 256, ...) can be analysed offline.

## Settings used on Qwen3.6-35B-A3B (2026-11-25 onward)

These are the settings of the current experiment. [Results and progress](qwen36-mxfp4-progress.md).

| Item | Setting |
|---|---|
| Prompts | `quick300.jsonl`: 300 prompts, 60 per domain (deepswe, long_document, math, non_latin, terminal_bench). `render` with the Qwen3.6 chat template, thinking at the template default (on) |
| Reference answers (`gen`) | BF16, triton MoE. 2 answers per prompt, model-card sampling (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5), seed `17 + d`, at most 8,192 tokens. 600 answers, 1,709,308 answer positions; 100 of the 120 math answers stop at the length limit |
| Reference (`ref`) | BF16, triton MoE, one prefill per answer. Top-2000 per answer position (26 GB) |
| Candidates (`score`) | One prefill per answer. MXFP4: `--moe-backend emulation --linear-backend emulation` (MoE `OCP_MXQuantizationEmulationTritonExperts`, dense `EmulationMxfp4LinearKernel`). `VLLM_ROCM_USE_AITER_MOE=0`. No vLLM source change. Runs before this option existed set `VLLM_DISABLED_KERNELS=AiterMxfp4LinearKernel`, which binds the same kernel |
| Engine | vLLM 0.34.0 (`vllm/vllm-rocm:v0.34.0`), one GPU, `gpu_memory_utilization` 0.80, prefix caching off, `language_model_only`, inductor `deterministic=True` |
| K | 20, 128 and 2000 from one reference |
| Repeats | Every candidate scored twice; `compare` must report every position identical |
| Main metric | Token mean, as in the QDA paper (`qda_kl/tally.py` `reading`): the sum of the per-position KL over all answer positions, divided by their number. The answer mean (mean over an answer's positions, then over the 600 answers) is reported as well |
| Self-check | BF16 scored against its own reference: KL 0 and no top-1 flip at all 1,709,308 positions |

### Offline analysis (`analyze`)

Every array row is one answer position, so `analyze` reads `score_*/seqs` without running a model again:

- **Length**: KL of the first L positions of every answer, L = 32 ... 8192.
- **Uncertainty**: paired bootstrap over prompts (4,000 draws). Resampling is stratified by domain, and the two answers
  of one prompt are drawn together. The 95% interval of the ln KL ratio of two models gives their detection threshold:
  the smallest relative KL difference that holds for another sample of similar prompts. It does not include
  calibration noise (rebuilding a recipe on other calibration data).
- **Prompt count**: the same bootstrap on stratified subsets of 48, 96 and 150 prompts.
- **K and aggregation**: thresholds for every K and for the answer, token and domain means.
- **Domains and positions**: KL per domain, KL per position bucket, share of the total KL inside the first L positions,
  and Kendall tau of the order at each length against the full-length order.
- **Accuracy agreement**: Kendall tau between -KL and accuracy, with the KL prompts and the benchmark questions
  resampled independently, an exact permutation test, and pairwise comparison against per-question paired
  bootstrap intervals of accuracy.

On the Qwen3.6 runs, `analyze` reproduces the experiment's earlier analysis script exactly: KL, intervals, thresholds
by length and by prompt count, GPQA accuracy and its intervals, tau and the permutation p-value.

Main findings on the five routed-expert MXFP4 checkpoints:
- Token mean, the main metric:
  - The detection threshold falls from 7.5% at 32 positions to 2.3% at 256, 1.1% at 1K, and 0.6% at 8,192.
    Longer answers keep adding resolution, because long answers carry most of the weight.
  - With 48 prompts it is 1.4%; with 300, 0.6%.
- Answer mean: the threshold stays at about 2.4% from 1K positions and shrinks about as 1/sqrt(prompts), from 6.2%
  with 48 prompts.
- Both metrics give the same order for the routed-expert checkpoints.
- K = 20, 128 and 2000 give the same order and the same threshold, and their KL differs by less than 2%.

### Cost (one MI355)

- Reference: 9.4 min to generate and 4.5 min to prefill, once.
- Each candidate score: 3.7-4.1 min, or 5-7 min including engine start.
- GPQA-Diamond x16 on the same card: 10-12 h.

## Smoke test (Qwen3.6-35B-A3B, 10 answers x <= 256 tokens, 2,353 positions)

- BF16 scored against its own reference: KL 0.0 for K = 20, 128, 2000; top-1 flips 0.
- MXFP4 GPTQ (emulation MoE) scored on GPU0 and GPU1: identical in all 2,353 positions; KL_2000 answer-mean 0.02287.
- Hook vs engine target log-probability: max abs difference 4.9e-6.
