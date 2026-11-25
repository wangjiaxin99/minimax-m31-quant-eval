# Long teacher-forced KL inside vLLM (`m31-eval prefill-kl`)

PR2 (`sparse_live`) scores 32 positions per prompt with one HTTP request per position, against a reference captured
during greedy decode. That does not scale to thousands of positions, and the reference (decode) and candidate
(prefill) take different numeric paths, which leaves a self-KL floor (about 0.012 on MiniMax MXFP8).

`prefill-kl` reads every answer token in one prefill, on both sides:

1. `gen`: the reference model answers each prompt `--answers` times (model-card sampling, seed `17 + d`,
   up to `--max-tokens`), once; the answers are cached as token ids.
2. `ref`: the reference model reads prompt + answer in one prefill; a vLLM worker extension takes the hidden states at
   the prompt-logprob step and keeps, per answer token, the Top-K log-probabilities, their ids and the log tail mass.
3. `score`: a candidate reads the same token sequences; the extension computes, on the GPU, the KL for each requested
   K (K <= stored K), whether the top-1 differs and the target token's log-probability. Only these per-position
   scalars leave the worker.
4. `compare`: bitwise equality of two score runs.

Engine settings: one GPU, prefix caching off, one sequence per call, torch inductor `deterministic=True`. Each
step checks that the hook's target log-probability matches the engine's own prompt log-probability (tolerance 1e-3).

The worker extension is adapted from QDA (`qda_kl/vllm_hooks.py`, https://github.com/humanfia/qda, Apache-2.0) and
needs vLLM's V2 GPU model runner (vLLM 0.34).

## Commands (inside the vLLM container)

```bash
export PYTHONPATH=$PWD/src HIP_VISIBLE_DEVICES=0 VLLM_ROCM_USE_AITER_MOE=0
python3 -m m31eval.prefill_kl.run gen   --model BF16 --moe-backend triton --prompts prompts.jsonl \
        --answers 2 --seed 17 --max-tokens 8192 --out answers.jsonl
python3 -m m31eval.prefill_kl.run ref   --model BF16 --moe-backend triton --answers answers.jsonl --topk 2000 --out reference
python3 -m m31eval.prefill_kl.run score --model CKPT --moe-backend emulation --answers answers.jsonl \
        --reference reference --ks 20,128,2000 --out score_r1
python3 -m m31eval.prefill_kl.run compare score_r1 score_r2
```

`prompts.jsonl` rows: `{id, domain, prompt_token_ids}` (already rendered with the model's chat template).

## Outputs

- `answers.jsonl`: `{key, id, domain, draw, seed, prompt_len, finish_reason, prompt_token_ids, answer_token_ids}`.
- `reference/seqs/NNNNN.npz`: `topv` f32 [n, K], `topi` i32 [n, K], `log_rest` f32 [n], `tgt` f32 [n].
- `score_*/seqs/NNNNN.npz`: `kl_<K>` f64 [n], `flip` bool [n], `tgt` f32 [n]; `manifest.json` holds a summary
  (answer-mean and token-mean KL per K, top-1 flip rate), the engine arguments, the vLLM version and the bound kernels.

Row j of every array belongs to answer token j, so any prefix length (32, 256, ...) can be analysed offline.

## Smoke test (Qwen3.6-35B-A3B, 10 answers x <= 256 tokens, 2,353 positions)

- BF16 scored against its own reference: KL 0.0 for K = 20, 128, 2000; top-1 flips 0.
- MXFP4 GPTQ (emulation MoE) scored on GPU0 and GPU1: identical in all 2,353 positions; KL_2000 answer-mean 0.02287.
- Hook vs engine target log-probability: max abs difference 4.9e-6.
