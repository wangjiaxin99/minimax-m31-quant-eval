# Quantization Fidelity Evaluation

[English](README.md) | [简体中文](README.zh-CN.md)

Compare a reference model and quantized candidates through OpenAI-compatible model servers. Cache the reference once, then measure how closely each candidate preserves its next-token probabilities and short greedy continuations.

The evaluator is model-independent when the reference and candidate share compatible tokenizers and chat templates. It has been used with MiniMax and Qwen models. Evaluation uses Python's standard library; a model server is the only runtime requirement beyond Python.

## MiniMax M3.1 results

Historical **300-prompt** evaluation, three rounds per model, with 32 generated tokens and all 32 positions requested. These are the original overall numbers; they are not measurements on the current public 240-prompt dataset. Only numerical aggregates from that historical prompt set are published here.

| Model | Mean bucketed KL (nat) ↓ | Shared-prefix Top-1 ↑ | Exact 32-token ↑ |
|---|---:|---:|---:|
| MXFP8 (rerun) | 0.01577440 | 97.55% | 49.56% |
| MXFP4 | 0.03158934 | 96.50% | 38.44% |
| MXFP4 RT | 0.03519367 | 96.20% | 38.67% |
| MXFP4 QADv1 | 0.02610190 | 96.86% | 44.33% |

Each candidate was evaluated in the same later engine environment against one fixed MXFP8 reference cache captured in an earlier environment. The MXFP8 row is a rerun against that cache, so its agreement is not 100%. Each model contributes 900 sequences and 28,800 shared-prefix positions across three rounds; all rounds completed with zero evaluation errors.

QADv1 reduces mean bucketed KL by **17.37%** versus ordinary MXFP4, improves Top-1 agreement by **0.36 percentage points**, and improves exact 32-token agreement by **5.89 percentage points**. Its three rounds took **44.81 minutes** in total, excluding server startup. This is evidence of higher fidelity on that historical prompt set, not a benchmark task-success result.

[Original overall results and scope](docs/minimax-results.md) · [Recomputed public-category results](docs/results.md)

## Quick start

```bash
python -m pip install -e .

# Check native tokenization and probability-scoring capabilities.
m31-eval doctor --base-url http://localhost:8000/v1 --probe \
  --output cache/reference-doctor.json
m31-eval doctor --base-url http://localhost:8001/v1 --probe \
  --output cache/candidate-doctor.json

# Capture the reference using every position in a 32-token continuation.
m31-eval reference \
  --dataset data/mixed/quick240.jsonl \
  --base-url http://localhost:8000/v1 \
  --reference-kind mxfp8 \
  --metric coarse --tokens 32 --positions {0..31} \
  --max-prompt-tokens 65536 --concurrency 2 \
  --time-budget 1800 --output cache/reference

# Compare a candidate with the same fixed reference.
m31-eval evaluate \
  --reference cache/reference \
  --base-url http://localhost:8001/v1 \
  --concurrency 2 --time-budget 1800 \
  --output logs/candidate-r1
```

The commands above assume Bash. For another shell, supply the integer positions `0` through `31` explicitly. The CLI otherwise defaults to eight scored positions. Set `--model` when an endpoint serves multiple models. Authentication reads `OPENAI_API_KEY`, or the environment variable selected with `--api-key-env`.

For example, run an MXFP8 model on port 8000 and an MXFP4 model on port 8001, capture MXFP8 as the reference, then evaluate MXFP4. The same workflow applies to BF16 references and algorithm-enhanced candidates; set `--reference-kind` accordingly.

## Server requirements

Token-level fidelity requires native `/tokenize`, `/detokenize`, and `/v1/completions` endpoints, including token IDs and raw next-token log probabilities. Coarse KL additionally needs the probabilities of specified reference token IDs, even when they are absent from the candidate's own Top-K list.

For vLLM, use `--logprobs-mode raw_logprobs` and an engine build that supports the requested-token scoring API. Check `tokenize` and `requested_token_logprobs` in the doctor output before evaluation. A chat-only proxy cannot provide token-level KL; the `chat` and `responses` modes measure message agreement instead.

The evaluator connects to an existing server. Model loading, quantization kernels, engine patches, GPU allocation, and server startup are managed separately.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| Mean bucketed KL (nat) | Difference between next-token probability distributions under the same reference prefix | Lower |
| Shared-prefix Top-1 agreement | Fraction of scored positions with the same highest-probability token under the same reference prefix | Higher |
| Exact 32-token agreement | Fraction of independently generated continuations whose 32 token IDs all match | Higher |

For coarse KL, the reference defines up to 20 individual token buckets plus one bucket for all remaining vocabulary probability mass. The same partition is used for the candidate. At each position, bucket contributions are **summed**, then valid positions are averaged within each prompt and prompts are averaged for the overall result. This is a lower bound on full-vocabulary KL.

Generation uses greedy decoding (`temperature=0`); probability scoring uses raw log probabilities. Near-deterministic reference distributions or zero candidate tail mass can prevent a finite KL calculation at some positions. Report finite-position coverage alongside KL. Top-1 and sequence agreement remain separate metrics.

The native summary includes task-cluster aggregation for Top-1 and KL. For direct-position Top-1, count `top1_agreement` in `samples.jsonl`; do not substitute the task-cluster mean. See [metric details](docs/metrics.md).

These metrics measure fidelity to a chosen reference. They do not directly measure benchmark accuracy, task success, or the quality of an answer. An improvement is a useful screening signal that should be checked against independent task evaluation.

## Fixed public dataset

The bundled dataset contains **240 prompts across four categories**, with 60 prompts per category.

| Category | Prompts | Source |
|---|---:|---|
| Terminal tasks | 60 | Terminal-Bench 4.0 instructions |
| Mathematics | 60 | NVIDIA OpenMathReasoning problems |
| Non-Latin languages | 60 | Global-MMLU-Lite: Arabic, Bengali, Hindi, Japanese, Korean, and Chinese |
| Long documents | 60 | CPython documentation and Project Gutenberg texts |

There are 200 original tasks/documents: the long-document category uses three windows from each of 20 documents. The development/holdout split contains 192/48 prompts and keeps all windows of a document in one split. These splits do not establish separation from a model's actual calibration or training data.

Inputs, source revisions, transformations, and licenses are retained in [data/mixed](data/mixed) and documented in [dataset details](docs/dataset.md). The public dataset has a distinct hash: capture a new reference for it rather than reusing a cache built from a different prompt list.

Use your own JSONL with `--dataset` to evaluate a different number of prompts or domain mix. See [custom datasets and rebuilding](docs/dataset.md).

## Repeated runs and comparison

Use separate output directories for each round while keeping the server settings, dataset, and reference cache fixed:

```bash
m31-eval evaluate --reference cache/reference \
  --base-url http://localhost:8001/v1 --concurrency 2 \
  --output logs/candidate-r2

m31-eval evaluate --reference cache/reference \
  --base-url http://localhost:8001/v1 --concurrency 2 \
  --output logs/candidate-r3

# Paired comparison of two candidates scored against the same reference.
m31-eval compare --old logs/candidate-r1 --new logs/improved-r1 \
  --aggregation domain_macro --output logs/comparison.json
```

Use repeated reference-model runs to establish an A/A noise floor. Keep engine versions and server parameters aligned, and examine per-category changes, valid KL coverage, and confidence intervals before interpreting a difference.

## Outputs and historical measurements

Every completed evaluation saves `run.json`, `summary.json`, and `samples.jsonl`. The samples retain generated token IDs/text, per-position probability evidence, KL, Top-1 agreement, and source identifiers. Local caches and run outputs are ignored by Git.

[Historical public-category results](docs/results.md) contain aggregate measurements only. They were recomputed from the four retained categories of earlier experiments, rather than rerun on a newly captured reference. Their dataset and reference scope are stated explicitly.

## Method background

This project follows the broad short-continuation, multi-domain fidelity strategy described in [Unsloth Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs#divergence-300-32). Its dataset, fixed-partition KL implementation, and reporting are defined here; they are not an exact reproduction of Unsloth's unpublished prompt list or Divergence-300@32 scoring formula.

Harbor is an evaluation framework and registry; Terminal-Bench is a benchmark that can run within it. The bundled terminal prompts come from Terminal-Bench 4.0. No full task containers or benchmark verifiers are executed by this evaluator.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
