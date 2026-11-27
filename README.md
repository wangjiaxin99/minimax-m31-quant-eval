# Quantization Fidelity Evaluation

[English](README.md) | [简体中文](README.zh-CN.md)

Compare a reference model and quantized candidates through OpenAI-compatible model servers. Cache the reference once, then measure how closely each candidate preserves its next-token probabilities and short greedy continuations.

The evaluator is model-independent when the reference and candidate share compatible tokenizers and chat templates. It has been used with MiniMax and Qwen models. Evaluation uses Python's standard library; a model server is the only runtime requirement beyond Python.

## Full-vocabulary KL on all 300 prompts

The evaluator scores the complete vocabulary at every one of the 32 continuation positions. MiniMax has 200,064 vocabulary entries. Start vLLM with `--max-logprobs -1 --logprobs-mode raw_logprobs`, then use `--metric full`. Full-vocabulary KL is the only supported KL mode and is the CLI default. Historical tables below retain their original bucketed-KL labels; they are archived measurements, not current full-vocabulary results.

Reference capture obtains all 32 full-vocabulary distributions in one greedy generation request. The cache retains complete raw log probabilities as uncompressed little-endian float64 arrays, with small per-prompt JSON metadata compressed at gzip level 1. Token IDs 0 through vocabulary-size minus one are implicit; other ID maps are stored explicitly. For MiniMax, 300 × 32 × 200,064 × 8 bytes requires 15.37 GB (14.31 GiB) of probability storage, plus small metadata. Candidates are scored with complete distributions in memory; output keeps per-position KL, Top-1, reference-token probability evidence, vocabulary size and a full-distribution digest, alongside generated token IDs/text. Candidate full distributions are not persisted.

## Long prefill KL inside vLLM (`prefill-kl`)

`prefill-kl` scores up to 8,192 answer tokens per prompt in one prefill, with the same prefill path on the reference
and candidate side, so BF16 against itself gives KL 0. It runs on vLLM 0.34 as an offline engine with a worker
extension: `render` -> `gen` -> `ref` -> `score` -> `compare`, plus an offline `analyze` report (NumPy only). [Method and settings](docs/prefill-kl.md) · [Qwen3.6-35B-A3B MXFP4 results and progress](docs/qwen36-mxfp4-progress.md)

## Dataset restored to 300 prompts — 2026-11-18

The default fidelity dataset again includes the original 60 DeepSWE prompts: five categories, 60 prompts each. Only the standalone DeepSWE task benchmark was removed. The November 16–18 reruns remain 240-prompt results; restoring inputs does not turn them into 300-prompt measurements.

## MiniMax LH and RT: public 240-prompt rerun

Fresh MXFP8 reference captured in the same environment; 240 public prompts, all 32 positions, three rounds per candidate (2026-11-16). Zero evaluation errors.

| Model | Mean KL (nat) ↓ | Shared-prefix Top-1 ↑ | Exact 32-token ↑ | Total evaluation (min) |
|---|---:|---:|---:|---:|
| LH | 0.03843885 | 96.21% | 34.31% | 36.79 |
| RT | 0.04018061 | 95.93% | 33.89% | 40.98 |

[New results README, rounds, categories and observed ranges](examples/minimax-lh-rt-public240-20261116/README.md)

## Historical MiniMax M3.1 results

Historical **300-prompt** evaluation, three rounds per model, with 32 generated tokens and all 32 positions requested. These are historical measurements with their original reference and environment. The original 300-prompt inputs have now been restored; the separate 240-prompt reruns use a different reference and scope.

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
  --dataset data/mixed/quick300.jsonl \
  --base-url http://localhost:8000/v1 \
  --reference-kind mxfp8 \
  --metric full --tokens 32 --positions {0..31} \
  --max-prompt-tokens 65536 --concurrency 2 \
  --time-budget 1800 --output cache/reference

# Compare a candidate with the same fixed reference.
m31-eval evaluate \
  --reference cache/reference \
  --base-url http://localhost:8001/v1 \
  --concurrency 2 --time-budget 1800 \
  --output logs/candidate-r1
```

The commands above assume Bash. For another shell, supply the integer positions `0` through `31` explicitly. The CLI defaults to all 32 scored positions. Set `--model` when an endpoint serves multiple models. Authentication reads `OPENAI_API_KEY`, or the environment variable selected with `--api-key-env`.

For example, run an MXFP8 model on port 8000 and an MXFP4 model on port 8001, capture MXFP8 as the reference, then evaluate MXFP4. The same workflow applies to BF16 references and algorithm-enhanced candidates; set `--reference-kind` accordingly.

To verify complete vocabulary support for MiniMax:

```bash
m31-eval doctor --base-url http://localhost:8000/v1 --probe --probe-full \
  --expected-vocab-size 200064 --output cache/full-doctor.json
```

Use the model config vocabulary size for other models. Every token ID must be returned, and probability mass must normalize.

## Server requirements

Token-level fidelity requires native `/tokenize`, `/detokenize`, and `/v1/completions` endpoints, including token IDs and raw next-token log probabilities. Every scored position must return the complete vocabulary.

For vLLM, use `--max-logprobs -1 --logprobs-mode raw_logprobs`. Check `tokenize` and `full_vocabulary_logprobs` with `doctor --probe-full --expected-vocab-size` before evaluation. A chat-only proxy cannot provide token-level KL; the `chat` and `responses` modes measure message agreement instead.

The evaluator connects to an existing server. Model loading, quantization kernels, engine patches, GPU allocation, and server startup are managed separately.

For full-vocabulary exports, see [server performance](docs/server-performance.md) and the token-ID-only vLLM patch that avoids internally decoding the entire vocabulary at each position.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| Mean full-vocabulary KL (nat) | KL summed across the complete vocabulary under the same reference prefix, averaged across positions and prompts | Lower |
| Shared-prefix Top-1 agreement | Fraction of scored positions with the same highest-probability token under the same reference prefix | Higher |
| Exact 32-token agreement | Fraction of independently generated continuations whose 32 token IDs all match | Higher |

At each reference-prefix position, compute `sum(P(v) * (log P(v) - log Q(v)))` across the complete vocabulary. Sum token contributions, average scored positions within each prompt, then average prompts for `prompt_mean_kl`. For 300 prompts and 32 positions, require 9,600 successful position scores.

Generation uses greedy decoding (`temperature=0`); probability scoring uses raw log probabilities. Missing token IDs, invalid probability mass, or zero candidate probability at positive reference mass are errors, not reasons to silently drop a position. Near-deterministic reference positions are retained. An incomplete run must not be reported as a complete result.

The native summary includes task-cluster aggregation for Top-1 and KL. For direct-position Top-1, count `top1_agreement` in `samples.jsonl`; do not substitute the task-cluster mean. See [metric details](docs/metrics.md).

These metrics measure fidelity to a chosen reference. They do not directly measure benchmark accuracy, task success, or the quality of an answer. An improvement is a useful screening signal that should be checked against independent task evaluation.

## Fixed fidelity dataset

The default `data/mixed/quick300.jsonl` contains **300 prompts across five categories**, with 60 prompts per category. `quick240.jsonl` remains available to reproduce the four-category reruns.

| Category | Prompts | Source |
|---|---:|---|
| DeepSWE | 60 | Frozen historical task and agent-context inputs |
| Terminal tasks | 60 | Terminal-Bench 4.0 instructions |
| Mathematics | 60 | NVIDIA OpenMathReasoning problems |
| Non-Latin languages | 60 | Global-MMLU-Lite: Arabic, Bengali, Hindi, Japanese, Korean, and Chinese |
| Long documents | 60 | CPython documentation and Project Gutenberg texts |

There are 260 original tasks/documents: the long-document category uses three windows from each of 20 documents. The development/holdout split contains 240/60 prompts and keeps all windows of a document in one split. These splits do not establish separation from a model's actual calibration or training data.

Inputs, source revisions, transformations, and licenses are retained in [data/mixed](data/mixed) and documented in [dataset details](docs/dataset.md). The restored 300-prompt set and retained 240-prompt set have different hashes. A 240-prompt reference cannot evaluate the additional 60 prompts; capture and freeze a 300-prompt reference before comparing new 300-prompt runs.

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

Every completed evaluation saves `run.json`, `summary.json`, and `samples.jsonl`. The samples retain generated token IDs/text, per-position probability evidence, KL, Top-1 agreement, and source identifiers. New reference arrays are under `distributions/*.logprobs.f64`; matching `samples/*.json.gz` files describe shape, dtype and paths. Candidate generation starts from the original prompt for Exact-32; candidate KL/Top-1 is separately scored under each reference continuation prefix. Local caches and run outputs are ignored by Git.

[Historical public-category results](docs/results.md) contain aggregate measurements only. They were recomputed from the four retained categories of earlier experiments, rather than rerun on a newly captured reference. Their dataset and reference scope are stated explicitly.

## Method background

This project follows the broad short-continuation, multi-domain fidelity strategy described in [Unsloth Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs#divergence-300-32). Its dataset, full-vocabulary KL implementation, and reporting are defined here; they are not an exact reproduction of Unsloth's unpublished prompt list or Divergence-300@32 scoring formula.

Harbor is an evaluation framework and registry; Terminal-Bench is a benchmark that can run within it. The bundled terminal prompts come from Terminal-Bench 4.0. No full task containers or benchmark verifiers are executed by this evaluator.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Fixed Top-2000 with stable GPU tail export is available as a separate [experimental workflow](docs/sparse-experiment.md). It retains all positions without the old minimum-tail pruning rule; the full-vocabulary CLI above is unchanged. No fixed-2000 model results are claimed yet.
