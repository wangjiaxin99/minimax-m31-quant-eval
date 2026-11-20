# Stable sparse KL experiment

The public CLI still uses full-vocabulary KL. `m31eval.sparse_experiment` is an
offline experimental module; it does not change the live vLLM API.

The experiment freezes a reference Top-K token set. It keeps all K tokens,
including reference Top-1, and computes the complementary tail using
log-sum-exp over the full input vector. The candidate uses exactly the same
token set. No minimum tail mass or reference-confidence-based position drop
is applied.

Complete input probability mass is validated within `2e-5`. Small mass
corrections are explicitly recorded as `normalization_log_mass`; malformed
mass is rejected. This normalization is specific to the experiment, not the
CLI's raw full-KL path. Comparison with full KL must use the same normalized
inputs and report any difference from raw full KL.

Exact zero reference probability contributes zero. Exact zero candidate
probability at positive reference mass gives infinite KL, rather than a
silently excluded position. Invalid data remains an error. Neither tail
clipping nor arbitrary epsilon replacement is used.

`compress_gpu` reduces a full one-dimensional GPU tensor to K selected log
probabilities and one tail log probability before copying sparse numeric
values to the CPU. This helper demonstrates arithmetic and serialization
feasibility; live sampler integration, TP layout, API schemas, and engine
latency still require separate validation. Loading cached full tensors onto
the GPU is an experiment step, not the intended production data path.

Useful checks are paired comparisons on the same P/Q vectors; confidence and
dataset strata; full-KL values at positions the old code excluded; maximum
and average approximation gaps; and ranking stability across candidates.
Top-1 NLL is a complementary metric, not a substitute labeled as KL.

The initial 10-prompt/5-position pilot cannot establish approximation quality
for all 300 prompts, all 32 positions, or different candidate models. The
first positions often contain formatting and reasoning opening text. A
synthetic degradation verifies a failure mechanism, but does not establish
its frequency or severity in real checkpoints.

## Experimental live GPU-tail integration

`m31eval.sparse_live` adds a separate opt-in live path. It requires the tested
server-side GPU-tail patch; the ordinary full-vocabulary CLI is unchanged.
For every reference position it freezes the reference token IDs with
fixed reference Top-2000 (with no coverage threshold). Every candidate scores
those same IDs on the reference prefix. Candidate raw argmax is returned
separately, even when it is outside the selected IDs.

The experimental V1 sampler gathers selected log probabilities and computes
the complementary tail and total normalization mass on the GPU in float64.
It returns them through a `sparse_fidelity` completion field, with no
full-vocabulary text decoding or full-distribution host transfer. JSON
serialization, saving, and KL reduction over the selected entries still run
on the CPU. This is not an entirely GPU-resident evaluation.

The opt-in API lifts the ordinary 128-entry `logprob_token_ids` limit. It
requires a nonempty unique valid ID list, token-ID output, one generated
token, and non-streaming responses. It does not change the ordinary `logprobs`
Top-K or full-vocabulary request semantics. Enabling it is experimental, not
a claim of upstream vLLM support. Large K can still approach full-vocabulary
transfer cost.

`prepare` creates immutable per-prompt reference arrays from the frozen full
reference, without a coverage audit. `evaluate` runs an independent greedy
32-token continuation for Exact-32, then scores all 32 reference prefixes for
KL and Same-Prefix Top-1. It preserves per-position probabilities, selected-ID
hashes, and normalization corrections. Missing or malformed responses mark
the run incomplete; they are not silently excluded from final statistics.
The reference trace and prepared arrays must be kept with candidate outputs.

Numerical checks cover tiny tails, empty tails, heterogeneous K, argmax
outside the selected set, input preservation, and the actual sampler-to-API
transport. Live preflight has also exercised an explicit 178,943-ID request.
These checks establish implementation behavior, not approximation accuracy
for every possible candidate or prompt.

## Fixed Top-2000 workflow (not an evaluation result)

Install the optional client dependency with `pip install -e '.[sparse]'`.
Apply `patches/vllm-full-logprobs-token-id-only.patch` first, then
`patches/vllm-sparse-fidelity-gpu-tail.patch` in the vLLM source root. The patches
target `vllm/vllm-openai-rocm:minimax-m31-flash` and its V1 sampler. Restart the
server with `VLLM_FIDELITY_SPARSE=1`, `VLLM_USE_V2_MODEL_RUNNER=0`,
`--logprobs-mode raw_logprobs`, and `--max-logprobs -1` in addition to the
model's required serving configuration. Different engine branches need
separate compatibility validation.

```bash
python -m m31eval.sparse_live prepare \
  --reference /path/to/frozen-full-reference \
  --output /path/to/reference-top2000 --top-k 2000
python -m m31eval.sparse_live evaluate \
  --prepared-reference /path/to/reference-top2000 \
  --base-url http://localhost:8000/v1 --concurrency 2 \
  --output /path/to/candidate-r1
```

The MiniMax client currently expects 32 positions and a 200,064-entry
vocabulary. At every position exactly 2,000 reference-selected IDs are kept;
all candidates use the identical set. No `1e-5` tail threshold removes any
token or position. The `2e-5` total-mass validity check is a separate input
sanity check, not a selection threshold. Stable float64 tail reduction and
explicit normalization are unchanged. True zero candidate probability at
positive reference mass produces an explicitly serialized `"Infinity"` KL.

Old adaptive-reference caches are rejected; create a separate prepared
reference and result directory. Existing adaptive evaluation results cannot
be relabeled as fixed-2000 results. This PR prepares the method only: no
new fixed-2000 model evaluation has been run. The original 240-prompt,
Top-20 implementation remains on main until the PRs are reviewed and merged.
