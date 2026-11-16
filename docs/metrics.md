# Fidelity metrics and API behavior

For a prompt `x` and reference-generated continuation `y`, position `t` is scored on the common prefix `x + y[:t]`. This keeps the conditioning input identical even if independently generated candidate output has already diverged.

## Fixed-partition KL

Let `P` denote reference probabilities and `Q` candidate probabilities. The reference chooses a fixed token set `S`, normally its Top-20 tokens. Each token in `S` is an individual bucket. All remaining vocabulary entries form one shared tail bucket:

```text
P(other) = 1 - sum(P(token) for token in S)
Q(other) = 1 - sum(Q(token) for token in S)

KL = sum(P(bucket) * ln(P(bucket) / Q(bucket)) for bucket in S + [other])
```

Natural logarithms give units of nats. The bucket contributions are summed, not averaged. Coarsening the vocabulary makes this a KL lower bound, so two candidates can differ in the tail without that difference being fully visible.

The candidate must return probabilities for the reference token set, rather than only its own Top-20 tokens. Do not renormalize just the selected tokens, replace missing probabilities with arbitrary values, or compare distributions conditioned on different prefixes.

A position can lack finite KL when the reference has no numerically resolvable partition or the returned candidate tail mass is zero. Preserve that evidence and report valid/missing coverage. `prompt_mean_kl` averages valid positions within each prompt and then averages prompts with a defined mean; inspect `kl_scored_prompts` and position counts.

## Agreement metrics

Shared-prefix Top-1 uses each position's argmax under the common reference prefix. For a complete 240-prompt, 32-position run, direct-position agreement is:

```text
sum(position['top1_agreement'] for every sample and scored position) / 7680
```

The native `same_context_top1_agreement.mean` is task-cluster weighted and may differ from this direct count because a source document can contribute multiple prompts. Keep the aggregation explicit.

Exact 32-token agreement compares independently generated token-ID sequences. One different token makes that sequence a non-match. Check generated lengths and EOS statistics; inspect `prefix_at_32` and token counts rather than treating text agreement as exact token agreement.

## Saved evidence

Reference caches contain rendered prompt IDs, reference continuation IDs/text, fixed scoring partitions, probabilities, source metadata, and a reference signature. Candidate samples contain independent generation and per-position requested probabilities and metrics. Cache signatures prevent mixing incompatible settings.

The CLI exposes coarse KL, full-vocabulary KL where the server supports it, token agreement, and chat/Responses message agreement. Full-vocabulary scoring can be much more expensive. A finite time budget can leave an incomplete run; always check `status`, `completed`, and `errors` before publishing metrics.

## Resuming an interrupted evaluation

Use `--resume-from` with a new output directory to reuse successful samples from an identical candidate/reference configuration and retry missing or failed samples. Existing outputs are not overwritten. Reference capture supports `--resume` for its own cache directory.

For comparisons, `--aggregation domain_macro` gives each domain equal weight with task/document resampling. Its paired confidence interval is distinct from a mean across repeated runs. Three repeated runs are useful for screening variability but provide limited evidence about long-term reproducibility or benchmark performance.
