# Fidelity metrics and API behavior

For a prompt `x` and reference-generated continuation `y`, position `t` is scored on the common prefix `x + y[:t]`. This keeps the conditioning input identical even if independently generated candidate output has already diverged.

## Full-vocabulary KL

Let `P` denote reference probabilities and `Q` candidate probabilities over the same complete vocabulary `V`:

```text
KL(P || Q) = sum(P(v) * (log P(v) - log Q(v)) for v in V)
```

Natural logarithms give units of nats. Token contributions are summed, not averaged. Full-vocabulary KL is the only supported KL calculation. MiniMax M3.1 has 200,064 vocabulary entries. Reference capture requests `logprobs=-1` once while greedily generating the continuation, then selects the requested positions from that response. Candidate scoring requests the same complete vocabulary under each reference prefix.

Both distributions must have matching token-ID sets and normalized probability mass (tolerance 0.00002). Missing IDs, malformed probabilities, or zero candidate probability at positive reference mass fail the position and mark its prompt incomplete. Near-deterministic reference positions remain included. No probability mass is renormalized or replaced with arbitrary values.

`prompt_mean_kl` averages positions within each successful prompt, then averages prompts. A complete 300-prompt, 32-position run requires 9,600 successful scores and zero failed positions/prompts. Partial-run aggregates are diagnostic only; check coverage and completion before comparing candidates. Actual vocabulary size should also be checked using `doctor --probe-full --expected-vocab-size`.

## Agreement metrics

Shared-prefix Top-1 uses each position's argmax under the common reference prefix. For a complete 300-prompt, 32-position run, direct-position agreement is:

```text
sum(position['top1_agreement'] for every sample and scored position) / 9600
```

The native `same_context_top1_agreement.mean` is task-cluster weighted and may differ from this direct count because a source document can contribute multiple prompts. Keep the aggregation explicit.

Exact 32-token agreement compares independently generated token-ID sequences. One different token makes that sequence a non-match. Check generated lengths and EOS statistics; inspect `prefix_at_32` and token counts rather than treating text agreement as exact token agreement.

## Saved evidence

Reference caches contain rendered prompt IDs, reference continuation IDs/text, complete per-position raw log probabilities, source metadata, and a reference signature. New probability files use uncompressed little-endian float64 arrays in `distributions/`. Small gzip-level-1 JSON metadata gives array shape, file paths and vocabulary-ID mapping. This preserves returned Python float values exactly, including negative infinity; it does not downcast to float32. Contiguous IDs starting at zero are implicit. MiniMax probability storage is 300 × 32 × 200,064 × 8 = 15,364,915,200 bytes (15.37 GB / 14.31 GiB), excluding small metadata. The loader can still read old full-distribution JSON caches. Capture strategy and storage format are part of the new reference signature; a changed format must not silently resume a previous capture. Candidate distributions are scored in memory. Samples retain independent generation, per-position metrics, reference-top-token probability evidence, vocabulary size and a digest of the complete candidate distribution; complete candidate distributions are not persisted. Cache signatures prevent mixing incompatible settings.

The CLI defaults to full-vocabulary KL and all 32 continuation positions. It also exposes token agreement and chat/Responses message agreement. Old probability-partition reference caches must be recaptured; they cannot be upgraded to full-vocabulary data. Full-vocabulary scoring can be much more expensive. A finite time budget can leave an incomplete run; always check `status`, `completed`, and `errors` before publishing metrics.

## Resuming an interrupted evaluation

Use `--resume-from` with a new output directory to reuse successful samples from an identical candidate/reference configuration and retry missing or failed samples. Existing outputs are not overwritten. Reference capture supports `--resume` for its own cache directory.

For comparisons, `--aggregation domain_macro` gives each domain equal weight with task/document resampling. Its paired confidence interval is distinct from a mean across repeated runs. Three repeated runs are useful for screening variability but provide limited evidence about long-term reproducibility or benchmark performance.

## Independent generation versus reference-prefix scoring

Both models start independent generation from the original prompt. Exact-32 compares those two free-running sequences. For KL and Top-1, the candidate is separately queried on `prompt + reference_tokens[:t]` at each position `t`. This is the same conditioning policy used by historical measurements. It isolates next-token probability changes from the consequences of an earlier generation divergence; it does not force the candidate's independent generation to follow the reference.
