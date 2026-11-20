# Full-vocabulary probability export performance

Returning full-vocabulary probabilities through an OpenAI-compatible endpoint is substantially more expensive than returning Top-K probabilities. A 32-position MiniMax M3.1 continuation contains 6,402,048 probability entries. Binary reference storage reduces local serialization work, but the serving engine also needs to avoid unnecessary work while exporting those entries.

## Token-ID-only log probabilities

In the tested MiniMax vLLM image, `return_tokens_as_token_ids=True` formats the final response with token ID keys but does not skip internal vocabulary detokenization. `LogprobsProcessor._update_sample_logprobs()` calls `convert_ids_list_to_tokens()`, which decodes each vocabulary token separately at every generated position. A 20-second live CPU stack sample found 84.15% of active samples within that conversion subtree and 13.04% within probability-object construction. These are sample proportions for that interval, not a full-run wall-clock breakdown.

[The narrow vLLM patch](../patches/vllm-full-logprobs-token-id-only.patch) forwards an ID-only flag from `CompletionRequest` only when both `logprobs=-1` and `return_tokens_as_token_ids=True` are requested. It disables the tokenizer in the probability processor for those requests. The ordinary output detokenizer is retained, as are raw probability values and model computation. Normal text-logprob requests and Top-K requests keep their existing path.

Apply the patch in the corresponding vLLM source root, then restart the service. The patch targets the package layout in `vllm/vllm-openai-rocm:minimax-m31-flash` used by this experiment; other vLLM revisions may require adaptation. It is a probability-export performance change, separate from any model-support PR.

Before using a new reference, verify all vocabulary IDs, probability normalization, all requested scoring positions, and token output. Use the same fixed prompt for before/after measurements. Engine numerical variability should be reported separately from exact preservation of the export operation.

## Reference storage

Reference capture requests all probabilities once while greedily generating 32 tokens. Full raw values are written as uncompressed little-endian float64 arrays; small metadata uses gzip level 1. Offline measurement on one 32-position cache reduced storage time from 19.58 seconds to 1.18 seconds while preserving every returned float value exactly. That storage-only result does not predict end-to-end server speed.
