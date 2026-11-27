# Self-generated calibration prompts: Qwen3.6-35B-A3B, A-small

These 280 prompts feed the self-generated calibration set **A-small**. The BF16 model answers each prompt once. The
answers, prompt included, are packed into calibration segments for GPTQ. The method follows the QDA paper, where
calibrating GPTQ on the model's own answers lowered the KL of this model by 30% at W4A16.

`prompts.jsonl` has one row per prompt: `{id, domain, source, used_in_a_small, messages}`. Every row is ready for
`m31-eval prefill-kl render`. `used_in_a_small` marks the 123 prompts whose answers were packed. The others were
generated as surplus so that each domain could reach its token share.

`manifest.json` records:
- the sources with pinned revisions, filters and counts;
- the decontamination;
- the generation settings and answer lengths;
- the packing.

## Composition

| Domain | Prompts | Packed | Token share | Source | License |
|---|---|---|---|---|---|
| math | 55 | 28 | 35.6% | NuminaMath-CoT (olympiads, cn_k12, math, synthetic_math; no AMC/AIME sources) | Apache-2.0 |
| science | 40 | 29 | 20.4% | OpenThoughts-114k, biology / chemistry / physics | Apache-2.0 |
| code | 30 | 14 | 17.9% | OpenThoughts-114k, code | Apache-2.0 |
| chat | 75 | 33 | 14.8% | UltraChat-200k first turn (50), Dolly-15k (25) | MIT; CC BY-SA 3.0 |
| tool | 40 | 9 | 5.9% | Hermes function-calling v1, system with tool signatures + first user turn | Apache-2.0 |
| multilingual | 40 | 10 | 5.3% | Aya dataset, non-Latin-script languages | Apache-2.0 |

## How it was built

1. **Sample.**
   - Datasets were streamed and shuffled with seed 0, then filtered by source, domain and length.
   - The revisions are the Hub heads on 2026-11-26, pinned in the manifest.
2. **Decontaminate.** A prompt was dropped if it shares any 13-gram of lowercased words, or its normalized text,
   with one of these sets:
   - `data/mixed/quick300.jsonl` (the KL prompts);
   - GPQA-Diamond;
   - IFBench;
   - AIME 2025 and 2026.

   No prompt was dropped. HMMT and LiveCodeBench were not checked. Neither is scored in this experiment.
3. **Generate.**
   - `prefill-kl render`, then `gen` with Qwen3.6-35B-A3B in BF16.
   - Thinking on, model-card sampling, seed 17, at most 8,192 tokens, one answer per prompt.
   - Mean answer length: math 6.3K, code 6.7K, science 3.6K, others about 2K tokens.
   - 29 math answers and 18 code answers reach 8,192 tokens.
4. **Pack.**
   - Samples are taken in a seeded order until each domain reaches its token target: math 36%, science 21%,
     code 18%, chat 15%, tool 5%, multilingual 5%.
   - The samples are shuffled and concatenated as prompt + answer + EOS.
   - The result is cut into 128 segments of 4,096 tokens (524,288 tokens) with no padding.
5. **Calibrate.** Quark's default pileval set, 512 x 512 and unchanged, is followed by the packed tokens. Every
   other GPTQ argument is the same as the pileval-only build.
   - Quark's layer-wise calibration reuses the first row's rotary embeddings for every row, so all rows must have the
     same length. Mixing 512-token and 4,096-token rows fails.
   - The 524,288 packed tokens were therefore cut into 1,024 rows of 512 tokens: the same tokens in the same order.
6. **Result.** On 2026-11-26 the build took 3.0 h, against 1.0 h with pileval alone (3x the positions). The prefill KL
   (K = 2000, token mean) was 0.07164, against 0.08159 for the pileval-only X1-GPTQ (-12.2%) and 0.08961 for
   X1-RTN. With the answer mean: 0.08433, 0.09308 (-9.4%) and 0.10253. Both KL scores were bitwise identical.

## License and attribution

The prompts keep the licenses of their sources, listed above and in the manifest. The Dolly-15k rows are from
`databricks/databricks-dolly-15k` (Databricks, CC BY-SA 3.0) and are redistributed under the same license. The rows
are the instruction and the context, unchanged.
