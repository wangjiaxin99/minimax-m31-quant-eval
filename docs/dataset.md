# Fidelity dataset

`data/mixed/quick300.jsonl` restores the original 300 fixed prompts, five categories with 60 prompts each, and 260 original tasks/documents. `manifest.json` records the original dataset hash, counts, source revisions, and splits. The 240-prompt variant is retained as `quick240.jsonl`, with `manifest240.json` and `holdout240.jsonl`.

Restoration on 2026-11-18 corrects an earlier removal of fidelity inputs: the requested removal concerned only the standalone DeepSWE task benchmark. Existing 240-prompt measurements keep their original scope.

## Sources

| Category | Selection | Source and license |
|---|---|---|
| DeepSWE | 60 distinct tasks; one frozen initial/middle/late agent context per task | [Task definitions](https://github.com/datacurve-ai/deep-swe), Apache-2.0; shared trace provenance separately applies, as retained in each input |
| Terminal-Bench 4.0 | 60 distinct tasks, stratified by category/difficulty; original instructions plus a bash tool schema | [Task repository](https://github.com/harbor-framework/terminal-bench), Apache-2.0 and retained task-specific notices |
| Mathematics | 60 additional problems, stratified using lexical topic proxies and input length; no expected answers or generated solutions | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), CC-BY-4.0 |
| Non-Latin languages | 10 questions per language: Arabic, Bengali, Hindi, Japanese, Korean, Chinese; original questions/options and localized response instructions | [Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite), Apache-2.0 |
| Long documents | 10 CPython documents and 10 Gutenberg works; three different windows per document | CPython/Python license and public-domain works with retained Gutenberg terms |

The long-document windows target 32K/64K/128K **characters**, not tokens. Source texts shorter than a target window retain their available content and record actual length. Use `mixed audit` with your tokenizer before evaluation.

The four public categories use fixed seed 31, source strata, and stable hashing, without selecting by candidate KL or task success. DeepSWE restores the historical input snapshot (`deepswe60.jsonl`), including its original messages, tools, stage, task and split. Its historical NVFP4 trace selection favored successful runs; that provenance bias remains. The 240/60 development/holdout split is grouped by original task/document; multilingual variants use original sample identifiers to avoid duplicated questions across languages. Calibration overlap is unverified unless separately checked against your calibration inputs. No training-data disjointness is claimed.

Source metadata and license notices are bundled in `data/mixed/licenses/` and `source-downloads.json`. Public availability does not override source license terms. MathArena is not bundled; the mathematical category uses OpenMathReasoning.

## Custom inputs

Each JSONL line represents one input, for example:

```json
{"id":"example-1","task":"example-1","domain":"general","split":"holdout","messages":[{"role":"user","content":"Explain why the sky is blue."}]}
```

Use unique `id` values. Group related windows under the same `task` for cluster-aware reporting. A row can include an OpenAI-style `tools` list when your server supports that template. Never include reference answers as part of an input intended to test independent generation.

Pass any such JSONL to `m31-eval reference --dataset ...`; its size need not be 300. Candidate evaluation reuses the frozen inputs from that reference cache. Changes to prompt text, selection, tokenizer, or reference-scoring settings require a new cache.

## Fetching and rebuilding

Evaluation of the bundled JSONL requires no dataset libraries. Rebuilding requires the optional dependencies:

```bash
python -m pip install -e '.[build]'
m31-eval mixed fetch --assets assets/public-sources
m31-eval mixed build --assets assets/public-sources --output data/rebuilt
```

The builder uses pinned Git/Hugging Face revisions and validates downloaded checksums. Gutenberg's dynamic download packaging can change upstream; use a checksum-verified source snapshot when exact rebuilding is required. The default builder appends the bundled frozen `deepswe60.jsonl`; `--deepswe-prompts PATH` selects an explicit snapshot. It does not import a benchmark harness or task-subset manifest. Use `--public-only` to rebuild the four-category 240-prompt variant.

Terminal-Bench 2.1 is an optional alternative selected with `--terminal-version 2.1`; it produces its own prompt list and needs a separate reference cache.

```bash
m31-eval mixed audit --dataset data/mixed/quick300.jsonl \
  --base-url http://localhost:8000/v1 \
  --max-prompt-tokens 65536 --output logs/token-audit.json
```

The evaluator refuses prompts beyond its configured input limit rather than silently truncating them.
