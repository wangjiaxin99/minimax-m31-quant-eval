# Original MiniMax M3.1 overall results

These are the original three-round overall results on the historical **300-prompt** input set. They are retained as numerical aggregates only. The original 300-prompt inputs were restored on 2026-11-18. These remain historical measurements with their original reference and environment, distinct from the later **240-prompt** reruns.

| Model | Mean bucketed KL (nat) ↓ | Shared-prefix Top-1 agreement ↑ | Exact 32-token agreement ↑ |
|---|---:|---:|---:|
| MXFP8 (rerun) | 0.01577440 | 97.55% | 49.56% |
| MXFP4 | 0.03158934 | 96.50% | 38.44% |
| MXFP4 RT | 0.03519367 | 96.20% | 38.67% |
| MXFP4 QADv1 | 0.02610190 | 96.86% | 44.33% |

Each model completed three rounds of 300 prompts with zero evaluation errors. Every prompt generated 32 tokens; all 32 positions were requested for common-prefix scoring. Direct-position Top-1 uses 28,800 positions per model, and exact 32-token agreement uses 900 sequences per model.

All candidates used the same later engine environment and one fixed MXFP8 reference captured in an earlier environment. The MXFP8 rerun is therefore not a comparison against a new cache captured in its own environment. Its nonzero KL and imperfect agreement include environment and repeated-execution differences.

QADv1 improved mean bucketed KL by 17.37% relative to ordinary MXFP4. Top-1 agreement increased by 0.36 percentage points; exact sequence agreement increased by 5.89 percentage points. QADv1's three rounds took 15.22, 14.81, and 14.78 minutes (44.81 minutes total), excluding model loading and server startup.

All 32 positions were requested, but finite KL was available at 6,383/9,600 positions in each QADv1 round under the existing numerical policy. Top-1 used all 9,600 positions. Fidelity improvements are not task-success measurements and do not isolate every possible checkpoint difference.

[Recomputed results for the four retained public categories](results.md) use a different aggregation scope: 240 prompts per round rather than 300. Use a matching reference for the selected dataset: the default is restored `quick300.jsonl`; `quick240.jsonl` is retained for the four-category scope.
