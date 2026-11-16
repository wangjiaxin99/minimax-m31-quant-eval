# Historical public-category results

These aggregate results were recomputed from the four public categories of earlier MiniMax M3.1 experiments. No model was rerun for this publication. Each model has three rounds over the same 240 retained prompts, totaling 720 sequences and 23,040 shared-prefix positions.

The reference was a fixed MXFP8 cache captured in an earlier engine environment. All candidates used the same later engine environment. MXFP8 below is a rerun against that earlier cache; its nonzero KL and imperfect agreement therefore include environment/repeated-execution differences. These values describe the retained categories of the historical protocol, rather than a new reference capture on the public-only dataset.

## Overall public-category aggregates

| Model | Mean bucketed KL (nat) ↓ | Shared-prefix Top-1 ↑ | Exact 32-token ↑ |
|---|---:|---:|---:|
| MXFP8 | 0.01749533 | 97.44% | 46.11% |
| MXFP4 | 0.03580128 | 96.30% | 34.58% |
| RT | 0.03926414 | 96.06% | 34.17% |
| QADv1 | 0.02929287 | 96.70% | 41.39% |

## Per-category aggregates

| Category | Model | Mean bucketed KL (nat) ↓ | Shared-prefix Top-1 ↑ | Exact 32-token ↑ |
|---|---|---:|---:|---:|
| Terminal-Bench 4.0 | MXFP8 | 0.00806755 | 99.10% | 75.56% |
| Terminal-Bench 4.0 | MXFP4 | 0.01658966 | 98.72% | 66.67% |
| Terminal-Bench 4.0 | RT | 0.01768301 | 98.70% | 64.44% |
| Terminal-Bench 4.0 | QADv1 | 0.01373138 | 98.87% | 70.00% |
| Mathematics | MXFP8 | 0.01486938 | 96.94% | 39.44% |
| Mathematics | MXFP4 | 0.03089298 | 95.42% | 30.00% |
| Mathematics | RT | 0.03501143 | 95.28% | 30.00% |
| Mathematics | QADv1 | 0.02304132 | 95.73% | 31.11% |
| Non-Latin languages | MXFP8 | 0.02814428 | 96.49% | 29.44% |
| Non-Latin languages | MXFP4 | 0.05984021 | 94.69% | 18.89% |
| Non-Latin languages | RT | 0.06383434 | 95.24% | 23.33% |
| Non-Latin languages | QADv1 | 0.05064490 | 95.56% | 26.11% |
| Long documents | MXFP8 | 0.01890013 | 97.24% | 40.00% |
| Long documents | MXFP4 | 0.03588229 | 96.39% | 22.78% |
| Long documents | RT | 0.04052779 | 95.03% | 18.89% |
| Long documents | QADv1 | 0.02975388 | 96.63% | 38.33% |

KL averages valid positions within a prompt before averaging prompts and rounds. Top-1 is counted directly across all requested positions. Exact sequence agreement requires all 32 generated token IDs to match. These are fidelity metrics, not benchmark pass rates.

The public export includes aggregate counts and numerical metrics only, without historical reference caches, prompts from removed sources, server logs, or machine-specific paths. Use the bundled public prompt list and create your own reference cache to reproduce the workflow.

[Aggregate CSV](../examples/historical-public-category-results.csv) contains full precision and finite-KL coverage.
