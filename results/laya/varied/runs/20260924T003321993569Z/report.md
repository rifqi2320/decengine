# Prompt-study fresh60: Laya MLX / JEV comparison

Run: `20260924T003321993569Z`
Fixture SHA-256: `316b3c2e1d0c5cf50356d0e236a5f4adf27fdac9ee6d6c6b56a14f5354ef6a9f`
Laya pinned checkpoint SHA-256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`

## Accuracy

| Model | Choice | Noul | Score | All three |
|---|---:|---:|---:|---:|
| Laya MLX | 34/60 (56.7%) | 45/60 (75.0%) | 16/60 (26.7%) | 10/60 (16.7%) |
| JEV | 51/60 (85.0%) | 59/60 (98.3%) | 47/60 (78.3%) | 38/60 (63.3%) |
| Qwen v2-baseline shared head, C=1 | 29/60 (48.3%) | 49/60 (81.7%) | 12/60 (20.0%) | 5/60 (8.3%) |
| Harrier v2-baseline shared head, C=0.1 | 31/60 (51.7%) | 57/60 (95.0%) | 13/60 (21.7%) | 8/60 (13.3%) |

Per-domain case supports and per-type accuracy for all four systems, plus choice-option-count slices, are in `metrics.json`.

Case p50/p95 latency: Laya three-call sum 917.3/10974.4 ms; JEV single network request 769.6/3568.2 ms.

Qwen/Harrier comparators are read-only, all-MLX v2-baseline shared-head outputs with exact CPU typed-output parity (Qwen C=1; Harrier C=0.1). Train-only selection remains the frozen baseline; no per-variant heads or other prompt variants are included in this primary comparison.

Pinned Laya root checkpoint ran on MLX/Metal. Every state received three independent typed Laya calls and one JEV typed multi-question request. Original question wires were used; top-level metadata and references were excluded from inputs.

Raw model/API responses and prediction files were saved before reference scoring. No prompt tuning or test-driven selection was performed.

Completed 60/60 cases, 180 Laya decisions, and 60 JEV requests; errors Laya 0, JEV 0.
