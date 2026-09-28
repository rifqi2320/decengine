# Explicit-evidence authored clarity intervention

Run: `20260923T153215418583Z`
Fixture SHA-256: `4cb002e1c29f5273ecc23fcdf742b093cb5433d661f5df0b50d209a534fb1409`
Pinned checkpoint SHA-256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`

This is an authored clarity intervention on the same 60 matched cases, not an independent holdout. Added evidence facts are included in state; evaluation metadata and references are excluded from model input.

## Overall accuracy

| Model | Choice | Noul | Score | All three |
|---|---:|---:|---:|---:|
| Laya MLX | 31/60 (51.7%) | 31/60 (51.7%) | 19/60 (31.7%) | 4/60 (6.7%) |
| JEV | 46/60 (76.7%) | 51/60 (85.0%) | 33/60 (55.0%) | 24/60 (40.0%) |

Per-domain and choice-option-count accuracies are in `metrics.json`.

## Paired outcomes vs native matched baseline

| Model / question | Prediction flips | Baseline correct | Explicit correct | Correct→incorrect | Incorrect→correct |
|---|---:|---:|---:|---:|---:|
| Laya MLX / response_plan | 25/60 | 19/60 | 31/60 | 3 | 15 |
| Laya MLX / immediate_intervention | 18/60 | 31/60 | 31/60 | 9 | 9 |
| Laya MLX / record_auditability | 7/60 | 19/60 | 19/60 | 3 | 3 |
| Laya MLX / all three | — | 3/60 | 4/60 | 1 | 2 |
| JEV / response_plan | 27/60 | 20/60 | 46/60 | 0 | 26 |
| JEV / immediate_intervention | 30/60 | 29/60 | 51/60 | 4 | 26 |
| JEV / record_auditability | 19/60 | 23/60 | 33/60 | 1 | 11 |
| JEV / all three | — | 4/60 | 24/60 | 1 | 21 |

Cases with any prediction flip: Laya 39/60; JEV 47/60.
Case-latency p50/p95 (ms): Laya 3-call sum 405.7/1626.9; JEV single request 387.2/538.8.

Per-call timings and errors are in `timings.jsonl`; raw responses and normalized outcomes are in `raw-results.jsonl`. Prediction files were saved before reference scoring. No prompt/model tuning was performed.
The fixture integrity validator confirmed unchanged question wires and targets after predictions were saved. `evaluation_metadata` and `reference` were not passed to either model.
This is an authored, label-informed clarity intervention, not an independent holdout and not evidence for a generalization-gain claim.
Errors: Laya 0, JEV 0.
