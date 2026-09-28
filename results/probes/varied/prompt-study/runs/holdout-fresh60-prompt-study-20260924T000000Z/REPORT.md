# Fresh60 prompt-variant holdout evaluation

**Status:** scored only after all eight prediction files were saved and hash-frozen. No fitting, tuning, or test-driven selection occurred. The probe is **mixed GPU/CPU**: MLX/Metal generated embeddings; the final frozen typed scorer was applied with CPU NumPy. This is not an all-GPU result.

Fixture SHA-256: `316b3c2e1d0c5cf50356d0e236a5f4adf27fdac9ee6d6c6b56a14f5354ef6a9f`. Frozen train config SHA-256: `305a124523f12e28686bd037d8be96719717ee36cbfc86bfc7b3f8c41b7fd325`.

## Per-model, per-variant summary

| Model | Variant | Type-balanced accuracy | Type-macro candidate-local F1 | All three correct |
|---|---|---:|---:|---:|
| qwen | v2-baseline | 0.500 | 0.212 | 5/60 (0.083) |
| qwen | readable-state | 0.456 | 0.194 | 5/60 (0.083) |
| qwen | rule-in-candidate | 0.494 | 0.198 | 5/60 (0.083) |
| qwen | joint-context-candidate | 0.439 | 0.177 | 4/60 (0.067) |
| harrier | v2-baseline | 0.561 | 0.241 | 8/60 (0.133) |
| harrier | readable-state | 0.511 | 0.209 | 5/60 (0.083) |
| harrier | rule-in-candidate | 0.522 | 0.210 | 7/60 (0.117) |
| harrier | joint-context-candidate | 0.650 | 0.269 | 12/60 (0.200) |

## Typed metrics

Accuracy CIs are Wilson 95% descriptive intervals. Score MAE compares the model's probability-weighted expected value with the numeric value of the reference rubric label; normalized MAE divides each error by that question's rubric span.

### qwen
| Variant | Type | Correct / n | Accuracy (Wilson 95% CI) | Candidate-local F1 | Expected-value MAE / normalized |
|---|---|---:|---:|---:|---:|
| v2-baseline | choice | 29/60 | 0.483 [0.362, 0.607] | 0.176 | — |
| v2-baseline | noul | 49/60 | 0.817 [0.701, 0.894] | 0.408 | — |
| v2-baseline | score | 12/60 | 0.200 [0.118, 0.318] | 0.052 | 4.308 / 0.375 |
| readable-state | choice | 28/60 | 0.467 [0.346, 0.591] | 0.172 | — |
| readable-state | noul | 43/60 | 0.717 [0.592, 0.815] | 0.358 | — |
| readable-state | score | 11/60 | 0.183 [0.106, 0.299] | 0.051 | 4.438 / 0.377 |
| rule-in-candidate | choice | 33/60 | 0.550 [0.425, 0.669] | 0.195 | — |
| rule-in-candidate | noul | 39/60 | 0.650 [0.524, 0.758] | 0.325 | — |
| rule-in-candidate | score | 17/60 | 0.283 [0.185, 0.408] | 0.074 | 3.754 / 0.324 |
| joint-context-candidate | choice | 30/60 | 0.500 [0.377, 0.623] | 0.173 | — |
| joint-context-candidate | noul | 35/60 | 0.583 [0.457, 0.699] | 0.292 | — |
| joint-context-candidate | score | 14/60 | 0.233 [0.144, 0.354] | 0.066 | 3.903 / 0.362 |

### harrier
| Variant | Type | Correct / n | Accuracy (Wilson 95% CI) | Candidate-local F1 | Expected-value MAE / normalized |
|---|---|---:|---:|---:|---:|
| v2-baseline | choice | 31/60 | 0.517 [0.393, 0.638] | 0.189 | — |
| v2-baseline | noul | 57/60 | 0.950 [0.863, 0.983] | 0.475 | — |
| v2-baseline | score | 13/60 | 0.217 [0.131, 0.336] | 0.058 | 4.624 / 0.425 |
| readable-state | choice | 33/60 | 0.550 [0.425, 0.669] | 0.202 | — |
| readable-state | noul | 42/60 | 0.700 [0.575, 0.801] | 0.350 | — |
| readable-state | score | 17/60 | 0.283 [0.185, 0.408] | 0.075 | 4.607 / 0.423 |
| rule-in-candidate | choice | 42/60 | 0.700 [0.575, 0.801] | 0.243 | — |
| rule-in-candidate | noul | 40/60 | 0.667 [0.541, 0.773] | 0.333 | — |
| rule-in-candidate | score | 12/60 | 0.200 [0.118, 0.318] | 0.053 | 5.094 / 0.435 |
| joint-context-candidate | choice | 44/60 | 0.733 [0.610, 0.829] | 0.247 | — |
| joint-context-candidate | noul | 60/60 | 1.000 [0.940, 1.000] | 0.500 | — |
| joint-context-candidate | score | 13/60 | 0.217 [0.131, 0.336] | 0.059 | 4.943 / 0.423 |

## Paired comparisons

Exact two-sided McNemar tests and normal-approximation paired accuracy-difference intervals compare each challenger with v2 baseline within the same model. They are descriptive, not a test-set selection mechanism.

### qwen
| Variant | Type | Variant-only correct | Baseline-only correct | Exact McNemar p | Paired delta [95% descriptive CI] |
|---|---|---:|---:|---:|---:|
| readable-state | choice | 0 | 1 | 1.0000 | -0.017 [-0.049, 0.016] |
| readable-state | noul | 5 | 11 | 0.2101 | -0.100 [-0.228, 0.028] |
| readable-state | score | 3 | 4 | 1.0000 | -0.017 [-0.103, 0.070] |
| rule-in-candidate | choice | 7 | 3 | 0.3438 | 0.067 [-0.035, 0.169] |
| rule-in-candidate | noul | 2 | 12 | 0.0129 | -0.167 [-0.281, -0.052] |
| rule-in-candidate | score | 10 | 5 | 0.3018 | 0.083 [-0.041, 0.208] |
| joint-context-candidate | choice | 8 | 7 | 1.0000 | 0.017 [-0.110, 0.143] |
| joint-context-candidate | noul | 2 | 16 | 0.0013 | -0.233 [-0.359, -0.108] |
| joint-context-candidate | score | 7 | 5 | 0.7744 | 0.033 [-0.080, 0.146] |

### harrier
| Variant | Type | Variant-only correct | Baseline-only correct | Exact McNemar p | Paired delta [95% descriptive CI] |
|---|---|---:|---:|---:|---:|
| readable-state | choice | 3 | 1 | 0.6250 | 0.033 [-0.031, 0.098] |
| readable-state | noul | 0 | 15 | 0.0001 | -0.250 [-0.360, -0.140] |
| readable-state | score | 7 | 3 | 0.3438 | 0.067 [-0.035, 0.169] |
| rule-in-candidate | choice | 13 | 2 | 0.0074 | 0.183 [0.066, 0.301] |
| rule-in-candidate | noul | 3 | 20 | 0.0005 | -0.283 [-0.423, -0.144] |
| rule-in-candidate | score | 7 | 8 | 1.0000 | -0.017 [-0.143, 0.110] |
| joint-context-candidate | choice | 15 | 2 | 0.0023 | 0.217 [0.094, 0.340] |
| joint-context-candidate | noul | 3 | 0 | 0.2500 | 0.050 [-0.005, 0.105] |
| joint-context-candidate | score | 9 | 9 | 1.0000 | 0.000 [-0.139, 0.139] |

## Train-only CV versus fresh60

The train column is the pre-existing five-fold OOF result for the same text variant (with the train runner's internally selected scorer). Fresh60 uses the single frozen v2-baseline scorer unchanged for every text variant, so this is descriptive cross-protocol comparison, not a test-time refit or selection.

### qwen
| Prompt variant | Train-only OOF type-balanced accuracy | Fresh60 type-balanced accuracy | Fresh60 − train OOF |
|---|---:|---:|---:|
| v2-baseline | 0.528 | 0.500 | -0.028 |
| readable-state | 0.556 | 0.456 | -0.101 |
| rule-in-candidate | 0.513 | 0.494 | -0.019 |
| joint-context-candidate | 0.500 | 0.439 | -0.061 |

| Prompt variant | Type | Train-only OOF accuracy | Fresh60 accuracy | Fresh60 − train OOF |
|---|---|---:|---:|---:|
| v2-baseline | choice | 0.400 | 0.483 | +0.083 |
| v2-baseline | noul | 0.723 | 0.817 | +0.094 |
| v2-baseline | score | 0.460 | 0.200 | -0.260 |
| readable-state | choice | 0.446 | 0.467 | +0.021 |
| readable-state | noul | 0.773 | 0.717 | -0.056 |
| readable-state | score | 0.450 | 0.183 | -0.267 |
| rule-in-candidate | choice | 0.404 | 0.550 | +0.146 |
| rule-in-candidate | noul | 0.645 | 0.650 | +0.005 |
| rule-in-candidate | score | 0.490 | 0.283 | -0.207 |
| joint-context-candidate | choice | 0.450 | 0.500 | +0.050 |
| joint-context-candidate | noul | 0.550 | 0.583 | +0.033 |
| joint-context-candidate | score | 0.500 | 0.233 | -0.267 |

### harrier
| Prompt variant | Train-only OOF type-balanced accuracy | Fresh60 type-balanced accuracy | Fresh60 − train OOF |
|---|---:|---:|---:|
| v2-baseline | 0.595 | 0.561 | -0.034 |
| readable-state | 0.587 | 0.511 | -0.076 |
| rule-in-candidate | 0.603 | 0.522 | -0.081 |
| joint-context-candidate | 0.611 | 0.650 | +0.039 |

| Prompt variant | Type | Train-only OOF accuracy | Fresh60 accuracy | Fresh60 − train OOF |
|---|---|---:|---:|---:|
| v2-baseline | choice | 0.496 | 0.517 | +0.021 |
| v2-baseline | noul | 0.755 | 0.950 | +0.195 |
| v2-baseline | score | 0.535 | 0.217 | -0.318 |
| readable-state | choice | 0.496 | 0.550 | +0.054 |
| readable-state | noul | 0.755 | 0.700 | -0.055 |
| readable-state | score | 0.510 | 0.283 | -0.227 |
| rule-in-candidate | choice | 0.525 | 0.700 | +0.175 |
| rule-in-candidate | noul | 0.759 | 0.667 | -0.092 |
| rule-in-candidate | score | 0.525 | 0.200 | -0.325 |
| joint-context-candidate | choice | 0.504 | 0.733 | +0.229 |
| joint-context-candidate | noul | 0.764 | 1.000 | +0.236 |
| joint-context-candidate | score | 0.565 | 0.217 | -0.348 |

## Provenance, cost, and limits

Every variant has an MLX/Metal feature export, text/profile/tokenizer/checkpoint manifest, prediction JSONL, feature SHA-256, prediction SHA-256, measured export wall time, exact query-plus-candidate token count, and estimated dense embedding FLOPs in `run-metadata.json`. Candidate score/order permutation is asserted on every question; explicit supported/unsupported polarity order checks are asserted for every noul question. Checkpoint hashes and exact feature text/token counts are in the exporter manifests.

FLOPs use `2 × 600,000,000 parameters × sum(query and candidate tokens)`. This is a transparent arithmetic estimate, not measured hardware FLOPs, and omits attention/cache and implementation details. Export wall timing includes exporter process startup, model/checkpoint verification/load, embedding, and output writing; it is not scorer latency.

The prompt variants are evaluated with the **unchanged training-frozen v2 typed weights** to isolate text/embedding changes without test fitting. Differences from train-only grouped CV are reported but do not alter the frozen selection. Embedding plus linear scoring does not acquire language-model reasoning just from prompt repetition.

Full per-family and per-domain total and per-type supports/correct counts/accuracies are in `holdout-metrics.json` and the flat `family-domain-metrics.csv`. All predictions were frozen before reference scoring in `PRE_REFERENCE_FREEZE.json`. No JEV/API was called.

All-MLX GPU parity is pending a separate agent: the nested-agent tool hit a depth limit, and the existing Metal runner does not directly accept these four isolated text variants. See `GPU-PARITY-HANDOFF.md`; this report remains mixed GPU/CPU.
