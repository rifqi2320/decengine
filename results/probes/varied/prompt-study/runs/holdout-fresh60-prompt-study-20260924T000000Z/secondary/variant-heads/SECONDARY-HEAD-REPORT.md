# Secondary fresh60 evaluation: per-variant frozen heads

This is a secondary comparison on the **same already-seen fresh60 holdout**, not an independent test. The scorer bundles were already frozen from clean300+multi120 train-only features before the first reference scoring. This stage performed no fitting, tuning, or test-based selection. Embeddings/features were reused; only the scorer head changed. CPU NumPy scoring is the reference; this report is not all-GPU.

Predictions were written to `predictions/{model}/{variant}.jsonl` and frozen in `variant-head-freeze.json` before this metric stage read references. Full model/head/scaler hashes and per-family/per-domain results are in `variant-head-run-metadata.json` and `secondary-head-metrics.json`.

| Model | Variant | C | Head SHA-256 | Type-balanced accuracy | Type-macro local F1 | All-three | Fixed-v2 prompt-only accuracy | Delta |
|---|---|---:|---|---:|---:|---:|---:|---:|
| qwen | v2-baseline | 1 | `5b7260f04f0b66784504b09526447b30a9f020a554abf611c57eb971f9fabd79` | 0.500 | 0.212 | 5/60 | 0.500 | +0.000 |
| qwen | readable-state | 1 | `b3d19ab873b2ffa90a4f8af63833a3b3195e1d17a8976a107021aafa259e8436` | 0.472 | 0.200 | 4/60 | 0.456 | +0.017 |
| qwen | rule-in-candidate | 0.01 | `62a8f7f663465f082231c885de0e6fc439d734901bc51e90aafe23eb4685f9ca` | 0.467 | 0.176 | 6/60 | 0.494 | -0.028 |
| qwen | joint-context-candidate | 0.01 | `35fa698c58583def77da4db762a9b2504ebb0adacd8bfa63471128eed23daa54` | 0.589 | 0.247 | 6/60 | 0.439 | +0.150 |
| harrier | v2-baseline | 0.1 | `5fb0727296cc89d41c9ebcec1664a7ef3a99e5c08c6da424d267f02ea6c22d30` | 0.561 | 0.241 | 8/60 | 0.561 | +0.000 |
| harrier | readable-state | 1 | `831a1558e197e3a949b02e667887abf8824c4196c2c31ab58ddb60ab1c83f49f` | 0.506 | 0.211 | 11/60 | 0.511 | -0.006 |
| harrier | rule-in-candidate | 0.1 | `c8b38381b695dc71269139171d1b4d83c6db641d8e6be4e581484b5d7b71ed4a` | 0.561 | 0.235 | 7/60 | 0.522 | +0.039 |
| harrier | joint-context-candidate | 0.1 | `9673793192b590b964e1384ea29a45a554625f68cb4868b00210991c7981c8f5` | 0.517 | 0.231 | 5/60 | 0.650 | -0.133 |

## Secondary per-type metrics

Wilson intervals are descriptive; score MAE is probability-weighted expected value versus the variable rubric's numeric target (raw / normalized by rubric span).

### qwen

| Variant | Type | Correct / n | Accuracy | Candidate-local F1 | Expected-value MAE raw / normalized |
|---|---|---:|---:|---:|---:|
| v2-baseline | choice | 29/60 | 0.483 | 0.176 | — |
| v2-baseline | noul | 49/60 | 0.817 | 0.408 | — |
| v2-baseline | score | 12/60 | 0.200 | 0.052 | 4.308 / 0.375 |
| readable-state | choice | 29/60 | 0.483 | 0.176 | — |
| readable-state | noul | 44/60 | 0.733 | 0.367 | — |
| readable-state | score | 12/60 | 0.200 | 0.056 | 4.394 / 0.371 |
| rule-in-candidate | choice | 30/60 | 0.500 | 0.171 | — |
| rule-in-candidate | noul | 30/60 | 0.500 | 0.250 | — |
| rule-in-candidate | score | 24/60 | 0.400 | 0.107 | 3.438 / 0.308 |
| joint-context-candidate | choice | 33/60 | 0.550 | 0.192 | — |
| joint-context-candidate | noul | 58/60 | 0.967 | 0.483 | — |
| joint-context-candidate | score | 15/60 | 0.250 | 0.064 | 4.034 / 0.356 |

### harrier

| Variant | Type | Correct / n | Accuracy | Candidate-local F1 | Expected-value MAE raw / normalized |
|---|---|---:|---:|---:|---:|
| v2-baseline | choice | 31/60 | 0.517 | 0.189 | — |
| v2-baseline | noul | 57/60 | 0.950 | 0.475 | — |
| v2-baseline | score | 13/60 | 0.217 | 0.058 | 4.624 / 0.425 |
| readable-state | choice | 31/60 | 0.517 | 0.191 | — |
| readable-state | noul | 45/60 | 0.750 | 0.375 | — |
| readable-state | score | 15/60 | 0.250 | 0.066 | 4.070 / 0.437 |
| rule-in-candidate | choice | 36/60 | 0.600 | 0.217 | — |
| rule-in-candidate | noul | 52/60 | 0.867 | 0.433 | — |
| rule-in-candidate | score | 13/60 | 0.217 | 0.053 | 4.543 / 0.414 |
| joint-context-candidate | choice | 19/60 | 0.317 | 0.130 | — |
| joint-context-candidate | noul | 60/60 | 1.000 | 0.500 | — |
| joint-context-candidate | score | 14/60 | 0.233 | 0.062 | 6.026 / 0.495 |

Detailed per-type accuracy/F1, score expected-value MAE, 95% Wilson intervals for all-three cases, per-family/domain accuracy, and paired McNemar/descriptive CIs are in `secondary-head-metrics.json`.

Embedding token counts, MLX export wall timings, feature hashes, and FLOP estimates are unchanged from `../run-metadata.json`; this secondary pass incurs CPU scorer execution only. The head files and all their mean/scale/weight array hashes, C values, per-variant CV config and train-feature provenance are in `variant-head-run-metadata.json`.

No test result changes the original v2-baseline train-only selection. In particular, this secondary exercise was prompted by a protocol correction after the prompt-only report and must not be presented as fresh confirmation.
