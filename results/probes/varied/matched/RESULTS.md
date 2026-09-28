# Frozen matched-set evaluation results

Train-only selection was frozen before the matched feature export/evaluation. The matched fixture was evaluated only after the selected models, train-input hashes, and copied feature manifests were checked. Raw predictions were written before the runner accessed test references. No test-driven retraining or tuning was performed.

## Frozen models

| Embedder | Frozen baseline | Frozen config SHA-256 | New models SHA-256 | New metadata SHA-256 |
|---|---:|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | per-type C=1.0 | `e92149799df56666251533333aa2518ad93ec66f4397a81bc24a66b003087b54` | `7ce3ecae127696bd1bbc929e84893f2e4ddb740106346910f43c84fe4cff6eb5` | `3003fa351ad1e4e92fbbf14002136fad11fdbe8c3a5c690c2920e30e0a89f1d0` |
| microsoft/harrier-oss-v1-0.6b | per-type C=0.1 | `02d233c8a32020e96bbb8d1f41684ec81dba75007fb8db196c76281b5701738d` | `077962ae2a20dea890ee8e6566c4730023b1a3d11ff2ca1be567aa8e1462b32f` | `3e37da1c7454e3401150d9a14e5e0c0cdb924284b24a3e6bc0160c9adce7b90d` |

The old Qwen `multi-qwen` artifact selected C=0.01 and was **not used**; Qwen was refit at frozen C=1.0. Harrier was refit at frozen C=0.1. Exact compatibility comparison against the previous Harrier artifact gave max absolute coefficient, feature-mean, and feature-scale differences of **0.0 for choice, noul, and score**. New immutable artifacts and train provenance are in `frozen-qwen/` and `frozen-harrier/`. The internal hash-bound selection record is `selection-freeze.json`.

## Matched holdout (60 cases / 180 questions)

| Embedder | Choice accuracy (n=60) | Noul accuracy (n=60) | Score label argmax (n=60) | Score expected-value MAE | All 3 correct cases |
|---|---:|---:|---:|---:|---:|
| Qwen | 0.333 | 0.483 | 0.350 | 5.897 | 6/60 |
| Harrier | 0.367 | 0.450 | 0.367 | 5.983 | 6/60 |

Candidate counts were variable: choice has 15 each at 2, 3, 4, and 5; noul has 60 two-outcome questions; score has 39 three-level, 18 four-level, and 3 five-level rubrics. Every model scored all 60 cases with all three types.

### Calibration (descriptive held-out point estimates)

| Embedder / type | Top-label ECE (10 bins) | Question-local Brier | Mean log loss |
|---|---:|---:|---:|
| Qwen / choice | 0.597 | 1.229 | 4.278 |
| Qwen / noul | 0.407 | 0.858 | 1.681 |
| Qwen / score | 0.486 | 0.993 | 2.685 |
| Harrier / choice | 0.416 | 0.940 | 2.029 |
| Harrier / noul | 0.201 | 0.573 | 0.787 |
| Harrier / score | 0.385 | 0.977 | 1.951 |

Calibration values are descriptive for this set only, using question-local candidate distributions. They imply no calibration guarantee or interval; Brier scores are candidate-count dependent.

### By domain

Each domain contains 5 cases. Values are per-type exact-match accuracy followed by all-three-correct case count. Full typed calibration and individual case outcomes are in each model `evaluation/metrics.json`.

| Domain | Qwen choice / noul / score | Qwen all 3 | Harrier choice / noul / score | Harrier all 3 |
|---|---:|---:|---:|---:|
| arts colleges | 0.60 / 0.40 / 0.20 | 0/5 | 0.60 / 0.80 / 0.60 | 2/5 |
| coastal research | 0.40 / 0.20 / 0.60 | 1/5 | 0.40 / 0.00 / 0.00 | 0/5 |
| community clinics | 0.60 / 0.40 / 0.20 | 1/5 | 0.20 / 0.60 / 0.40 | 0/5 |
| community energy | 0.20 / 0.80 / 0.60 | 0/5 | 0.40 / 0.60 / 0.20 | 0/5 |
| credit unions | 0.20 / 0.20 / 0.40 | 0/5 | 0.40 / 0.40 / 0.00 | 0/5 |
| food cooperatives | 0.20 / 0.40 / 0.40 | 1/5 | 0.40 / 0.60 / 0.20 | 0/5 |
| municipal permitting | 0.40 / 0.60 / 0.40 | 2/5 | 0.40 / 0.20 / 0.80 | 1/5 |
| museum conservation | 0.40 / 0.60 / 0.40 | 1/5 | 0.40 / 0.40 / 0.40 | 1/5 |
| public libraries | 0.20 / 0.20 / 0.00 | 0/5 | 0.20 / 0.40 / 0.40 | 1/5 |
| regional farms | 0.20 / 0.40 / 0.60 | 0/5 | 0.40 / 0.60 / 0.60 | 1/5 |
| regional transit | 0.20 / 0.60 / 0.20 | 0/5 | 0.40 / 0.40 / 0.40 | 0/5 |
| small manufacturers | 0.40 / 1.00 / 0.20 | 0/5 | 0.20 / 0.40 / 0.40 | 0/5 |

### Runtime, leakage, and output provenance

| Embedder | Family overlap | Scorer-only mean / p50 / p95 (µs per question) | Runner wall time (s; excludes feature export) | Raw predictions SHA-256 | Metrics SHA-256 |
|---|---:|---:|---:|---|---|
| Qwen | 0 | 6.63 / 6.10 / 7.54 | 0.169 | `6983bfe6f1a3237bdede0ac3ddd56e43300ebd8d8bc9661dfaddc92fe1ea7fdf` | `ebdab89dc85af1e0684ab0735268e54c29ad6f5ce7ed433fe581c86829b94206` |
| Harrier | 0 | 6.57 / 6.04 / 7.50 | 0.166 | `14c133acd3632f8ea89b326f5bd32c7d1118a4dd903292182a4d800866962545` | `7afbbfc64553db0bad8393a357a5d43758a65ed72e1c31d055d9996ff46f34d9` |

Latency covers frozen NumPy scoring only; it excludes MLX feature export, embedding inference, parsing, and reference scoring. Provenance includes the matched fixture/features/manifests, selection freeze, model/metadata, predictions, and outcome/metric hashes in `evaluation/provenance.json`.

## Artifact paths

- `selection-freeze.json`: internal selection record and exact saved model/metadata approvals.
- `frozen-qwen/`, `frozen-harrier/`: read-only C=1.0 and C=0.1 final typed scorer runs; includes frozen config, copied train manifests, metadata, and provenance.
- `runs/qwen/matched.features.jsonl` and `runs/harrier/matched.features.jsonl`: real MLX feature exports.
- `runs/{qwen,harrier}/evaluation/predictions.jsonl`: raw predictions persisted before label evaluation.
- `runs/{qwen,harrier}/evaluation/{question_outcomes.jsonl,case_outcomes.jsonl,metrics.json,provenance.json}`: scored artifacts and full per-domain results.
