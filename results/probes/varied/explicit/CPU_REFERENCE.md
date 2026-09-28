# Explicit-state context — CPU parity reference only

**Scoring backend is NumPy CPU.** Embeddings were exported by real MLX/Metal, but these paired scores are not a GPU-resident or end-to-end GPU result. They are a parity reference for the independent GPU typed-scorer agent. Do not publish them as the all-MLX comparison until GPU prediction parity and synchronized end-to-end timing are verified.

The authored manifest/validator confirms 60 cases / 180 unchanged questions and targets, aligned to matched cases by transformed/source IDs. It explicitly cautions that the added evidence was authored with reference access, makes the task easier by construction, and is not an independent holdout. Results below are descriptive, not an unbiased transfer/generalization gain. No fit or tuning used either test condition.

## Frozen models

| Model | Per-type C | Frozen selection config SHA-256 | Frozen `models.json` SHA-256 |
|---|---:|---|---|
| Qwen/Qwen3-Embedding-0.6B | 1.0 | `e92149799df56666251533333aa2518ad93ec66f4397a81bc24a66b003087b54` | `7ce3ecae127696bd1bbc929e84893f2e4ddb740106346910f43c84fe4cff6eb5` |
| microsoft/harrier-oss-v1-0.6b | 0.1 | `02d233c8a32020e96bbb8d1f41684ec81dba75007fb8db196c76281b5701738d` | `077962ae2a20dea890ee8e6566c4730023b1a3d11ff2ca1be567aa8e1462b32f` |

Qwen uses the fresh C=1.0 model; the older C=0.01 run was rejected. Harrier uses the fresh C=0.1 model, exact-compatible with its previous C=0.1 baseline.

## Paired CPU-reference results (60 cases / 180 questions)

| Model | Type | Matched accuracy | Explicit accuracy | Delta | Prediction switches | Explicit-only correct / matched-only correct |
|---|---|---:|---:|---:|---:|---:|
| Qwen | choice | 0.333 | 0.350 | +0.017 | 1/60 | 1 / 0 |
| Qwen | noul | 0.483 | 0.583 | +0.100 | 6/60 | 6 / 0 |
| Qwen | score | 0.350 | 0.400 | +0.050 | 4/60 | 3 / 0 |
| Qwen | All-three case outcomes | 6/60 | 8/60 | +2 cases | 11/180 questions | — |
| Harrier | choice | 0.367 | 0.400 | +0.033 | 6/60 | 4 / 2 |
| Harrier | noul | 0.450 | 0.600 | +0.150 | 11/60 | 10 / 1 |
| Harrier | score | 0.367 | 0.467 | +0.100 | 13/60 | 6 / 0 |
| Harrier | All-three case outcomes | 6/60 | 8/60 | +2 cases | 30/180 questions | — |

Absolute explicit metrics and expected-value metrics are in `runs/{qwen,harrier}/cpu-reference/evaluation/metrics.json`. Paired per-question and case transitions are in `paired-questions.jsonl` / `paired-cases.jsonl`; `paired-cpu-reference.json` gives per-domain deltas. Candidate counts are matched across conditions: choice 15 each at 2/3/4/5 candidates; noul 60 two-outcome questions; score 39 three-level, 18 four-level, 3 five-level questions.

## Timing scope and provenance

| Model | CPU ranker mean / p50 / p95 (µs/question) | Runner wall (s) | Explicit fixture SHA-256 | Feature SHA-256 | CPU explicit raw predictions SHA-256 | Paired CPU report SHA-256 |
|---|---:|---:|---|---|---|---|
| Qwen | 6.95 / 6.44 / 7.75 | 0.172 | `4cb002e1c29f5273ecc23fcdf742b093cb5433d661f5df0b50d209a534fb1409` | `539815e6f30278d1b3c6715c3e6d36cb4dd5bd283bbb625caca82dd960fe28fd` | `d183dc3b5a4cfb325bf4224f9b63dc1e2eee0847b6874507d54195d65ea3e85c` | `5f746157af1821f4071b9802b6f8acf5877e628c0c525dde242980d07771928d` |
| Harrier | 6.78 / 6.29 / 7.85 | 0.174 | `4cb002e1c29f5273ecc23fcdf742b093cb5433d661f5df0b50d209a534fb1409` | `3746fc6d41e450c02171ab84a6ac5ba4be8bc74a0de0a44f1a153d1315e16634` | `692ed5f294ab5e4bddf8a1d06cf201157c545fe5d85360011b46ae95dd52ad9a` | `beda063630e27345a744f345b70c0b47cb57bf95c347aab5360e191a68720cbd` |

CPU latency scopes the NumPy frozen scorer only and excludes MLX embeddings, feature export, file IO/parsing, and reference scoring. Runner wall time excludes feature export. Each explicit run’s `evaluation/provenance.json` contains its feature-manifest, freeze/model, prediction and metrics hashes.

## Next: GPU parity before publishing

GPU typed-scorer agent handoff: [`GPU_PARITY_HANDOFF.md`](GPU_PARITY_HANDOFF.md). Required next step is to load exactly these frozen model arrays, compare all explicit question probabilities/selections/score expected values against the CPU predictions, and measure synchronized MLX/Metal embedding+scoring end to end. The GPU artifact/report must be separate from this CPU reference.
