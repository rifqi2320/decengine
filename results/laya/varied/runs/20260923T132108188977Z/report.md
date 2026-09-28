# Laya MLX / JEV varied multi-question heldout run

- Run: `20260923T132108188977Z`
- Fixture: `benchmarks/cases/varied/multi/test-60.jsonl` (60 cases, 180 typed questions). Its SHA-256 remained `66a858680dd1e6e16bf12859ac18d67b24db11accb0efe23542a9b9c81fbbac3`.
- Laya: real root checkpoint `convaiinnovations/laya`, revision `5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`, pinned weights SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`; MLX Metal `Device(gpu, 0)`, float32, port commit `0a859518634112655cb97c745dbf04f5191aaf13`. Verified runner path: `results/laya/mlx/compare_300_mlx.py`; varied-schema adapter is scoped to `results/laya/varied/`.
- Requests: 3 independent typed MLX forward calls per case; exactly 1 JEV API request per shared state carrying choice, noul, and score, using the existing JEV typed wire translation and `jev-latest`.
- Protocol mapping: choice receives the exact fixture options mapping; noul is thresholded at 0.5; score gets the full ordered label+criterion rubric and scalar label is probability argmax (not expected-value binning). No prompt tuning or posthoc review.
- Validation: 60/60 case IDs; 180/180 Laya decisions and 60/60 JEV requests succeeded; zero errors. JEV returned HTTP 200 in every case. Provider response contained no response/request ID field; case IDs and local run/provenance IDs are retained. Credential was read only from the authorized key file and is absent from persisted artifacts.

## Accuracy

| Model | Choice (next_step) | Noul (human_gate) | Score (evidence_readiness) | All three correct |
| --- | ---: | ---: | ---: | ---: |
| Laya MLX | 29/60 (48.3%) | 39/60 (65.0%) | 31/60 (51.7%) | 13/60 (21.7%) |
| JEV | 26/60 (43.3%) | 22/60 (36.7%) | 20/60 (33.3%) | 8/60 (13.3%) |
| Qwen probe baseline | — | — | — | 9/60 |
| Harrier probe baseline | — | — | — | 1/60 |

The Qwen/Harrier figures are contextual results from their existing probe pipelines, not a controlled model-equivalence comparison. Reference scoring was computed only after prediction files were written.

## Timing (ms)

Per-decision Laya timings and per-request JEV timings are in `timings.jsonl` and summarized in `metrics.json`. MLX inference mean (min–max): `evidence_readiness 38.09 (33.52–120.00), human_gate 25.73 (23.41–46.58), next_step 41.56 (26.07–126.87)`; JEV roundtrip mean (min–max): 501.5 (417.7–666.4).

Raw responses and normalized typed outcomes: `raw-results.jsonl`. Comparable predictions: `laya-predictions.jsonl`, `jev-predictions.jsonl`. Full provenance/checksums are in `run-metadata.json` and `checksums.json`.
