# Matched-domain / unseen-question-family test-60 run

Run: `20260923T151228074560Z`
Fixture SHA-256: `48a057dd3ebedc6d76b7ca55f0782069f49cd31151576c2bc01932744a6ed614`
Pinned checkpoint SHA-256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`

## Overall accuracy

| Model | Choice | Noul | Score | All three |
|---|---:|---:|---:|---:|
| Laya MLX | 19/60 (31.7%) | 31/60 (51.7%) | 19/60 (31.7%) | 3/60 (5.0%) |
| JEV | 20/60 (33.3%) | 29/60 (48.3%) | 23/60 (38.3%) | 4/60 (6.7%) |

Per-domain question-type and all-three accuracies are reported for each of the 12 five-case domains in `metrics.json`. Choice-option-count slices (2/3/4/5 candidates, 15 cases each) are also in `metrics.json`; per-call timing is in `timings.jsonl`.

## Prior combined-shift fixture (descriptive only)

| Prior model / run `20260923T132108188977Z` | Choice | Noul | Score | All three |
|---|---:|---:|---:|---:|
| Laya MLX | 29/60 | 39/60 | 31/60 | 13/60 |
| JEV | 26/60 | 22/60 | 20/60 | 8/60 |

This is a different combined domain/question-family shift fixture, so comparison is descriptive, not controlled.
Raw model/API responses and normalized typed outcomes were saved before reference scoring. No prompt/model selection or reference-based inference was performed.
Errors: Laya 0, JEV 0.
