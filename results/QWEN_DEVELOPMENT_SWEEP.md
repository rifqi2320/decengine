# Qwen prompt tuning: development-only sweep

This sweep ran only against `case-001` through `case-050`. The runner reads the first
50 lines only from the fixture, archived Qwen run, and archived judgment ledger; it
does not read the remaining lines or any separate holdout file. Reference labels are
used only to calculate metrics, never passed to model inference. No JEV/API requests
were made.

## Reproduction

```sh
cargo test -p decengine-models -p decengine-compiler
cargo build --release -p decengine-ffi --no-default-features --features mlx
PYTHONPATH=python:results uv run --python 3.14 python results/qwen_development_sweep.py
```

Run output (real MLX inference, 50 cases per variant):
`results/qwen-development-sweeps/20260923T090036783462Z/`. It includes model/profile
provenance in `metadata.json`, every raw response/distribution in each variant JSONL,
and per-field exact metrics plus errors in `summary.json`. The two-case smoke run is
separate at `20260923T090022014491Z` and is not used for findings.

## Findings / selection

| Variant | Owner | Urgent | Impact | Total |
|---|---:|---:|---:|---:|
| Baseline | 22/50 | 24/50 | 26/50 | 72/150 (48.0%) |
| Official-format task instruction | 22/50 | 24/50 | 29/50 | 75/150 (50.0%) |
| Choice scope candidate phrasing | 13/50 | 24/50 | 26/50 | 63/150 (42.0%) |
| Impact candidate phrasing | 22/50 | 24/50 | 25/50 | 71/150 (47.3%) |
| Explicit no-risk/workaround binary texts | 22/50 | 24/50 | 26/50 | 72/150 (48.0%) |
| Combined task-aware candidates | 8/50 | 24/50 | 21/50 | 53/150 (35.3%) |
| Combined candidates + spaced query | 8/50 | 24/50 | 21/50 | 53/150 (35.3%) |

The best development score was the task-instruction variant at 75/150 (+3 fields,
2.0 percentage points). All gain was impact (+3); owner and urgency did not improve.
The explicit urgency rubric did not move urgent accuracy, while owner candidate
phrasing and the combined changes substantially regressed. This is not a robust,
across-field improvement, so no new tuned production profile is selected or created.
Do not interpret the development-best variant as a winning model until it has been
evaluated once on a separately supplied, untouched holdout.

## Implementation

The compiler now supports optional per-kind query instructions and separate choice /
score candidate format strings, alongside the existing Noul candidate text. These
are populated only when explicitly supplied through FFI engine options; absent
options use defaults matching the previous compiled text. The candidate content and
templates are included in cache-key material. Tests cover these defaults/overrides.

No Harrier files, judgments, source dataset, base Qwen manifest, or prior experiment
outputs were changed. No experimental profile was added from this sweep because no
robust development improvement was found. The existing earlier
`qwen3-embedding-0.6b-direct-rubric.experimental.json` remains a separate exploratory
artifact and is not selected by this sweep.
