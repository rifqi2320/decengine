# Protocol-correction addendum: per-variant frozen-head comparison

The original `REPORT.md` and `holdout-metrics.json` are preserved as the first
**prompt-text-only** comparison (all four text variants scored with the unchanged
train-frozen v2 head). A prior blocker note was superseded after finding the
pre-existing per-variant bundles under `metal/frozen-scorers/`.

## Provenance and timing

The eight scorer JSONs/sidecars were created at 2026-09-23 16:46:56–16:47:02 UTC; the
first reference-scoring artifact, `holdout-metrics.json`, was written at 16:53:03 UTC.
The secondary runner also enforced `head_mtime < first_reference_score_file_mtime`,
checked each sidecar SHA-256, and verified the clean300/multi120 fixture, feature,
manifest, variant-CV config and selected-model provenance in every bundle. The bundles
are deterministic full-train `train_cv.fit_variant` refits on clean300 + multi120 only,
using each text variant's already-recorded CV-selected C and the per-type baseline
objective. No refit, tuning, or model selection was performed by this secondary test
evaluation. Baseline bundles were bit-exact with the published frozen model weights.

All eight new scorer-specific prediction files were saved and frozen in
`secondary/variant-heads/variant-head-freeze.json` before the secondary metric stage
read references. Their v2-baseline outputs match the original prompt-only v2 outputs on
all 180 questions/model with identical labels and max probability difference **0.0**.

## Secondary results, using corresponding pre-frozen variant heads

This is the same already-seen 60-case holdout, not independent confirmation. The table
is deliberately separate from the original fixed-v2-head results.

| Model | Variant | C | Type-balanced accuracy | Type-macro local F1 | All three correct | Original prompt-only fixed-v2 accuracy |
|---|---|---:|---:|---:|---:|---:|
| Qwen | v2 baseline | 1.0 | 0.500 | 0.212 | 5/60 | 0.500 |
| Qwen | readable state | 1.0 | 0.472 | 0.200 | 4/60 | 0.456 |
| Qwen | rule in candidate | 0.01 | 0.467 | 0.176 | 6/60 | 0.494 |
| Qwen | joint context + candidate | 0.01 | 0.589 | 0.247 | 6/60 | 0.439 |
| Harrier | v2 baseline | 0.1 | 0.561 | 0.241 | 8/60 | 0.561 |
| Harrier | readable state | 1.0 | 0.506 | 0.211 | 11/60 | 0.511 |
| Harrier | rule in candidate | 0.1 | 0.561 | 0.235 | 7/60 | 0.522 |
| Harrier | joint context + candidate | 0.1 | 0.517 | 0.231 | 5/60 | 0.650 |

Per-type metrics, score expected-value MAE, family/domain slices, Wilson intervals and
paired McNemar/descriptive CIs are in `secondary/variant-heads/secondary-head-metrics.json`.
Every variant head has the same reused MLX feature export/token cost as the primary
report; only the frozen CPU NumPy scorer changed. All 8 model/variant bundles, exact C,
per-type mean/scale/weight fingerprints, prediction paths and output hashes are in
`secondary/variant-heads/variant-head-run-metadata.json` and
`secondary/variant-heads/gpu-variant-head-handoff.json`.

The separate all-MLX prompt-study Metal path has already produced fresh60 predictions
for the shared-v2-head prompt-only condition under
`metal/runs/fresh60/shared-v2-baseline/`. Those outputs do not score the new
variant-specific heads. `GPU-PARITY-HANDOFF.md` sends the exact per-variant bundles and
secondary CPU prediction paths to the GPU agent for that additional parity condition;
this CPU scorer did not launch or claim a per-variant-head GPU run.

The original train-only selection remains v2 baseline for both models; neither this
secondary comparison nor the earlier prompt-only holdout results change that decision.
