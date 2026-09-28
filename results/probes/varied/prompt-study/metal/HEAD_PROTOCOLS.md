# Prompt-study scorer-head protocols / correction record

The fresh60 CPU reference run's `run-metadata.json` specifies
`selected_training_variant: "v2-baseline scorer applied unchanged to every prompt text variant"`.
Its per-model `final_fit_sha256` is exactly the byte SHA-256 of the corresponding
`trainonly-clean300-20260924T002000Z/frozen/{qwen,harrier}/models.json`. Thus the parity
protocol uses one exact frozen v2 head per model for all four text variants: Qwen C=1.0,
Harrier C=0.1, with the same weights, feature means, and scales in every variant.

## Discrepancy found before GPU holdout inference

The first `freeze_trainonly_scorers.py` implementation used each text variant's own CV
selected C and refit its typed scorer on that variant's training features. Those bundles
implement a distinct prompt+retrain protocol and **do not match** the CPU reference
protocol. In particular, C differed for Qwen `rule-in-candidate` and `joint-context-candidate`
(0.01 instead of 1.0) and Harrier `readable-state` (1.0 instead of 0.1); even where C
coincided, weights and scaler arrays were variant-specific. They were not used in the
primary parity run.

Before any GPU holdout run, `freeze_shared_v2_heads.py` packaged the unchanged source
frozen weights/scalers/C into `frozen-scorers/shared-v2-baseline/{model}/{variant}.json`.
The prompt Metal runner checks each bundle SHA, source `models.json` SHA, exact JSON array
equality of every type's weights/mean/scale, C, source selection, profile/tokenizer and
installed checkpoint hashes, and the pre-reference fixture hash. The old candidate-head
bundles remain under `frozen-scorers/{model}/` as the separate prompt+retrain protocol;
do not use them when claiming parity to the current CPU reference.

Primary all-MLX GPU measurements/CPU prediction-output parity therefore are like-for-like
with the primary reference: **prompt-only variation, one unchanged train-frozen v2 baseline
head**. A separate secondary per-variant protocol was also run using the original
`frozen-scorers/{model}/{variant}.json` bundles, each verified against the secondary
handoff's exact file and per-array fingerprints and matched only to its secondary CPU
prediction file. The secondary run does not change or replace the primary shared-head
comparison.

No holdout fit or selection is performed in either inference run. Reference prediction
files contain model outputs, not target labels; both parity analyzers read only these
outputs after raw GPU predictions have been written. Final reports are
`runs/fresh60/FINAL_SUMMARY.md` plus the protocol-specific JSON/Markdown reports alongside
each set of GPU predictions.
