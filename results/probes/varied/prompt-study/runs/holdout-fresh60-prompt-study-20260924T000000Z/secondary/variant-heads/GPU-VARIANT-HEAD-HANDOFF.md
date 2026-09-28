# All-MLX parity handoff: per-variant frozen train-only heads

The secondary CPU scorer run used existing frozen bundles only. This file transmits the exact head hashes, fixed C, scorer normalization/weight fingerprints, and CPU prediction paths for a separate GPU parity invocation. No model was refit or selected using fresh60; no new GPU run was launched here.

Fresh60 SHA-256: `316b3c2e1d0c5cf50356d0e236a5f4adf27fdac9ee6d6c6b56a14f5354ef6a9f`  
First reference-scoring file time (UTC): `2026-09-23T16:53:03.212262+00:00`  
All listed head JSON/sidecars have mtimes before that time; `variant-head-run-metadata.json` records provenance to the train-only features, CV config, frozen selection, and full-train fit.

## Frozen bundle and prediction path map

| Model | Variant | C | Bundle SHA-256 | Bundle mtime (UTC) | CPU variant-head prediction | Proposed isolated Metal output |
|---|---|---:|---|---|---|---|
| qwen | v2-baseline | 1 | `5b7260f04f0b66784504b09526447b30a9f020a554abf611c57eb971f9fabd79` | 2026-09-23T16:46:56.073937+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/qwen/v2-baseline.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/qwen/v2-baseline/predictions.jsonl` |
| qwen | readable-state | 1 | `b3d19ab873b2ffa90a4f8af63833a3b3195e1d17a8976a107021aafa259e8436` | 2026-09-23T16:46:57.087278+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/qwen/readable-state.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/qwen/readable-state/predictions.jsonl` |
| qwen | rule-in-candidate | 0.01 | `62a8f7f663465f082231c885de0e6fc439d734901bc51e90aafe23eb4685f9ca` | 2026-09-23T16:46:57.845801+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/qwen/rule-in-candidate.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/qwen/rule-in-candidate/predictions.jsonl` |
| qwen | joint-context-candidate | 0.01 | `35fa698c58583def77da4db762a9b2504ebb0adacd8bfa63471128eed23daa54` | 2026-09-23T16:46:58.596414+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/qwen/joint-context-candidate.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/qwen/joint-context-candidate/predictions.jsonl` |
| harrier | v2-baseline | 0.1 | `5fb0727296cc89d41c9ebcec1664a7ef3a99e5c08c6da424d267f02ea6c22d30` | 2026-09-23T16:46:59.625593+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/harrier/v2-baseline.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/harrier/v2-baseline/predictions.jsonl` |
| harrier | readable-state | 1 | `831a1558e197e3a949b02e667887abf8824c4196c2c31ab58ddb60ab1c83f49f` | 2026-09-23T16:47:00.708952+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/harrier/readable-state.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/harrier/readable-state/predictions.jsonl` |
| harrier | rule-in-candidate | 0.1 | `c8b38381b695dc71269139171d1b4d83c6db641d8e6be4e581484b5d7b71ed4a` | 2026-09-23T16:47:01.621507+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/harrier/rule-in-candidate.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/harrier/rule-in-candidate/predictions.jsonl` |
| harrier | joint-context-candidate | 0.1 | `9673793192b590b964e1384ea29a45a554625f68cb4868b00210991c7981c8f5` | 2026-09-23T16:47:02.497133+00:00 | `/Users/IDSP35241/Documents/Code/decengine/results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z/secondary/variant-heads/predictions/harrier/joint-context-candidate.jsonl` | `results/probes/varied/prompt-study/metal/runs/fresh60/per-variant-heads/harrier/joint-context-candidate/predictions.jsonl` |

Each type head uses 4096-dimensional pair features and stores the exact train-only `mean`, `scale`, and `weights`; SHA-256 fingerprints for each are under the JSON handoff's `normalization_and_weights`. The normalization is the frozen per-type standardizer `[x - mean] / scale`, algebraically folded into Metal scorer weights by the `prompt-study-metal` runner. Bundle file contents, not an OOF fold head, are the final full-train scorers.

## Existing prompt-only all-MLX run versus requested head parity

`metal/runs/fresh60/shared-v2-baseline/` already contains all eight all-MLX predictions for the four prompt texts using a shared v2 head. That is not the requested per-variant-head comparison. Preserve it unchanged. The separate parity run should use `metal/frozen-scorers/{model}/{variant}.json` and write to the isolated `per-variant-heads/` destinations in the table. The CLI shape from `metal/README.md` is:

```sh
target/release/decengine --home "$STORE" prompt-study-metal \
  --input benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl \
  --output <isolated-gpu-prediction-path> \
  --scorer-bundle results/probes/varied/prompt-study/metal/frozen-scorers/<model>/<variant>.json \
  --study-run results/probes/varied/prompt-study/runs/trainonly-clean300-20260924T002000Z \
  --variant <variant> --model <model-id>
```

Score candidate/probability/expected-value deltas against the CPU secondary files listed above only after GPU prediction files are saved. Do not fit or select based on fresh60. The per-head C choices reflect their already-recorded train-only CV configs; they do not replace or alter the original frozen prompt selection.
