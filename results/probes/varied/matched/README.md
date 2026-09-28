# Matched multi-question evaluation (train-only selection frozen)

`evaluate_matched.py` is evaluation-only. It reuses the frozen saved `typed_scorers` and
normalization arrays from `frozen-qwen/` or `frozen-harrier/`; it does not train, tune,
or select variants. The orchestrator froze train-only baseline Qwen at C=1.0 and
Harrier at C=0.1. Qwen's previous `multi-qwen` C=0.01 model was rejected and refit.
Harrier's previous C=0.1 parameters were exact-compatible with a fresh baseline refit
(zero difference in coefficients, means, and scales for all three types). The new
read-only run directories are bound by `selection-freeze.json`. The fixture was exported
and scored only after those frozen artifacts were verified; raw prediction files were
persisted before references were accessed.

## Freeze gate

The required `--selection-freeze` JSON binds approval to the exact immutable artifacts.
The experiment record is already prepared at `selection-freeze.json`; it has this shape:

```json
{
  "selection_status": "frozen",
  "selection_id": "orchestrator-assigned-id",
  "approved_runs": {
    "qwen": {
      "selection_status": "frozen",
      "model": "Qwen/Qwen3-Embedding-0.6B",
      "run_dir": "/absolute/path/to/results/probes/varied/matched/frozen-qwen",
      "metadata_sha256": "sha256-of-that-run-metadata.json",
      "models_sha256": "sha256-of-that-run-models.json"
    },
    "harrier": {
      "selection_status": "frozen",
      "model": "microsoft/harrier-oss-v1-0.6b",
      "run_dir": "/absolute/path/to/results/probes/varied/matched/frozen-harrier",
      "metadata_sha256": "sha256-of-that-run-metadata.json",
      "models_sha256": "sha256-of-that-run-models.json"
    }
  }
}
```

Only include model variants approved by train-only selection. The runner checks
the status, selected run directory, and both saved-artifact hashes **before opening
the fixture or its feature vectors**. The gate is intentionally fail-closed.

## Safe preparation smoke test

This touches only synthetic in-memory questions, synthetic vectors, and a temporary
prediction file; it does not need a model, freeze file, or benchmark fixture:

```sh
python3 results/probes/varied/matched/evaluate_matched.py --self-test
```

## Matched export and evaluation

The matched feature export used the explicit installed store (the default store is not
the intended one). These are the reproducible commands used for the completed run:

```sh
STORE=/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store
# The freeze is verified; export with the approved embedder:
target/release/decengine --home "$STORE" varied-export-features \
  --input benchmarks/cases/varied/multi/test-matched-60.jsonl \
  --output results/probes/varied/matched/runs/qwen/matched.features.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B

python3 results/probes/varied/matched/evaluate_matched.py \
  --run-key qwen --selection-freeze results/probes/varied/matched/selection-freeze.json \
  --features results/probes/varied/matched/runs/qwen/matched.features.jsonl \
  --out results/probes/varied/matched/runs/qwen/evaluation
```

Use `--run-key harrier` plus its approved model and a separate output directory for
Harrier. The runner requires 60 cases / 180 questions by default and checks exporter
identity against the frozen training manifest. It writes `predictions.jsonl` before
accessing reference targets, then writes per-question outcomes, per-domain and typed
metrics, plus all-three-correct case outcomes. Calibration reports descriptive
10-bin ECE, Brier, and log loss on this test set only (no calibration guarantee or
interval); calibration is question-local across variable candidate sets. Domain-level
typed reports include calibration and score expected-value MAE where applicable.
Latency is scorer-only NumPy time per question; it excludes MLX embedding export, file
IO, parsing, and label scoring. A separate wall-clock duration covers the runner after
approval but still excludes feature export. Provenance hashes the fixture, feature
export, freeze file, saved model/metadata, predictions, and scored artifacts.

## Results

Train-only config/model hashes were verified before MLX export; matched predictions
were then persisted before reference scoring. See [`RESULTS.md`](RESULTS.md) for the
per-type, all-three, per-domain, calibration, latency-scope, and artifact-hash report.
