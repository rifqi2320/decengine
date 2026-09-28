# Locked Qwen holdout evaluation

The selected development variant and baseline were frozen before inference. Both
were run with real local MLX inference over all 60 cases in
`benchmarks/cases/decision-cases-holdout.jsonl`. The runner retained only request
inputs during inference, wrote/flushed both complete prediction JSONLs, and only
then opened the file again to calculate reference-based metrics. No JEV/API calls
were made.

## Reproduction

```sh
cargo test -p decengine-models -p decengine-compiler
cargo build --release -p decengine-ffi --no-default-features --features mlx
PYTHONPATH=python:results uv run --python 3.14 python results/run_qwen_locked_holdout.py
```

Successful run artifacts:
`results/qwen-holdout/20260923T090609039852Z/`

- `metadata.json`: SHA-256 hashes of the holdout dataset, locked development metadata,
  both inference configs, active Qwen manifest, installed model record, and release
  native library; runtime and no-network provenance.
- `baseline.jsonl` and `official-format-task-instruction.jsonl`: all 60 raw native
  responses, per-question distributions, confidence, predictions, and per-case latency.
- `summary.json`: exact metrics, errors, confidence summaries and latency summaries.

An earlier attempt wrote both prediction JSONLs but failed when it assumed the
holdout references used the development set's nested `reference.decision` shape. Its
raw outputs and pending metadata remain preserved in
`results/qwen-holdout/20260923T090501302613Z/`; the runner now accepts either direct
reference fields or the nested shape. The successful run above uses the corrected
runner and contains the completed metrics.

## Results

| Profile | Owner exact | Urgent exact | Impact exact | Total exact |
|---|---:|---:|---:|---:|
| Baseline | 24/60 (40.0%) | 27/60 (45.0%) | 37/60 (61.7%) | 88/180 (48.9%) |
| Locked task instruction | 28/60 (46.7%) | 27/60 (45.0%) | 41/60 (68.3%) | 96/180 (53.3%) |
| Delta | +4 | 0 | +4 | +8 (+4.4 pp) |

Urgency had **27 reference-true and 33 reference-false** cases. Both configurations
predicted urgent for all 60: TP 27, FP 33, TN 0, FN 0. Thus sensitivity was 100%,
but specificity was 0%; the locked prompt did not fix the urgent false-positive
failure. This is not robust across questions and is not a basis for production
promotion.

Median per-case latency was 78.2 ms baseline and 101.6 ms locked; mean latency was
80.7 ms and 104.8 ms, respectively. Reported confidence is the engine's normalized
entropy confidence, not a calibrated probability of correctness. Detailed per-case
errors and confidence summaries are in `summary.json`.
