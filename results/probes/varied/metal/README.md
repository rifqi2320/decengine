# Varied typed scorer, resident on Metal

`decengine varied-metal` is the real inference path for the frozen generic typed scorer. It
loads only the approved frozen `models.json`/`metadata.json` artifacts and requires their
hashes to match `matched/selection-freeze.json`. The `ModelStore` verifies the installed
checkpoint files against `install.json`. Do not use exporter JSONL feature vectors for this
path.

For example, using the explicitly installed model store:

```sh
STORE=/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store
cargo build --release -p decengine-cli --features mlx
target/release/decengine --home "$STORE" varied-metal \
  --input benchmarks/cases/varied/multi/test-matched-60.jsonl \
  --output results/probes/varied/metal/runs/qwen/predictions.jsonl \
  --run-dir results/probes/varied/matched/frozen-qwen \
  --model Qwen/Qwen3-Embedding-0.6B
```

Use `frozen-harrier`, `microsoft/harrier-oss-v1-0.6b`, and a separate output path for
Harrier. The compiler follows the generic prompt semantics documented in
`../README.md`; choice candidate order follows serde's object order, score preserves rubric
order, and binary outcomes are proposition-conditioned. Question names, count and case
count are not labels; input records may contain multiple questions of any of the three
supported wire types. `reference` is never read by the inference runner.

At inference, token IDs are the only model input transferred from the tokenizer to MLX.
Query and candidate embeddings are pooled/L2-normalized on MLX; the 4096-dimensional
`[query, candidate, query*candidate, abs(query-candidate)]` pair rows are assembled as MLX
arrays, and the frozen linear scorer and softmax execute on `Device(gpu, 0)`. Only final
typed probability vectors are copied to host. There is no embedding JSONL or host feature
construction/classification. Frozen scaler/coefficients are loaded as model parameters;
the scorer folds standardization into float64 host-side weights before transfer, which is
algebraically equivalent and avoids Metal flushing Harrier subnormal scales to zero.

The runner writes predictions, per-question synchronized stage timings and per-case
aggregate timings. It warms one complete inference for each typed scorer before timing.
`/usr/bin/time -l` reports total invocation resource footprint including load/warmup.
`analyze_runs.py` checks the matched60 run against archived NumPy predictions and records
probability drift, exact selected-label/boolean/score-argmax parity, timing percentiles,
installed checkpoint hashes, frozen-artifact hashes, runtime/device, and process memory in
each run's `summary.json`.

## Recorded matched60 results

| Model | max probability abs diff | exact selected/boolean/score argmax | case total p50/p95 | embed p50/p95 | GPU scorer p50/p95 |
|---|---:|---|---:|---:|---:|
| Qwen | 6.4262e-7 | yes (180/180) | 527.15 / 580.34 ms | 174.86 / 222.05 ms | 0.234 / 0.872 ms |
| Harrier | 5.1068e-7 | yes (180/180) | 298.26 / 319.11 ms | 97.57 / 121.75 ms | 0.215 / 0.287 ms |

Float32 GPU outputs cause maximum expected-value differences of `2.3478e-5` (Qwen) and
`9.1047e-6` (Harrier); all score argmax/selected rubric labels still match exactly. Full
resource measurements and SHA-256 provenance are in `runs/{qwen,harrier}/summary.json`.

The initial matched timing anomaly was checked with two additional sequential fresh
processes per model, in alternating model order. See
[`runs/timing-stability/report.md`](runs/timing-stability/report.md): both repeated
Qwen runs were about 296 ms/case p50 (embedding ~97 ms/question), close to Harrier's
~296–298 ms/case (embedding ~97 ms/question), versus the preserved initial Qwen result
of 527 ms/case (~175 ms embedding/question). Query-plus-candidate p50 token counts are
391 for every run. Repeated prediction files are byte-identical by model. The original
Qwen run has no environment snapshot, and the repeats show host background activity, so
the cause of the first-run outlier is unresolved rather than attributed to architecture.
