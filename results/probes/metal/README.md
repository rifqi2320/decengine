# MLX/Metal frozen probe

This is inference only: it loads the archived `models.json` probes/scaler parameters and does not train or tune anything. The CLI compiles the case request with the installed profile, tokenizes the compiled owner/urgent/impact queries on CPU, then performs embedding normalization, standardization, logistic linear algebra, and softmax on MLX `Device(gpu, 0)`. Only the final 2/3/7 class-probability vectors cross back to CPU.

After both checkpoints are installed and the independent CPU-probe benchmark has completed, run from the repository root:

```sh
cargo build --release -p decengine-cli --features mlx
mkdir -p 'results/probes/metal/Qwen+Metal probe' 'results/probes/metal/Harrier+Metal probe'
target/release/decengine probe-decide \
  --model-store /private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store \
  --model Qwen/Qwen3-Embedding-0.6B \
  --input benchmarks/cases/probe-test-300.jsonl \
  --probe-dir results/probes/runs/20260923T094618Z/qwen/task-specific \
  --output 'results/probes/metal/Qwen+Metal probe/predictions-all300.jsonl'
target/release/decengine probe-decide \
  --model-store /private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store \
  --model microsoft/harrier-oss-v1-0.6b \
  --input benchmarks/cases/probe-test-300.jsonl \
  --probe-dir results/probes/runs/20260923T094618Z/harrier/task-specific \
  --output 'results/probes/metal/Harrier+Metal probe/predictions-all300.jsonl'
python3 results/probes/metal/compare_metal_probe.py
python3 results/probes/metal/benchmark_metal_probe.py
```

The prediction sidecar records probe/model/profile/checkpoint hashes and numerical drift. Because archived sklearn probabilities use CPU float64 and Metal calculations use float32, comparison requires exact labels and probability absolute error at most `1e-4`; the tolerance documents numeric precision drift, not a change to trained parameters. The pre-existing CPU probe artifacts and comparison outputs are inputs only and are not rewritten.

The probe path embeds exactly three compiled task queries per case (`owner`, `urgent`, `impact`); it never exports or evaluates the unused fourth state vector. The CLI times these 300 unique cases only after loading and one warmup execution for each classifier. GPU arrays are synchronized at embedding and classifier boundaries. Each prediction JSONL has a corresponding `.timings.jsonl`; the benchmark script writes mean/p50/p95 in milliseconds for end-to-end case time, tokenization, embedding, and classifier to each system's `timing-summary.json`. It also records whole-invocation elapsed time (including load/warmup), process maximum RSS, and peak memory footprint; macOS unified-memory process counters are not a GPU-only allocation measurement. The benchmark isolates Metal-probe inference only; compare against the CPU benchmark's three-query path and Laya's three-call path separately, not against the earlier four-vector CPU exporter amortized numbers.

## Recorded run

Both models completed one-case smoke tests and isolated 300-case runs. `compare_metal_probe.py` verified all 300 labels exactly; maximum absolute probability errors versus the archived CPU probe predictions were `3.8872691e-7` (Qwen) and `1.9744154e-7` (Harrier), below `1e-4`.

| System | E2E mean / p50 / p95 (ms) | Embed mean / p50 / p95 (ms) | Classifier mean / p50 / p95 (ms) | Max RSS | Peak process footprint |
|---|---:|---:|---:|---:|---:|
| Qwen+Metal probe | 80.032 / 79.168 / 89.528 | 78.864 / 78.021 / 88.300 | 0.682 / 0.672 / 0.792 | 1,366,556,672 B | 3,868,935,848 B |
| Harrier+Metal probe | 79.973 / 79.205 / 88.973 | 78.811 / 78.083 / 87.601 | 0.678 / 0.673 / 0.770 | 1,358,118,912 B | 3,858,810,560 B |

Per-case measurements exclude model loading and warmup; whole CLI process resource reports include both. Detailed tokenization and other stage values, timing distributions, and resource elapsed/user/sys figures are in the corresponding `timing-summary.json` files. The one-case smoke predictions and metadata are preserved alongside the 300-case outputs.
