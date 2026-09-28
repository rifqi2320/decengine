# Matched60 timing stability

No references/targets were opened. All repeated MLX invocations were sequential, with a fresh model load and one warmup per type; order alternated between repetitions. AC status, load snapshot and resource reports are stored per run. The OS page cache was not cleared.

| Model/run | case p50/p95 ms | embedding p50/p95 ms/question | scorer p50/p95 ms/question | query+candidate tokens p50 | embed/token Pearson |
|---|---:|---:|---:|---:|---:|
| qwen | 527.15/580.34 | 174.86/222.05 | 0.234/0.872 | 391 | 0.475 |
| replicate-1/qwen | 296.82/312.21 | 97.30/121.16 | 0.189/0.225 | 391 | 0.670 |
| replicate-2/qwen | 295.41/312.01 | 97.55/121.34 | 0.199/0.236 | 391 | 0.661 |
| harrier | 298.26/319.11 | 97.57/121.75 | 0.215/0.287 | 391 | 0.655 |
| replicate-1/harrier | 295.76/319.16 | 97.73/121.77 | 0.212/0.322 | 391 | 0.665 |
| replicate-2/harrier | 297.49/314.32 | 97.47/120.85 | 0.196/0.277 | 391 | 0.661 |

The first Qwen all-MLX run is substantially slower than both repeats; those repeats cluster tightly with Harrier, despite identical p50 query+candidate token count (391). The initial run lacks a contemporaneous host-load snapshot, so the cause cannot be attributed to model architecture, cache, thermal, or host contention. Repetitions were sequential and on AC but snapshots show variable background activity and nonzero involuntary context switches; OS page cache was left intact. CPU instructions/cycles do not establish GPU clock/state. Treat the initial 527 ms p50 as an outlier, not a model-level Qwen-vs-Harrier performance difference.

All measurements and exact process-condition snapshots are in `report.json`; initial run outputs were not rewritten.
