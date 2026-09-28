# Fresh60 prompt variants: per-variant frozen heads, all-MLX parity

Matched against the separate secondary CPU prediction artifacts. Each per-variant full-train scorer bundle/hash and C is bound to the secondary handoff; no holdout labels were read.

| Model | Variant | head C | typed outputs exact (180/180) | all-three | max prob abs diff | max score EV diff | case p50/p95 ms | GPU scorer p50/p95 ms/question | tokens/case p50/p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen | v2-baseline | 1 | True | 60/60 | 6.8224421e-07 | 3.50768062e-06 | 301.32/333.87 | 0.196/0.236 | 1162/1240 |
| qwen | readable-state | 1 | True | 60/60 | 7.6433974e-07 | 3.67552241e-06 | 305.31/342.44 | 0.196/0.235 | 1184/1261 |
| qwen | rule-in-candidate | 0.01 | True | 60/60 | 2.64231817e-07 | 4.02600644e-06 | 402.83/452.17 | 0.206/0.306 | 1675/1853 |
| qwen | joint-context-candidate | 0.01 | True | 60/60 | 1.8590361e-07 | 2.48665599e-06 | 838.91/1008.59 | 0.217/0.276 | 4234/5152 |
| harrier | v2-baseline | 0.1 | True | 60/60 | 4.27953704e-07 | 7.82269129e-06 | 301.40/332.32 | 0.194/0.232 | 1162/1240 |
| harrier | readable-state | 1 | True | 60/60 | 8.24426022e-07 | 6.65283427e-06 | 305.19/340.97 | 0.198/0.238 | 1184/1261 |
| harrier | rule-in-candidate | 0.1 | True | 60/60 | 8.90373577e-07 | 1.28469205e-05 | 402.99/451.62 | 0.205/0.246 | 1675/1853 |
| harrier | joint-context-candidate | 0.1 | True | 60/60 | 5.94419074e-07 | 7.68074664e-06 | 854.64/1014.15 | 0.216/0.259 | 4234/5152 |
