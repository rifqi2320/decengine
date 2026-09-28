# Fresh60 prompt variants: shared v2 head, all-MLX parity

Inference mode: one unchanged v2-baseline frozen head (weights, means, scales, C) per model reused across all four prompt-only text variants. Compared with the pre-reference-frozen CPU NumPy prediction files; only predictions were read, no targets/labels.

| Model | Prompt variant | all 180 selected/typed values exact | all-three cases | max probability abs diff | max score EV diff | case p50/p95 ms | embed p50/p95 ms/question | GPU scorer p50/p95 ms/question | tokens/case p50/p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen | v2-baseline | True | 60/60 | 6.8224421e-07 | 3.50768062e-06 | 300.30/332.79 | 96.09/118.61 | 0.196/0.228 | 1162/1240 |
| qwen | readable-state | True | 60/60 | 6.7617155e-07 | 2.45758301e-06 | 307.78/342.26 | 97.93/124.57 | 0.200/0.271 | 1184/1261 |
| qwen | rule-in-candidate | True | 60/60 | 7.09677178e-07 | 1.06980917e-05 | 403.04/452.19 | 126.96/172.68 | 0.204/0.244 | 1675/1853 |
| qwen | joint-context-candidate | True | 60/60 | 5.21806729e-07 | 1.20743976e-05 | 893.24/1122.32 | 268.46/450.59 | 0.219/0.268 | 4234/5152 |
| harrier | v2-baseline | True | 60/60 | 4.27953704e-07 | 7.82269129e-06 | 301.07/331.43 | 96.21/118.88 | 0.195/0.240 | 1162/1240 |
| harrier | readable-state | True | 60/60 | 4.27599456e-07 | 1.44913174e-05 | 305.05/339.90 | 97.62/123.55 | 0.196/0.248 | 1184/1261 |
| harrier | rule-in-candidate | True | 60/60 | 4.9375028e-07 | 9.75031768e-06 | 403.24/450.55 | 126.67/172.59 | 0.205/0.249 | 1675/1853 |
| harrier | joint-context-candidate | True | 60/60 | 5.012404e-07 | 8.37456264e-06 | 849.23/1022.35 | 261.23/402.39 | 0.220/0.362 | 4234/5152 |
