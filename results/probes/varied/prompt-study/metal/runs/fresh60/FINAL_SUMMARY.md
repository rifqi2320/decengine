# Prompt-study fresh60 all-MLX results

The fresh60 fixture remained unchanged. Both protocols ran as separate sequential, isolated MLX/Metal processes using the explicit model store; no holdout labels or targets were read. Raw GPU predictions were persisted before prediction-only CPU-reference comparison.

## Protocol A — shared frozen v2 head for every text variant

CPU reference protocol: unchanged Qwen C=1.0 / Harrier C=0.1 train-frozen weights, means, and scales applied to all four prompt texts. Every variant/model has exact selected-label, boolean, and score-argmax parity on all 180 typed predictions and all 60 three-question cases.

| Model | Variant | max probability abs diff | max score EV abs diff | case p50/p95 ms | scorer p50/p95 ms/question | query+candidate tokens/case p50/p95 |
|---|---|---:|---:|---:|---:|---:|
| harrier | joint-context-candidate | 5.012404e-07 | 8.37456264e-06 | 849.23/1022.35 | 0.220/0.362 | 4234/5152 |
| harrier | readable-state | 4.27599456e-07 | 1.44913174e-05 | 305.05/339.90 | 0.196/0.248 | 1184/1261 |
| harrier | rule-in-candidate | 4.9375028e-07 | 9.75031768e-06 | 403.24/450.55 | 0.205/0.249 | 1675/1853 |
| harrier | v2-baseline | 4.27953704e-07 | 7.82269129e-06 | 301.07/331.43 | 0.195/0.240 | 1162/1240 |
| qwen | joint-context-candidate | 5.21806729e-07 | 1.20743976e-05 | 893.24/1122.32 | 0.219/0.268 | 4234/5152 |
| qwen | readable-state | 6.7617155e-07 | 2.45758301e-06 | 307.78/342.26 | 0.200/0.271 | 1184/1261 |
| qwen | rule-in-candidate | 7.09677178e-07 | 1.06980917e-05 | 403.04/452.19 | 0.204/0.244 | 1675/1853 |
| qwen | v2-baseline | 6.8224421e-07 | 3.50768062e-06 | 300.30/332.79 | 0.196/0.228 | 1162/1240 |

## Protocol B — own frozen train-only head for each variant

Compared against the distinct secondary CPU predictions with per-variant heads; original head bundle hashes/C/array fingerprints are verified in `secondary/variant-heads/gpu-variant-head-handoff.json`. No heldout fit/tuning. Each model/variant matches all 180 labels/booleans/score argmaxes and all 60 three-question cases.

| Model | Variant | C | max probability abs diff | max score EV abs diff | case p50/p95 ms | scorer p50/p95 ms/question | query+candidate tokens/case p50/p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| harrier | joint-context-candidate | 0.1 | 5.94419074e-07 | 7.68074664e-06 | 854.64/1014.15 | 0.216/0.259 | 4234/5152 |
| harrier | readable-state | 1 | 8.24426022e-07 | 6.65283427e-06 | 305.19/340.97 | 0.198/0.238 | 1184/1261 |
| harrier | rule-in-candidate | 0.1 | 8.90373577e-07 | 1.28469205e-05 | 402.99/451.62 | 0.205/0.246 | 1675/1853 |
| harrier | v2-baseline | 0.1 | 4.27953704e-07 | 7.82269129e-06 | 301.40/332.32 | 0.194/0.232 | 1162/1240 |
| qwen | joint-context-candidate | 0.01 | 1.8590361e-07 | 2.48665599e-06 | 838.91/1008.59 | 0.217/0.276 | 4234/5152 |
| qwen | readable-state | 1 | 7.6433974e-07 | 3.67552241e-06 | 305.31/342.44 | 0.196/0.235 | 1184/1261 |
| qwen | rule-in-candidate | 0.01 | 2.64231817e-07 | 4.02600644e-06 | 402.83/452.17 | 0.206/0.306 | 1675/1853 |
| qwen | v2-baseline | 1 | 6.8224421e-07 | 3.50768062e-06 | 301.32/333.87 | 0.196/0.236 | 1162/1240 |

Both case totals include synchronized query/candidate embedding, pair-feature construction, frozen scorer, and softmax over each case's three questions. Token totals and dense embedding FLOP estimates, process-level load/warmup footprint, conditions, model/profile/bundle/CPU-output hashes, and detailed stage percentiles are in the two JSON reports. The FLOP figure is the analytical `2*600M*tokens` estimate, not a measured device counter.

Several metadata-only preflight launches were correctly rejected before the engine/model load; they are retained but excluded from GPU timing. Full separate reports: `shared-v2-baseline/PARITY_TIMING_REPORT.md` and `per-variant-heads/PARITY_TIMING_REPORT.md`. `HEAD_PROTOCOLS.md` records why the two head protocols must not be mixed.
