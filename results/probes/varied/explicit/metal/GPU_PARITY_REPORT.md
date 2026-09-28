# Explicit60 all-MLX prediction parity

Compared only against the archived CPU prediction outputs; no target/reference labels were read. Same frozen typed scorers, no fitting or tuning. Every question has exact candidate identifiers.

| Model | Type | n | selected/boolean/score exact | max probability abs diff |
|---|---|---:|---:|---:|
| qwen | choice | 60 | True (60/60) | 3.55884952e-07 |
| qwen | noul | 60 | True (60/60) | 5.17025325e-07 |
| qwen | score | 60 | True (60/60) | 8.4313149e-07 |
| qwen | all-three case prediction parity | 60 | 60/60 | — |
| qwen | 3-question GPU case time p50/p95 | 60 | — | 393.89/422.85 ms |
| harrier | choice | 60 | True (60/60) | 1.92530713e-07 |
| harrier | noul | 60 | True (60/60) | 5.15010486e-07 |
| harrier | score | 60 | True (60/60) | 4.04866288e-07 |
| harrier | all-three case prediction parity | 60 | 60/60 | — |
| harrier | 3-question GPU case time p50/p95 | 60 | — | 377.20/400.16 ms |
