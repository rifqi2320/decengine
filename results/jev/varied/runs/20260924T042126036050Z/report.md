# JEV-only LoRA fresh60 evaluation

Run: `20260924T042126036050Z`
Fixture SHA-256: `2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629`

JEV only: one typed API request per state with one choice, one noul, and one score question. No Laya/MLX/GPU execution was started.

| Type | Correct / 60 | Accuracy |
|---|---:|---:|
| choice | 26/60 | 43.3% |
| noul | 56/60 | 93.3% |
| score | 14/60 | 23.3% |
| All three | 6/60 | 10.0% |

Completed 60/60 requests (180/180 normalized outcomes), errors 0.
Per-case single-request latency p50/p95: 1064.4/2259.0 ms.
Per-domain supports/accuracies are in `metrics.json`; raw bodies are in `raw-results.jsonl`.
Top-level reference/metadata fields were excluded from JEV inputs. Raw predictions were persisted before any scoring. A postprocessing-order bug meant the first scoring pass preceded the hash-freeze file; reported metrics were then regenerated after the freeze from unchanged prediction files. No inference was repeated and no labels were used for tuning. Results remain separate from frozen LoRA training.
