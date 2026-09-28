# Frozen model comparison: LoRA fresh-60 (six frozen systems + Laya + separate JEV)

Fixture SHA-256: `2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629`  
All runs are aligned to the exact ordered case IDs `mq-lora-fresh-001` through `mq-lora-fresh-060` (60 cases, 180 typed outcomes). The existing six-system comparison remains preserved at [`comparison-core-six.md`](comparison-core-six.md); JEV remains an independently run remote system, not part of the LoRA model family.

| Model | Choice | Noul | Score | All three | Score expected-value MAE | Score ordinal MAE |
|---|---:|---:|---:|---:|---:|---:|
| qwen_lora | 15/60 (25.0%) | 30/60 (50.0%) | 20/60 (33.3%) | 0/60 (0.0%) | 30.132 | 0.933 |
| harrier_lora | 15/60 (25.0%) | 30/60 (50.0%) | 20/60 (33.3%) | 0/60 (0.0%) | 30.161 | 0.933 |
| qwen_frozen_baseline_full660 | 20/60 (33.3%) | 37/60 (61.7%) | 10/60 (16.7%) | 2/60 (3.3%) | 52.076 | 1.583 |
| harrier_frozen_baseline_full660 | 16/60 (26.7%) | 37/60 (61.7%) | 16/60 (26.7%) | 2/60 (3.3%) | 38.113 | 1.267 |
| qwen_headonly_same460 | 21/60 (35.0%) | 37/60 (61.7%) | 6/60 (10.0%) | 2/60 (3.3%) | 44.825 | 1.500 |
| harrier_headonly_same460 | 17/60 (28.3%) | 35/60 (58.3%) | 20/60 (33.3%) | 3/60 (5.0%) | 35.873 | 1.133 |
| **laya_root_mlx_float32** | **18/60 (30.0%)** | **33/60 (55.0%)** | **19/60 (31.7%)** | **1/60 (1.7%)** | n/a | n/a |
| **jev_latest (separate remote system)** | **26/60 (43.3%)** | **56/60 (93.3%)** | **14/60 (23.3%)** | **6/60 (10.0%)** | n/a | n/a |

Laya is fully scored, but its results are weak: only 1/60 cases are correct on all three outcomes. These data do not support a claim that LoRA improved over the frozen baselines. `n/a` means the Laya and JEV scorers do not expose the same score-value/error quantities used for the six frozen systems' expected-value and ordinal MAE calculations; the directly comparable score-label accuracy is shown.

## Laya run and guard

- Pinned root checkpoint `convaiinnovations/laya@5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`, float32, MLX 0.32.2 / Metal, exact root temperatures, unchanged fixture prompt translation.
- The mandatory first-four-case smoke completed 12/12 decisions. All 11 previously preserved typed probabilities/labels matched exactly (maximum absolute probability difference `0.0`); full inference then continued once in the same process. No reference scoring was performed until predictions were frozen.
- 60/60 cases and 180/180 decisions completed with zero errors. Process physical-footprint peak: `4,298,246,880` bytes (< 6 GiB early stop); swap growth from run baseline: `0` bytes (< 256 MiB guard). The 8 GiB absolute cap remained in force.
- Run artifacts: [`20260924T054309737898Z-cacheguard`](../../../../../laya/varied/runs/20260924T054309737898Z-cacheguard/). Raw output SHA-256: `680e085f6968216fe87c141c0f5ef286277b9db3cd9836aced260385ee231816`; prediction SHA-256: `0273e0c8b9a453de8688f0542f1382820e984613dcaafe09aa1b8c348bd78181`.

## Timing and interpretation caveats

Latency values below are reported with their own scopes and are not a normalized ranking. The Laya case values are runner wall time from before the first question through the end of the case, excluding model load but **including** the three forward/evaluation steps, per-call footprint/swap sampling, cache cleanup/GC, and incremental artifact writes: p50 `494.8 ms`, p95 `520.7 ms`, mean `494.0 ms`. The archived JEV scope is one remote API request for all three questions per case (thus includes service/network effects): p50 `1,064.4 ms`, p95 `2,259.0 ms`, mean `1,276.7 ms`. Per-system scopes for the six frozen systems remain in [`comparison-core-six.md`](comparison-core-six.md).

The LoRA holdout reached 375 input tokens; observed training maximum was 266, with a configured training guard of 288 and inference cap of 512. No truncation was recorded. This is an authorized length extrapolation and remains a caveat for comparing the LoRA systems.

JEV is kept separate from local MLX runs. Its immutable raw outcomes were verified against the same 60 case IDs and fixture hash; its archive notes that an initial scoring pass preceded the freeze artifact, then metrics were deterministically regenerated after freezing unchanged predictions. No inference was repeated or tuned.

Machine-readable Laya/JEV metrics and provenance are in [`comparison-laya-jev-addendum.json`](comparison-laya-jev-addendum.json).
