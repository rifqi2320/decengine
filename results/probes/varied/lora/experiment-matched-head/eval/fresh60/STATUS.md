# Corrected matched-head fresh-60 evaluation — completed after resource-policy correction

## Superseding follow-up: clarified 10 GiB total guard

The earlier blocker below accurately records the initial `SWAP512` run, but that policy
was overconstrained: the standalone 512 MiB swap-growth ceiling was an agent-added guard,
not the user's requested stop condition. It stopped Qwen head at 5.25 GB total and
prevented the other arms from starting. That policy/result remains archived and is not
treated as the final run.

Per the corrected instruction, a new hash-bound policy removed the standalone swap
ceiling. The hard total guard is process peak footprint plus positive system-swap growth
from each run's baseline `< 10 GiB`; process early-stop remains 6 GiB and the absolute
process cap 8 GiB. `mx.clear_cache()` and Python GC run after every case. The fixture,
weights, training configuration, freeze and 512-token non-truncating inference limit
were unchanged. The current Harrier policy is
`../../MATCHED-INFERENCE-POLICY-512-TOTAL10G-FINAL.json` (SHA-256
`57455ae8d88322ea7703d4d67b1ee317ad198b57071b35ebc57be29a01f1a590`). Qwen head and
Qwen LoRA policy snapshots are retained separately because the evaluator source changed
between those sequential runs; each snapshot pins its own evaluator hash. No JEV/Laya
inference was repeated.

All four selected checkpoints completed sequentially with 180 request-only predictions
each. Qwen head's first seven cases were checked against the preserved earlier partial
before any reference scoring: exact raw logits, probabilities, candidate schema and
normalized predictions (`parity-prefix-check.json`, references not read). No resource
guard was reached. Full predictions and run metadata are under the corresponding
`*-total10g/` directories; raw prediction files were frozen before this report read any
references. The full comparison is `FINAL-COMPARISON.md` and machine-readable
`FINAL-COMPARISON.json`. The four corrected rows are separate from all earlier
three-epoch LoRA results.

| Corrected selected checkpoint | Result | Case p50/p95 ms | Max process peak | Max positive swap growth | Max process peak + growth |
|---|---|---:|---:|---:|---:|
| Qwen matched head, epoch 2 | 180/180 predictions; 7-case exact parity | 387.8 / 426.8 | 2,376,025,864 B | 0 B | 2,376,025,864 B |
| Qwen matched LoRA, epoch 9 | 180/180 predictions | 391.0 / 428.4 | 2,405,861,176 B | 447,280,579 B | 2,853,141,755 B |
| Harrier matched head, epoch 9 | 180/180 predictions | 388.4 / 426.3 | 2,377,189,152 B | 0 B | 2,377,189,152 B |
| Harrier matched LoRA, epoch 9 | 180/180 predictions | 390.5 / 427.8 | 2,385,774,344 B | 0 B | 2,385,774,344 B |

The results remain exploratory because fresh60 was previously scored. `FINAL-COMPARISON`
includes per-type and all-three accuracy, score MAEs, candidate-position distributions,
structural checks, provenance/hashes, resource summaries, and the previously archived
full-660, same-460, Laya and JEV systems. Scoring does not alter checkpoint selection.

## Initial SWAP512 policy and stop (historical, superseded)

The 9-epoch train-only freeze was verified at SHA-256
`ecb46f3a952d48a78a9004ad0213e70e57da0e9bd84e183fda31a5b63186eb1f`. It declares
`TRAIN_ONLY_SELECTION_FROZEN`, `SEALED_NOT_ACCESSED`, and the four intended selected
checkpoints: Qwen head epoch 2, Qwen LoRA epoch 9, Harrier head epoch 9, Harrier LoRA
epoch 9. The initial inference-only policy was
`../../MATCHED-INFERENCE-POLICY-512-SWAP512.json` (SHA-256
`b164ae4614ac4faffe5a40c5e25de467cc51fa280335e0f1134a652a2ab7043d`). It pins the
freeze, evaluator, compiler, matched/shared trainers, adapter, selected checkpoints and
fixture. It keeps the 288-token training guard and frozen weights unchanged, authorizes
inference up to 512 without truncation, retains 6 GiB early-stop/8 GiB absolute process
limits, and allows positive swap growth only below 512 MiB while process peak plus
positive swap growth remains under 10 GiB. The previously scored fixture makes any new
comparison exploratory. Exact compact profiles, generic-v2 compiler, tokenizer,
config/checkpoint hashes, selection epochs and train-only manifest records passed
preflight for all four arms.

The first Qwen-head partial attempt used the previous zero-swap-growth policy
`../../MATCHED-INFERENCE-POLICY-512-attempt.json` (SHA-256
`7fe05b947a00b0fbc6a32554c1e9a848c5b9278f4662f6d4cca75764f1b53391`) and remains
preserved under `qwen-head/` (3/180 rows). The swap512 policy is separate and adds
source-code pins and the explicitly authorized resource budget.

The request-only preflights covered all 60 ordered IDs, 180 questions and 750 compiled
strings per arm. Training's observed maximum is 266 tokens; the fresh60 maximum is 375.
All 750 strings fit the separate 512-token inference policy. No reference or target was
read by the new evaluator.

## Concrete blocker at that time (superseded by follow-up above)

Only `qwen-head` was started under the new swap512 policy. It persisted 21/180 raw and
normalized request-only predictions (the first seven cases) before stopping at
`608,509,624 B` positive swap growth, beyond the 512 MiB ceiling. Process peak was
`4,643,292,936 B`; process peak plus growth was `5,251,802,560 B`, both below their
6/10 GiB limits. The model process exited. The other three arms were not started and
no reference scoring was performed. Do not continue the sequence after this guard
breach.

The first partial remains at `qwen-head/` (3 rows). New-policy partials are at
`qwen-head-epoch02-swap512/`: raw predictions SHA-256
`5c758dc92448ef758bd033e5c8133704c1eccb71f7fdb465bafc7c03271ec147`, normalized
predictions SHA-256 `6fd46285dfa8e2f64493533d274da41057b514ecbda66754a92aea9f47e289db`.
The stop marker, per-question resource/swap log and external 0.5-second watch accompany
both attempts. Neither partial is scored or may be listed as a complete Qwen-head result.

Request-only structural checks confirmed fixture candidate order and valid normalized
probabilities for all 21 new partial Qwen-head rows and the archived outputs. Within
those seven cases, selected candidate positions were choice `{0:1, 1:3, 2:1, 3:2}`,
noul `{0:7}`, and score `{2:7}`. This is a small partial sample, so it is not a
full-run collapse conclusion, but the noul/score selections are position-collapsed in
the partial sample. The archived full-660, same-460, JEV, and Laya bundles each have
non-singleton candidate-position distributions for all three types. No full
corrected-arm parity/collapse conclusion is possible because the other three arms have
no predictions.

### Initial incomplete snapshot — not the final result

| Corrected checkpoint | Selected epoch | Fresh60 status at initial stop | Scores at initial stop | Case p50/p95 |
|---|---:|---|---|---|
| Qwen matched head | 2 | Stopped after 7 cases on 512 MiB swap ceiling; 21/180 rows (plus earlier preserved 3-row attempt) | Not scored | Not available |
| Qwen matched LoRA | 9 | Not started (sequence stopped after Qwen-head guard breach) | Not scored | Not available |
| Harrier matched head | 9 | Not started | Not scored | Not available |
| Harrier matched LoRA | 9 | Not started | Not scored | Not available |

## Verified reusable same-fixture results (not rerun)

The fixture SHA-256 is
`2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629`; all checks below
confirmed its expected 60 ordered IDs. JEV and Laya each have 60 case rows in fixture
order and verified checksums. Full-660 and same-460 baseline outputs each have the
exact 180 unique case/question pairs and the same fixture hash. No JEV/Laya APIs, MLX,
Metal or GPU were invoked in this attempt.

| Existing system (separate from corrected 9-epoch arms) | Choice | Noul | Score | All three | Score expected-value MAE | Score ordinal MAE | Case p50/p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen frozen baseline, full 660 | 20/60 | 37/60 | 10/60 | 2/60 | 52.076 | 1.583 | 320.9 / 344.8 |
| Harrier frozen baseline, full 660 | 16/60 | 37/60 | 16/60 | 2/60 | 38.113 | 1.267 | 320.5 / 344.9 |
| Qwen same-460 head-only linear | 21/60 | 37/60 | 6/60 | 2/60 | 44.825 | 1.500 | 322.1 / 348.2 |
| Harrier same-460 head-only linear | 17/60 | 35/60 | 20/60 | 3/60 | 35.873 | 1.133 | 322.6 / 347.6 |
| Laya archived run | 18/60 | 33/60 | 19/60 | 1/60 | Not available (no probabilities) | Not reported | 494.8 / 520.7 |
| JEV archived run | 26/60 | 56/60 | 14/60 | 6/60 | Not available (no probabilities) | Not reported | 1064.4 / 2259.0 |

The full-660 results are under `results/probes/varied/lora/eval/fresh60/predictions/`
(`qwen-frozen-baseline-full660/`, `harrier-frozen-baseline-full660/`). Laya's archived
run is `results/laya/varied/runs/20260924T054309737898Z-cacheguard/`; JEV's is
`results/jev/varied/runs/20260924T042126036050Z/`. Their own historical metrics remain
the source for those archived scores. The earlier three-epoch LoRA results are not
the corrected matched-head checkpoints and are excluded from this table. No combined
final comparison report was produced because the four corrected arms lack complete
predictions.

Archived prediction SHA-256: Qwen full-660 `1baeb0cd61470529e80d33473a7908bd4a71e1eae6269f97632b09221aef1e12`,
Harrier full-660 `b86bc07f2bd3ff1f5fc119a1e8cc59e630398a01dc164a790c3a8a396f1d2754`,
Qwen same-460 linear `a704c39005e5fe0cbec4f78331d0238a625e27fd436fd6b2af70c7692fb46ae8`,
Harrier same-460 linear `798491a9741934fe5c35d04ba22fdf7400e5c72f36ec4672ed452ad9ee9463c9`,
Laya `0273e0c8b9a453de8688f0542f1382820e984613dcaafe09aa1b8c348bd78181`, and
JEV `b0f8e192e1b0effa0f9c7e5c665ccdd09a719dc6f1abe7188cf353e1693fa86f`. JEV's
metadata notes that its post-reference freeze was repaired by regenerating metrics from
the unchanged, hash-frozen prediction file; no inference was retried or tuned.
