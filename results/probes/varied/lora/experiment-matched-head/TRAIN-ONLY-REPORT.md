# Matched-head experiment: train-only report

## Scope and protocol

CPU-only post-run audit of the four completed runs. No MLX/model was loaded, no inference/training was run, and no heldout/test labels, predictions, or hashes were accessed.
The fixed split has 460 train questions (300 cases, 11 families) and 200 dev questions (120 cases, 4 families). The deterministic grouped split has nine connected components, of which the dev set contains only two; this is a small and noisy basis for selection.
Both arms use the same profile-compiled generic `decengine-varied-pair-text-v2-generic-types` inputs per model, the same 192-wide GELU typed head initialization, optimizer/loss/order, and the same case/family split. Qwen's export manifest is profile version 1 and Harrier's is profile version 2; neither arm changes that model-specific export. Token inputs reject above 288 tokens (no truncation).
Primary selection: maximize equal-weight macro exact accuracy across choice, noul, and score; ties break by lower mean dev objective loss, lower normalized score expected-value MAE, then earlier epoch. All epochs 1–9 are retained; no selection uses heldout data.

## Frozen selections

| Model | Arm | Selected epoch | Dev choice | Dev noul | Dev score | Type-balanced macro | Score norm. MAE | Mean dev loss | Selected checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen | head | 2 | 0.350 | 0.533 | 0.325 | 0.4028 | 0.3291 | 0.9703 | `58bef9ac575faf62344d7e176e5c80910fff7370dd8314295cc2d9a23d0437b8` |
| qwen | lora | 9 | 0.330 | 0.667 | 0.500 | 0.4989 | 0.3165 | 1.2567 | `cc6ceb9cd13819cd162ec494630660c159aa27e2e37cf5b82391b36ef5f6bf06` |
| harrier | head | 9 | 0.310 | 0.700 | 0.525 | 0.5117 | 0.3140 | 1.0356 | `92145ec36c4eeec1d89e70818313f57a758bd696b121ddcd20d8cd1153b59c28` |
| harrier | lora | 9 | 0.310 | 0.683 | 0.575 | 0.5228 | 0.2971 | 1.0282 | `06f1f13eb90fb84ce8bfac8a444fad7af2312e53c2c23745342256aa567bc10f` |

## Per-epoch dev metrics

### qwen

#### head

| Epoch | Choice acc. | Noul acc. | Score acc. | Macro acc. | Score norm. MAE | Mean dev loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.340 | 0.533 | 0.325 | 0.3994 | 0.3292 | 0.95595 |
| 2 | 0.350 | 0.533 | 0.325 | 0.4028 | 0.3291 | 0.97027 |
| 3 | 0.300 | 0.533 | 0.325 | 0.3861 | 0.3290 | 0.99668 |
| 4 | 0.330 | 0.533 | 0.325 | 0.3961 | 0.3286 | 1.01384 |
| 5 | 0.340 | 0.533 | 0.325 | 0.3994 | 0.3281 | 1.03431 |
| 6 | 0.280 | 0.533 | 0.325 | 0.3794 | 0.3276 | 1.05967 |
| 7 | 0.330 | 0.533 | 0.325 | 0.3961 | 0.3271 | 1.08475 |
| 8 | 0.280 | 0.533 | 0.325 | 0.3794 | 0.3263 | 1.12685 |
| 9 | 0.280 | 0.533 | 0.325 | 0.3794 | 0.3257 | 1.16093 |

Train/dev loss and macro trajectory (do not interpret training loss alone as convergence):

| Epoch | Train mean loss | Dev mean loss | Dev macro acc. |
|---:|---:|---:|---:|
| 1 | 0.69680 | 0.95595 | 0.3994 |
| 2 | 0.60865 | 0.97027 | 0.4028 |
| 3 | 0.57729 | 0.99668 | 0.3861 |
| 4 | 0.55957 | 1.01384 | 0.3961 |
| 5 | 0.54041 | 1.03431 | 0.3994 |
| 6 | 0.52971 | 1.05967 | 0.3794 |
| 7 | 0.51261 | 1.08475 | 0.3961 |
| 8 | 0.49637 | 1.12685 | 0.3794 |
| 9 | 0.48214 | 1.16093 | 0.3794 |

#### lora

| Epoch | Choice acc. | Noul acc. | Score acc. | Macro acc. | Score norm. MAE | Mean dev loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.340 | 0.533 | 0.325 | 0.3994 | 0.3293 | 0.96745 |
| 2 | 0.330 | 0.533 | 0.325 | 0.3961 | 0.3296 | 1.02840 |
| 3 | 0.320 | 0.550 | 0.325 | 0.3983 | 0.3295 | 1.03408 |
| 4 | 0.330 | 0.650 | 0.325 | 0.4350 | 0.3292 | 1.06618 |
| 5 | 0.330 | 0.667 | 0.325 | 0.4406 | 0.3282 | 1.12285 |
| 6 | 0.330 | 0.617 | 0.325 | 0.4239 | 0.3268 | 1.13342 |
| 7 | 0.330 | 0.617 | 0.325 | 0.4239 | 0.3258 | 1.18812 |
| 8 | 0.360 | 0.633 | 0.450 | 0.4811 | 0.3222 | 1.22483 |
| 9 | 0.330 | 0.667 | 0.500 | 0.4989 | 0.3165 | 1.25669 |

Train/dev loss and macro trajectory (do not interpret training loss alone as convergence):

| Epoch | Train mean loss | Dev mean loss | Dev macro acc. |
|---:|---:|---:|---:|
| 1 | 0.68820 | 0.96745 | 0.3994 |
| 2 | 0.60514 | 1.02840 | 0.3961 |
| 3 | 0.62168 | 1.03408 | 0.3983 |
| 4 | 0.60116 | 1.06618 | 0.4350 |
| 5 | 0.58726 | 1.12285 | 0.4406 |
| 6 | 0.60486 | 1.13342 | 0.4239 |
| 7 | 0.58459 | 1.18812 | 0.4239 |
| 8 | 0.57580 | 1.22483 | 0.4811 |
| 9 | 0.55108 | 1.25669 | 0.4989 |

### harrier

#### head

| Epoch | Choice acc. | Noul acc. | Score acc. | Macro acc. | Score norm. MAE | Mean dev loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.350 | 0.533 | 0.325 | 0.4028 | 0.3284 | 0.94753 |
| 2 | 0.340 | 0.533 | 0.325 | 0.3994 | 0.3274 | 0.95079 |
| 3 | 0.290 | 0.533 | 0.325 | 0.3828 | 0.3263 | 0.96705 |
| 4 | 0.290 | 0.533 | 0.325 | 0.3828 | 0.3249 | 0.96964 |
| 5 | 0.320 | 0.533 | 0.375 | 0.4094 | 0.3232 | 0.97542 |
| 6 | 0.310 | 0.533 | 0.450 | 0.4311 | 0.3210 | 0.98729 |
| 7 | 0.300 | 0.533 | 0.500 | 0.4444 | 0.3190 | 0.99550 |
| 8 | 0.290 | 0.700 | 0.525 | 0.5050 | 0.3163 | 1.01729 |
| 9 | 0.310 | 0.700 | 0.525 | 0.5117 | 0.3140 | 1.03559 |

Train/dev loss and macro trajectory (do not interpret training loss alone as convergence):

| Epoch | Train mean loss | Dev mean loss | Dev macro acc. |
|---:|---:|---:|---:|
| 1 | 0.69478 | 0.94753 | 0.4028 |
| 2 | 0.60687 | 0.95079 | 0.3994 |
| 3 | 0.57547 | 0.96705 | 0.3828 |
| 4 | 0.55875 | 0.96964 | 0.3828 |
| 5 | 0.54207 | 0.97542 | 0.4094 |
| 6 | 0.52974 | 0.98729 | 0.4311 |
| 7 | 0.51578 | 0.99550 | 0.4444 |
| 8 | 0.50040 | 1.01729 | 0.5050 |
| 9 | 0.48647 | 1.03559 | 0.5117 |

#### lora

| Epoch | Choice acc. | Noul acc. | Score acc. | Macro acc. | Score norm. MAE | Mean dev loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.360 | 0.533 | 0.325 | 0.4061 | 0.3285 | 0.94957 |
| 2 | 0.360 | 0.533 | 0.325 | 0.4061 | 0.3276 | 0.97242 |
| 3 | 0.350 | 0.533 | 0.325 | 0.4028 | 0.3263 | 0.97406 |
| 4 | 0.340 | 0.533 | 0.325 | 0.3994 | 0.3241 | 0.98795 |
| 5 | 0.340 | 0.700 | 0.400 | 0.4800 | 0.3209 | 1.00960 |
| 6 | 0.340 | 0.700 | 0.500 | 0.5133 | 0.3162 | 1.00637 |
| 7 | 0.340 | 0.517 | 0.525 | 0.4606 | 0.3119 | 1.02737 |
| 8 | 0.300 | 0.633 | 0.575 | 0.5028 | 0.3034 | 1.01556 |
| 9 | 0.310 | 0.683 | 0.575 | 0.5228 | 0.2971 | 1.02822 |

Train/dev loss and macro trajectory (do not interpret training loss alone as convergence):

| Epoch | Train mean loss | Dev mean loss | Dev macro acc. |
|---:|---:|---:|---:|
| 1 | 0.69102 | 0.94957 | 0.4061 |
| 2 | 0.59990 | 0.97242 | 0.4061 |
| 3 | 0.58747 | 0.97406 | 0.4028 |
| 4 | 0.56719 | 0.98795 | 0.3994 |
| 5 | 0.55463 | 1.00960 | 0.4800 |
| 6 | 0.55614 | 1.00637 | 0.5133 |
| 7 | 0.52148 | 1.02737 | 0.4606 |
| 8 | 0.50443 | 1.01556 | 0.5028 |
| 9 | 0.47246 | 1.02822 | 0.5228 |

## Interpretation and convergence limits

- Qwen frozen-head selected epoch 2 at macro 0.4028; Qwen LoRA selected epoch 9 at 0.4989 (Δ +0.0961). Qwen LoRA macro was still rising from epoch 8 (0.4811) to 9 (0.4989), while dev mean loss worsened from 1.2248 to 1.2567. This is not evidence of convergence; the accuracy/loss divergence and very small grouped dev basis warrant caution.
- Harrier frozen-head selected epoch 9 at 0.5117; Harrier LoRA selected epoch 9 at 0.5228 (Δ +0.0111). Both arms' macro accuracy rose from epoch 8 to 9 (head 0.5050→0.5117; LoRA 0.5028→0.5228); LoRA dev loss worsened 1.0156→1.0282. Neither has demonstrated convergence by the 9-epoch cap.
- The dev loss rose materially across training for all arms even where primary macro accuracy improved. Per-type prediction histograms and confidence are saved in each epoch's JSONL metrics and `COMPARISON.json`; early constant-class predictions in noul/score were detected by the ≥90% concentration diagnostic. The small two-component dev split makes these fluctuations particularly uncertain.
- Historical explicit cached-embedding C=1 head-only dev accuracies were Qwen 0.480 and Harrier 0.570. This is a different linear architecture/training recipe, so it is shown only as context and is not the matched frozen-backbone typed-head control.

## Integrity and resources

All four runs are complete (9 epochs / 4,140 updates each). Initial head safetensors hashes match; within each model, initial train-sample logits match between arms at absolute tolerance 1e-6. Compiled-text parity reports passed. Every step has finite nonzero head gradients; frozen-head LoRA gradients are zero at every step, and LoRA-arm gradients are finite/nonzero at every step.

| Run | Process peak (GiB) | 6 GiB stop | Swap baseline (B) | Swap min (B) | Swap max (B) | Swap increased? |
|---|---:|---:|---:|---:|---:|---|
| qwen_head | 2.197 | 6.000 | 1366294528 | 1341128704 | 1366294528 | no |
| qwen_lora | 2.255 | 6.000 | 1341128704 | 1324351488 | 1341128704 | no |
| harrier_head | 2.198 | 6.000 | 1324351488 | 1315962880 | 1324351488 | no |
| harrier_lora | 2.254 | 6.000 | 1315962880 | 1307574272 | 1315962880 | no |

The process peaks are below the 6 GiB early-stop and 8 GiB hard cap. Swap never exceeded the per-run start baseline; decreases are allowed by the guard. All code/config/source/checkpoint/log hashes and split-set hashes are recorded in `TRAIN-ONLY-FREEZE.json`.
The pre-correction interrupted Qwen-head attempt is preserved under `runs/qwen-head-interrupted-attempt1/` and excluded from this report/freeze.

## Files

- `TRAIN-ONLY-FREEZE.json`: hash-bound train/dev selections, integrity/resource audit, and epoch metrics.
- `COMPARISON.md` / `COMPARISON.json`: pairwise per-epoch model comparisons and diagnostics.
- `runs/{qwen,harrier}-{head,lora}/`: configs, logs, immutable epoch checkpoints, best/latest checkpoints, optimizer state, and summaries.
- `verify_and_freeze.py`: reproducible CPU-only validator/freeze generator; it does not import MLX or read any test/heldout path.

No additional training is authorized by this report; the selected train-only artifacts are frozen pending the user's/orchestrator's next decision.
