# LoRA fresh-60 quantitative degeneracy audit

**Scope:** read-only audit of the saved Qwen/Harrier fresh-60 raw and normalized predictions, train/dev artifacts and the trainer/evaluator source. No model was loaded or inference run; no GPU/API was used. I did not inspect the fixture source or revisit subjective reference labels. No tuning or source edits were made.

## Bottom line

The saved outputs are not numerically identical, and the choice scorer is **not** choosing the first candidate. However, both models have the same striking decision collapse on this set: **60/60 Noul answers are `false` (unsupported, candidate index 0), and 60/60 score answers select `workable` (candidate index 2)**. Their choice selections also agree case/question-keyed on all 60 questions, although logits differ. The reported matching type accuracies are therefore a consequence of matching decisions against the same evaluation set, not proof that model outputs or weights are identical.

I found **no demonstrated candidate-index, Noul-polarity, score-value mapping, or checkpoint-loading implementation defect**. The output is plausible execution of a weak/under-discriminating scorer: probabilities are input-dependent and not uniform, but Noul is nearly a coin flip and score is close to flat while its argmax is constant. Treat the raw files as genuine recorded model outputs, but **do not treat fresh-60 metrics as evidence of useful Noul/score discrimination or as a trustworthy model comparison**. This is an output-quality degeneracy, not a discovered evaluator accounting bug.

## Prediction-level checks

Raw rows were paired by `(id, question_name)`; each model has 180 rows (60 per type). Candidate positions below are zero-based indices in `candidate_keys`; Noul index 0 is `unsupported`/`false`, index 1 is `supported`/`true`. Probability min/max is over all candidate probability elements for that model/type; logit min/max is over all raw logit elements. Entropy is per-row categorical entropy in nats, summarized across 60 rows.

| Model | Type | Selected candidate position counts | Selected value counts | Probability min–max | Raw logit min–max | Per-row entropy min / mean / max |
|---|---|---:|---|---:|---:|---:|
| Qwen | choice | 0:22, 1:17, 2:11, 3:6, 4:4 | — | 0.1255–0.6574 | -0.6469–0.3779 | 0.6427 / 1.1708 / 1.5892 |
| Harrier | choice | 0:22, 1:17, 2:11, 3:6, 4:4 | — | 0.1271–0.6772 | -0.6005–0.4088 | 0.6290 / 1.1593 / 1.5833 |
| Qwen | Noul | 0:60, 1:0 | false:60, true:0 | 0.4741–0.5259 | -0.1212–0.0364 | 0.6918 / 0.6925 / 0.6929 |
| Harrier | Noul | 0:60, 1:0 | false:60, true:0 | 0.4758–0.5242 | -0.0507–0.0859 | 0.6920 / 0.6924 / 0.6928 |
| Qwen | score | 0:0, 1:0, 2:60 (no winner at later positions) | workable:60 | 0.1871–0.3517 | -0.1625–-0.0283 | 1.0975 / 1.3639 / 1.6089 |
| Harrier | score | 0:0, 1:0, 2:60 (no winner at later positions) | workable:60 | 0.1869–0.3606 | -0.2063–-0.0516 | 1.0969 / 1.3636 / 1.6087 |

The outputs are not literally uniform or input-independent: each type has 60 distinct per-row logit vectors (probabilities are derived from these); Qwen/Harrier probabilities vary. But the Noul logits stay very close to a tie: mean entropy is about `ln(2)` for both and the probability of the chosen unsupported class stays in a narrow 0.474–0.526 / 0.476–0.524 band. Score rows have nonzero candidate separation, but the same third candidate is the argmax every time and mean entropy is high relative to the available 3–5 levels. Choice winners span positions 0–4 rather than always position zero.

## Qwen vs Harrier keyed comparison

| Type | Same selected label/value on paired rows | Identical raw-logit vectors | Elementwise absolute logit difference: min / mean / max |
|---|---:|---:|---:|
| choice | 60/60 | 0/60 | 0.00142 / 0.08551 / 0.27798 |
| Noul | 60/60 | 0/60 | 0.00146 / 0.04278 / 0.10621 |
| score | 60/60 | 0/60 | 0.00030 / 0.04329 / 0.11266 |

Thus the models are not sharing identical raw outputs or a single loaded adapter. Yet their winning labels match on every keyed question, explaining why the saved aggregate accuracies match exactly (`choice 15/60`, `noul 30/60`, `score 20/60`, `all-three 0/60`). The score expected-value MAE is also recorded as `30.1316` for both. The matching 15/60 choice accuracy alongside a diverse winner-position distribution also disproves an always-first-choice explanation.

## Training, gradients, heads, and selection

- `SELECTION-FREEZE.json` records completed 1,380-step train/dev runs for both selected checkpoints, best epoch 1, and nonzero finite learned LoRA tensors. The selected Qwen/Harrier best-checkpoint hashes differ (`092f…d9ca` and `d6c2…ab1e`). Recorded summaries give Qwen best dev loss/accuracy `0.9594 / 0.385`, Harrier `0.9483 / 0.395`; dev accuracy increases in later epochs while dev loss worsens, so selecting epoch 1 by minimum aggregate loss does not ensure strong per-type behavior. The freeze explicitly notes per-type dev predictions/metrics were not saved.
- Both frozen manifests record nonzero LoRA/head tensor counts and optimizer moments. Their recorded per-step LoRA gradient norm ranges are Qwen `0.00033898–0.12006755` for score, `0.00052050–0.49331234` for Noul, and `0.00282102–5.63092975` for choice; Harrier `0.00011966–0.15966646`, `0.00021751–0.09444280`, and `0.00066419–4.28400106`, respectively. The selected run logs contain all 1,380 steps. This rules out “LoRA never trained” but not poor generalization/calibration.
- The logs save combined parameter gradient norm and LoRA-only norm, not a separate per-step typed-head gradient norm. The smoke checks explicitly reject zero total/LoRA gradients and require a real minibatch loss decrease. Nonzero serialized head parameters and candidate-dependent predictions establish head loading/use; the records do not support a claim that every head tensor had nonzero gradients on every step.
- Training has meaningful candidate-wise objectives: `TypedScorer` computes a score for each paired query/candidate (`train.py:307–320`); `preference_loss` applies local softmax CE for choice/Noul and a candidate-valued ordinal CE + expected-value Huber objective for score (`train.py:398–416`). Noul target encoding maps true to `supported` and false to `unsupported` (`train.py:102–108`). So the graph does carry candidate-dependent gradients; these saved logs do not show gradient vanishing at the whole-model level.

## Source-path defect review

- **Noul polarity:** evaluator candidate order is explicitly `unsupported`, then `supported` (`evaluate_frozen.py:300–304`); prediction returns `bool(winner)` (`:259–266`), consistent with training's `supported` index 1 for true. The 60 false predictions reflect the learned winning side, not a reversed boolean decoder.
- **Score-level numeric mapping:** compiled level labels and integer values preserve the same `question["levels"]` order (`evaluate_frozen.py:296–300`, `:675–678`). Expected value zips that ordered level array with its matching probability vector (`:824–827`); the post-pass MAE maps gold labels to their configured numeric values (`:855–862`). Trainer likewise aligns `keys`, values, and gold index from that level order (`train.py:109–116`), using those values in its candidate-aware score loss (`train.py:405–416`). No evident position/value inversion.
- **Checkpoint/head loading:** evaluator validates the exact LoRA + typed-head key set, shapes, and FP32 dtype (`evaluate_frozen.py:313–331`, invoked at `:604–613`), injects the one selected LoRA layer, creates the typed scorer, then loads the frozen best weights (`:737–744`). It also verifies hashes and source/config provenance before model load. The distinct model hashes and distinct paired raw logits are consistent with successful separate checkpoints.
- **Candidate scoring:** the trainer broadcasts query per candidate, concatenates candidate-specific `[q, c, q*c, abs(q-c)]` features, applies a shared projection and the named type head (`train.py:316–320`). This is a candidate-dependent scorer, not a query-only scalar broadcast.

No concrete implementation bug and therefore no code fix is justified by this audit. A minimal *guardrail* (not an algorithm fix) would be to persist per-type grouped-dev prediction/selection histograms and flag a type with a single selected label or near-chance entropy before freezing/evaluating. Current grouped dev artifacts retain only aggregate loss/accuracy, so they cannot independently diagnose whether these same collapses were already present at selection time.

## References inspected

- Saved raw and normalized results: `eval/fresh60/predictions/{qwen-lora,harrier-lora}/{raw-predictions.jsonl,predictions.jsonl,metrics.json}`.
- Freeze and training records: `SELECTION-FREEZE.json`, `runs/{qwen-local-lowmem-v2,harrier-local-lowmem-v2}/{summary.json,epochs.jsonl,steps.jsonl}`.
- Source: `train.py:102–120,307–338,398–416,641–683`; `evaluate_frozen.py:259–267,280–310,313–331,604–613,737–744,803–862`.
