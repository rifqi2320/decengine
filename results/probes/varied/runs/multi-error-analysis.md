# Multi-task varied probe: archived error diagnosis

## Scope and reproducibility

Read-only analysis of `multi-qwen` and `multi-harrier` archived `predictions.jsonl`, `question_outcomes.jsonl`, `metrics.json`, and `metadata.json`, plus aggregate input/target distributions in `benchmarks/cases/varied/multi/{train-120,test-60}.jsonl`. No labels were rereviewed, predictions regenerated, or APIs called. All figures below can be reproduced by grouping the JSONL rows on `question_type`; correctness/confidence by question uses matching `(id, question_name)` rows. Candidate position is zero-based in archived `candidate_keys` (choice) or fixture option/level insertion order (train distribution). The two models share the same train/test cases, as confirmed by metadata input hashes.

## Observations

### Overall and per-type error

| Type (60 each) | Qwen correct | Harrier correct | Test target distribution |
|---|---:|---:|---|
| Choice | 23/60 (38.3%) | 22/60 (36.7%) | 5 answer keys: quartz 22, cobalt 17, birch 11, ember 6, moss 4 |
| Noul / boolean | 30/60 (50.0%) | 25/60 (41.7%) | false 39, true 21 |
| Score exact class | 29/60 (48.3%) | 12/60 (20.0%) | ordinal 0: 0, 1: 22, 2: 38 |

The type-specific chance baselines are not identical: the test score target is 0/22/38 across three ordinal levels, and dominant-class accuracy is 63.3%; choice dominant-key is 36.7%; noul dominant-class is 65.0%. Thus raw accuracy alone overstates relative score/choice performance and understates boolean weakness relative to the majority baseline. All-three-correct case counts are 9/60 Qwen and 1/60 Harrier (from archived metrics).

### Class imbalance and systematic preferences

- Test choice labels are imbalanced (quartz 22/60; moss only 4/60). Prediction counts show marked quartz preference: Qwen predicts quartz 38 times and never predicts cobalt; Harrier predicts quartz 44 times, never cobalt, and predicts ember once. This fails to cover an important gold class. Candidate positions of gold labels are 0:14, 1:10, 2:16, 3:10, 4:10; predicted choice position counts are Qwen `[22,13,16,4,5]`, Harrier `[19,12,19,5,5]`. Accuracy by gold position: Qwen `[7/14,5/10,6/16,1/10,4/10]`, Harrier `[6/14,5/10,5/16,3/10,3/10]`. There is no monotonic “always pick first” signature, but both avoid later slots and the 5th option (only 5 predictions each), consistent with a position/cardinality interaction worth checking.
- Noul targets are 39 false/21 true. Confusion counts `(gold,predicted)` are Qwen: `(F,F)=25,(F,T)=14,(T,F)=16,(T,T)=5`; Harrier: `(F,F)=17,(F,T)=22,(T,F)=13,(T,T)=8`. Qwen leans false (41 false predictions); Harrier predicts true 30 times, a substantial polarity/calibration shift against a 65%-false target base rate. The archive maps false to `unsupported` and true to `supported`; metadata records polarity checks passed. These results are not explained by an obvious wire-level polarity inversion, although the operating threshold/model preference is wrong in opposite directions.
- Test score gold ordinal positions are 1:22 and 2:38 (no position 0 targets); expected values are independently heterogeneous: 0:8, 1:7, 2:13, 3:7, 4:12, 8:13. Train score targets have five options/levels: ordinal counts `[32,24,40,17,7]` (train cases also have four/five-level score questions). In test, Qwen predicts position 2 for 44 and position 0 for 16, never position 1; Harrier predicts 0:38, 1:10, 2:12. Both miss the 1st ordinal level frequently; Harrier has a pronounced bottom-level collapse. Score exact accuracy and expected-value MAE: Qwen 29/60, 1.694; Harrier 12/60, 2.653. For Qwen the archived label-argmax accuracy equals exact accuracy; expected-value scoring is derived from probabilities per metadata.

### Binary polarity, score scale, candidate semantics

The score case labels are `band_<case>_<ordinal>` while their numeric score values vary per case, and the test uses only three candidate levels versus train’s 3/4/5. The test golds occupy ordinals 1 and 2 only, despite value range 0–8; train score values/criteria and test values are not on a single common numeric grid. Therefore an expected-value difference can combine ordinal-class error and a scale/rubric-shift effect. The observed MAEs support a Harrier scale/ranking problem, but do not by themselves prove the model’s scalar outputs are mis-scaled: there is no per-case common value scale.

Archived confidence is weakly discriminative for correctness, especially for Harrier score: mean confidence/correct vs incorrect confidence is Qwen choice `.552 / .574 / .538`, noul `.671 / .670 / .671`, score `.737 / .763 / .714`; Harrier choice `.792 / .821 / .775`, noul `.706 / .709 / .703`, score `.719 / .623 / .743`. At confidence ≥.8, errors were Qwen 14/35 overall (choice 1/3, noul 3/8, score 10/24) and Harrier 50/71 (choice 22/37, noul 8/13, score 20/21). Harrier score is particularly overconfident on wrong predictions; Qwen confidence provides almost no discrimination for noul. This is descriptive calibration evidence, not a full calibration estimate (n=60/type; no proper scoring/Brier/ECE analysis).

### Train/test and domain/family separation

Metadata: 420 train cases/660 questions from 15 families versus 60 test cases/180 questions from 10 families, with zero family overlap and zero domain overlap allowed/observed. Train fixtures cover 12 domains at 10 cases/domain; test covers 60 distinct domains at one case/domain. Every test domain and family is therefore test-only, and the two factors change together. The test is an unseen-family **and** unseen-domain evaluation, not a controlled domain-generalization experiment. Per-domain correctness has only one observation per model/type/domain, so it is not interpretable as stable domain performance. Family-level test accuracy ranges substantially: Qwen all-type family totals range 1–5/6; Harrier 0–4/6. Test choice gold positions are reasonably spread, but count per position is small (10–16).

The biggest evident task-distribution shift is candidate cardinality: train has 2/3/4/5-option choice counts 0/30/30/30 (for train’s `response_plan`; 30 choices each at 2,3,4,5), versus test 3/4/5 counts 20/20/20; train score has 3/4/5 levels 40 each versus test 3 levels only. Noul remains binary. Train target rates: noul false/true 72/48 (60% false) versus test 39/21 (65% false); train score ordinal counts across up to five categories differ from test’s 0/22/38 across three. The targeted shared scorer is therefore being evaluated under changed class/cardinality and rubric/value construction, not just new domain names.

### Leakage/artifact checks

No direct train/test family or domain overlap is present in metadata; it reports train-only grouped CV and records candidate-permutation invariance and noul polarity checks as passing. Archived score candidate strings visibly encode case number and ordinal (`band_01_0` etc.). That is a possible shortcut/artifact because the ordinal suffix is correlated with ordered level; it is not evidence of target leakage by itself, and the test labels have a different per-case prefix, so memorizing a family prefix should not transfer. Likewise generated choice keys encode route/index-like strings in training, but the test uses unrelated canonical keys; no direct target copied into the archived prediction outputs was found. Input feature rows include query/candidate text and embeddings, so any learned association with generated labels or repeated rubric phrasing remains a plausible representation shortcut. The run records test references read only after raw predictions were written; no post-prediction leakage signal is evident from metadata.

## Hypotheses (not established causes)

1. **Score regime mismatch is a leading explanation**, particularly for Harrier: train sees wider cardinality and a different ordinal/value mapping; its test predictions collapse to ordinal 0 while test labels are only 1/2. Different MAEs across models suggest model-dependent feature/scale interaction. The existing numbers cannot isolate rubric text, label-token artifacts, class prior, or numeric scale.
2. **Noul operating-point/semantic bias** is likely more important than literal polarity reversal: opposite false/true prediction skews for Qwen versus Harrier, with both weak relative to the majority baseline. This may reflect the learned scorer’s threshold under shifted class prior or prompt/template encoding.
3. **Choice key and candidate position bias** may contribute to quartz overprediction / missing cobalt and weak last-slot use. Because test keys are shared semantic labels and test domains/families are wholly novel, key/position effects are confounded with underlying decision semantics and domain/family.
4. **Family/domain confounding prevents causal attribution.** Each test family appears six times (three types across cases, as family assignment/rows indicate), but each domain has one case; both family and domain are unseen. Need controlled matched-domain/unseen-family data, not more inspection of the same heldout set, to separate these explanations.
5. **Confidence is not a safe abstention signal for Harrier** in this set, with 20/21 wrong Harrier score calls at confidence ≥.8. This could be reduced by train-only calibration, but the six-per-family clustering and small test make this unsuitable for fitting or choosing calibration against test.

## Train-side-only experiments (do not tune against test)

- Within training only, report per-type class priors, balanced accuracy/macro-F1, and grouped CV confusion matrices by candidate count and position; select any change through the existing grouped family/case CV protocol, keeping all folds component-separated.
- Add/rebalance training score examples to cover each ordinal level and 3/4/5 candidate cardinalities, and normalize numeric score targets/rubric semantics per question (e.g. ordinal classification separate from within-question value regression). Compare with the baseline in grouped CV; do not use test classes to pick weights.
- Train with candidate-order permutation augmentation and position-balanced sampling / explicit invariant candidate scoring. Validate each through the train-only permutation tests and grouped CV; current metadata already verifies permutation invariance at inference, which does not rule out learned semantic/key bias.
- Use train-only per-type class weighting or prior/threshold selection for noul; consider temperature/Platt calibration per type fit only from out-of-fold training predictions. Freeze decisions before any fresh test evaluation.
- Audit feature construction on training examples for generated identifiers, label suffixes, rubric ordinal words, and repeated template cues; run ablations (drop raw candidate key/ID tokens; retain meaning-bearing criterion text) using grouped CV. This tests shortcut dependence without changing fixtures or reviewing test labels.

## Evaluation design recommendation

Yes: add **new** test data to isolate causes. To estimate domain effect, use same domains with newly held-out families/cases (domain matched, family disjoint); to estimate family/structure effect, include unseen domains while matching task, candidate cardinality, label prevalence, score value mapping, and question templates to training. Ideally cross these as a small factorial (seen-domain/unseen-family and unseen-domain/unseen-family) with multiple cases per domain/family, balanced target classes, randomized candidate positions, and score rubrics with known shared or explicitly normalized scales. Keep current `test-60` untouched as a single combined-shift holdout; do not tune on it.
