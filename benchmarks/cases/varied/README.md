# Varied-question domain-generalization fixture (authoring seed)

This directory is a **small authoring seed**, not an evaluation dataset or evidence
that current decengine heads generalize. `seed.jsonl` contains independently worded
fictional examples in several domains and question families. `schema.json` describes
the record contract; `validate.py` checks structural and leakage invariants without
training, inference, external calls, or inspecting model outputs.

## Record contract

Each line has a unique `id`, domain, opaque random-looking `question_family_id`,
`request` (`state` plus exactly one protocol question), and `reference` (typed target
and rationale). Questions are actual wire shapes: `choice` has 2–16 keyed options,
`noul` has an explicit operational yes/no prompt, and `score` has 2–16 named levels
with strictly increasing numeric values and criteria. Reference targets are:

* choice: `selected` candidate key;
* noul: boolean `value`;
* score: both `label` and its corresponding numeric `value`.

Rationales and labels are adjudication metadata only and MUST NOT be copied into
`state`, prompts, candidate descriptions, IDs, or other inference input. Candidate
keys/labels are legitimate model inputs, so use arbitrary keys and avoid a universal
label convention mapping to the preferred answer. IDs for families are opaque (not
semantic names), assigned randomly, and kept in a separately access-controlled
family map during evaluation. Families should denote the underlying task/construct,
not a surface template. One case per family here is illustrative only; useful
within-family training/evaluation needs multiple independent cases.

Run `python3 benchmarks/cases/varied/validate.py` (or pass a fixture path). The
validator rejects repeated family IDs only if duplicated case IDs? No: multiple
examples per family are expected; IDs must be unique. It detects target-token leakage
and request/reference schema mistakes, not subtler semantic leakage; human review is
still required.

## Scoring across variable questions (proposal, not implemented)

The existing probes are fixed task-specific logistic heads over exported state/task
embeddings (`results/probes/train_probe.py`, `predict_probe.py`); they do not learn
question-conditioned decisions and must not be presented as generalized to unseen
questions. A suitable next scorer exports frozen embeddings for state/context `s`,
prompt/rubric `q`, and each candidate description `c_i` and creates shared pair
features such as `[s, q, c_i, s⊙q, s⊙c_i, q⊙c_i, |q-c_i|]`, optionally adding
candidate-count/position-independent scalar features. Fit one shared regularized
linear/MLP scorer `g(features)` across choice candidate pairs; normalize logits with
softmax **within each question**, selecting the highest candidate. Candidate-set
permutations must not change scores. For binary questions, use the same shared pair
scorer for the two semantic outcomes (explicitly supplied as candidates) and sigmoid
or two-way softmax; specify which semantic outcome is positive. For ordinal rubrics,
score each level criterion and use a within-question softmax, with predicted expected
value `Σ p_i v_i`; report both level classification and normalized ordinal error.
Fit feature scaling on training folds only. Calibrate probabilities (e.g. temperature
chosen on development data) without evaluation labels; also report uncalibrated
metrics. Compare against fixed-head probes as a baseline, not as a solution to unseen
question families.

## Leakage-resistant evaluation

1. Before authoring labels, assign opaque family IDs; freeze a family-to-construct map
   separately. Keep scenario/case siblings, paraphrases, and templated variants in
   one group. Audit duplicate and near-duplicate text across folds.
2. Split by **question family**, not row or domain alone: outer leave-one-family-out
   (LOQFO) folds train on all other families, select hyperparameters/calibration only
   with grouped inner folds, then predict the held-out family once. Never expose held
   out targets/rationales to feature export, fitting, threshold selection, or prompt
   tuning. Save prediction artifacts before a separate metrics pass reads labels.
3. Also report leave-one-domain-out (LODO) as a distinct axis; cross family × domain
   where sample size permits. State whether a test family has any same-family training
   support (LOQFO intentionally has none). Report per-family and per-domain results,
   macro-average families, plus pooled metrics; for choice include log loss/Brier and
   candidate-count slices, noul include balanced accuracy/F1/AUROC, and score include
   ordinal MAE/quadratic weighted kappa. Give bootstrap intervals clustered by family
   and domain. Compare to majority, lexical/rule baselines, and fixed task heads.
4. Keep a final family-and-domain locked test set untouched until the protocol and
   model are frozen. Publish fixture hashes, split assignments, model/embedding
   versions, predictions, and annotation adjudication records.

## Size and next steps

Six examples cannot support a meaningful generalization claim. Target at least 30–50
independent families and 20–30 adjudicated scenarios per family (roughly 600–1,500
labels), across at least 8–10 domains; reserve at least 8–10 wholly held-out families
and multiple held-out domains. This is a pragmatic minimum, not a power calculation:
estimate family-level variance and perform a blinded power analysis before fixing the
final sample size. Use two independent annotators plus adjudication, document rubric
agreement, then build the frozen pair-feature exporter, grouped split manifest,
question-conditioned trainer/predictor and calibration/evaluation suite. Do not extend
or alter the current owner/urgent/impact LOIO artifacts as part of this track.
