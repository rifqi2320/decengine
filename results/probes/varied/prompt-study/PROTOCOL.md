# Predeclared generic text-serialization study

Study scope: MLX embeddings plus the existing generic typed pairwise scorer, fitted
independently for Qwen3-Embedding-0.6B and Harrier OSS v1 0.6B. This is not an
instruction-following or reasoning evaluation: prompt text only changes the vector
representation that a linear scorer receives.

## Variants frozen before CV

* `v2-baseline`: existing v2 compiler text exactly, including compact canonical JSON
  state, generic per-wire query rule, profile templates, and proposition-conditioned
  binary candidates.
* `readable-state`: identical prompt/template/candidates, but state object is emitted
  recursively as readable `key: value` / `- item` lines (sorted object keys; scalar
  JSON escaping retained). Structured paths are explicit; no metadata is added.
* `rule-in-candidate`: same query and compact state as baseline, but each candidate also
  contains the generic wire-type selection rule and exact question prompt. The rule is
  generic (`choice`: evaluate supplied criteria; `noul`: test proposition against
  evidence; `score`: apply supplied rubric), not a task/family/domain hint.
* `joint-context-candidate`: baseline query unchanged; candidate text includes the
  exact prompt, compact state, generic rule, and candidate criterion in one string.
  This intentionally repeats query context per candidate and may cost more tokens.

Candidate labels/options remain request data and arbitrary IDs are not interpreted as
targets. Noul polarity always maps false/true to unsupported/supported using the full
question proposition, not task-specific vocabulary. Score uses each wire's ordered
levels and numeric values unchanged. No reference, gold target, rationale, question
family, domain, or case metadata may be serialized into model input. Every variant
records full exact text, tokenizer token counts, bytes, input/checkpoint/profile hashes,
and actual embedding call count. Any overlength text fails closed; no truncation.

## Train-only fitting and selection

Inputs are exactly the isolated `data/train-clean-300.jsonl` (single-question clean300)
and `benchmarks/cases/varied/multi/train-120.jsonl` (multi train120), each with its
matching variant/model embeddings. No test fixture, heldout labels, prior heldout
predictions, or archived evaluation is accepted by the runner. Five deterministic
folds use connected components joining case IDs and `question_family_id`; all three
questions in a multi case and every shared family stay together. CV OOF is computed
once per variant/model using the established typed linear scorer and bounded optimizer.
Selection score is equally weighted wire-type exact-match accuracy plus candidate-local
macro-F1 (for each question, exact hit contributes `1/K`, miss zero; arbitrary labels
are not pooled into a bogus global class F1). Challenger must strictly beat the best
baseline on both aggregate measures and on both measures in >=4/5 folds; otherwise
select baseline. Final scorer is refit on all 420 training cases only after selection.

Candidate permutation is checked by reversing candidate rows and verifying scores and
probabilities reverse exactly; no label-index feature is allowed. Noul polarity is
checked by swapping candidate order while preserving semantic labels. Score expected
value/error uses wire-provided numeric values, never fixed rubric vocabulary. Each
variant's token burden is reported as query+candidate total tokens and expected
embedding FLOPs as `2 * parameter_count * total_tokens` (estimate only; ignores
attention/MLP details and KV/cache effects). Feature call counts are explicit.

All selection artifacts (fold map, OOF rows, metrics, model/profile/checkpoint/text
hashes, feature hashes, call/token/FLOP counts, and final fitted model) are scoped to
this directory. A fresh independent holdout author owns any new heldout evaluation;
this code will not read heldout data. Freeze selected configuration before any
orchestrator-coordinated heldout run.
