# Frozen-embedding probes

`train_probe.py` is the reproducible, downstream-only evaluator for the 300/90
probe split. It does not run an encoder, create substitute features, tune against
test data, or modify input datasets. The following is only for the deprecated combined
export format; prefer the separate train/test commands in the real-data section below:

```sh
python3 -m venv .venv-probes
. .venv-probes/bin/activate
python -m pip install -r results/probes/requirements.txt
python results/probes/train_probe.py \
  --train benchmarks/cases/probe-train-300.jsonl \
  --test benchmarks/cases/probe-test-90.jsonl \
  --embeddings path/to/qwen-embeddings.jsonl --model qwen \
  --expected-counts --out results/probes/runs/qwen
```

Repeat with the Harrier export and a distinct output directory. Use actual exporter
outputs only; this repository does not check in generated/dummy features. `--expected-counts`
enforces exactly 300/90 records. All input file hashes, ID hashes, versions, parameters,
coefficients, per-record predictions/probabilities, metrics, and the random seed are saved.

For split-specific exporter files, pass `--train-embeddings TRAIN.jsonl` and
`--test-embeddings TEST.jsonl` separately. They are checked against train/test IDs
respectively, and their model IDs/dimensions must agree. This ensures test embeddings
are never passed to train-only CV fitting.

## Inputs

The dataset is JSONL, one object per question/case, with unique `id` and gold labels
under `reference.owner`, `reference.urgent` (JSON boolean), and `reference.impact`.
Those references are the only label source. The embedding export is JSONL records shaped
`{"id":"...","model":"mlx-community/Qwen3-Embedding-0.6B-4bit","embeddings":{"state":[...],"owner":[...],"urgent":[...],"impact":[...]}}`.
A JSON array of records or a JSON object with an `embeddings` array is also accepted.
Each classifier consumes its matching task-specific vector (`owner`, `urgent`, or
`impact`); `--state-only` is an optional control that uses `embeddings.state` for all
three classifiers. Legacy records with a single flat `embedding` vector remain
supported and share that vector across tasks. `--model` accepts the exact exported ID,
or a case-insensitive alias whose alphanumeric tokens uniquely identify one exported
model ID (e.g. `qwen`); ambiguous/non-matching aliases fail. Records for other model
IDs are ignored. Every train/test ID must have exactly one finite 1024-dimensional
vector for each selected task. Logistic
regression receives per-feature standardized vectors; the scaler is fitted separately
inside each CV fold and on all training data for the final model, then applied unchanged
to test data. No encoder or test-fitted transform is used; scaler parameters are saved
with model coefficients. The output metadata records the resolved full model ID and
which embedding key was used for each classifier.

Records may declare `group_id` or `group` at top level or in `metadata`; those groups
are kept together in CV. In addition, records whose normalized scenario/request text
has token-set Jaccard similarity >=0.88 are linked into near-duplicate groups. Any
such group crossing train/test is a hard error. CV uses shuffled `StratifiedGroupKFold`
(up to five folds, deterministic seed); if the data have no repeated groups it uses
stratified folds. The regularization grid is L2 logistic regression `C` in
`[.001,.01,.1,1,10,100]`, selected only by mean training-fold macro F1; ties select
smaller C. The fixed solver is `lbfgs` (5000 iterations maximum). CV class support/group
count must permit stratification, otherwise execution fails rather than silently leaking
or using test data. Separate train/test embedding files are preferred; an older combined
`--embeddings` file remains accepted for compatibility. Adjacent exporter manifests are
checked for model identity and their contents/hashes are captured in metadata.

## Outputs and comparisons

Each run writes `metrics.json`, `predictions.jsonl`, `models.json`, and `metadata.json`.
Metrics include accuracy, confusion matrix, per-class precision/recall/F1/support,
macro F1 with a deterministic percentile bootstrap interval, and an always-majority
baseline. Urgency includes Brier score and probability-bin calibration. The hard-label
previous cosine comparison can be added using `--previous-predictions PATH`; its JSONL
must have exactly the test IDs and `owner`, `urgent`, `impact` label fields. Because
hard-label outputs contain no probabilities, no calibration score is invented for it.
All confidence intervals are test-set bootstrap summaries, not guarantees of
population uncertainty; the probe set is small and should not be used alone for
deployment decisions.

Install versions are pinned in `requirements.txt`. The script requires no API/key,
network access, or model files once the exports are available.

## Locked real-data run (2026-09-23)

The real exporter outputs were used in separate train/test files. Reproduce the
task-specific probes and the state-only controls with:

```sh
FEATURES=/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/probe-features
BASE=results/probes/baselines/20260923T094641685361000Z
RUN=results/probes/runs/20260923T094618Z
python results/probes/train_probe.py --train benchmarks/cases/probe-train-300.jsonl \
  --test benchmarks/cases/probe-test-90.jsonl \
  --train-embeddings "$FEATURES/Qwen-Qwen3-Embedding-0-6B-train300.jsonl" \
  --test-embeddings "$FEATURES/Qwen-Qwen3-Embedding-0-6B-test90.jsonl" \
  --model qwen --expected-counts --previous-predictions "$BASE/qwen-predictions.jsonl" \
  --out "$RUN/qwen/task-specific"
python results/probes/train_probe.py --train benchmarks/cases/probe-train-300.jsonl \
  --test benchmarks/cases/probe-test-90.jsonl \
  --train-embeddings "$FEATURES/microsoft-harrier-oss-v1-0-6b-train300.jsonl" \
  --test-embeddings "$FEATURES/microsoft-harrier-oss-v1-0-6b-test90.jsonl" \
  --model harrier --expected-counts --previous-predictions "$BASE/harrier-predictions.jsonl" \
  --out "$RUN/harrier/task-specific"
# Repeat those two commands with --state-only and output dirs qwen/state-only and
# harrier/state-only to reproduce the state-vector controls.
```

The fixture SHA-256 values are train `ae8c3cbbd58b1a38db65e72ff638728bd5e5bae7df8be5a78b2ebed9c9aba434`
and test `1ded62e864666e324a3154d86114ade18a816a2453b5e2ace1593a2c77d3f587`.
The train/test ID hashes in run metadata are respectively
`4819948066f6e61ebdf9de13a13696cf6cb2002adf1e133cd2502989150a52e1` and
`ac9df5599f59f6b24dc86c7e88173c3e6d36fe232d0546bff8ea435020e0c8cd`; the ID sets are
disjoint. Every split-specific embedding file contained exactly the corresponding
fixture IDs. Qwen export SHA-256 (train/test) is
`1458cdd430a045735a367f4c4f519403cde5a4a3b70012f5d1fcf868623968b3` /
`89cd447bf8b733cfe9bc40df8d4e81c96540cd0f4eb7c91beab96a0037153588`; Harrier is
`79d15b67904c759cae4237ef366991add97d83981f24f7cc6da2ed8d7958c33a` /
`5efe6f50440771091228dc4f4c34ab7e7347d1a48e286ff6a82daccea02fbbd9`. Adjacent exporter
manifests had matching model IDs and profile hashes (Qwen
`a008dee8e9e7e279486982b2f8128c664335f50203b8dfd58c68ac5b3f3700bc`; Harrier
`96ec8ee07b71d1e34a006f0f5ba8c4d42df77d9808d54b4d4beb8ce7b7112c4e`). Run metadata
also includes all complete manifest contents, artifact hashes, resolved model IDs,
Python/library versions, trainer and requirements hashes, and exact arguments.

All figures below are on the untouched 90-case test set. `overall` is the unweighted
mean of the three task-level metrics; it is a summary, not a pooled example-level
statistic. Parentheses show 95% percentile bootstrap CIs (2,000 resamples, fixed seed).
Majority baseline accuracy / macro F1 is listed for each task. Urgency sensitivity is
the urgent/true recall; specificity is the non-urgent/false recall.

| Features | Task | Accuracy (95% CI) | Macro F1 (95% CI) | Selected C | Majority acc / F1 | Existing cosine acc / F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen task-specific | owner | .711 (.611–.800) | .698 (.586–.781) | 10 | .311 / .068 | .422 / .341 |
| Qwen task-specific | urgent | .844 (.767–.911) | .828 (.737–.906) | 10 | .656 / .396 | .344 / .256 |
| Qwen task-specific | impact | .711 (.622–.800) | .707 (.608–.796) | .001 | .333 / .167 | .333 / .167 |
| **Qwen task-specific** | **overall mean** | **.756** | **.744** | — | — | **.367 / .255** |
| Harrier task-specific | owner | .578 (.478–.678) | .540 (.433–.625) | .1 | .311 / .068 | .411 / .389 |
| Harrier task-specific | urgent | .922 (.867–.978) | .913 (.844–.972) | .01 | .656 / .396 | .344 / .256 |
| Harrier task-specific | impact | .844 (.767–.911) | .842 (.762–.911) | .01 | .333 / .167 | .456 / .440 |
| **Harrier task-specific** | **overall mean** | **.781** | **.765** | — | — | **.404 / .362** |

Urgency rates (95% bootstrap CI): Qwen task-specific sensitivity `.774` (`.607–.919`),
specificity `.881` (`.792–.956`); Harrier task-specific sensitivity `.871`
(`.741–.971`), specificity `.949` (`.885–1.000`). State-only control overall means
were Qwen accuracy `.822` / macro F1 `.805` and Harrier `.811` / `.795`; full per-task
metrics, selected C values, CIs, and cosine comparisons are in their `state-only/metrics.json`
files.

Each train split formed 300 independent groups under the declared-group/text near-
duplicate rules, so the fixed five-fold `StratifiedGroupKFold` produced 60 held-out
training examples per fold (owner fold counts are in `metrics.json`; all classes are
represented). Hyperparameters were selected only against those train-fold macro F1
scores; no test examples, features, or labels were passed to CV or fitting. Test CIs
were computed only after models were locked. The previous local decengine cosine-decision
baseline was evaluated separately on the same 90-case fixture (run
`results/probes/baselines/20260923T094641685361000Z`); its 90 Qwen and 90 Harrier outputs
were passed only to the post-fit hard-label comparison. Baseline errors were zero; Qwen
and Harrier prediction hashes are in its `summary.json`, and their source-file hashes
are captured in each probe `metadata.json`. It predicted urgent/true for every case
(sensitivity 1.0, specificity 0.0). No API or key was used by the probe training/evaluation.

The host's OpenBLAS build emits spurious divide/overflow/invalid RuntimeWarnings from
finite `matmul` operations (confirmed even for multiplying by an all-zero coefficient
matrix, whose result was finite all-zero output). The trainer narrowly filters those
messages during sklearn fit/predict and hard-fails on non-finite coefficients,
probabilities, invalid probability sums, max-iteration exhaustion, or convergence
warnings. The four completed run logs have empty stderr.

## Inference from saved probes

`predict_probe.py` performs inference only: it loads each task's saved classes, selected
`C`, coefficients, intercept, and train-fitted scaler from `models.json`, then applies
those values without fitting or tuning. It accepts **unlabeled** JSONL records of the
same nested exporter form and requires the adjacent exporter manifest (or an explicit
`--manifest`) so model ID, profile version/hash, text version, and 1024-D feature size
must match the trained artifact. It never accesses `reference`; no labels are required.
Predictions contain each task's class label, complete probability mapping, and
`confidence` (the highest class probability; not a calibration guarantee). A
`<output>.metadata.json` sidecar hashes input, manifest, artifacts, and predictor code.

```sh
FEATURES=/path/to/new-export
RUN=results/probes/runs/20260923T094618Z
python results/probes/predict_probe.py \
  --artifacts "$RUN/qwen/task-specific" \
  --embeddings "$FEATURES/qwen-new-cases.jsonl" \
  --out results/probes/new-cases/qwen-predictions.jsonl
# Use "$RUN/qwen/state-only" for the state-vector control, or the analogous
# "$RUN/harrier/task-specific" / "$RUN/harrier/state-only" directories.
```

Reproducibility check (also inference-only; comparison uses IDs, labels, and
probabilities from saved output files, not dataset references):

```sh
python results/probes/verify_predict_probe.py \
  --expected "$RUN/qwen/task-specific/predictions.jsonl" \
  --actual "$RUN/qwen/task-specific/inference-test-predictions.jsonl" \
  --report "$RUN/qwen/task-specific/verification.json"
```

All four saved artifact variants were replayed against their respective 90-case
exporter JSONLs. Every label matched exactly (270 task predictions per variant); all
1,080 probability values per variant were within `1e-12` of the trained-run values,
with maximum absolute difference `1.1102230246251565e-16` (not bitwise-identical due to
the standalone softmax/sigmoid arithmetic). Reports are saved at
`results/probes/runs/20260923T094618Z/{qwen,harrier}/{task-specific,state-only}/verification.json`.

## CPU-side probe benchmark

`benchmark_probe.py` runs the existing MLX feature exporter once per model (models are
processed sequentially), then times all three saved task-specific scaler/logistic
classifiers separately for each of the 300 test cases. It checks every predicted label
and probability against the archived 300-case probe output before writing
`benchmark.json` and per-model prediction JSONL. Reproduce with a new output directory:

```sh
python results/probes/benchmark_probe.py --model all \
  --out results/probes/benchmarks/$(date -u +%Y%m%dT%H%M%SZ)
```

The current `decengine export-features` CLI has no per-case encoder timings and always
embeds state in addition to the three task query vectors. Accordingly, reports identify
the encoder as a four-vector run, include process wall time (including model load,
first-call warmup and serialization), and do **not** claim encoder/end-to-end p50 or p95.
Only classifier inference has per-case p50/p95/mean. Export wall amortized over 300 is
not a per-case latency distribution and is not directly comparable to the separate
3-call Laya timing or the original full cosine-decision timing. No probe is retrained.

## Varied-question pairwise scorer

`varied/train_pairwise.py` is a separate, question-conditioned experiment; it does not
modify the fixed owner/urgent/impact probe or its exporter. For the single-question
LOQFO study it uses a quality-screened 300-case training subset and the untouched
200-case test split (one question per case). Qwen and Harrier are fit independently:

```sh
python results/probes/varied/train_pairwise.py \
  --train path/to/train-clean-300.jsonl --test benchmarks/cases/varied/test-200.jsonl \
  --train-embeddings path/to/train-clean-300-qwen.jsonl \
  --test-embeddings path/to/varied-test-qwen.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B --expected-counts --out results/probes/varied/runs/qwen
# Repeat with exact Harrier model ID / corresponding embeddings and a separate output.
```

Dataset rows follow the varied-question case contract: `id`, `domain`,
`question_family_id`, `request.questions` (exactly one choice/noul/score question), and
`reference.target`. Choice candidates are variable (the clean training fixture includes
one six-option family); Noul is represented by explicit
proposition outcomes `unsupported` and `supported`; score uses 3–5 named rubric levels.
The current MLX exporter schema is `{"id":"...","model":"...","query_embedding":[...],
"candidates":[{"candidate_id":"...","embedding":[...],...}]}`. Candidate IDs must
match option keys / score labels; Noul target booleans map to the semantic IDs. The
trainer also accepts the early prototype query/map shape for local smoke tests. Query
and candidate vectors must be same-sized finite vectors. Adjacent manifests are checked
for model/profile/text identity; the trainer does not create embeddings or call an encoder.

One L2-regularized linear scorer is fit over `[query, candidate, query*candidate,
abs(query-candidate)]` features using a bounded-memory L-BFGS solver (finite checks,
300-iteration and 24-step line-search bounds) to avoid the earlier gradient-descent
stall. Each question's candidate logits are normalized by
within-question softmax; the selected option is argmax probability. Score questions
additionally report expected numeric value `sum(p_i * value_i)`, separately from level
classification. Scaling is fitted from training candidates only. `C` is selected from
the bounded `C` grid `[.01,.1,1,10]` by deterministic five-fold grouped CV restricted to
training records; question families and any cross-family near-duplicate context links
stay intact within folds. Selection is mean held-out-fold exact-match accuracy (candidate
labels differ across questions, so pooled candidate-index F1 would be invalid). The script rejects family
overlap and exact/near-duplicate scenario text (state plus prompt, token Jaccard >= .88)
between train and sealed test. Domain overlap is expected under leave-question-family-out
(LOQFO), is explicitly listed in metadata, and is not rejected; domain-level results are
descriptive slices. Leave-one-domain-out requires a separate split/fold protocol and is
not produced here. Reported macro-F1 is mean
within-question macro-F1 over local candidate labels (a one-item exact match contributes
`1/K`, otherwise zero); exact-match accuracy is primary. No test labels participate in
CV or fitting. Test prediction JSONL is written before the metrics function reads any
test target. Output includes per-case probabilities/entropy inputs and expected values,
per-type/family/domain/candidate-count metrics, test family-cluster bootstrap intervals,
CV results, model coefficients, and input/code/ID hashes. Bootstrap intervals are
descriptive with only ten held-out families; they do not imply strong precision.
Choice outputs expose the selected candidate key; Noul outputs expose a boolean
`label`/`value`; score outputs expose `selected_level` and the separate probability-weighted
`expected_value`.

### Locked clean LOQFO run (2026-09-23)

The merged training manifest flags exact scenario-context reuse across families
`qf-00000001`–`qf-00000005` (100 records, two repeated contexts). Those entire families
were removed before model selection, yielding 300 rows / 15 train families
(`qf-00000006`–`qf-00000014`). The untouched test remains 200 rows / 10 test families
(`qf-00000015`–`qf-0000001e`). No exact or >=.88 state+prompt Jaccard duplicates were
found spanning different retained training families or across train/test; CV also unions any
near-duplicate family groups. The original 400-case split was not scored or used for a
generalization claim. No previous dirty 400-row run artifacts existed to overwrite.

Train-only optimizer benchmark on 60 cases / 3 families completed the 4-C, 3-fold CV
smoke in 0.386 s; candidate-order reversal preserved keyed probabilities to <1e-12.
Full models use the same generic shared scorer and no test labels during fit/CV. Results:

| Embedding | Selected C | Train CV mean accuracy | Test accuracy (200) | Choice (80) | Noul (60) | Score level (60) | Score expected-value MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | 1.0 | .500 | .495 | .525 | .517 | .433 | .842 |
| Harrier | .01 | .527 | .465 | .450 | .567 | .383 | .840 |

Family accuracy is saved for all ten test families in each `metrics.json` and
`summary.json`; it ranges Qwen `.25–.60` and Harrier `.20–.80`. Domain-level metrics for
all test domain strings are also saved. Exact string overlap in these merged fixtures is
only two domain labels (`online retail`, `telecommunications`), despite the broader
shared-domain design note; overlap is allowed under LOQFO and does not constitute LODO.
Those overlapping-domain test accuracies are respectively Qwen `0.0` / `.583` and Harrier
`1.0` / `.583` (n=2 / 12); most domain slices have only two records and are descriptive.
Family-cluster 95% bootstrap intervals for overall accuracy are Qwen `.43–.55` and
Harrier `.36–.58` (ten clusters; descriptive only). The Qwen mean log loss is 2.36 versus
Harrier 1.07; Qwen is particularly overconfident (mean max probability .87), so accuracy
alone does not characterize probability quality.

Artifacts are under `results/probes/varied/runs/loqfo-clean300-20260923/`: `qwen-final/`
and `harrier-final/` each contain prediction JSONL (200 records), metrics, coefficients,
and metadata with CV fold scores / input-output hashes. `preparation/filtered-id-manifest.json`
records the exclusion reason and filtered fixture/embedding hashes (manifest SHA-256
`45b723f9c3b6315bc885bba8c86a51489e61b1545f42082d315eaccdcc5a0d17`; clean fixture
SHA-256 `fe8ec1ce7661f2ba483e79b96d6cacd97e3ad04da42d86524bc9e14b374c1e8a`); feature files are
preserved in `/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/varied-loqfo-clean300/`.
`preparation/train-only-benchmark.json`, `training-attempts.json`, and `summary.json`
preserve runtime/permutation validation, failed attempts, and compact results. One
six-option choice family remains in the filtered fixture, so the scorer accepts 2–6
options; this is documented as a fixture deviation from the initial 2–5 target.

These results support only this small single-question LOQFO fixture. They do not establish
that existing fixed task heads generalize to varied or unseen questions, nor do they
estimate leave-one-domain-out performance. Test labels are read only after predictions
are persisted for post-fit evaluation. No JEV API or key was used.
