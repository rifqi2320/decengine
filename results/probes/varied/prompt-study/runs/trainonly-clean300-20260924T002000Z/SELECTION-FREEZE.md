# Prompt-study selection freeze

**Status: frozen for independent holdout coordination.** This selection was made only
from the clean300 single-question training fixture and multi train120. No heldout
fixture/reference/prediction was opened or scored. In particular, the fresh holdout
remains untested; this run does not authorize a test until the orchestrator coordinates.

## Frozen configurations

| Embedder | Frozen prompt serialization | Internal scorer | Type-balanced OOF accuracy | Type-macro candidate-local F1 | Robust challenger result |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3 Embedding 0.6B | `v2-baseline` | typed, per-type, C=1.0 | 0.5276 | 0.2071 | retain baseline; no challenger passed |
| Harrier OSS v1 0.6B | `v2-baseline` | typed, per-type, C=0.1 | 0.5951 | 0.2279 | retain baseline; no challenger passed |

Both require strict aggregate gains in **both** type-balanced accuracy and
candidate-local macro-F1, plus both metrics strictly better in at least 4 of 5 grouped
folds. Neither embedder had an eligible challenger. Strictly-better-fold counts for
`readable-state`, `rule-in-candidate`, `joint-context-candidate`, respectively, were
Qwen **3/5, 1/5, 0/5** and Harrier **3/5, 3/5, 2/5**. Thus even an aggregate-only
lead could not overrule the preregistered robustness gate.

## Selected OOF results

OOF support counts sum to 660 questions: choice 240, noul 220, score 200. Accuracy and
candidate-local F1 are reported separately by wire type; the latter gives an exact hit
`1/K` and a miss `0` per question, so arbitrary user option labels are never treated as
one pooled class vocabulary. Score MAE is expected-value error normalized by each
question's numeric rubric span.

| Embedder | Type | Correct / n | Accuracy | Candidate-local macro-F1 | Normalized score MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | choice | 96 / 240 | 0.4000 | 0.1317 | — |
| Qwen | noul | 159 / 220 | 0.7227 | 0.3614 | — |
| Qwen | score | 92 / 200 | 0.4600 | 0.1284 | 0.2953 |
| Harrier | choice | 119 / 240 | 0.4958 | 0.1607 | — |
| Harrier | noul | 166 / 220 | 0.7545 | 0.3773 | — |
| Harrier | score | 107 / 200 | 0.5350 | 0.1458 | 0.2549 |

## Train-only CV and checks

Each variant/model ran five deterministic folds (seed 271828), grouping connected
components over case IDs and question-family IDs; the fold count is 5 and component
count is 9. All 660 OOF rows per model were retained. Audit rechecked that no case or
family split across folds, OOF metrics match the saved selection metrics, feature and
manifest hashes match, all checkpoint sizes/SHA-256s match the installed models, and
the final typed scorer weights/scales are finite and converged. Final weights were fit
on all 420 training cases after selection only.

Final-fit artifact SHA-256:

* `frozen/qwen/models.json`: `ce4873347a6fd0de6df98f9becc2f045d27af0308c189186a8dd73d4a89b27af`
* `frozen/harrier/models.json`: `66fa11995f0c032d2c3169126071636bca6d2d3aca68a5df9a44ee7ef9ee1421`
* `frozen-config.json`: `305a124523f12e28686bd037d8be96719717ee36cbfc86bfc7b3f8c41b7fd325`

Permutation audit passed on 660 final-fit questions per model. The noul polarity/order
audit passed on all 220 binary propositions per model: reversing candidates only
reversed positions, not the `unsupported`/`supported` score mapping. CV prediction code
also checks row-order invariance during each OOF prediction pass.

## Prompt token and arithmetic-cost estimates

Each row below is the total over 300 clean-single plus 360 multi-train questions for
**one embedder and one variant**. Query plus all candidate tokens are tokenizer counts;
there are 2,788 actual query/candidate embedding calls. FLOPs are a transparent dense
estimate `2 × 600,000,000 parameters × token_count`; this is not measured MLX FLOPs and
omits attention/cache and other implementation details.

| Variant | Tokens | Embedding calls | Estimated dense FLOPs |
| --- | ---: | ---: | ---: |
| v2 baseline | 162,574 | 2,788 | 1.951e14 |
| readable state | 164,414 | 2,788 | 1.973e14 |
| question + rule in candidate | 306,038 | 2,788 | 3.672e14 |
| joint context + candidate | 486,069 | 2,788 | 5.833e14 |

Candidate/rule context is materially more expensive in tokens; the joint-context
variant is about 2.99× baseline token cost for no robust OOF gain. Feature JSONL and
sidecar manifests retain exact per-query/candidate text, token/byte lengths, feature
hashes, embedding-call counts, model/profile/tokenizer/checkpoint versions and hashes.

## Provenance

* Clean300 single train SHA-256: `fe8ec1ce7661f2ba483e79b96d6cacd97e3ad04da42d86524bc9e14b374c1e8a`
* Multi train120 SHA-256: `f26aaca6278274f6b3094737f8288dd3ad79ed1c70a49f93c1c9caaaf8a25c50`
* Protocol SHA-256: `954aa5543cdb29b7be5b9f5e8ede16d58d32788ad19a8c258d1256b5fe31598d`
* Variant exporter source SHA-256: `e90589b689551797edec7630084c91090658da6027e1a43353d599ae8231084b`
* Exporter executable SHA-256: `32743c7642545c213fe7dd27737bef3db48531db502b482e8aea46ded70a2c4c`
* CV runner SHA-256: `742fd95473f26781a50b7df972b414f8574727d310294c3d8307e848cfb11528`
* Study runner SHA-256: `8efbdbac0131df3322e57fdc2ba4119cc21c875f71149c7c3e083feac5648000`

Audit details and all input/output/checkpoint hashes are in `audit.json`, CV metrics and
OOF files are under `cv/{qwen,harrier}/<variant>/`, and final training-only scorers are
in `frozen/{qwen,harrier}/models.json`. The separately marked
`trainonly-20260924T000000Z` artifact directory is invalid/aborted because the wrong
300-record, three-question fixture was initially selected; it was stopped before CV or
selection and was not used. This clean300 run uses the verified one-question-per-row
training fixture; do not consume the invalid directory.

Finally, an embedding plus a linear typed scorer does not gain language-model
reasoning merely by repeating a rule in prompt text. This experiment tests only whether
generic text serialization changes a frozen embedding representation enough to help
the train-only linear probe.
