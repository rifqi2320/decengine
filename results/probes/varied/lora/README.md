# Frozen LoRA fresh-60 evaluation

## Freeze and attestations

The orchestrator selection manifest was verified against its supplied SHA-256:
`results/probes/varied/lora/SELECTION-FREEZE.json`,
`5f7dd19638df1f7bdcb56ecd44a9371ae5a0ac32a6314538146e324ac8a2e48f`.
It declares `TRAIN_DEV_SELECTION_FROZEN`, both runs `train_dev_complete`, and the
holdout `SEALED_NOT_ACCESSED` (no hash, labels, or predictions read during training).
The generated LoRA freeze attestations use the actual manifest artifact hashes.
Independent full-660 and newly fit same-460 baselines, plus both LoRA models, were run
sequentially. References were read for scoring only after each prediction bundle was
durably written and hashed.
Per-run evaluator attestations and compact profiles were then generated from its
actual artifact entries at `eval/fresh60/freeze/`; they are tied back to that exact
manifest hash. Qwen is epoch 1 / 1380 steps / dev loss `0.9594359870254994`, profile
v1 hash `a008dee8e9e7e279486982b2f8128c664335f50203b8dfd58c68ac5b3f3700bc`. Harrier
is epoch 1 / 1380 steps / dev loss `0.9483493425697088`, profile v2 hash
`96ec8ee07b71d1e34a006f0f5ba8c4d42df77d9808d54b4d4beb8ce7b7112c4e`.

The compact profile files are reconstructed with `ModelProfile` serde defaults/order
from the bundled profile manifests. Their raw hashes match the per-model training
feature manifests and the base-manifest hashes recorded in the selection freeze.
“v2” refers to `decengine-varied-pair-text-v2-generic-types`; the Qwen model profile
itself is version 1. The evaluator reproduces the existing training exporter text,
not a different runtime compiler variant.

Create/verify attestations without MLX or opening the fixture data (only its manifest
is checked):

```sh
python3 results/probes/varied/lora/evaluate_frozen.py \
  --prepare-freeze results/probes/varied/lora/SELECTION-FREEZE.json \
  --freeze-sha256 5f7dd19638df1f7bdcb56ecd44a9371ae5a0ac32a6314538146e324ac8a2e48f \
  --prepared-dir results/probes/varied/lora/eval/fresh60/freeze
```

The generated per-model attestations copy run/model, grouped-dev selection, profile,
and frozen artifact hashes from the verified manifest; profile/compiler hashes are
recomputed from their checked sources. The selected epoch remains 1—no test-driven
selection or tuning is allowed.

## Token policies and longest-input smoke

The initial CPU preflights applied the frozen training guard of 288 and correctly
rejected both models before loading them: all 180 queries exceeded 288, max 375. These
original reports remain at `eval/fresh60/preflight/{qwen,harrier}.json`. After explicit
authorization, a separate hash-bound policy `eval/fresh60/freeze/inference-policy-512.json`
(SHA-256 `d1fb0c2e142c066e135705cdd7d77a313d65e5eb730d2b4ed6459f7ee044a32c`) preserved
training max 288 and frozen weights/profile, while allowing non-truncating inference up
to 512 tokens (the installed backbone context limit is larger). Every run pins this
policy separately; >512 still fails closed.

Recreate the policy only after regenerating attestations from the verified selection
manifest; the command hash-binds the policy into each per-model attestation:

```sh
python3 results/probes/varied/lora/evaluate_frozen.py --prepare-inference-policy \
  --freeze-manifest results/probes/varied/lora/SELECTION-FREEZE.json \
  --policy-path results/probes/varied/lora/eval/fresh60/freeze/inference-policy-512.json \
  --prepared-dir results/probes/varied/lora/eval/fresh60/freeze
```

| Model | Questions | Text strings | Max tokens | Over training 288 | Over inference 512 |
|---|---:|---:|---:|---:|---:|
| Qwen LoRA | 180 | 750 | 375 | 180 | 0 |
| Harrier LoRA | 180 | 750 | 375 | 180 | 0 |

The first over-288 query was `mq-lora-fresh-001/response_plan` at 335 tokens. The later
512-token preflights are `eval/fresh60/preflight/{qwen,harrier}-inference512.json`;
all 750 strings are within that explicit inference-only limit. Inputs are not shortened
or truncated and no training artifact was changed. This is unseen-length extrapolation
beyond the training guard and should be disclosed with any interpretation.

The evaluator checks model/run, config/best/summary hashes, profile/compiler and
policy hashes, and frozen training status before loading. Its safetensors header
validator requires the exact 12 FP32 trainable tensors; the loader allows absent
frozen backbone tensors only. Before full inference, one longest case/question was
run as a request-only Qwen smoke: `mq-lora-fresh-049/record_auditability`, max 375
tokens. All four 1024-dimensional embeddings were finite, the score-head logits and
probabilities were finite/normalized, and no reference or score was read. Smoke raw
output and metadata are under `predictions/qwen-lora-longest-smoke/` (process peak
1,908,295,480 B; swap stayed at 1,087,635,456 B).

Both full models were then run sequentially. Process-footprint guard is 6 GiB early
stop / 8 GiB absolute; system swap may not rise over each process's pre-load baseline.
Raw per-question predictions were durably flushed and hashed before references were
opened for scoring. Per-question and three-question per-case p50/p95 scopes exclude
model load, tokenizer/compiler work, warmup and reference scoring.

Safe checks (no fixture/model/MLX access):

```sh
python3 -m py_compile results/probes/varied/lora/evaluate_frozen.py results/probes/varied/lora/compare_frozen.py
python3 -m py_compile results/probes/varied/lora/fit_head_only_460.py
python3 results/probes/varied/lora/evaluate_frozen.py --self-test
```

## Frozen full-660 baseline inference

The independent previously frozen `varied-metal` Qwen C=1.0 and Harrier C=0.1
baselines were run sequentially on the same sealed
request fixture after freeze verification. These historical heads are fitted/selected
on the full 660 questions, **not** the LoRA 460-question training subset; they are
clearly labeled and must not be presented as the equivalent same-split comparator.
Outputs are 180 per-question predictions in the handoff schema, plus original engine
outputs, question/case timings, resource-watch logs, and metadata:

* Qwen: `eval/fresh60/predictions/qwen-frozen-baseline-full660/`
  * normalized `predictions.jsonl` SHA-256:
    `1baeb0cd61470529e80d33473a7908bd4a71e1eae6269f97632b09221aef1e12`
  * question latency p50/p95: 102.0 / 127.5 ms; case sum: 320.9 / 344.8 ms
  * peak process footprint: 3,991,783,248 B; swap unchanged from 1,112,801,280 B
* Harrier: `eval/fresh60/predictions/harrier-frozen-baseline-full660/`
  * normalized `predictions.jsonl` SHA-256:
    `b86bc07f2bd3ff1f5fc119a1e8cc59e630398a01dc164a790c3a8a396f1d2754`
  * question latency p50/p95: 102.1 / 126.9 ms; case sum: 320.5 / 344.9 ms
  * peak process footprint: 3,986,884,408 B; swap unchanged from 1,112,801,280 B

`run-metadata.json` records fixture, predictions, frozen scorer/config/model/profile/compiler
hashes, C value, runtime, timing scopes, compute assumptions, and resource guards.
`varied-metal-raw.jsonl` preserves original runner outputs. No concurrent model
process was run.

The freeze manifest said same-460 coefficients were not saved. `fit_head_only_460.py`
fit a separate C=1.0 typed head from cached training embeddings and targets on exactly
the frozen 460 `train_indices`; dev and holdout labels were not used in fitting. Its
models, metadata, frozen config, and separate `selection-freeze.json` are under
`eval/fresh60/headonly-same460/`. Those frozen coefficients were then used in separate
`varied-metal` inference runs (never conflated with the full-660 C=1/.1 baseline):

* Qwen C=1.0: `eval/fresh60/predictions/qwen-headonly-same460/`, predictions SHA-256
  `a704c39005e5fe0cbec4f78331d0238a625e27fd436fd6b2af70c7692fb46ae8`; case p50/p95
  322.1 / 348.2 ms; max process footprint 3,985,753,912 B.
* Harrier C=1.0: `eval/fresh60/predictions/harrier-headonly-same460/`, predictions
  SHA-256 `798491a9741934fe5c35d04ba22fdf7400e5c72f36ec4672ed452ad9ee9463c9`; case
  p50/p95 322.6 / 347.6 ms; max process footprint 3,986,474,808 B.

Both runs retained the pre-load swap baseline of 1,112,801,280 B. Their fit bundles
record 460 training questions (140 choice, 160 noul, 160 score), C=1.0 for each head,
training input/feature/profile hashes, and the fit-script hash. No train/dev split
selection or holdout scoring was performed by the comparator fit.

## LoRA full fresh-60 runs and scores

* Qwen LoRA: `eval/fresh60/predictions/qwen-lora/`, 180 rows; prediction SHA-256
  `c47fb09e32e0ce9bbac2cda44cc016d587adbef8aa0e024ad9791f80397be2cc`.
  Choice 15/60, noul 30/60, score 20/60, all-three 0/60; score expected-value MAE
  30.132 rubric points and ordinal-level MAE 0.933. Question p50/p95 126.8/164.9 ms;
  case p50/p95 397.8/439.2 ms. Process peak 5,712,676,712 B; swap remained
  1,087,635,456 B.
* Harrier LoRA: `eval/fresh60/predictions/harrier-lora/`, 180 rows; prediction SHA-256
  `e7ce1c8c7912ad896d38cae6cb7d7a0ca6ad29318d7d2d52b37a5746d01bc3b9`.
  Choice 15/60, noul 30/60, score 20/60, all-three 0/60; score expected-value MAE
  30.161 rubric points and ordinal-level MAE 0.933. Question p50/p95 126.3/164.4 ms;
  case p50/p95 397.3/436.5 ms. Process peak 5,715,199,848 B; swap never rose above
  its 1,087,635,456 B baseline (it dipped below briefly).

Both metric files are colocated with predictions. External process-footprint/swap
watches, internal MLX allocator snapshots, logs, and original raw logits are retained.
`comparison-core-six.json` and `.md` score both LoRAs, both full-660 baselines and
both same-460 comparators on the same fixture after prediction freeze. The report
includes per-type/all-three and per-domain counts, score expected-value MAE and
ordinal-level MAE, timing scopes, resource guards, and provenance.

## Remaining comparisons and blockers

`compare_frozen.py` consumes already frozen 180-row normalized bundles with matching
fixture hashes, exact 60×3 ID/question coverage, and `predictions_sha256`. Required
core systems are Qwen/Harrier LoRA, full-660 Qwen/Harrier baselines, and same-460
Qwen/Harrier head-only comparators. Pinned Laya MLX and JEV can be supplied as
optional bundles. It reports
per-type, all-three, per-domain, score ordinal, timing, and
provenance summaries only after prediction files are frozen. It makes no model or API
calls.

Use bundle names `qwen_lora`, `harrier_lora`, `qwen_frozen_baseline_full660`,
`harrier_frozen_baseline_full660`, `qwen_headonly_same460`, and
`harrier_headonly_same460`; add optional `laya_mlx` or `jev` if their paths are
available. Laya GPU inference was left to the separate agent; JEV was not run here.

Current blockers:

1. LoRA inference used a separately hash-bound, explicitly authorized 512-token
   inference-only policy. Inputs over the unchanged training limit of 288 are an
   out-of-training-length extrapolation and should be disclosed.
2. Laya GPU inference was left to the separate agent. JEV was not rerun here; any
   separately produced JEV score/output remains outside this six-system report.
