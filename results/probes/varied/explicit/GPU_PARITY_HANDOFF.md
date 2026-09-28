# Handoff: GPU-resident typed scorer parity for explicit contexts

Owner/coordination target: GPU typed-scorer work in session
`ses_f311f5c6dffejaeZNNFBHfwxQF`. This handoff is read-only guidance; it does not
modify or assume ownership of that agent’s files.

## Frozen parameters — do not fit/tune

| Embedder | Frozen run | C | Frozen config SHA-256 | `models.json` SHA-256 |
|---|---|---:|---|---|
| Qwen | `results/probes/varied/matched/frozen-qwen/` | 1.0 | `e92149799df56666251533333aa2518ad93ec66f4397a81bc24a66b003087b54` | `7ce3ecae127696bd1bbc929e84893f2e4ddb740106346910f43c84fe4cff6eb5` |
| Harrier | `results/probes/varied/matched/frozen-harrier/` | 0.1 | `02d233c8a32020e96bbb8d1f41684ec81dba75007fb8db196c76281b5701738d` | `077962ae2a20dea890ee8e6566c4730023b1a3d11ff2ca1be567aa8e1462b32f` |

Each file contains exactly the generic `choice`, `noul`, and `score` scorer arrays,
including feature means/scales. Harrier’s final fresh refit was exactly compatible
(zero max absolute delta for coefficients, means, scales) with the prior C=0.1 typed
baseline. Qwen’s previous C=0.01 artifact is not approved; use only the frozen C=1.0
run.

## Explicit CPU-reference feature/prediction inputs

- Qwen features: `runs/qwen/explicit.features.jsonl`
  - SHA-256 `539815e6f30278d1b3c6715c3e6d36cb4dd5bd283bbb625caca82dd960fe28fd`
- Harrier features: `runs/harrier/explicit.features.jsonl`
  - SHA-256 `3746fc6d41e450c02171ab84a6ac5ba4be8bc74a0de0a44f1a153d1315e16634`
- Both use text version `decengine-varied-pair-text-v2-generic-types`, L2-normalized
  1024d embeddings, and match each frozen training exporter profile.
- CPU reference predictions (MLX embeddings + NumPy ranker):
  `runs/{qwen,harrier}/cpu-reference/evaluation/predictions.jsonl`.
- Pair rows map explicit IDs `mq-explicit-001..060` back to matched IDs using
  `benchmarks/cases/varied/multi/test-explicit-60.manifest.json`.

Questions/options/rubrics/references were preserved by the author validator. The
exporter consumes only `request.state` and `request.questions`: `evaluation_metadata`
and `reference` must remain absent from inference text. Keep original explicit IDs in
raw GPU outputs, with a separate source-ID mapping for paired comparisons.

## Required GPU work before publishing comparisons

1. Load the frozen scorer arrays above without updating/re-fitting them.
2. Implement the same pointwise features/order (`query`, `candidate`, `query*candidate`,
   `abs(query-candidate)`), per-type normalization, candidate-local softmax, and score
   expected-value calculation on the resident MLX/Metal path.
3. Check every GPU question against the matching NumPy CPU prediction. Preserve choice
   and noul candidate probabilities/selection, score label argmax and expected numeric
   value. Report maximum probability/logit/expected-value deltas and exact selected-label
   parity; fail visibly if parity fails.
4. Persist raw GPU predictions before reference scoring, then compute explicit/matched
   paired accuracy, candidate switches, and all-three case transitions only after GPU
   parity is established.
5. Measure synchronized end-to-end MLX/Metal latency with warmup and explicit device
   synchronization; include state/query/candidate embeddings plus typed scoring. Report
   separately from exporter setup, serialization, and any CPU reference timing. Do not
   relabel the CPU reference scorer as GPU-resident.

The explicit fixture was authored as an easy/clarity intervention with access to labels;
even paired improvements are descriptive and not an unbiased generalization/transfer
gain. Do not select/tune variants on this fixture.
