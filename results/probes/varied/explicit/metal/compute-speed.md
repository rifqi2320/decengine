# Explicit-60 arithmetic extension

Extends the Qwen/Harrier matched-run arithmetic comparison with the **actual
per-query and per-candidate tokenizer lengths logged for the 60-case explicit Metal
run**. No inference or API calls were made. Token counts and frozen model configs are
inputs to a transparent analytical FLOP formula; these are **not measured device/GPU
FLOPs** and do not imply equal effective hardware operations.

## Qwen / Harrier embedding arithmetic

The timing logs contain 180 typed questions and 534 variable candidates (714 separate
query/candidate texts). Candidate counts/question: 2:75, 3:54, 4:33, 5:18. Explicit
token totals are 95,538 (78,705 query + 16,833 candidate); matched totals were 69,942
(53,109 query + 16,833 candidate), so explicit adds 25,596 tokens (+36.6%). Same
Qwen3 architecture shape and same token lengths yield identical arithmetic estimates
for Qwen and Harrier; their weights differ.

| Scope | Explicit mean | p50 | p95 | Matched mean → explicit delta |
|---|---:|---:|---:|---:|
| Per question: query plus each candidate embedded independently | 416.35 GFLOPs | 417.98 | 439.98 | 301.82 → +114.53 GFLOPs (+37.95%) |
| Per case: three typed questions | 1,249.05 GFLOPs | 1,247.99 | 1,303.57 | 905.46 → +343.59 GFLOPs (+37.95%) |

Per-question p50/p95 matched values were 303.54/326.49 GFLOPs; explicit values
417.98/439.98, respectively. Per-case matched p50/p95 were 904.71/945.46; explicit
values 1,247.99/1,303.57. Model-by-model details and all 180 question / 60 case records
are in `compute-speed.json`.

| Typed question (`n=60`) | Explicit mean / p50 / p95 GFLOPs | Mean delta vs matched |
|---|---:|---:|
| immediate_intervention (noul) | 420.33 / 419.61 / 430.06 | +114.51 (+37.45%) |
| record_auditability (score) | 418.18 / 413.09 / 439.36 | +114.62 (+37.76%) |
| response_plan (choice) | 410.54 / 411.02 / 444.86 | +114.46 (+38.66%) |

Formula is the **same** as
[`../../matched/compute-speed.json`](../../matched/compute-speed.json): for each separate
unpadded text of `n` tokens, `L * [n*(4H² + 4H*Hkv + 6H*I) + 4H*n(n+1)/2]`, with
`1 MAC = 2 FLOPs`; sum query and candidate strings. Both configs verify 28 layers,
H=1024, I=3072, 8 KV heads × 128, full causal attention. This includes dense projections,
gated MLP and QK/AV arithmetic; excludes embedding lookup, norm/RoPE/activation/softmax
and O(H) final pooling/normalization. No padding or quantization/sparsity discount.
The exact per-question typed means (choice / noul / score) are available in the JSON's
question samples; percentiles above pool all 180 questions.

## Laya: defensible matched and explicit arithmetic estimates

**Available and calculated.** Both saved Laya MLX runs have 60 per-case records and
exact `decisions.*.input_tokens`; run metadata matches each fixture SHA and the same
pinned Laya revision/checkpoint. The verified ModernBERT encoder config and previously
documented head formula are available. The calculation uses only those token counts,
config and frozen architecture—no prediction, label, or target values.

| Saved Laya run | Input tokens (3 calls/case) | Per question mean / p50 / p95 GFLOPs | Per case mean / p50 / p95 GFLOPs |
|---|---:|---:|---:|
| Matched-60 | 66,515 | 283.18 / 294.07 / 327.66 | 849.53 / 848.01 / 884.98 |
| Explicit-60 | 90,245 | 387.46 / 395.94 / 395.94 | 1,162.37 / 1,161.91 / 1,174.34 |

Explicit Laya adds 23,730 input tokens (+35.68%); estimated mean per-case arithmetic
increases 312.85 GFLOPs (+36.83%). The estimates are not measured GPU FLOPs. The
documented formula copies `results/laya/mlx/compute_comparison.py`: encoder dense MACs
`L*N*(4H²+3H*I)`; per-layer attention pairs are `N²` for full layers and the pinned
local-window pair count for sliding layers; the two-block decision head, scorer and
small act MLP use the formula recorded in the JSON. All multiply-accumulate arithmetic
is counted as 2 FLOPs; norms, activations and other lower-order operations remain
omitted as in the prior calculation.

The older `~354 GFLOPs` estimate for a different 300-case test fixture is **not** reused
for matched or explicit-60; the same-fixture token lengths support the estimates above.
Even these architecture-specific estimates are arithmetic models, not proof of equal
effective GPU work or a hardware-throughput comparison.

**Provenance / guardrails:** Qwen/Harrier token counts come from
`runs/{qwen,harrier}/predictions.jsonl.timings.jsonl`; their feature manifests, checkpoint
identity, config hashes, input fixture hash and source hashes are included in the JSON.
Laya matched and explicit input token counts are selected from
`results/laya/varied/runs/{20260923T151228074560Z,20260923T153215418583Z}/raw-results.jsonl`
(`decisions.*.input_tokens` only). Laya run metadata confirms fixture hashes, 60 cases,
model revision, weights hash, MLX Metal runtime and GPU device. No fixture, parameter,
prediction, label, or reference was changed/read for this analysis.

**Cross-links:** [matched arithmetic and speed report](../../matched/compute-speed.md),
[matched JSON](../../matched/compute-speed.json).
