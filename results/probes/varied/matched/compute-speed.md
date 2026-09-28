# Matched 60 × 3 speed and arithmetic snapshot

**Scope:** fixture `benchmarks/cases/varied/multi/test-matched-60.jsonl` (SHA-256
`48a057dd…a6ed614`), 60 cases / 180 questions, matched for the Qwen/Harrier probe
exports and Laya/JEV timings. The Qwen/Harrier implementation is **MLX/Metal embeddings
followed by host-vector export and CPU NumPy scoring**—not an all-GPU probe. See
[`compute-speed.json`](compute-speed.json) for complete provenance, per-case/per-question
arithmetic estimates, and exact sampled timing summaries.

| System / scope | p50 | p95 | Arithmetic / notes |
|---|---:|---:|---|
| Qwen probe, MLX export process | 23.04 s (n=1) | — | 714 synchronized query/candidate embeddings; includes process startup, model load/init, host copies, JSON export; no isolated load/warmup or per-case samples |
| Qwen frozen scorer | 6.10 µs/question | 7.54 µs/question | **CPU NumPy only**, saved `n=180` timer; not GPU or end-to-end |
| Harrier probe, MLX export process | 23.46 s (n=1) | — | Same scope and call count as Qwen |
| Harrier frozen scorer | 6.04 µs/question | 7.50 µs/question | **CPU NumPy only**, saved `n=180` timer; not GPU or end-to-end |
| Laya MLX, 3 separate calls summed/case | 443.96 ms/case | 4,648.91 ms/case | `n=60`; large tail from the saved run; GPU inference timings |
| Laya MLX, individual calls | 103.33 / 132.83 / 128.65 ms p50 | 457.22 / 1,221.88 / 2,673.24 ms p95 | `immediate_intervention` / `record_auditability` / `response_plan`, each `n=60` |
| JEV, one HTTP request/case | 390.79 ms/case | 600.56 ms/case | `n=60`; includes network and service latency; API compute is unobservable |

| Qwen = Harrier encoder arithmetic estimate | p50 | p95 | Mean |
|---|---:|---:|---:|
| Per question (query + its variable candidates) | 303.54 GFLOPs | 326.49 GFLOPs | 301.82 GFLOPs |
| Per case (all three questions) | 904.71 GFLOPs | 945.46 GFLOPs | 905.46 GFLOPs |

Both embedding checkpoints have the same verified Qwen3 config (28 layers, H=1024,
I=3072, 8 KV heads × 128). The matched pairwise exporter makes **180 query + 534
candidate = 714 embeddings**; it makes **no separate state embedding**. State is included
in each question query, so it is encoded three times per case. Questions have 2–5
candidates (75 × 2, 54 × 3, 33 × 4, 18 × 5). Tokenization is per independent string,
unpadded: 69,942 total tokens, query mean 295.05 tokens, candidate mean 31.52. The
estimate includes dense Q/K/V/O projections, gated MLP and full causal QK/AV arithmetic;
it excludes lower-order norm/RoPE/activation/pooling operations and is **not measured
GPU FLOPs**. Formula and exclusions are in the JSON.

The saved scorer's 4,096-D pair-feature construction and standardization/linear scoring
are CPU operations. Approximate arithmetic is ~19,456 ops/candidate (10.39 M total for
534 candidates, softmax nonlinear/reductions excluded); only standardization/dot/softmax
fall inside the saved ~6 µs timer. This is distinct from encoder FLOPs and should not be
represented as GPU classifier work. For a future MLX-resident generic scorer, keep
embedding vectors on Metal and port the existing `concat(q, c, q*c, abs(q-c))`, frozen
per-type standardization/linear score, and candidate-local softmax as a separate stage;
variable candidate counts require question-local normalization. That is an architecture
recommendation, **not an implemented or benchmarked path**.

The prior `~354 GFLOPs` Laya three-call estimate is from another 300-case fixture and
does not represent this matched input set. A separate follow-up recovered the exact
matched Laya per-call token counts from saved `raw-results.jsonl` and applied the
verified Laya architecture formula; matched and explicit-60 Laya estimates and
Qwen/Harrier explicit deltas are in
[`../explicit/metal/compute-speed.md`](../explicit/metal/compute-speed.md) and its
JSON. Even on matched input, the probe embeds query plus variable candidate strings
while Laya runs three full decision calls, so these are not like-for-like operations;
no FLOP ratio is claimed. JEV FLOPs cannot be inferred from API latency.

**Timing provenance:** Qwen/Harrier CPU scorer summaries are the saved evaluation
`provenance.json` files; their explicit scope excludes feature export, parsing, and I/O.
The one-off synchronized MLX export wall timings were measured in a fresh subprocess per
model with explicit store
`/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store`;
temporary exports and hashes are recorded in the JSON. No explicit warmup was performed
and OS/device cache state was not controlled. One sample/model does not support p50/p95.
Laya/JEV values are recomputed from all 60 rows of
`results/laya/varied/runs/20260923T151228074560Z/timings.jsonl`; runtime/device and pinned
Laya revision are in `run-metadata.json`. No prediction/reference labels were consulted.
