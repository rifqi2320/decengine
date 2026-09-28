# All-MLX parity handoff

The CPU comparison artifacts remain labelled **mixed GPU/CPU**: MLX/Metal generated
the embeddings; frozen typed weights were scored with CPU NumPy. A separate prompt-study
Metal path exists at `results/probes/varied/prompt-study/metal/runner.rs` and has already
produced the all-MLX fresh60 predictions under
`metal/runs/fresh60/shared-v2-baseline/` for all eight model/text-variant combinations.
Those existing GPU predictions use the single shared v2 head and are the prompt-only
comparison, not the per-variant-head secondary run.

The protocol-correction secondary CPU evaluation now has eight pre-frozen, train-only
per-variant heads and eight prediction files. To compare GPU output against that second
condition, use the bundles and paths in
`secondary/variant-heads/GPU-VARIANT-HEAD-HANDOFF.md` and
`secondary/variant-heads/gpu-variant-head-handoff.json`:

* Scorer bundles: `metal/frozen-scorers/{qwen,harrier}/<variant>.json`; each is SHA-bound
  by its adjacent `.sha256`, with variant-specific C and all type-head mean/scale/weight
  fingerprints in the handoff metadata.
* Secondary CPU reference predictions:
  `secondary/variant-heads/predictions/{qwen,harrier}/<variant>.jsonl`.
* Suggested isolated GPU destinations:
  `metal/runs/fresh60/per-variant-heads/{qwen,harrier}/<variant>/predictions.jsonl`.

The scorer bundles predate the first reference-scoring artifact (00:46:56–00:47:02 UTC
versus 00:53:03 UTC), bind only train fixtures/features/CV configs, and are the existing
full-train heads; no secondary refit was made. All eight secondary CPU predictions were
hash-frozen before their scoring stage. Preserve the existing `shared-v2-baseline` run
and all original reports. GPU inference can run using the already available
`prompt-study-metal` command; compare labels, probabilities and score expected values
only after its prediction files are persisted. Do not tune or choose a text/head using
fresh60.
