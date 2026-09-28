# Explicit-state clarity condition

This is an authored, label-informed context intervention, not an independent
generalization holdout. The fixture author’s manifest says the intended targets and
rationales were available while adding explicit evidence; do not describe any observed
accuracy change as an unbiased transfer gain.

`evaluate_explicit.py` uses the immutable frozen typed parameters only (Qwen C=1.0,
Harrier C=0.1). It does not fit or tune. The real MLX/Metal exporter reads `request.state`
and `request.questions`; top-level `evaluation_metadata` and `reference` are not used
to form query/candidate text. The supplied validator passed, including exact matched-to-
explicit question/reference preservation and case-ID mapping.

## CPU parity reference and all-MLX parity

Current paired artifacts under `runs/{qwen,harrier}/cpu-reference/` deliberately use
MLX/Metal embeddings followed by the frozen NumPy CPU scorer. These are useful as a
reference for the all-MLX typed scorer, **not** a claim that this path is GPU-resident end
to end. See `CPU_REFERENCE.md` for paired changes, switches, and scoped timings. The
separate frozen all-MLX run is now under `metal/runs/{qwen,harrier}/`; it was compared to
these archived CPU prediction outputs without opening test targets. See
[`metal/GPU_PARITY_REPORT.md`](metal/GPU_PARITY_REPORT.md) for typed/all-three parity,
probability drift, synchronized timing, and provenance.

The GPU handoff is complete using the identical frozen model arrays. Artifacts remain
under their own explicit-run subtree; the CPU reference and existing `matched/` artifacts
were not rewritten.

## Frozen inputs and outputs

- `../matched/frozen-qwen/` and `../matched/frozen-harrier/`: read-only model, metadata,
  train manifests, and frozen selection records.
- `runs/{qwen,harrier}/explicit.features.jsonl`: real MLX/Metal embedding exports with
  text version `decengine-varied-pair-text-v2-generic-types`.
- `runs/{qwen,harrier}/cpu-reference/evaluation/predictions.jsonl`: raw CPU-reference
  predictions, persisted before the evaluator reads reference targets.
- `runs/{qwen,harrier}/cpu-reference/paired-cpu-reference.json` plus the paired question
  and case JSONL files: aligned to original matched IDs via the explicit fixture
  manifest’s source/transformed-ID map.

No weights were refit or tuned on explicit or matched test data. No JEV/API was used.
