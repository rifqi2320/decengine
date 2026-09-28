# Prompt-study MLX/Metal inference

This isolated path embeds the exact prompt-study text variants and runs the frozen typed
pair scorer on `Device(gpu, 0)`. Query/candidate embeddings, the 4096d pair features,
scaler-folded linear score, and candidate-local softmax remain MLX arrays. Only final
probabilities return to host; no feature vectors are read or transferred by inference.
Prompt text follows the non-editing reference serializer in
`../src/exporter_base.rs` for `v2-baseline`, `readable-state`, `rule-in-candidate`, and
`joint-context-candidate`.

## Head protocols and provenance

The CPU reference protocol is explicitly **one unchanged v2-baseline head per model
applied to each of the four prompt-only text variants**. Qwen C=1.0 and Harrier C=0.1;
the exact same coefficients, means, and scales apply across all variants. `HEAD_PROTOCOLS.md`
records a scorer mismatch found before inference: the earlier candidate head bundles were
variant-specific train-only refits/C values, and do not match the CPU reference. They
remain separate under `frozen-scorers/{model}/` and are not used for primary parity.

`freeze_shared_v2_heads.py` packages the exact published frozen `models.json` arrays into
`frozen-scorers/shared-v2-baseline/{model}/{variant}.json`. It performs no fit. The runner
checks exact weight/mean/scale JSON equality and source hashes, C, profile/tokenizer and
installed checkpoint hashes, and the pre-reference fixture hash before inference.

Run one model/variant combination using the explicit installed model store:

```sh
STORE=/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store
target/release/decengine --home "$STORE" prompt-study-metal \
  --input benchmarks/cases/varied/multi/test-prompt-fresh-60.jsonl \
  --output results/probes/varied/prompt-study/metal/runs/fresh60/shared-v2-baseline/qwen/v2-baseline/predictions.jsonl \
  --scorer-bundle results/probes/varied/prompt-study/metal/frozen-scorers/shared-v2-baseline/qwen/v2-baseline.json \
  --study-run results/probes/varied/prompt-study/runs/trainonly-clean300-20260924T002000Z \
  --variant v2-baseline --model Qwen/Qwen3-Embedding-0.6B
```

The reference exports and predictions are present under
`runs/holdout-fresh60-prompt-study-20260924T000000Z/`. Both head protocols have now been
run once per model/variant, sequentially and without concurrent MLX model loads:

* **Shared v2 head / primary CPU reference:** `runs/fresh60/shared-v2-baseline/`
  compares one exact frozen Qwen C=1.0 or Harrier C=0.1 head across all four texts.
* **Per-variant head / secondary CPU reference:** `runs/fresh60/per-variant-heads/`
  compares the pre-frozen full-train variant head and C for each text, with the original
  scorer bundle hashes validated against the separate secondary handoff.

The combined results are in `runs/fresh60/FINAL_SUMMARY.md` and `.json`; separate detailed
parity/timing, stages, tokens, resource footprints and provenance are in the two protocol
subdirectories. All model/variant combinations have exact 180/180 selected label, binary
value, and score argmax parity (60/60 all-three cases); probability and score expected
value differences are numerical float32-vs-float64 drift. `HEAD_PROTOCOLS.md` documents
the distinction and the pre-run correction of the unlike-head discrepancy. No holdout
targets were accessed and no weights were fitted/selected on the holdout.

Run each remaining variant/model in a separate sequential invocation and distinct
directory. The runner warms each typed scorer once, then writes raw predictions,
synchronized per-question stages/token counts, and three-question case totals. It never
reads `reference`, `target`, `rationale`, or evaluation metadata. Wait for all eight
fresh60 CPU-reference feature exports to be declared ready before starting the eight GPU
runs; do not run simultaneous model processes. Post-run parity/provenance analysis belongs
under `runs/fresh60/` and must compare only after GPU predictions have been persisted.
