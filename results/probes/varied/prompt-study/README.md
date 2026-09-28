# Prompt serialization study (isolated)

This directory owns its exporter, preregistered protocol, CV runner and outputs. It
does not modify the frozen v2 exporter, Rust compiler/model files, fixtures, or any
heldout artifacts. The exporter is a small standalone Rust package that links the
existing MLX engine/model store and embeds actual generated strings through MLX.

Run from the repository root on Apple Silicon, with both installed model checkpoints:

```sh
python3 results/probes/varied/prompt-study/run_study.py \
  --out results/probes/varied/prompt-study/runs/trainonly-<unique-id>
```

This creates 4 variants × 2 models × 2 train splits, then runs only the train-only
five-fold grouped CV. It reads clean300 and multi-train120, not test fixtures. It writes
the per-variant feature JSONL plus exact text/token lengths/manifests, OOF predictions,
CV selection and final training-only fitted weights. `--skip-export` is only for a
complete set of preexisting matching feature exports and is not a way to skip subsets.

The FLOP count is explicitly an estimate (`2 × 600M × tokenizer token count`), not
measured MLX FLOPs. Embedding calls include one query and each candidate per named
question. Qwen and Harrier are independently selected. No result is considered frozen
until the run has completed and its `frozen-config.json` and model-specific `selection`
files are saved; no heldout evaluation is run here.
