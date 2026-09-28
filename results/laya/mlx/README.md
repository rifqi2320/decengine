# Laya root-checkpoint MLX comparison

`compare_300_mlx.py` is an integration runner for the pinned native MLX port at
`mizorewww/laya-mlx` commit `0a859518634112655cb97c745dbf04f5191aaf13`.
It uses the original `convaiinnovations/laya` English root checkpoint revision
`5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b`; it refuses to load unless
`model.safetensors` has SHA-256
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
No converted or alternate checkpoint is accepted.

## Environment and commands

Python 3.12.12 environment and the pinned port checkout are under the approved
temporary directory `/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/laya-mlx-port`.
It currently uses MLX 0.32.2, tokenizers 0.23.2, Hugging Face Hub 1.32.0,
safetensors 0.8.0, Transformers 5.17.0. The original checkpoint files are in
`/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/checkpoint`.

```sh
python3.12 results/laya/mlx/compare_300_mlx.py --dry-run
HF_HOME=/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/hf-cache \
  /private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/laya-mlx-port/.venv/bin/python \
  results/laya/mlx/compare_300_mlx.py --case-id test-001 --dtype float32 --check-parity
```

The mandatory parity smoke compares exact input token IDs, raw option logits,
raw action logits, and the SDK-shaped output against
`results/laya/reference/cpu-reference.json` (official CPU implementation from
the same Hub revision). Measured Metal parity on `test-001` passed for all
three questions: token IDs and SDK outputs were exact; maximum raw option-logit
absolute difference was `8.74e-6` (owner), `4.30e-6` (urgent), and `5.01e-6`
(impact); max act-logit difference was `0.00196`. Float32 is used for this
reference-quality run. The specific results are in
`runs/parity-verified-f32/parity.json`.

After parity passes, run all three independent questions for every test case:

```sh
HF_HOME=/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/hf-cache \
  /private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/laya-mlx-port/.venv/bin/python \
  results/laya/mlx/compare_300_mlx.py --dtype float32
```

The benchmark makes three single-question inference calls per case (no batching),
matching the standalone PyTorch Laya comparison. Outputs go to a fresh
`results/laya/mlx/runs/<UTC timestamp>/` directory, or `--out` can select a
new run directory. Artifacts include:

* `raw-results.jsonl`: raw option/action logits, SDK-style raw responses,
  tokenized input IDs, scalar normalized labels, errors, and per-question timing;
* `laya-mlx-predictions.jsonl`: comparable `id`, `owner`, `urgent`, `impact` labels;
* `timings.jsonl`, `latency-summary.json`: preprocessing/inference and summed
  three-question times, with p50/p95/mean;
* `metrics.json`: fixture accuracy and output agreement/accuracy versus existing
  Qwen, Harrier and JEV pooled-300 results;
* `flops-estimate.json`: transparent dense-projection FLOPs lower-bound derived
  from the measured token counts (not unrelated accelerator/card claims);
* `run-metadata.json`, `checksums.json`: pinned model/revision/hash, runtime,
  memory and checksums.

Temperature calibration is taken from the original `rl_agent_config.json`
without the laya-mlx port's `[0.5, 5]` clamp. Outputs follow original root Laya
semantics: `rl_agent: {act_probability: ...}` (not `action`), choice argmax,
urgent as `noul >= 0.5` with no noul confidence, and impact as argmax of the
ordinal score probabilities (the complete rubric label and criterion are
passed through to the score head). The scalar impact mapping is for benchmark
comparison, not a claim that ordinal scoring is identical to a scalar classifier.

The integration runner uses the external port's verified native MLX encoder
and decision head. `encoder.py`, `head.py`, and `laya_mlx.py` in this directory
are separate strict reference implementations/helpers; no CPU or PyTorch
inference fallback is used for the MLX benchmark.

## Completed 300-case run

The float32 real-Metal run is recorded in
`runs/20260923T190000Z/` (300 cases, 900 independent forward calls, zero errors).
Accuracy against fixture references was owner **59.0%** (177/300), urgent
**61.0%** (183/300), impact **41.0%** (123/300), and complete-case **18.0%**
(54/300). For context, pooled-300 comparison artifacts report Qwen probe
44.7%/42.3%/29.3% and Harrier 45.0%/42.3%/42.3% (owner/urgent/impact); JEV
is 81.3%/91.0%/85.3%. These are accuracy results, not claims of model equivalence.

Per-decision MLX latency p50/p95 was owner **35.5/45.0 ms**, urgent
**25.0/31.1 ms**, impact **35.1/39.3 ms**; summed three-forward case p50/p95
was **95.7/123.6 ms**. The comparison run records Qwen p50/p95 **84.5/104.4
ms**, Harrier **81.0/93.0 ms**, JEV **421.9/580.3 ms** on its fresh 210-case
slice. Those baseline timings are included in `latency-summary.json` with a
non-equivalence caveat because hardware/mode and the number of forwards per
measurement differ. From measured average sequence length 156.50 and the
checkpoint's 421,296,830 parameters, the stated dense projection estimate is
**131.87 GFLOPs per forward** (attention and other overhead excluded); see
`flops-estimate.json` for its derivation.
