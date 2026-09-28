# LoRA preference-training feasibility (phase 1)

Status: checkpoint adapter and end-to-end trainer implemented. Both real-checkpoint
gradient/loss smoke tests and compiled-text parity gates passed. An unrestricted Qwen
run was stopped after two epochs at 17.3 GB process peak. A later guarded Qwen run
stopped at step 10 when its peak exceeded a 6 GiB early-stop threshold (but remained
below 8 GiB). Harrier received smoke only; no full Harrier run occurred. No final
selected model or full experiment claim is made. No holdout fixture or label was
opened. Existing frozen model files and inference code were not changed.

## Host and dependencies

- Host is Apple Silicon (`arm64`, Darwin 25.6.0), with 24 GiB unified memory.
- Python 3.9.6's installed packages include `mlx==0.29.3` and
  `mlx-lm==0.29.1`; `mlx.core` reports `Device(gpu, 0)`. MLX provides
  `value_and_grad` and `grad`. PyTorch is not installed, but is not needed for an
  MLX path. `mlx-lm` includes LoRA tuner/trainer modules.
- Both installed checkpoints are about 1.1 GiB and have 310 tensors. Direct MLX
  inspection reports all checkpoint tensors are BF16. Both configs describe a
  28-layer Qwen3 decoder, hidden width 1024, 16 attention heads / 8 KV heads, and
  intermediate width 3072. Qwen declares `Qwen3ForCausalLM`; Harrier declares
  `Qwen3Model`. The Rust backend implements last-token pooling and L2 normalization
  over the Qwen3 decoder (see `crates/decengine-engine-mlx/src/lib.rs`).
- The explicitly named train sources are available: clean300 at
  `/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/varied-loqfo-clean300/train-clean-300.jsonl`
  and multi train120 at `benchmarks/cases/varied/multi/train-120.jsonl`. The
  requested combined count is 660 questions. This inspection did not read labels
  from them or access any test split.

## Concrete integration blocker and remedy

`mlx_lm.load()` cannot load either installed checkpoint as-is. Running it against
the local Qwen path fails in `mlx.nn.Module.load_weights(strict=True)` with 310
unexpected names such as `layers.0.self_attn.q_proj.weight`: the Rust/HF embedding
checkpoint uses unprefixed Qwen3 decoder keys, while the installed
`mlx_lm.models.qwen3.Model` nests the decoder under `model.*`. This is an explicit
checkpoint namespace mismatch, not a missing MLX/autograd/hardware capability.

Smallest viable remedy: a separate worker is implementing the narrowly scoped
checkpoint adapter in `checkpoint_adapter.py`. It must instantiate MLX-LM Qwen3 from
local config, map decoder keys (`layers.*`, `embed_tokens.*`, `norm.*`) to nested
`model.*`, validate all tensor names/shapes, and verify Rust parity. This session has
implemented `train.py` against that adapter boundary. It uses `model.model(input_ids)`
(not causal-LM vocabulary logits) for last-token embeddings, four upper-layer q/v LoRA
modules, frozen BF16 base parameters, FP32 LoRA/head parameters, and grouped
train/dev. The training launch is intentionally gated on a passing adapter parity
report and is not yet complete.

The initial direct-load error was resolved. `train.py` reads compiled train texts only and
strips cached embedding arrays before JSON decoding; those arrays never enter the
LoRA training computation. Do not treat prior cached feature vectors or
head-only experiments as a substitute for the requested backbone-adapter experiment.
No heldout/test results should be opened until train-only selection is explicitly
reported to the orchestrator.

### Real training-path smoke and fixed train-only comparison (2026-09-24)

`train.py --smoke-only` completed sequentially on Qwen and Harrier, using one actual
question minibatch after strict checkpoint loading, profile-compiled train-text
parity, and FP32 trainable LoRA/head setup. It asserted finite nonzero full-model and
LoRA gradients and a lower post-update loss. Compiled query/candidate embedding
cosines and absolute differences were checked against the Rust MLX train feature
export, not against raw text:

| Model | Query cosine / max abs | Candidate cosine / max abs | minibatch loss before -> after | LoRA gradient norm |
| --- | ---: | ---: | ---: | ---: |
| Qwen 0.6B (profile 1) | 0.999822 / 0.003089 | 0.999894 / 0.001962 | 1.385405 -> 1.357829 | 0.015185 |
| Harrier 0.6B (profile 2) | 0.999897 / 0.003359 | 0.999942 / 0.001964 | 1.386138 -> 1.361579 | 0.006994 |

Pre-run, same grouped train/dev head-only baselines (fixed C=1, train split n=460,
dev n=200) were measured separately from the LoRA path. These cached-vector metrics
are only a comparator; cached vectors are never consumed by LoRA updates:

| Model | Choice accuracy (n=100) | Noul accuracy (n=60) | Score accuracy (n=40) | Score normalized value MAE | Macro accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 0.270 | 0.800 | 0.525 | 0.244254 | 0.531667 |
| Harrier | 0.380 | 0.833 | 0.650 | 0.185557 | 0.621111 |

The initial full run used one question per optimizer step and only one model at a
time, but did not yet include OS-footprint monitoring or periodic Metal cache clearing.
At 09:29 local time the Qwen process reached a 17.3 GB physical-footprint peak
(17 GB `IOAccelerator (graphics)`), while system swap was 27.9 GB used of 28.7 GB and
system-wide free memory was 9%. This is unsafe for the user's later 8 GiB cap. The
operator sent SIGINT to Python PID 53934 (wrapper shell PID 53931); it exited 130 at
an MLX gradient evaluation. No Harrier full run started. Qwen `latest.json` records
the last complete checkpoint boundary as epoch 2 / step 920; its epoch 3 had 4
additional log rows and was not committed. `best.json` selects epoch 1 by grouped-dev
loss (0.97537, exact accuracy 0.40); epoch 2 was worse on loss (1.12008, accuracy
0.46). These are partial, train/dev-only diagnostics, not final selection.

After process exit, no LoRA training/inference process remained. Swap settled to
about 1.57 GB of a 3 GB configured total and free memory returned to 83%. Device
information from the local MLX query: Apple M4 Pro, 24 GiB unified memory,
19,069,665,280-byte recommended Metal working set. The actual per-process maximum
observed is far above the new 8 GiB hard limit and must not be normalized as
acceptable.

The revised trainer first ran an instrumented one-minibatch smoke; both models passed
below 2 GiB. Per the user's approval, a capped Qwen attempt then stopped at step 10
on its early memory guard. The conservative configuration remains:
micro-batch one question, accumulation one, two upper q/v LoRA layers at rank 4,
profile-text maximum 288 tokens (the 2,788 train query/candidate strings per model
have p50=24, p90=167, p95=179, p99=257, max=266), reject any longer string rather
than truncate, clear Metal cache and sample OS footprint/system swap every 10 steps,
and stop at a 6 GiB measured process peak to retain at least 2 GiB under the user's
8 GiB cap; a 4 GiB total system-swap limit is also enforced. The trainer records
activation checkpointing as disabled/unverified; this path would need a separate
low-resource implementation review before any later run. User approved a bounded
run only if the instrumented smoke stayed below cap and swap remained stable; both
smoke checks passed. A Qwen capped run was started sequentially, then stopped by its
6 GiB guard at step 10, before an epoch checkpoint; no Harrier full run started.

The repeated one-question smoke measured process peak including model load and one
real optimizer update:

| Model | Process footprint peak | MLX peak allocator | Loss before -> after | LoRA grad norm | Swap during smoke |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 1,934,902,904 B (1.80 GiB) | 1,601,603,872 B | 1.383586 -> 1.355799 | 0.006790 | 1,640,622,981 B unchanged |
| Harrier | 1,936,524,944 B (1.80 GiB) | 1,602,144,544 B | 1.384173 -> 1.358870 | 0.003146 | 1,640,622,981 B unchanged |

This confirms an actual low-resource one-step path with batch=1, but does not
predict multi-epoch allocator growth. In the capped Qwen run, after 10 updates,
process peak was 6,472,418,960 B (about 6.03 GiB), above the 6 GiB stop but below
the hard 8 GiB cap. Current process footprint was 3,355,248,272 B; MLX active/peak
allocator was 1,204,928,404 / 1,892,488,166 B; system swap was unchanged at
1,640,622,981 B. `latest.json` remains at epoch 0 / step 0; only ten step rows were
written. The run's PID was not captured; future logger now includes PID. This quick
growth despite cache clearing indicates the allocator/cached Metal footprint needs
investigation; do not launch another full run until the orchestrator/user clarifies.

## Reproducibility inputs recorded

- Host Python: `3.9.6`; MLX: `0.29.3`; MLX-LM: `0.29.1`.
- Checkpoint configuration and tensor dtype/shape inspection were read locally;
  no network/API call was made.
- At the time of the original inspection, existing source files, frozen weights,
  fixtures, and inference code were not modified. This report was then the only
  new artifact in this task's allowed output directory.

## Adapter implementation and train-only validation addendum (2026-09-24)

The namespace blocker above is addressed by `checkpoint_adapter.py`; this does not
constitute a training run, checkpoint, adapter selection, or heldout evaluation.
The adapter reads only the explicitly installed local store
`/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store`.
Both installed model directories load: `Qwen--Qwen3-Embedding-0.6B` (profile 1)
and `microsoft--harrier-oss-v1-0.6b` (profile 2). For each, the validated mapping
has 310 tensors, with no skipped/extra weights. A complete strict comparison checks
source/destination key count, one-to-one key uniqueness, shape, and homogeneous
BF16 or FP32 dtype before calling MLX-LM `load_weights(strict=True)`. No
causal-LM `lm_head` is loaded or used: hidden states come from
`loaded.model.model(input_ids)`.

### Importable trainer contract

Import `load_checkpoint(model_dir, max_length=32768)` from
`results/probes/varied/lora/checkpoint_adapter.py`. The resulting
`Qwen3Embedding` exposes:

* `tokenize(text) -> mlx.core.array` with shape `(1, sequence)`;
* `forward_hidden(input_ids) -> mlx.core.array` with shape `(batch, sequence, 1024)`;
* `embed(text_or_ids, pooling_dtype="float32"|"bfloat16") -> mlx.core.array`
  with shape `(1, 1024)`, last-token pooling, and L2 normalization. `float32`
  pooling mirrors the Rust path; BF16 mode returns a BF16 normalized vector;
* `inject_upper_lora(loaded, layers=4, projections=("q_proj", "v_proj"),
  rank=8, alpha=16.0) -> list[str]`. It freezes the base model and injects
  trainable FP32 low-rank parameters in the selected upper layers. They are
  discoverable in `tree_flatten(loaded.model.trainable_parameters())` (or the
  regular parameter tree paths containing `.lora_a` / `.lora_b`).

Example objective wiring for MLX-LM's `nn.value_and_grad`:

```python
loaded = load_checkpoint(model_dir)
inject_upper_lora(loaded)
ids = loaded.tokenize(text)
def objective(model):
    vector = model.model(ids)[:, -1, :].astype(mx.float32)
    return pairwise_loss(vector, labels)
loss, grads = nn.value_and_grad(loaded.model, objective)(loaded.model)
```

The helper initializes LoRA B to zero, so its first-step gradient can be nonzero
while A's first-step gradient is zero; this is the standard identity-preserving
LoRA initialization, not a disconnected graph. In the real Qwen run (one upper
layer, q_proj, rank 2), `lora_b` gradient L1 was 89.4425 and the loss was
12.13598. The focused unit gradient test also verifies nonzero autograd to LoRA
weights. No optimizer/trainer behavior was tested.

### Inference comparison (not a shape-only check)

The input was the `question` and `options.blue` candidate from the first row of the
explicit **train-clean-300** source
`/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/varied-loqfo-clean300/train-clean-300.jsonl`.
No reference/label field, dev row, or test/heldout split was read. Harrier is the
installed profile-v2 model; Qwen was also measured as a profile-1 comparison. Rust
vectors were obtained by a temporary local harness calling
`MlxDecisionEngine::load_model` and `embed_text` against each installed model; MLX
vectors were computed with `load_checkpoint(...).embed(..., pooling_dtype=...)`.
Both paths used the same tokenizer file, add-special-tokens behavior, raw question
or candidate text, last-token pooling, and L2 normalization.

| Installed model / profile | Text | Pooling | max abs diff vs Rust | cosine vs Rust |
| --- | --- | --- | ---: | ---: |
| Harrier / 2 | question | FP32 | 0.00204743 | 0.99991800 |
| Harrier / 2 | candidate | FP32 | 0.00168370 | 0.99994072 |
| Harrier / 2 | question | BF16 | 0.00210798 | 0.99991434 |
| Harrier / 2 | candidate | BF16 | 0.00149523 | 0.99993817 |
| Qwen / 1 | question | FP32 | 0.00173585 | 0.99990569 |
| Qwen / 1 | candidate | FP32 | 0.00338033 | 0.99987489 |
| Qwen / 1 | question | BF16 | 0.00171202 | 0.99990329 |
| Qwen / 1 | candidate | BF16 | 0.00355005 | 0.99987306 |

This establishes a real numerical comparison on this train-only sample, not bitwise
identity or a generalized parity claim. No semantic attention discrepancy was
found on inspection: Rust uses default non-traditional RoPE (`RopeBuilder`, base
from `rope_theta`), Q/K RMSNorm precedes RoPE, and causal scaled-dot-product
attention; mlx-lm Qwen3 uses `traditional=False`, the same Q/K norm and RoPE order,
and its causal attention mask. The remaining max-absolute differences (up to
0.00355) are not explained by a demonstrated RoPE/mask mismatch; likely backend or
BF16 intermediate-rounding differences remain unlocalized. Do not claim exact
parity based on these near-unity cosine values.

Reproduction commands used:

```sh
python3 -m pytest -q results/probes/varied/lora/test_checkpoint_adapter.py
# Output: 3 passed, 1 LibreSSL urllib3 warning; includes loading both installed
# checkpoints and validating the 1024-D normalized embedding output.
```

The actual Rust vectors came from a temporary harness (not added to the repository)
using `cargo run --manifest-path /private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/lora-parity/Cargo.toml -- <installed-model-dir> '<train question or candidate>'`;
the Python metric calculation used `numpy.abs(mlx_vector-rust_vector).max()` and
`numpy.dot(mlx_vector,rust_vector)/(numpy.linalg.norm(mlx_vector)*numpy.linalg.norm(rust_vector))`.
Autograd was separately exercised with the same adapter on the real Qwen checkpoint.
No API/network call was made. No trainer, fixtures, Rust sources, heldout/test data,
or model-store files were changed.

## Follow-up resource guard stop

User approved the bounded run conditional on smoke staying below budget and swap
stability. The capped smoke met both: each model's process peak was about 1.94 GB and
swap stayed at 1.64 GB. The subsequent single Qwen process reached step 10, where the
10-step sample found process peak `6,472,418,960 B` (~6.03 GiB), above the configured
6 GiB early-stop threshold. It was stopped before the 8 GiB hard cap and without
system-swap growth (still `1,640,622,981 B`). The process exited; no other training
or inference process remained. `runs/qwen-capped/STOPPED_MEMORY_GUARD.md` records
details; its checkpoint boundary remains epoch 0/step 0 and its 10 step rows are not
a trained checkpoint. Harrier full training was never started.

The initial 17.3 GB peak, followed by this growth from ~1.9 GB one-step peak to
~6.03 GiB after 10 steps, means this trainer cannot currently guarantee the requested
8 GiB cap across a full run. `mx.clear_cache()` every 10 steps did not prevent the
increase; MLX active allocator was only ~1.20 GB at the stop, suggesting the excess
is outside currently active MLX arrays (e.g. Metal/IOAccelerator reservations/cache),
but this remains a hypothesis. No more training or inference was started. Candidate
lightweight plan for discussion only: instrument/stop every step, fixed sequence
buckets to reduce variable-shape kernel/cache growth, consider rematerializing the
last one or two blocks, keep micro-batch one/rank 2-4, and retain reject-only 288
tokens (all 2,788 train texts max at 266; shorter caps would exclude evidence-bearing
inputs and are not proposed). Revalidate memory with approval before any new run.
