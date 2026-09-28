# Implementation status

This repository is a tested development vertical slice, not a releasable v1 binary.

## Green

- Stable `Choice`, `Noul`, and `Score` domain validation and result shapes.
- Versioned compiler, deterministic canonical state, and semantic candidate cache keys.
- Coarse `DecisionEngine` boundary with no device types above it.
- One shared `DecisionService` for C ABI and HTTP.
- Panic-contained C ABI with opaque handles and explicit string ownership.
- Pure-Python CFFI ABI-mode wrapper with typed objects and exception mapping.
- Allowlisted profiles for both required checkpoints.
- Explicit Hugging Face pull, local source install, SHA-256 records, verification, list,
  and remove lifecycle.
- Loopback-safe HTTP server and required routes.
- Deterministic fixture engine for application development, unit tests, and UAT.
- Source-only packaging checks.

## Red release gate

The dense Qwen3 forward path in `decengine-engine-mlx` is not implemented. The crate links
only on macOS/aarch64 and deliberately returns `model_load_failed`; it does not pretend that
fixture results are model results and never falls back remotely.

Required work before any binary release:

1. Implement tokenizer batching and left-padding masks for both profiles.
2. Implement dense Qwen3 forward using `mlx-rs` 0.32.0 and upstream safetensors.
3. Keep pooling, L2 normalization, candidate matrix, similarity, temperature, softmax, score
   reduction, and confidence in MLX.
4. Generate official golden vectors and pass both hardware parity suites.
5. Instrument evaluation boundaries, synchronizations, GPU time, and peak unified memory.
6. Replace this red status only with evidence from `tests/hardware/run.sh`.

The release script checks this status marker and exits until the gate is intentionally
removed in the same reviewed change that supplies the implementation and evidence.
