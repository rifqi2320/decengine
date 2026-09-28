# decengine-engine-mlx

This crate is the only workspace crate allowed to depend on `mlx-rs`. Version `0.32.0`
is pinned and Metal is enabled only on `macos/aarch64`.

Current gate: the coarse engine boundary and fail-closed platform/model checks are in
place. The dense Qwen3 forward implementation is intentionally not represented as
complete: `load_model` returns `model_load_failed` until golden embedding parity passes
for both required checkpoints. See `docs/IMPLEMENTATION_STATUS.md` and
`tests/hardware/run.sh`.

No release artifact may be built while this gate is red.
