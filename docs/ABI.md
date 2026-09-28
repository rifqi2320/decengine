# C ABI contract

The public declarations are in `include/decengine.h`. ABI version 1 exposes six symbols:

- `de_abi_version`
- `de_version`
- `de_engine_create`
- `de_engine_decide_json`
- `de_engine_destroy`
- `de_string_free`
- `de_last_error`

`de_engine_t` is opaque. Inputs are NUL-terminated UTF-8. A successful decision returns a
Rust-allocated UTF-8 string; the caller must release it exactly once with
`de_string_free`. Engine handles are destroyed exactly once with `de_engine_destroy`.
Null destroy/free calls are safe no-ops.

No panic, exception, Rust struct, allocator assumption, MLX type, or GPU buffer may cross
this boundary. A nonzero status is accompanied by a thread-local diagnostic available from
`de_last_error` until the next ABI call on that thread.

The Python package checks `de_abi_version()` before engine creation and fails clearly when
the independently installed library is incompatible.
