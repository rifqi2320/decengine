# Development guide

## Prerequisites

- Rust 1.88 with `rustfmt` and `clippy`
- Python 3.10+
- A C compiler for the independent ABI fixture
- For MLX work: Apple Silicon, macOS 14+, full Xcode, CMake, and Metal toolchain

## Portable workflow

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
make lint
make test
make uat
```

The default Cargo feature is a deliberately named fixture backend. It provides stable,
deterministic integration behavior but does not measure semantic model quality.

## Application development

Run a local development server:

```bash
cargo run -p decengine-cli --features fixture-engine -- \
  serve --engine fixture --port 8787
```

Build the ABI test library and use the real Python CFFI path:

```bash
python tests/native/build_fixture.py
export DECENGINE_LIB_PATH="$PWD/build/test-native/libdecengine.so" # Linux test host
PYTHONPATH=python python examples/support_router.py
```

On macOS the fixture filename is `libdecengine.dylib`.

## Model lifecycle

Explicit network install:

```bash
decengine pull Qwen/Qwen3-Embedding-0.6B
decengine list
```

Offline/local install for controlled environments:

```bash
decengine pull Qwen/Qwen3-Embedding-0.6B --from /path/to/snapshot
```

The source must contain profile-required tokenizer/config files and safetensors weights.
The store verifies its generated checksums every time a production model is resolved.

## Architectural rules

Run `python scripts/check-boundaries.py`. A change fails if MLX symbols leak outside the
engine crate or if inference crates acquire known remote-inference clients. Network access
belongs only to explicit model installation.
