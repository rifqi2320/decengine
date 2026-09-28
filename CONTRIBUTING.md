# Contributing

Use Rust 1.88 and Python 3.10 or newer. Before a pull request:

```bash
make lint
make test
make uat
```

Changes to compiler templates, pooling, temperature, model profiles, tokenizer behavior,
or `mlx-rs` require the Apple hardware suite for both supported models. Include the
generated test report in the pull request, but do not commit benchmark output, model
weights, shared libraries, or build directories.

Public API changes require an entry in `CHANGELOG.md`. C ABI changes additionally require
an ABI version bump and an update to `include/decengine.h`.
