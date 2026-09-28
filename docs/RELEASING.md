# Release procedure

No release is permitted while `docs/IMPLEMENTATION_STATUS.md` contains the red MLX gate.

After that gate is resolved:

1. Run portable lint, unit, contract, and UAT suites.
2. Run both checkpoint parity/regression suites on Apple Silicon with networking disabled.
3. Run residency, latency, CFFI overhead, HTTP overhead, and unified-memory benchmarks.
4. Build with `--no-default-features --features mlx`; verify no fixture strings or features
   are present.
5. Build `decengine`, `libdecengine.dylib`, and install header.
6. Run Python UAT against the release dylib, then an approved System One SDK against the
   release server.
7. Package, sign, notarize if applicable, generate SHA-256 checksums, and attach the
   supported-model matrix and benchmark report.

The Python wheel is pure Python and is published independently. It must not contain the
native library or any Rust build hook.
