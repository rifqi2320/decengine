# Test strategy

Testing separates deterministic software correctness from hardware/model quality.

| Suite | Environment | Claims |
|---|---|---|
| Rust unit tests | Portable | Types, compiler, cache, store, runtime, ABI, routes, CLI |
| Python unit tests | Portable | Public API, validation, CFFI calls, ownership, exceptions |
| Developer UAT | Portable | A user can integrate and branch on typed results |
| HTTP black-box UAT | Portable with Rust | SDK-like wire client can use all routes/results |
| Model parity | Apple Silicon | Embeddings agree with official references |
| Decision regression | Apple Silicon | Both models preserve frozen semantic fixtures |
| Residency/performance | Apple Silicon | No avoidable host tensor path; latency/memory measured |

## Commands

```bash
make test
make uat
python scripts/check-boundaries.py
```

Hardware suite (explicit local models, no implicit download):

```bash
DECENGINE_QWEN_DIR=/path/to/qwen \
DECENGINE_HARRIER_DIR=/path/to/harrier \
bash tests/hardware/run.sh
```

## CFFI test fidelity

Python tests compile a tiny C library implementing `include/decengine.h`, then use the
production `ffi.dlopen` wrapper. This catches ABI declarations, status mappings, pointer
ownership, context management, JSON decoding, and typed results without linking a Rust
crate into Python. Rust ABI tests separately call the exported functions around the real
shared core.

## UAT persona

The acceptance persona is an application developer routing a support ticket. They install
only the wrapper, point it at an independently installed library, submit all three question
types, branch on typed results, repeat a schema to observe a cache hit, and close the engine.

## What a passing portable suite does not claim

It does not claim that Qwen or Harrier load, that embeddings match references, that work
stays GPU-resident, or that System One proprietary behavior is equivalent. Those claims
belong only to the target hardware gates.
