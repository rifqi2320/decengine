# decengine v1.0 — Product Requirements

Status: frozen input specification. Implementation status is tracked separately in
`IMPLEMENTATION_STATUS.md`.

## Product

`decengine` lets Python applications make narrow, typed semantic decisions locally on
Apple Silicon. It accepts application state and one or more questions and returns one of
three result types:

- `Choice`: one candidate, a candidate distribution, and directional confidence.
- `Noul`: support for a proposition as a value in `[0, 1]`.
- `Score`: an expected ordinal score, level distribution, legend, and confidence.

It does not generate text, execute tools, train models, call cloud inference, or silently
fall back to a remote provider.

## Users and interfaces

The primary user is a Python developer on an Apple-Silicon Mac. The primary interface is
an in-process pure-Python wrapper:

```text
Python → CFFI ABI mode → extern C → libdecengine.dylib → Rust core
```

The Rust library is installed independently. Installing the Python package must not invoke
Cargo, Xcode, MLX, or a native compiler.

The secondary interface is `decengine serve`, exposing:

- `POST /v1/systemone`
- `GET /v1/models`
- `GET /healthz`

Python and HTTP must converge on the same Rust decision implementation.

## Platform and models

v1 targets macOS on Apple Silicon and exactly two release-blocking checkpoints:

- `Qwen/Qwen3-Embedding-0.6B`
- `microsoft/harrier-oss-v1-0.6b`

Both use a dense Qwen3 0.6B embedding path, model-profile-controlled query formatting,
last non-padding token pooling, and L2 normalization. Arbitrary fine-tunes are not a v1
promise.

## Runtime promises

- MLX is accessed through exactly pinned `mlx-rs`.
- Tensor-heavy forward, pooling, normalization, similarity, temperature, softmax, score
  expectation, and simple confidence calculations stay in the engine/GPU where practical.
- Candidate representations may stay GPU-resident and are keyed by every semantic input
  that could change their vectors.
- After explicit model installation, inference is completely offline.
- Telemetry is absent and no cloud fallback exists.
- Confidence is deterministic, bounded, and directional; it is not `P(correct)`.
- The server binds only to loopback by default.

## Required CLI

```text
decengine serve
decengine pull <model>
decengine list
decengine rm <model>
```

## Distribution

The minimum release is a source-independent Apple-Silicon archive containing:

- `bin/decengine`
- `lib/libdecengine.dylib`
- `include/decengine.h`
- Python wrapper/install guidance
- checksums, licenses, and supported-model metadata

The Python and native lifecycles remain independent.

## Acceptance criteria

A v1 release requires:

1. Both checkpoints load through the same `mlx-rs` Qwen3 implementation.
2. Tokenization, pooling, normalization, embeddings, rankings, and similarity matrices
   meet documented reference tolerances for both models.
3. Choice, Noul, Score, multi-question, English, Indonesian, clear, and ambiguous fixtures
   pass for both models.
4. Python CFFI, HTTP wire, offline inference, cache reuse, and local-error paths pass.
5. No tensor-heavy avoidable CPU scoring path or accidental host materialization is found.
6. Latency and peak unified-memory benchmarks are published.

## Non-goals

Android, iOS, Linux/Windows product distribution, generation, agents, workflows, training,
fine-tuning, reranking, vector databases, ANN, arbitrary architectures, external inference,
vLLM, calibrated correctness guarantees, OOD guarantees, and behavioral equivalence with
proprietary System One models are deferred.
