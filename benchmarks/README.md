# Benchmarks

Hardware release benchmarks must report model load, cold/warm decision, single and
multi-question batches, cache hit/miss, tokenization, GPU execution, CFFI and HTTP overhead,
and peak unified memory for both required checkpoints.

Write generated results under `benchmarks/results/`; that directory is ignored because
benchmark output is a release artifact, not source. Publish the reviewed report alongside
the release instead.
