# Golden model fixtures

Release-blocking golden files are intentionally not fabricated in the source repository.
Generate one JSON file per model from the approved official reference implementation:

- `Qwen--Qwen3-Embedding-0.6B.json`
- `microsoft--harrier-oss-v1-0.6b.json`

Each fixture must record reference package versions, checkpoint revision and file hashes,
token IDs, attention masks, pooled normalized embeddings (or approved digest/sample), norms,
pairwise rankings, and similarity matrices for short English, Indonesian, long,
multilingual, mixed-padding, query-instruction, and candidate/document cases.

Fixtures are test evidence, not model weights. Review numerical tolerance from measured
BF16/MLX behavior; never choose a tolerance merely to make a failing port pass.

The hardware gate first loads each configured checkpoint through `MlxDecisionEngine` and
then requires its golden fixture. File presence is only the harness bootstrap; replace the
bootstrap assertion with numerical comparisons as part of the same change that implements
the forward path. The red release marker must remain until those comparisons exist and pass.
