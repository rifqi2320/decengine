# Qwen3-Embedding-0.6B prompt experiment

This experiment uses the real installed Qwen checkpoint through the release MLX
backend. It does not call JEV. The baseline run is compared to the archived Qwen
predictions and to already archived JEV judgments; those archived files are read-only
references, not rerun.

## Reproduce

From the repository root on Apple Silicon macOS, with the local model installed at
the model-store path below:

```sh
cargo test -p decengine-models -p decengine-compiler
cargo build --release -p decengine-ffi --no-default-features --features mlx
PYTHONPATH=python:results uv run --python 3.14 python results/qwen_prompt_experiment.py
```

The runner verifies the fixture against archived inputs, records dataset/archive,
profile, and installed-record SHA-256 values, writes every raw response and score
distribution to a variant JSONL, then writes per-field exact accuracy and full error
lists to `summary.json`. Use `--variant <name>` to run one profile or `--limit N` only
for an explicitly non-final smoke test. `--help` lists flags. No JEV client is called.

## Run and findings

The reproducible full run is in
`results/qwen-prompt-experiments/20260923T085222704948Z/` (metadata and one JSONL
per variant). The 100 cases were split by fixed ID: 001-050 development, 051-100
holdout. The baseline on holdout was 69/150 fields (46.0%); the direct binary rubric
variant was 75/150 (50.0%), +6 fields, all six in urgency (28/50 vs 22/50). Owner
remained 23/50 and impact remained 24/50. The spaced `Query: ` format scored 70/150;
the label-only binary phrasing 71/150; prefixing candidates with `Passage:` scored
48/150. On development, baseline and direct rubric tied at 72/150; the label-only
phrasing scored 73/150. Thus the direct rubric did not beat baseline on the
development split.

For context on the same holdout cases, archived Qwen baseline is 69/150 (46.0%) and
archived JEV is 127/150 (84.7%; per field owner 40/50, urgent 47/50, impact 40/50).
Across all 100 cases, archived JEV is 258/300 (86.0%), versus Qwen baseline 141/300
(47.0%). These are exact-match descriptive metrics on this small synthetic fixture,
not calibrated confidence or deployment-suitability claims.

**Selection caveat:** every fixed variant was run over all 100 cases, and the holdout
numbers have now been inspected. The holdout therefore is no longer an untouched
selection set for future iterations. The experimental direct-rubric config is
preserved as opt-in only because it showed an exploratory holdout urgency gain; the
development tie and small test size do not establish generalization. Validate on a
new, independently labeled set before any production adoption. Temperature remains
at the manifest 0.05 and the Noul threshold remains 0.5; no calibration was attempted.

## Opt-in configuration and code changes

`models/profiles/qwen3-embedding-0.6b-direct-rubric.experimental.json` records the
experimental settings; it does not replace the shipped Qwen manifest or create a
new installed model identity. The native options now accept optional query/candidate
templates, Noul candidate texts, and temperature; omitted Noul strings default to
the original exact wording. The compiler applies Noul candidates from its profile,
and candidate content remains included in cache keys. The experiment passes these
overrides only when running its named variant. Harrier files and its runner were not
edited.
