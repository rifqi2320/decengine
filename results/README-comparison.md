# JEV / decengine comparison runner

`compare_jev.py` reads `benchmarks/cases/decision-cases-100.jsonl` by default.
Each line retains the dataset worker's complete case, including `scenario`,
`reference`, and `metadata`; the execution request is nested at `request`:

```json
{"id":"case-001","request":{"model":"Qwen/Qwen3-Embedding-0.6B","state":{"context":"..."},"questions":{"urgent":{"type":"noul","prompt":"..."}}},"reference":{"urgent":true},"metadata":{"source_type":"original_synthetic_case"}}
```

The runner uses the request model unless `--model` overrides it. The native
engine is explicitly selected as MLX with `options={"engine":"mlx",
"model_store":"..."}`. JEV calls use TypeSafe's documented
`POST https://api.typesafe.ai/v1/systemone` contract: Bearer authorization,
JSON body with string `state`, JEV `model`, and typed questions. State is sent as
stable compact JSON text. Choice criteria map directly; Noul prompts map to
JEV instructions. Ordered Score levels map to JEV's ordered textual criteria
(`label: criterion (value=number)`) because JEV scores an ordinal scale rather
than decengine's custom numeric values. Normalized JEV scores include the source
level mapping when possible, in addition to the raw ordinal and probabilities.

## Validation and execution

Use the supplied benchmark venv and run dry validation first:

```bash
PYTHONPATH=python /private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-benchmark-venv/bin/python \
  results/compare_jev.py --dry-run
```

Dry-run validates all 100 cases and does not read the key, initialize MLX, or
contact JEV. The default real-run native library is `target/release/libdecengine.dylib`;
the model store defaults to
`/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store`.
These can be overridden with `--native-library` and `--model-store`.

To run one explicit live smoke case (not the bulk dataset):

```bash
PYTHONPATH=python /private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-benchmark-venv/bin/python \
  results/compare_jev.py --limit 1
```

The default key is read from `~/Documents/Code/jev.api.key` only during a live
run. The key and Authorization header are never printed or persisted; persisted
records are additionally scrubbed for any key echo. A run writes to a unique
`results/comparison-runs/<UTC timestamp>/` directory:

- `metadata.json`: run configuration and environment, without credentials
- `cases.jsonl`: complete original fixture case (including all annotations),
  exact transmitted JEV payload without auth, raw JEV response, decengine
  response, normalized decisions, timings, and errors

The 100-case default is enforced unless `--limit` is explicitly supplied.
Calls are sequential and errors are recorded per engine per case. The smoke
command is intentionally limited to one case; orchestrator coordination is
required before the full dataset run.

Smoke validation completed against `case-001` with both decengine MLX and JEV
successful (HTTP 200): `results/comparison-runs/20260923T082046349428Z/`.
The artifact was checked for the expected response objects, transmitted JEV
payload (without auth), preserved fixture annotations, and absence of the key.
