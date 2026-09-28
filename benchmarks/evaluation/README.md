# MLX vs JEV case-by-case evaluation

The ledger is one JSON object per case **per judge per run**, and includes an
explicit judgment for each of the dataset's three questions: `owner`, `urgent`,
and `impact`. The 100 fixture records are `case-001`…`case-100`; each carries
`scenario.industry` and `reference.{owner,urgent,impact,rationale}`. The runner's
raw archive is separate and immutable; ledger output fields contain only
normalized answer values plus archive locators, never raw response payloads.

## 10-agent workflow

After the paired runner completes and its archive is stable:

1. Record its manifest `run_id` and a stable SHA-256 digest for the run/archive
   as `source_run.run_hash`. Runner archives live at
   `results/comparison-runs/<run_id>/cases.jsonl`; one line per case contains
   `case_id`, `input`, and backend objects `decengine` / `jev`, each with
   `normalized_decisions`, `error`, and raw response. Preserve that archive
   untouched. For an output, `archive_ref` should identify the exact file and
   case/backend, e.g. `results/comparison-runs/<run_id>/cases.jsonl#case-001.jev`.
2. Make ten **non-overlapping** chunks of ten fixture IDs (case-001–010, …,
   case-091–100) and give one chunk to each Luna agent. Each assignment includes
   the fixture's industry, reference decision and rationale, and access to both
   archived outputs. Keep candidate IDs and answer normalization visible. A
   judge produces one JSONL record per assigned case, with its own stable
   `judgment_id`, `agent.identity`, actual `agent.session_id`, UTC timestamp,
   and the same `source_run` values on every row.
3. Use `judgment.schema.json` and the record example below. Normalize each
   backend answer in `outputs.<backend>.normalized_predictions` to a scalar:
    owner = exact candidate key; urgent = boolean (`noul` value >= 0.5); impact =
    the fixture level label selected by probability-distribution argmax. For
    decengine, distribution keys are fixture labels. For JEV, probability keys
    are ordinal strings (`"0"`, `"1"`, `"2"`): parse as indices into
    `request.questions.impact.levels`, then use that level's `label`. Do **not**
    use JEV's raw `score` as a level index or rely on `mapped_level`: scores can
    be fractional (e.g. 1.18), leaving `mapped_level` null even when
    probabilities identify a label. If probabilities are absent/malformed,
    contain invalid/out-of-range keys or values, or have a tied maximum, use
    Treat probabilities as valid only if every key maps uniquely to one level,
    every value is a finite number in [0,1], and their sum is within 0.001 of
    1.0; otherwise normalize to null. Keep other malformed/absent/unmappable answers null too. Capture
    reported confidence only when explicitly present on that question's archived
    answer; do not infer confidence from probabilities.
4. Each question gets an independent correctness label and relative outcome.
   The case-level assessment/outcome is a headline judgment of the complete
   three-answer case, not a substitute for question assessments. Include concise
   rationale for each backend/question and each comparison. Missing, errored,
   or null prediction => `indeterminate` correctness for that backend/question;
   `outputs.<backend>.status` is `error` or `missing` and its `error` explains
   the situation. Do not call a missing answer incorrect. If only a single
   question is missing, the other two can still be judged. Use a `null`
   archive locator only if no archive entry exists; record a descriptive
   locator (`missing:<reason>`) since `archive_ref` is intentionally required.
5. Collect chunk JSONLs in a fresh output file (do not overwrite earlier runs):

   ```sh
   cat chunk-*.jsonl > benchmarks/results/mlx-jev-judgments.jsonl
   python benchmarks/evaluation/evaluate.py validate benchmarks/results/mlx-jev-judgments.jsonl
   python benchmarks/evaluation/evaluate.py summary benchmarks/results/mlx-jev-judgments.jsonl
   ```

   To add a single correction/additional judgment later, save one record in a
   JSON file and use `evaluate.py append LEDGER RECORD.json`; it takes an
   exclusive lock and rejects duplicate judgment IDs. Never rewrite a judgment
   silently: use a new ID and explain the superseded one in `notes`. Separate
   per-agent chunk files are preferred while workers are active.

## Adjudication rules

- Compare semantic meaning to the corresponding reference field, using only
  case evidence and the reference rationale. `owner` requires the best initial
  response function, not merely a plausible secondary team. `urgent` means
  same-day action to prevent likely harm, material disruption, or a near-term
  missed deadline; emphatic language alone is not enough. `impact` is the
  highest supported `low`/`moderate`/`high` level under the fixture's criteria.
- Labels: `correct` means the normalized answer matches the intended meaning;
  `partial` is reserved for a multi-part/graded response with material correct
  and incorrect portions (not a near miss on a single categorical answer);
  `incorrect` means a present, judgeable answer is materially wrong;
  `indeterminate` means missing/error/null, ambiguous evidence, or no defensible
  reference-based determination. Cite ambiguity in rationale.
- `questions.<name>.comparison.outcome` is `mlx`/`jev` only when that backend is
  clearly better on that question; `tie` when materially equivalent (including
  both right or both wrong); `indeterminate` when evidence does not permit a
  comparison, including both answers missing. A sole present correct answer
  beats an indeterminate one; a present wrong answer does not automatically
  beat an indeterminate answer—use rationale and tie/indeterminate as warranted.
- `comparison.outcome` is the case-level winner over the three answers. Prefer
  the backend with more correct/partial answers and fewer wrong answers; use
  `tie` when overall quality is equivalent, `indeterminate` when the case cannot
  be fairly compared. Explain mixed question results. `assessment` is each
  backend's case-level correctness (`correct` all three, `partial` mixed
  judgeable performance, `incorrect` judgeable and no correct answers,
  `indeterminate` only when the case cannot be assessed); include rationale.
- Confidence is not correctness. Only enter a probability explicitly reported
  by that backend for that exact scope, basis `reported`; otherwise set null and
  `unavailable`. The runner does not synthesize case-level confidence: the
  case-level assessment must use unavailable/null unless the archive explicitly
  reports case-level confidence. Never copy or average question confidence into
  a case-level value. Single-case confidence is not calibration. Summary Brier scores are calculated
  only for reported-confidence answers labelled binary `correct`/`incorrect`;
  partial and indeterminate judgments are excluded.
- Per-case metadata must match fixture (`industry` and reference fields).
  Preserve exact normalized predictions for transparent per-category scoring.
  Never add raw model responses or credentials to the ledger.

## Record format

`judgment.schema.json` documents the full record. Example abbreviated to one
question only for display; actual record requires all three questions:

```json
{"schema_version":1,"judgment_id":"case-001-luna-ses_example","case_id":"case-001","industry":"healthcare","timestamp":"2026-09-23T12:00:00Z","agent":{"identity":"Luna-1","session_id":"ses_example"},"source_run":{"run_id":"20260923T120000Z","run_hash":"sha256:..."},"reference":{"decision":{"owner":"safety_compliance","urgent":true,"impact":"high"},"rationale":"..."},"outputs":{"mlx":{"archive_ref":"results/comparison-runs/<run>/cases.jsonl#case-001.decengine","status":"ok","normalized_predictions":{"owner":"safety_compliance","urgent":true,"impact":"high"},"error":null},"jev":{"archive_ref":"results/comparison-runs/<run>/cases.jsonl#case-001.jev","status":"ok","normalized_predictions":{"owner":"operations","urgent":false,"impact":"moderate"},"error":null}},"assessment":{"mlx":{"correctness":"correct","confidence":null,"confidence_basis":"unavailable","rationale":"All three match; no case-level confidence is reported."},"jev":{"correctness":"incorrect","confidence":null,"confidence_basis":"unavailable","rationale":"All three differ from reference."}},"comparison":{"outcome":"mlx","rationale":"MLX matches all fields; JEV misses all three."},"questions":{"owner":{"assessment":{"mlx":{"correctness":"correct","confidence":0.8,"confidence_basis":"reported","rationale":"Matches the reference owner."},"jev":{"correctness":"incorrect","confidence":null,"confidence_basis":"unavailable","rationale":"Different owner."}},"comparison":{"outcome":"mlx","rationale":"MLX matches; JEV does not."}},"urgent":{"assessment":{"mlx":{"correctness":"correct","confidence":0.8,"confidence_basis":"reported","rationale":"Same-day response is warranted."},"jev":{"correctness":"incorrect","confidence":null,"confidence_basis":"unavailable","rationale":"Misses urgency."}},"comparison":{"outcome":"mlx","rationale":"MLX matches the reference."}},"impact":{"assessment":{"mlx":{"correctness":"correct","confidence":0.8,"confidence_basis":"reported","rationale":"High is supported."},"jev":{"correctness":"incorrect","confidence":null,"confidence_basis":"unavailable","rationale":"Moderate understates likely harm."}},"comparison":{"outcome":"mlx","rationale":"MLX matches the reference."}}},"notes":""}
```

Summary output gives case winners and, for each question/backend, correctness,
exact accuracy, breakdowns by reference and predicted category and by industry,
reported confidence counts, and Brier score. Keep the fixture count/completeness
check in mind: the summary describes only judgments actually in the ledger.
