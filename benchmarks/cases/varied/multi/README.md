# Multi-question training cases

`train-120.jsonl` contains 120 independent fictional state records, with three
questions per shared state: one `choice`, one `noul`, and one `score`. The top-level
record shape is:

```json
{
  "id": "mq-train-001",
  "domain": "...",
  "request": {
    "state": {},
    "questions": {
      "response_plan": {"type": "choice", "prompt": "...", "options": {}},
      "immediate_intervention": {"type": "noul", "prompt": "..."},
      "record_auditability": {"type": "score", "prompt": "...", "levels": []}
    }
  },
  "reference": {
    "response_plan": {"target": {"selected": "..."}, "rationale": "...", "question_family_id": "qf-00000006"},
    "immediate_intervention": {"target": {"value": false}, "rationale": "...", "question_family_id": "qf-00000007"},
    "record_auditability": {"target": {"label": "...", "value": 0}, "rationale": "...", "question_family_id": "qf-00000008"}
  }
}
```

Choice sets contain 2–5 options; score rubrics contain 3–5 ordered levels. Family
IDs are in `qf-00000006` through `qf-00000014`; each question construct rotates
across three family IDs. All answer-bearing details are grounded in `request.state`;
references and rationales are adjudication metadata, not inference input.

Run `python3 benchmarks/cases/varied/multi/validate.py` to check the schema, targets,
IDs, family range, case/question counts, distinct states, and mechanical token-Jaccard
overlap against the sibling varied train/test fixtures. The checker reads those
comparison fixtures only programmatically and does not print their contents.

## Explicit-evidence intervention

`test-explicit-60.jsonl` is a new, label-informed easiness intervention matched to
`test-matched-60.jsonl`; the frozen matched fixture is not modified. Case IDs are
`mq-explicit-001` through `mq-explicit-060`. Each state adds per-question
`explicit_assessment_facts`; `evaluation_metadata.evidence_map` points to the
corresponding fact without changing prompts, candidates/rubrics, targets, rationales,
or family IDs. The manifest records the source hash and transformation caveats.

This is not an independent holdout and must not be used to claim fair generalization
gains: authors saw original reference targets/rationales to construct the added
evidence. It is an authored clarity condition, not naturalistic ground truth. Run
`python3 benchmarks/cases/varied/multi/validate_explicit.py` to check source linkage,
question/target invariance, fact mappings, and mechanical rationale/answer-string
leakage safeguards.
