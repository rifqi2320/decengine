# Probe test set provenance

`probe-test-90.jsonl` is a newly authored, fictional held-out set containing 90
cases (`test-001` through `test-090`). Each case has its own scenario and
human-written owner, urgency, and impact reference with a concise rationale.
The cases cover ten industries evenly (nine per industry) and reuse the shared
decision request question wording and candidate definitions from the original
benchmark. All scenarios are synthetic and should not be treated as verified
incidents or authoritative operational, clinical, legal, or safety guidance.

## Coverage

- Owner: account_access 10, billing 10, operations 11, safety_compliance 28,
  sales 9, technical_support 12, other 10.
- Urgency: yes 31, no 59.
- Impact: low 23, moderate 37, high 30.
- Industries: healthcare, financial services, retail/e-commerce,
  transport/logistics, education, manufacturing, energy/utilities, software/SaaS,
  hospitality, and public services (nine cases each).

This is a held-out evaluation set. Do not inspect its cases for prompt, training,
or hyperparameter tuning; leave it untouched until the training approach and
hyperparameters are frozen. Do not run model/API inference on it during
development. Evaluate only after the approach is locked.

## Validation and duplicate screening

The 90 records were parsed as JSONL, checked for sequential unique IDs and
reference labels, and validated using the public `Choice`, `Noul`, and `Score`
question constructors against the shared request shape. A normalized exact-text
and sequence-similarity screen was run against the original 100-case and
60-case fixtures and within this test set: no exact duplicate or pair with a
`SequenceMatcher` ratio of 0.86 or greater was found. The screen is mechanical
and does not establish semantic independence; cases were independently
composed to avoid repeating or paraphrasing original scenarios.
