# Decision case dataset (v1)

`decision-cases-100.jsonl` is a deterministic, UTF-8 JSON Lines fixture: exactly
100 independent cases, one JSON object per line, with stable IDs `case-001` through
`case-100`. It is intended to provide the same request inputs to the MLX engine and
the real JEV API for a paired comparison. Keep inference inputs identical between
backends; the `reference` and `metadata` fields are evaluation annotations, not part
of the engine request.

## Record shape

- `id`: stable unique fixture ID.
- `scenario`: industry, case summary, and a coarse `standard` / `edge_case` tag.
- `request`: the `systemone-v1` decision request shape (`model`, `state`, and
  `questions`). Its `state` is descriptive JSON, and its questions use the engine's
  supported `choice`, `noul`, and `score` variants. The shared question wording,
  candidate keys/descriptions, and score levels are held constant across cases to
  support like-for-like model comparison.
- `reference`: intended `owner` candidate, `urgent` boolean, `impact` level, and a
  concise human-authored rationale. Urgent means same-day action to prevent likely
  harm, material disruption, or a near-term missed deadline; emphatic wording alone
  is insufficient. Impact levels correspond to the criteria in the score question.
- `metadata`: authorship/provenance and interpretation caveat.

## Coverage and interpretation

Ten industries are represented with ten cases each: healthcare, financial services,
retail/e-commerce, transport/logistics, education, manufacturing, energy/utilities,
software/SaaS, hospitality, and public services. References contain 46 urgent and
54 non-urgent cases; impact labels are low 41, moderate 22, high 37. Owner labels are
naturally less balanced (some functions are more common in these scenarios): billing
16, operations 22, safety/compliance 22, technical support 20, sales 9, account access
8, other 3. Do not treat raw accuracy across owner labels as a balanced-class score;
report per-class results or use a clearly stated class-balanced metric as well.

Cases are original, fictionalized scenarios grounded in common operational patterns,
not sourced incident reports, verified facts, or legal/clinical/safety advice. The
reference is a benchmark target, not an authoritative disposition. Several cases
intentionally hinge on scope, available workaround, stated deadline, or separating
an actual hazard from a routine inquiry. Review references with domain owners before
using results to make consequential deployment decisions. This set evaluates
decision behavior on shared text inputs; it does not by itself establish that either
backend's scores/probabilities are calibrated or that backend-specific preprocessing
is equivalent.


## Held-out decision set

`decision-cases-holdout.jsonl` contains 60 independently authored fictional scenarios
with IDs `holdout-001` through `holdout-060`. It preserves the exact `request.questions`
wording, candidate sets, and score levels from the v1 fixture so either backend can score
it without semantic changes. The fixture has its own owner/urgency/impact references and
rationales; treat them as evaluation annotations only. Keep this file held out from prompt
tuning and use it only for the locked evaluation. Labels are urgency 27 yes / 33 no and
impact low 21 / moderate 13 / high 26; owner counts are billing 9, operations 11,
safety/compliance 16, technical support 13, account access 3, sales 3, and other 5.
Owner prevalence reflects the fictional scenarios and should not be interpreted as a
balanced-class sample.
