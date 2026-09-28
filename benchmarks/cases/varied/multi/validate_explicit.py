#!/usr/bin/env python3
"""Integrity and leakage checks for the authored explicit-evidence variant."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "test-matched-60.jsonl"
FIXTURE = HERE / "test-explicit-60.jsonl"
MANIFEST = HERE / "test-explicit-60.manifest.json"


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


source_bytes = SOURCE.read_bytes()
source = read(SOURCE)
rows = read(FIXTURE)
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
assert hashlib.sha256(source_bytes).hexdigest() == manifest["source_sha256"], "source fixture hash changed"
assert len(source) == len(rows) == 60
assert manifest["cases"] == 60 and manifest["question_count"] == 180
assert manifest["source_fixture"] == SOURCE.name

expected_map = {
    "response_plan": "request.state.explicit_assessment_facts.response_plan.decision_evidence",
    "immediate_intervention": "request.state.explicit_assessment_facts.immediate_intervention.authority_evidence",
    "record_auditability": "request.state.explicit_assessment_facts.record_auditability.record_evidence",
}
for i, (old, new) in enumerate(zip(source, rows), 1):
    assert new["id"] == f"mq-explicit-{i:03d}"
    assert old["id"] == f"mq-test-matched-{i:03d}"
    assert old["domain"] == new["domain"]
    assert old["request"]["questions"] == new["request"]["questions"], f"case {i}: question changed"
    assert old["reference"] == new["reference"], f"case {i}: target/reference changed"
    assert set(new) == {"id", "domain", "request", "reference", "evaluation_metadata"}
    assert set(new["request"]["state"]) == set(old["request"]["state"]) | {"explicit_assessment_facts"}
    assert new["evaluation_metadata"] == {"evidence_map": expected_map}
    facts = new["request"]["state"]["explicit_assessment_facts"]
    assert set(facts) == set(expected_map)
    for name, fact in facts.items():
        assert len(fact) == 1
        fact_text = " ".join(fact.values())
        rationale = old["reference"][name]["rationale"]
        a, b = words(fact_text), words(rationale)
        assert len(a & b) / max(1, len(a | b)) < 0.55, f"case {i}/{name}: rationale overlap too high"
        assert rationale.casefold() not in fact_text.casefold(), f"case {i}/{name}: rationale copied"
    choice_fact = facts["response_plan"]["decision_evidence"].casefold()
    selected = old["reference"]["response_plan"]["target"]["selected"]
    selected_text = old["request"]["questions"]["response_plan"]["options"][selected]
    assert selected_text.casefold() not in choice_fact, f"case {i}: selected option copied into evidence"
    score_fact = facts["record_auditability"]["record_evidence"].casefold()
    score_label = old["reference"]["record_auditability"]["target"]["label"]
    assert score_label.casefold() not in score_fact, f"case {i}: score label copied into evidence"
    assert type(old["reference"]["immediate_intervention"]["target"]["value"]) is bool

assert manifest["transformed_case_ids"] == [r["id"] for r in rows]
assert manifest["source_case_ids"] == [r["id"] for r in source]
assert {r["domain"] for r in rows} == {r["domain"] for r in source}
assert len({r["reference"][q]["question_family_id"] for r in rows for q in r["reference"]}) == 10
print("explicit fixture integrity checks passed: 60 cases / 180 unchanged questions and targets")
