#!/usr/bin/env python3
"""Structural and mechanical overlap checks for the multi-question training set."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("train-120.jsonl")
FAMILY = re.compile(r"^qf-000000(0[6-9]|1[0-4])$")
ID = re.compile(r"^mq-train-(\d{3})$")


def tokens(value):
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def states(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["request"]["state"] for line in f if line.strip()]


def validate(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 120, f"expected 120 cases; found {len(rows)}"
    ids, domains, types, families, state_texts = set(), Counter(), Counter(), Counter(), []
    for i, row in enumerate(rows, 1):
        assert set(row) == {"id", "domain", "request", "reference"}, f"case {i}: invalid keys"
        match = ID.fullmatch(row["id"])
        assert match and int(match.group(1)) == i and row["id"] not in ids, f"case {i}: bad ID/order"
        ids.add(row["id"])
        assert isinstance(row["domain"], str) and row["domain"].strip()
        domains[row["domain"]] += 1
        req, ref = row["request"], row["reference"]
        assert set(req) == {"state", "questions"} and isinstance(req["state"], dict)
        assert set(req["questions"]) == {"response_plan", "immediate_intervention", "record_auditability"}
        assert set(ref) == set(req["questions"])
        state_texts.append(json.dumps(req["state"], ensure_ascii=False))
        for name, q in req["questions"].items():
            r = ref[name]
            assert set(r) == {"target", "rationale", "question_family_id"}
            assert FAMILY.fullmatch(r["question_family_id"]), f"case {i}: family out of range"
            assert len(r["rationale"].strip()) >= 20
            families[r["question_family_id"]] += 1
            assert isinstance(q.get("prompt"), str) and len(q["prompt"]) >= 20
            kind, target = q.get("type"), r["target"]
            types[kind] += 1
            if name == "response_plan":
                assert kind == "choice" and set(q) == {"type", "prompt", "options"}
                opts = q["options"]
                assert 2 <= len(opts) <= 5 and set(target) == {"selected"} and target["selected"] in opts
            elif name == "immediate_intervention":
                assert kind == "noul" and set(q) == {"type", "prompt"}
                assert set(target) == {"value"} and type(target["value"]) is bool
            else:
                assert kind == "score" and set(q) == {"type", "prompt", "levels"}
                lev = q["levels"]
                assert 3 <= len(lev) <= 5
                vals = [x["value"] for x in lev]
                assert all(a < b for a, b in zip(vals, vals[1:]))
                assert set(target) == {"label", "value"}
                assert any(x["label"] == target["label"] and x["value"] == target["value"] for x in lev)
    assert len(set(state_texts)) == 120, "duplicate states in fixture"
    assert types == {"choice": 120, "noul": 120, "score": 120}, types

    # Read held-out/train fixtures only for token-based overlap comparison; do not
    # print or otherwise expose their content.
    new_sets = [tokens(x) for x in state_texts]
    comparisons = {}
    for filename in ("test-200.jsonl", "train-400.jsonl"):
        old = states(ROOT / filename)
        old_sets = [tokens(json.dumps(x, ensure_ascii=False)) for x in old]
        max_jaccard = 0.0
        for a in new_sets:
            for b in old_sets:
                score = len(a & b) / max(1, len(a | b))
                max_jaccard = max(max_jaccard, score)
                assert score < 0.72, f"near-duplicate state detected against {filename} (Jaccard {score:.2f})"
        comparisons[filename] = max_jaccard
    return rows, domains, types, families, comparisons


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    rows, domains, types, families, overlap = validate(path)
    print(f"valid: {len(rows)} cases / {sum(types.values())} questions")
    print("question types:", dict(types))
    print("domains:", dict(domains))
    print("family distribution:", dict(sorted(families.items())))
    print("maximum state-token Jaccard overlap:", {k: round(v, 3) for k, v in overlap.items()})
