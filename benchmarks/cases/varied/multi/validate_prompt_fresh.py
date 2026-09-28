#!/usr/bin/env python3
"""Mechanical checks for the independently authored fresh multi-question holdout."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "test-prompt-fresh-60.jsonl"
MANIFEST = HERE / "test-prompt-fresh-60.manifest.json"
TRAIN_DOMAINS = {
    "arts colleges", "coastal research", "community clinics", "community energy",
    "credit unions", "food cooperatives", "municipal permitting", "museum conservation",
    "public libraries", "regional farms", "regional transit", "small manufacturers",
}
PRIOR = [HERE / name for name in (
    "train-120.jsonl", "test-60.jsonl", "test-matched-60.jsonl", "test-explicit-60.jsonl"
)]
FAMILIES = {f"qf-{n:08x}" for n in range(0x29, 0x33)}
Q_NAMES = ("response_plan", "immediate_intervention", "record_auditability")
STOP = set("a an and are as at be by can current for from has have in is it of on or the to with".split())


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP


def similarity(a, b):
    return len(a & b) / max(1, len(a | b))


def validate(write_manifest=False):
    raw = FIXTURE.read_bytes()
    rows = read_rows(FIXTURE)
    assert len(rows) == 60
    domains, families, target_positions, boolean_values, option_counts = Counter(), Counter(), Counter(), Counter(), Counter()
    state_texts, prompt_texts, positions = [], [], []
    choice_outcomes, score_outcomes = Counter(), Counter()
    for index, row in enumerate(rows, 1):
        assert set(row) == {"id", "domain", "request", "reference", "metadata"}
        assert row["id"] == f"mq-fresh-{index:03d}"
        assert row["domain"] in TRAIN_DOMAINS
        domains[row["domain"]] += 1
        assert set(row["request"]) == {"state", "questions"}
        assert set(row["request"]["questions"]) == set(Q_NAMES)
        assert set(row["reference"]) == set(Q_NAMES)
        assert set(row["metadata"]) == {"evidence_map"}
        assert isinstance(row["metadata"]["evidence_map"], dict)
        assert "evidence_map" not in row["request"]
        state = row["request"]["state"]
        assert set(state) == {"service_area", "asset", "incident_note", "operational_status", "response_authority", "decision_rules", "record_facts", "audit_measure"}
        assert state["service_area"] == row["domain"]
        state_texts.append(json.dumps(state, sort_keys=True, ensure_ascii=False))
        for name in Q_NAMES:
            question, reference = row["request"]["questions"][name], row["reference"][name]
            assert set(reference) == {"target", "rationale", "question_family_id"}
            family = reference["question_family_id"]
            assert family in FAMILIES
            families[family] += 1
            assert len(reference["rationale"]) >= 30
            prompt = question["prompt"]
            assert len(prompt) >= 30
            prompt_texts.append(prompt)
            target = reference["target"]
            if name == "response_plan":
                assert question["type"] == "choice" and set(question) == {"type", "prompt", "options"}
                options = question["options"]
                assert 2 <= len(options) <= 5 and set(target) == {"selected"}
                option_counts[len(options)] += 1
                assert target["selected"] in options
                pos = list(options).index(target["selected"])
                positions.append(pos)
                target_positions[f"{len(options)}:{pos}"] += 1
                choice_outcomes[target["selected"]] += 1
            elif name == "immediate_intervention":
                assert question["type"] == "noul" and set(question) == {"type", "prompt"}
                assert set(target) == {"value"} and type(target["value"]) is bool
                boolean_values[target["value"]] += 1
            else:
                assert question["type"] == "score" and set(question) == {"type", "prompt", "levels"}
                levels = question["levels"]
                assert 3 <= len(levels) <= 5
                values = [level["value"] for level in levels]
                assert all(type(v) is int for v in values)
                assert all(a < b for a, b in zip(values, values[1:]))
                assert set(target) == {"label", "value"}
                assert any(level["label"] == target["label"] and level["value"] == target["value"] for level in levels)
                score_outcomes[target["label"]] += 1
    assert domains == Counter({domain: 5 for domain in TRAIN_DOMAINS}), domains
    assert families == Counter({family: 18 for family in FAMILIES}), families
    assert boolean_values == Counter({True: 30, False: 30}), boolean_values
    assert option_counts == Counter({2: 15, 3: 15, 4: 15, 5: 15}), option_counts
    for size in range(2, 6):
        counts = [target_positions.get(f"{size}:{position}", 0) for position in range(size)]
        assert max(counts) - min(counts) <= 1, f"answer-position imbalance for {size} options: {counts}"
    assert len(set(state_texts)) == 60
    assert len(set(prompt_texts)) == len(prompt_texts), "duplicate prompt in fresh fixture"

    # Mechanical-only checks: old fixture text is read for comparisons, never emitted.
    old_rows = [(path, read_rows(path)) for path in PRIOR]
    prior_families = set()
    prior_states, prior_prompts = [], []
    for _, old in old_rows:
        for row in old:
            prior_states.append(tokens(json.dumps(row["request"]["state"], ensure_ascii=False)))
            for name, question in row["request"]["questions"].items():
                prior_prompts.append(tokens(question["prompt"]))
            prior_families.update(v["question_family_id"] for v in row["reference"].values())
    assert not FAMILIES & prior_families, FAMILIES & prior_families
    fresh_state_tokens = [tokens(value) for value in state_texts]
    fresh_prompt_tokens = [tokens(value) for value in prompt_texts]
    max_state, max_prompt = 0.0, 0.0
    for new in fresh_state_tokens:
        for old in prior_states:
            score = similarity(new, old)
            max_state = max(max_state, score)
            assert score < 0.72, f"near-duplicate state found (Jaccard {score:.3f})"
    for new in fresh_prompt_tokens:
        for old in prior_prompts:
            score = similarity(new, old)
            max_prompt = max(max_prompt, score)
            assert score < 0.72, f"near-duplicate prompt found (Jaccard {score:.3f})"

    digest = hashlib.sha256(raw).hexdigest()
    manifest = {
        "fixture": FIXTURE.name,
        "sha256": digest,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in old_rows},
        "case_count": len(rows),
        "questions_per_case": 3,
        "domains": dict(sorted(domains.items())),
        "question_families": dict(sorted(families.items())),
        "answer_positions_by_option_count": dict(sorted(target_positions.items())),
        "boolean_outcomes": {str(k).lower(): v for k, v in sorted(boolean_values.items())},
        "choice_outcomes": dict(sorted(choice_outcomes.items())),
        "score_outcomes": dict(sorted(score_outcomes.items())),
        "maximum_prior_state_token_jaccard": round(max_state, 4),
        "maximum_prior_prompt_token_jaccard": round(max_prompt, 4),
        "prior_fixtures_compared": [path.name for path, _ in old_rows],
        "annotation_note": "Labels and evidence maps are authored synthetic judgments, not externally adjudicated ground truth. Evidence maps are metadata and excluded from request/inference.",
    }
    if write_manifest:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = validate(write_manifest="--write-manifest" in sys.argv)
    print(f"valid: {result['case_count']} cases; sha256={result['sha256']}")
    print("domains/families/positions and mechanical overlap checks passed")
