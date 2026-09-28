#!/usr/bin/env python3
"""Mechanical integrity and corpus-overlap checks for the sealed LoRA holdout."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "test-lora-fresh-60.jsonl"
MANIFEST = HERE / "test-lora-fresh-60.manifest.json"
DOMAINS = {
    "arts colleges", "coastal research", "community clinics", "community energy",
    "credit unions", "food cooperatives", "municipal permitting", "museum conservation",
    "public libraries", "regional farms", "regional transit", "small manufacturers",
}
PRIOR = [HERE / name for name in (
    "train-120.jsonl", "test-60.jsonl", "test-matched-60.jsonl",
    "test-explicit-60.jsonl", "test-prompt-fresh-60.jsonl",
)] + [HERE.parents[1] / "probe-train-300.jsonl"]
FAMILIES = {f"qf-{n:08x}" for n in range(0x33, 0x3D)}
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
    domains, families, positions = Counter(), Counter(), Counter()
    booleans, score_values, option_counts, score_level_counts = Counter(), Counter(), Counter(), Counter()
    state_texts, prompts = [], []
    for i, row in enumerate(rows, 1):
        assert set(row) == {"id", "domain", "request", "reference", "metadata"}
        assert row["id"] == f"mq-lora-fresh-{i:03d}"
        assert row["domain"] in DOMAINS
        domains[row["domain"]] += 1
        assert set(row["request"]) == {"state", "questions"}
        assert set(row["request"]["questions"]) == set(Q_NAMES)
        assert set(row["reference"]) == set(Q_NAMES)
        assert set(row["metadata"]) == {"evidence_map"}
        assert set(row["metadata"]["evidence_map"]) == set(Q_NAMES)
        state = row["request"]["state"]
        assert state["service_area"] == row["domain"]
        for key in ("incident_note", "operational_status", "decision_rules", "response_authority", "record_facts", "follow_up"):
            assert isinstance(state[key], str) and len(state[key]) >= 30
        state_texts.append(json.dumps(state, sort_keys=True, ensure_ascii=False))
        for name in Q_NAMES:
            question, reference = row["request"]["questions"][name], row["reference"][name]
            assert set(reference) == {"target", "rationale", "question_family_id"}
            assert reference["question_family_id"] in FAMILIES
            families[reference["question_family_id"]] += 1
            assert len(reference["rationale"]) >= 30
            prompts.append(question["prompt"])
            if name == "response_plan":
                assert question["type"] == "choice" and set(question) == {"type", "prompt", "options"}
                options, target = question["options"], reference["target"]
                assert 2 <= len(options) <= 5 and set(target) == {"selected"}
                option_counts[len(options)] += 1
                assert target["selected"] in options
                pos = list(options).index(target["selected"])
                positions[f"{len(options)}:{pos}"] += 1
            elif name == "immediate_intervention":
                assert question["type"] == "noul" and set(question) == {"type", "prompt"}
                assert set(reference["target"]) == {"value"} and type(reference["target"]["value"]) is bool
                booleans[reference["target"]["value"]] += 1
            else:
                assert question["type"] == "score" and set(question) == {"type", "prompt", "levels"}
                levels = question["levels"]
                assert 3 <= len(levels) <= 5
                score_level_counts[len(levels)] += 1
                values = [level["value"] for level in levels]
                assert all(type(v) is int for v in values) and all(a < b for a, b in zip(values, values[1:]))
                assert values[0] == 0 and values[-1] == 100
                target = reference["target"]
                assert set(target) == {"label", "value"}
                assert any(level["label"] == target["label"] and level["value"] == target["value"] for level in levels)
                score_values[target["value"]] += 1
    assert domains == Counter({d: 5 for d in DOMAINS}), domains
    assert families == Counter({f: 18 for f in FAMILIES}), families
    assert booleans == Counter({True: 30, False: 30}), booleans
    assert option_counts == Counter({2: 15, 3: 15, 4: 15, 5: 15}), option_counts
    assert score_level_counts == Counter({3: 20, 4: 20, 5: 20}), score_level_counts
    assert max(score_values.values()) - min(score_values.values()) <= 12, score_values
    for size in range(2, 6):
        counts = [positions.get(f"{size}:{j}", 0) for j in range(size)]
        assert max(counts) - min(counts) <= 1, (size, counts)
    assert len(set(state_texts)) == 60
    assert len(set(prompts)) == 180

    # Read existing fixtures solely inside the checker; never print or emit their contents.
    prior_rows = [(path, read_rows(path)) for path in PRIOR]
    prior_families, prior_states, prior_prompts = set(), [], []
    for _, old_rows in prior_rows:
        for old in old_rows:
            prior_states.append(tokens(json.dumps(old["request"]["state"], ensure_ascii=False)))
            prior_prompts.extend(tokens(q["prompt"]) for q in old["request"]["questions"].values())
            prior_families.update(
                ref["question_family_id"] for ref in old["reference"].values()
                if isinstance(ref, dict) and "question_family_id" in ref
            )
    assert not (FAMILIES & prior_families), FAMILIES & prior_families
    max_state = max((similarity(tokens(text), old) for text in state_texts for old in prior_states), default=0)
    max_prompt = max((similarity(tokens(text), old) for text in prompts for old in prior_prompts), default=0)
    assert max_state < 0.72, f"near-duplicate state (Jaccard {max_state:.3f})"
    assert max_prompt < 0.72, f"near-duplicate prompt (Jaccard {max_prompt:.3f})"

    manifest = {
        "fixture": FIXTURE.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in prior_rows},
        "case_count": len(rows), "questions_per_case": 3,
        "domains": dict(sorted(domains.items())),
        "question_families": dict(sorted(families.items())),
        "answer_positions_by_option_count": dict(sorted(positions.items())),
        "boolean_outcomes": {str(k).lower(): v for k, v in sorted(booleans.items())},
        "score_outcomes_by_numeric_value": {str(k): v for k, v in sorted(score_values.items())},
        "score_levels_per_question": dict(sorted((str(k), v) for k, v in score_level_counts.items())),
        "maximum_prior_state_token_jaccard": round(max_state, 4),
        "maximum_prior_prompt_token_jaccard": round(max_prompt, 4),
        "prior_fixtures_compared": [path.name for path, _ in prior_rows],
        "annotation_note": "Targets and rationales are authored synthetic judgments, not externally adjudicated ground truth. Rationales/evidence_map are metadata and excluded from the request; no model predictions or training-derived labels were used.",
        "seal_note": "Keep this fixture sealed until the orchestrator confirms the LoRA train-only configuration is frozen.",
    }
    if write_manifest:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = validate(write_manifest="--write-manifest" in sys.argv)
    print(f"valid: {result['case_count']} cases; sha256={result['sha256']}")
    print("schema, balance, family isolation, and mechanical overlap checks passed")
