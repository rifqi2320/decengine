#!/usr/bin/env python3
"""Deterministically curate a 160-case source set to 90 cases.

Inputs are the original and holdout fixture JSONLs with their objective
three-way comparison-metrics JSONs. This tool only reads inputs and writes
new artifacts; it never edits fixtures or run archives.

Policy: exclude the 70 easiest cases. First prioritize cases on which all
three systems have 3/3 exact matches. If there are fewer than 70 (as in the
known original-100 results), fill the remaining exclusions by descending
sum of exact decisions across the three systems (0..9). Ties prefer cases
with less model-score spread, retaining cases that discriminate between
systems. A greedy guard enforces proportional lower-bound quotas for each
industry and individual reference label (owner, urgent, impact). Final ties
use case ID ascending. This preserves representation without altering labels
or case content.

Example (run only after all 160 cases have three-way metrics; defaults use the recorded fixtures/runs):
  python3 results/analysis/curate_holdout.py \
    --cases combined-160.jsonl --metrics combined-metrics.json \
    --output-dir results/analysis/curated-90
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any


MODELS = ("harrier", "qwen_mlx", "jev")
QUESTIONS = ("owner", "urgent", "impact")
EXPECTED_INPUT = 160
TARGET = 90
EXCLUSION_COUNT = EXPECTED_INPUT - TARGET
EXPECTED_INDUSTRIES = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain objects: {path}")
    return rows


def validate_and_index(cases: list[dict[str, Any]], metrics: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    case_by_id: dict[str, dict[str, Any]] = {}
    for row in cases:
        case_id = row.get("id") or row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each case needs a nonempty id or case_id")
        if case_id in case_by_id:
            raise ValueError(f"duplicate case ID in source JSONL: {case_id}")
        case_by_id[case_id] = row

    records = metrics.get("per_case")
    if not isinstance(records, list):
        raise ValueError("metrics JSON must contain a per_case array")
    metric_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id in metric_by_id:
            raise ValueError(f"invalid or duplicate metrics case_id: {case_id!r}")
        scores = record.get("exact_matches_out_of_3")
        if not isinstance(scores, dict) or any(model not in scores for model in MODELS):
            raise ValueError(f"{case_id}: needs exact_matches_out_of_3 for {MODELS}")
        for model in MODELS:
            if type(scores[model]) is not int or not 0 <= scores[model] <= 3:
                raise ValueError(f"{case_id}: invalid {model} exact-match score")
        metric_by_id[case_id] = record

    if len(case_by_id) != EXPECTED_INPUT:
        raise ValueError(f"expected exactly {EXPECTED_INPUT} source cases, got {len(case_by_id)}")
    if set(case_by_id) != set(metric_by_id):
        missing = sorted(set(case_by_id) - set(metric_by_id))
        extra = sorted(set(metric_by_id) - set(case_by_id))
        raise ValueError(f"source/metrics IDs differ; missing metrics={missing}, extra metrics={extra}")
    return case_by_id, metric_by_id


def strata(row: dict[str, Any]) -> list[tuple[str, str]]:
    scenario = row.get("scenario") or {}
    reference = row.get("reference") or row.get("input", {}).get("reference") or {}
    industry = scenario.get("industry") or row.get("metadata", {}).get("industry") or "<missing>"
    keys = [("industry", str(industry))]
    for question in QUESTIONS:
        value = reference.get(question, "<missing>")
        # JSON scalar spelling makes bool labels unambiguous and reproducible.
        label = json.dumps(value, sort_keys=True, separators=(",", ":"))
        keys.append((question, label))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-cases", type=Path, default=Path("benchmarks/cases/decision-cases-100.jsonl"))
    parser.add_argument("--holdout-cases", type=Path, default=Path("benchmarks/cases/decision-cases-holdout.jsonl"))
    parser.add_argument("--original-metrics", type=Path, default=Path("results/harrier-runs/20260923T084828012314Z/comparison-metrics.json"))
    parser.add_argument("--holdout-metrics", type=Path, default=Path("results/holdout-comparison-runs/20260923T090938207411Z/comparison-metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/curation"))
    parser.add_argument("--curated-jsonl", type=Path, default=Path("benchmarks/cases/decision-cases-curated-90.jsonl"))
    args = parser.parse_args()

    original_cases_path = args.original_cases.resolve()
    holdout_cases_path = args.holdout_cases.resolve()
    original_metrics_path = args.original_metrics.resolve()
    holdout_metrics_path = args.holdout_metrics.resolve()
    output_dir = args.output_dir.resolve()
    curated_jsonl_path = args.curated_jsonl.resolve()
    source_paths = (original_cases_path, holdout_cases_path, original_metrics_path, holdout_metrics_path)
    if any(output_dir == path.parent for path in source_paths):
        raise ValueError("output directory must not be an input's parent directory")
    if curated_jsonl_path in source_paths:
        raise ValueError("curated JSONL output must not overwrite an input")
    original_cases = read_jsonl(original_cases_path)
    holdout_cases = read_jsonl(holdout_cases_path)
    cases = original_cases + holdout_cases
    source_set_by_id = {
        (row.get("id") or row.get("case_id")): name
        for name, rows in (("original", original_cases), ("holdout", holdout_cases))
        for row in rows
    }
    original_metrics = json.loads(original_metrics_path.read_text(encoding="utf-8"))
    holdout_metrics = json.loads(holdout_metrics_path.read_text(encoding="utf-8"))
    metrics = {"per_case": original_metrics.get("per_case", []) + holdout_metrics.get("per_case", [])}
    case_by_id, metric_by_id = validate_and_index(cases, metrics)

    # Validate the real fixture's question definitions through decengine's
    # public constructors before any selected artifact is persisted.
    package_parent = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(package_parent))
    # Import the standalone public question module without initializing the
    # top-level engine module (which unnecessarily requires the CFFI runtime).
    if "decengine" not in sys.modules:
        package = types.ModuleType("decengine")
        package.__path__ = [str(package_parent / "decengine")]
        sys.modules["decengine"] = package
    from decengine.questions import Choice, Noul, Score, ScoreLevel

    def validate_questions(case: dict[str, Any]) -> None:
        questions = case["request"]["questions"]
        expected = set(QUESTIONS)
        if set(questions) != expected:
            raise ValueError(f"{case['id']}: expected exactly {sorted(expected)} questions")
        for name, wire in questions.items():
            kind = wire.get("type")
            if kind == "choice":
                Choice(prompt=wire["prompt"], options=wire["options"])
            elif kind == "noul":
                Noul(prompt=wire["prompt"])
            elif kind == "score":
                Score(prompt=wire["prompt"], levels=[
                    ScoreLevel(label=level["label"], criterion=level["criterion"], value=level["value"])
                    for level in wire["levels"]
                ])
            else:
                raise ValueError(f"{case['id']}/{name}: unsupported question type {kind!r}")

    for case in cases:
        validate_questions(case)
    for case_id, record in metric_by_id.items():
        if set(record.get("predictions", {})) != set(MODELS):
            raise ValueError(f"{case_id}: metrics must contain predictions for all three models")
        if any(set(record["predictions"][model]) != set(QUESTIONS) for model in MODELS):
            raise ValueError(f"{case_id}: one or more model predictions are incomplete")
        fixture_reference = case_by_id[case_id].get("reference", {})
        if record.get("reference") != {question: fixture_reference.get(question) for question in QUESTIONS}:
            raise ValueError(f"{case_id}: metrics reference disagrees with source fixture")

    # A case is fully solved iff all models are exact on all three decisions.
    full_solved = {
        case_id for case_id, record in metric_by_id.items()
        if all(record["exact_matches_out_of_3"][model] == 3 for model in MODELS)
    }
    remaining_strata = Counter(
        key for row in case_by_id.values() for key in strata(row)
    )
    industries = {value for (kind, value) in remaining_strata if kind == "industry"}
    if len(industries) != EXPECTED_INDUSTRIES or "<missing>" in industries:
        raise ValueError(
            f"expected {EXPECTED_INDUSTRIES} named industries in source, got {sorted(industries)}"
        )
    # Preserve proportional representation: each industry and answer label
    # keeps at least its floor-proportional share of the 90-case target, and
    # every rare stratum retains at least one case.
    minimum_remaining = {
        key: max(1, (TARGET * count) // EXPECTED_INPUT)
        for key, count in remaining_strata.items()
    }
    exclusions: dict[str, str] = {}

    def try_exclude(case_id: str, reason: str) -> bool:
        row = case_by_id[case_id]
        case_strata = strata(row)
        if any(remaining_strata[key] - 1 < minimum_remaining[key] for key in case_strata):
            return False
        exclusions[case_id] = reason
        for key in case_strata:
            remaining_strata[key] -= 1
        return True

    # Phase 1: all jointly mastered cases precede any secondary exclusions.
    # If there are over 70, exact-90 and removing every such case conflict;
    # deterministic tie-breaks choose only the first 70 for the curation.
    fully_solved_order = sorted(full_solved)
    for case_id in fully_solved_order:
        if len(exclusions) == EXCLUSION_COUNT:
            break
        try_exclude(case_id, "all_three_models_exact_3_of_3")
    if len(exclusions) < len(full_solved):
        raise ValueError(
            "coverage quotas prevent removing every jointly solved case before "
            "secondary exclusions; agreed policy cannot be satisfied"
        )

    # Phase 2: preserve high-difficulty/discriminative cases; remove cases
    # with the strongest aggregate correctness, then weakest score spread.
    candidates = sorted(
        (case_id for case_id in case_by_id if case_id not in exclusions),
        key=lambda case_id: (
            -sum(metric_by_id[case_id]["exact_matches_out_of_3"][model] for model in MODELS),
            max(metric_by_id[case_id]["exact_matches_out_of_3"][m] for m in MODELS)
            - min(metric_by_id[case_id]["exact_matches_out_of_3"][m] for m in MODELS),
            case_id,
        ),
    )
    for case_id in candidates:
        if len(exclusions) == EXCLUSION_COUNT:
            break
        if try_exclude(case_id, "secondary_high_three_model_exact_match_total_low_discrimination"):
            continue
        # Retain a quota-limited case; later deterministic exchanges can
        # rebalance already-excluded cases without violating any quota.
    # A one-pass greedy can leave a tiny quota dead-end. Resolve it with
    # deterministic exchanges: restore one least-preferred secondary
    # exclusion and exclude two highest-ranked retained cases, netting one
    # additional exclusion while preserving every quota. Jointly solved
    # exclusions stay locked in phase 1.
    rank = {case_id: index for index, case_id in enumerate(candidates)}
    while len(exclusions) < EXCLUSION_COUNT:
        exchange = None
        retained_candidates = sorted(
            (case_id for case_id in case_by_id if case_id not in exclusions),
            key=lambda case_id: (rank.get(case_id, len(rank)), case_id),
        )
        restore_candidates = sorted(
            (case_id for case_id in exclusions if case_id not in full_solved),
            key=lambda case_id: (rank.get(case_id, -1), case_id), reverse=True,
        )
        for first_index, first_id in enumerate(retained_candidates):
            first_strata = strata(case_by_id[first_id])
            for second_id in retained_candidates[first_index + 1:]:
                second_strata = strata(case_by_id[second_id])
                for restore_id in restore_candidates:
                    restore_strata = strata(case_by_id[restore_id])
                    proposed = remaining_strata.copy()
                    for key in first_strata + second_strata:
                        proposed[key] -= 1
                    for key in restore_strata:
                        proposed[key] += 1
                    if all(proposed[key] >= minimum_remaining[key] for key in minimum_remaining):
                        exchange = (first_id, second_id, restore_id, proposed)
                        break
                if exchange:
                    break
            if exchange:
                break
        if exchange is None:
            break
        first_id, second_id, restore_id, remaining_strata = exchange
        del exclusions[restore_id]
        exclusions[first_id] = "secondary_high_three_model_exact_match_total_low_discrimination"
        exclusions[second_id] = "secondary_high_three_model_exact_match_total_low_discrimination"
    if len(exclusions) != EXCLUSION_COUNT:
        blocked = []
        for case_id in candidates:
            if case_id in exclusions:
                continue
            deficits = [
                {"stratum": key, "remaining": remaining_strata[key], "minimum": minimum_remaining[key]}
                for key in strata(case_by_id[case_id])
                if remaining_strata[key] - 1 < minimum_remaining[key]
            ]
            if deficits:
                blocked.append({"case_id": case_id, "deficits": deficits})
        raise ValueError(
            f"could only exclude {len(exclusions)} cases without dropping the last "
            f"case below a representation quota; cannot reach exactly 90. "
            f"Blocked candidates and quota deficits: {blocked[:20]}"
        )

    selected_ids = sorted(set(case_by_id) - set(exclusions))
    # Preserve original source row content/order; do not reconstruct/edit data.
    selected_rows = [row for row in cases if (row.get("id") or row.get("case_id")) in set(selected_ids)]
    if len(selected_rows) != TARGET:
        raise AssertionError("internal selection cardinality mismatch")
    if len(exclusions) != EXCLUSION_COUNT or len(set(selected_ids)) != TARGET:
        raise AssertionError("selection/exclusion cardinality invariant failed")
    if any(case_id not in case_by_id for case_id in selected_ids):
        raise AssertionError("selected ID is absent from source fixtures")
    if any(case_id in full_solved for case_id in selected_ids):
        raise AssertionError("selected set contains a case fully solved by all three models")
    if set(selected_ids) | set(exclusions) != set(case_by_id) or set(selected_ids) & set(exclusions):
        raise AssertionError("selected and excluded IDs do not partition source cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "selected-90-ids.json"
    manifest_path = output_dir / "curation-manifest.json"
    jsonl_bytes = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected_rows).encode("utf-8")
    curated_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    curated_jsonl_path.write_bytes(jsonl_bytes)
    ids_path.write_text(json.dumps(selected_ids, indent=2) + "\n", encoding="utf-8")

    def distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for name in ("industry", *QUESTIONS):
            counts = Counter(dict(strata(row))[name] for row in rows)
            result[name] = dict(sorted(counts.items()))
        return result

    def distributions_by_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            source_name: distributions([row for row in rows if source_set_by_id[row["id"]] == source_name])
            for source_name in ("original", "holdout")
        }

    manifest = {
        "schema_version": 1,
        "policy": {
            "source_cases": EXPECTED_INPUT,
            "target_cases": TARGET,
            "exclusions": EXCLUSION_COUNT,
            "phase_1": "jointly solved: each of Harrier, Qwen MLX, and JEV has 3/3 exact matches; deterministic case_id order if more than the exclusion budget",
            "phase_2": "descending sum of model exact matches (0..9), then ascending max-minus-min model score spread, then case_id; exclude high-scoring/low-discrimination cases first",
            "representation_guard": "for every industry and individual owner/urgent/impact reference-label stratum, retain at least max(1, floor(90 * input_stratum_count / 160)) cases; input has 10 named industries",
            "fully_solved_count": len(full_solved),
            "fully_solved_ids": sorted(full_solved),
        },
        "inputs": {
            "original_cases": {"path": str(original_cases_path), "sha256": sha256(original_cases_path)},
            "holdout_cases": {"path": str(holdout_cases_path), "sha256": sha256(holdout_cases_path)},
            "original_metrics": {"path": str(original_metrics_path), "sha256": sha256(original_metrics_path)},
            "holdout_metrics": {"path": str(holdout_metrics_path), "sha256": sha256(holdout_metrics_path)},
        },
        "artifacts": {
            "curated_jsonl": {"path": str(curated_jsonl_path), "sha256": hashlib.sha256(jsonl_bytes).hexdigest()},
            "selected_ids": {"path": ids_path.name, "sha256": sha256(ids_path)},
        },
        "counts": {
            "input": len(cases), "selected": len(selected_rows), "excluded": len(exclusions),
            "selected_by_industry_and_label": distributions(selected_rows),
            "input_by_industry_and_label": distributions(cases),
            "excluded_by_industry_and_label": distributions([
                case_by_id[case_id] for case_id in exclusions
            ]),
            "selected_by_source_and_industry_label": distributions_by_source(selected_rows),
            "input_by_source_and_industry_label": distributions_by_source(cases),
            "excluded_by_source_and_industry_label": distributions_by_source([
                case_by_id[case_id] for case_id in exclusions
            ]),
            "selected_by_source": dict(sorted(Counter(source_set_by_id[row["id"]] for row in selected_rows).items())),
            "excluded_by_source": dict(sorted(Counter(source_set_by_id[case_id] for case_id in exclusions).items())),
            "selected_difficulty": {
                "exact_matches_out_of_3_by_model": {
                    model: sum(metric_by_id[row["id"]]["exact_matches_out_of_3"][model] for row in selected_rows)
                    for model in MODELS
                },
                "exact_matches_total": sum(
                    sum(metric_by_id[row["id"]]["exact_matches_out_of_3"][model] for model in MODELS)
                    for row in selected_rows
                ),
                "decisions_total": len(selected_rows) * len(MODELS) * len(QUESTIONS),
                "fully_solved_by_all_three": sum(
                    row["id"] in full_solved for row in selected_rows
                ),
            },
        },
        "selected_ids": selected_ids,
        "selected_cases": [
            {"case_id": row["id"], "source": source_set_by_id[row["id"]],
             "reference": row["reference"], "predictions": metric_by_id[row["id"]]["predictions"],
             "exact_matches_out_of_3": metric_by_id[row["id"]]["exact_matches_out_of_3"],
             "combined_exact_matches": sum(metric_by_id[row["id"]]["exact_matches_out_of_3"][model] for model in MODELS),
             "model_score_spread": max(metric_by_id[row["id"]]["exact_matches_out_of_3"][model] for model in MODELS)
             - min(metric_by_id[row["id"]]["exact_matches_out_of_3"][model] for model in MODELS)}
            for row in selected_rows
        ],
        "exclusions": [
            {"case_id": case_id, "source": source_set_by_id[case_id], "reason": exclusions[case_id],
             "reference": case_by_id[case_id]["reference"],
             "predictions": metric_by_id[case_id]["predictions"],
             "exact_matches_out_of_3": metric_by_id[case_id]["exact_matches_out_of_3"],
             "combined_exact_matches": sum(metric_by_id[case_id]["exact_matches_out_of_3"][model] for model in MODELS),
             "model_score_spread": max(metric_by_id[case_id]["exact_matches_out_of_3"][model] for model in MODELS)
             - min(metric_by_id[case_id]["exact_matches_out_of_3"][model] for model in MODELS)}
            for case_id in sorted(exclusions)
        ],
        "provenance_note": "Selected JSONL rows are copied from the combined input without changing fixture labels/content. Existing fixtures and run archives are not modified.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"selected={len(selected_ids)} excluded={len(exclusions)} fully_solved={len(full_solved)}")
    print(f"curated_jsonl={curated_jsonl_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
