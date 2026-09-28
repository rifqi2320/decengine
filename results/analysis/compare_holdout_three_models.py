#!/usr/bin/env python3
"""Offline exact-match metrics for the 60-case Harrier/Qwen/JEV holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "benchmarks/cases/decision-cases-holdout.jsonl"
DEFAULT_HOLDOUT_RUN = ROOT / "results/holdout-comparison-runs/20260923T090938207411Z"
DEFAULT_QWEN_RUN = ROOT / "results/qwen-holdout/20260923T090609039852Z"
ENGINES = ("harrier", "qwen_mlx", "jev")
QUESTIONS = ("owner", "urgent", "impact")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected an object")
                rows.append(row)
    return rows


def indexed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        case_id = str(row[key])
        if case_id in result:
            raise ValueError(f"duplicate case ID: {case_id}")
        result[case_id] = row
    return result


def argmax_unique(distribution: Any, labels: dict[str, str] | None = None) -> str | None:
    if not isinstance(distribution, dict):
        return None
    values = [(str(label), value) for label, value in distribution.items()
              if isinstance(value, (int, float)) and math.isfinite(value)]
    if not values:
        return None
    best = max(value for _, value in values)
    winners = [label for label, value in values if value == best]
    if len(winners) != 1:
        return None
    return labels.get(winners[0], winners[0]) if labels else winners[0]


def predict_mlx(decisions: dict[str, Any]) -> dict[str, Any]:
    owner, urgent, impact = (decisions.get(k, {}) for k in QUESTIONS)
    value = urgent.get("value")
    return {
        "owner": owner.get("selected"),
        "urgent": value >= 0.5 if isinstance(value, (int, float)) else None,
        "impact": argmax_unique(impact.get("distribution")),
    }


def predict_jev(decisions: dict[str, Any], levels: list[dict[str, Any]]) -> dict[str, Any]:
    owner, urgent, impact = (decisions.get(k, {}) for k in QUESTIONS)
    value = urgent.get("value")
    labels = {str(index): str(level["label"]) for index, level in enumerate(levels)}
    return {
        "owner": owner.get("selected"),
        "urgent": value >= 0.5 if isinstance(value, (int, float)) else None,
        "impact": argmax_unique(impact.get("probabilities"), labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--holdout-run", type=Path, default=DEFAULT_HOLDOUT_RUN)
    parser.add_argument("--qwen-run", type=Path, default=DEFAULT_QWEN_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.holdout_run / "comparison-metrics.json"

    fixture_path = args.fixture
    h_cases_path = args.holdout_run / "cases.jsonl"
    h_meta_path = args.holdout_run / "metadata.json"
    h_summary_path = args.holdout_run / "summary.json"
    q_baseline_path = args.qwen_run / "baseline.jsonl"
    q_meta_path = args.qwen_run / "metadata.json"
    fixtures = read_jsonl(fixture_path)
    harrier = read_jsonl(h_cases_path)
    qwen = read_jsonl(q_baseline_path)
    hmeta = json.loads(h_meta_path.read_text(encoding="utf-8"))
    hsummary = json.loads(h_summary_path.read_text(encoding="utf-8"))
    qmeta = json.loads(q_meta_path.read_text(encoding="utf-8"))

    fixture_by_id = indexed(fixtures, "id")
    harrier_by_id = indexed(harrier, "case_id")
    qwen_by_id = indexed(qwen, "case_id")
    case_ids = [str(row["id"]) for row in fixtures]
    if len(case_ids) != 60 or len(set(case_ids)) != 60:
        raise ValueError(f"expected exactly 60 unique fixture cases, found {len(case_ids)}")
    if set(case_ids) != set(harrier_by_id) or set(case_ids) != set(qwen_by_id):
        raise ValueError("fixture, Harrier/JEV run, and Qwen baseline case IDs differ")
    if hmeta.get("case_count") != 60 or hsummary.get("case_count") != 60 or qmeta.get("case_count") != 60:
        raise ValueError("run metadata must describe exactly 60 holdout cases")
    if hmeta.get("qwen_baseline_sha256") != digest(q_baseline_path):
        raise ValueError("holdout run provenance does not match the supplied Qwen baseline")
    if hmeta.get("fixture_sha256") != digest(fixture_path):
        raise ValueError("holdout run provenance does not match the supplied fixture")
    for case_id in case_ids:
        if harrier_by_id[case_id].get("input") != fixture_by_id[case_id]:
            raise ValueError(f"Harrier run input differs from fixture for {case_id}")
        if harrier_by_id[case_id].get("harrier", {}).get("error") is not None:
            raise ValueError(f"Harrier error recorded for {case_id}")
        if harrier_by_id[case_id].get("jev", {}).get("error") is not None:
            raise ValueError(f"JEV error recorded for {case_id}")
        if qwen_by_id[case_id].get("error") is not None:
            raise ValueError(f"Qwen error recorded for {case_id}")

    per_case = []
    correct_totals = Counter()
    decision_totals = Counter()
    confusion: dict[str, dict[str, dict[str, Counter[str]]]] = {
        q: {engine: defaultdict(Counter) for engine in ENGINES} for q in QUESTIONS
    }
    fully_correct = Counter()
    all_three_full = 0
    for case_id in case_ids:
        case = fixture_by_id[case_id]
        reference = case["reference"]
        levels = case["request"]["questions"]["impact"]["levels"]
        hrow, qrow = harrier_by_id[case_id], qwen_by_id[case_id]
        predictions = {
            "harrier": predict_mlx(hrow["harrier"]["normalized_decisions"]),
            "qwen_mlx": predict_mlx(qrow["raw_response"]["results"]),
            "jev": predict_jev(hrow["jev"]["normalized_decisions"], levels),
        }
        correctness: dict[str, dict[str, bool | None]] = {}
        for engine in ENGINES:
            correctness[engine] = {}
            for question in QUESTIONS:
                predicted, expected = predictions[engine][question], reference[question]
                correct = predicted == expected if predicted is not None else None
                correctness[engine][question] = correct
                if correct is not None:
                    decision_totals[engine] += 1
                    correct_totals[engine] += int(correct)
                confusion[question][engine][str(expected)][
                    "<missing>" if predicted is None else str(predicted)
                ] += 1
            if all(correctness[engine][q] is True for q in QUESTIONS):
                fully_correct[engine] += 1
        all_three = all(fully_correct_marker for fully_correct_marker in (
            all(correctness[engine][q] is True for q in QUESTIONS) for engine in ENGINES
        ))
        all_three_full += int(all_three)
        per_case.append({
            "case_id": case_id,
            "reference": {q: reference[q] for q in QUESTIONS},
            "predictions": predictions,
            "correctness": correctness,
            "exact_matches_out_of_3": {
                engine: sum(correctness[engine][q] is True for q in QUESTIONS) for engine in ENGINES
            },
            "fully_correct": {
                engine: all(correctness[engine][q] is True for q in QUESTIONS) for engine in ENGINES
            },
            "all_three_fully_correct": all_three,
        })

    metrics = {
        "schema_version": 1,
        "method": {
            "source": "recorded model outputs compared exactly with holdout fixture references; offline only",
            "owner": "choice selected label",
            "urgent": "true iff scalar value >= 0.5",
            "impact": "unique argmax label from recorded probability distribution; null on missing/tied max",
            "all_three_fully_correct": "true iff each model exactly matches all three reference decisions",
            "confusion_matrix": "rows are reference labels; columns predictions; <missing> denotes unavailable/tied prediction",
        },
        "case_count": len(per_case),
        "zero_error_validation": {
            "harrier": sum(row["harrier"].get("error") is not None for row in harrier),
            "qwen_mlx": sum(row.get("error") is not None for row in qwen),
            "jev": sum(row["jev"].get("error") is not None for row in harrier),
        },
        "overall": {
            engine: {"exact_matches": correct_totals[engine], "decisions": decision_totals[engine],
                    "accuracy": correct_totals[engine] / decision_totals[engine] if decision_totals[engine] else None,
                    "fully_correct_cases": fully_correct[engine]}
            for engine in ENGINES
        },
        "all_three_fully_correct_cases": all_three_full,
        "by_question": {
            question: {
                "reference_count": len(per_case),
                "models": {
                    engine: {
                        "exact_matches": sum(row["correctness"][engine][question] is True for row in per_case),
                        "decisions": sum(row["correctness"][engine][question] is not None for row in per_case),
                        "confusion": {ref: dict(counts) for ref, counts in sorted(confusion[question][engine].items())},
                    } for engine in ENGINES
                },
            } for question in QUESTIONS
        },
        "run_references": {
            "holdout_run_id": hmeta.get("run_id"), "harrier_model": hmeta.get("harrier_model"),
            "harrier_jev_run_dir": str(args.holdout_run), "qwen_run_id": qmeta.get("run_id"),
            "qwen_model": qmeta.get("model_id"), "qwen_run_dir": str(args.qwen_run),
            "fixture": str(fixture_path), "network_or_api_calls": False,
        },
        "hashes_sha256": {
            "fixture": digest(fixture_path), "harrier_jev_cases": digest(h_cases_path),
            "harrier_jev_metadata": digest(h_meta_path), "harrier_jev_summary": digest(h_summary_path),
            "qwen_baseline": digest(q_baseline_path), "qwen_metadata": digest(q_meta_path),
        },
        "per_case": per_case,
    }
    if any(metrics["zero_error_validation"].values()):
        raise ValueError(f"expected zero errors in all model runs: {metrics['zero_error_validation']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "overall": metrics["overall"],
                      "all_three_fully_correct_cases": all_three_full,
                      "zero_error_validation": metrics["zero_error_validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
