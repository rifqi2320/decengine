#!/usr/bin/env python3
"""Recompute exact-match metrics from a comparison archive and its fixture.

No network access or credentials are used. By default this reads the validated
20260923T082226631235Z run and writes corrected-metrics.json beside it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "results/comparison-runs/20260923T082226631235Z"
DEFAULT_FIXTURE = ROOT / "benchmarks/cases/decision-cases-100.jsonl"
QUESTIONS = ("owner", "urgent", "impact")
ENGINES = ("mlx", "jev")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def raw_prediction(case: dict[str, Any], engine: str, question: str,
                   fixture_case: dict[str, Any]) -> Any:
    """Convert raw backend output to the fixture's discrete answer labels."""
    raw = case["decengine" if engine == "mlx" else "jev"]["raw_response"]
    if engine == "mlx":
        results = raw["results"]
        if question == "owner":
            return results[question]["selected"]
        if question == "urgent":
            return results[question]["value"] >= 0.5
        # The score endpoint returns a probability distribution over its
        # authored levels; the runner/judgment normalized class is its argmax.
        distribution = results[question]["distribution"]
        return max(distribution, key=distribution.__getitem__)

    answers = raw["body"]["answers"]
    if question == "owner":
        return answers[question]["choice"]
    if question == "urgent":
        return answers[question]["noul"] >= 0.5
    # JEV's score distribution uses integer level indices as keys.
    levels = fixture_case["request"]["questions"][question]["levels"]
    probabilities = answers[question]["probabilities"]
    top_index = int(max(probabilities, key=lambda key: probabilities[key]))
    return levels[top_index]["label"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    fixture_rows = read_jsonl(args.fixture.resolve())
    fixture_by_id = {row["id"]: row for row in fixture_rows}
    archive_rows = read_jsonl(run_dir / "cases.jsonl")
    judgments = read_jsonl(run_dir / "judgments.jsonl")
    archive_by_id = {row["case_id"]: row for row in archive_rows}
    judgment_by_id = {row["case_id"]: row for row in judgments}

    if len(archive_by_id) != len(archive_rows) or len(fixture_by_id) != len(fixture_rows):
        raise ValueError("duplicate case IDs in archive or fixture")
    if set(archive_by_id) != set(fixture_by_id) or set(archive_by_id) != set(judgment_by_id):
        raise ValueError("archive, fixture, and judgment case IDs do not match")

    question_metrics: dict[str, dict[str, Any]] = {}
    case_scores: dict[str, dict[str, int]] = {
        case_id: {engine: 0 for engine in ENGINES} for case_id in archive_by_id
    }
    assessment_disagreements: dict[str, dict[str, list[str]]] = {
        engine: {question: [] for question in QUESTIONS} for engine in ENGINES
    }
    matching_but_indeterminate: dict[str, dict[str, list[str]]] = {
        engine: {question: [] for question in QUESTIONS} for engine in ENGINES
    }

    for question in QUESTIONS:
        question_metrics[question] = {}
        for engine in ENGINES:
            matches = 0
            mismatches: list[str] = []
            for case_id, case in archive_by_id.items():
                fixture_case = fixture_by_id[case_id]
                reference = fixture_case["reference"][question]
                prediction = raw_prediction(case, engine, question, fixture_case)
                # Guard against drift between the run's embedded fixture and
                # the benchmark fixture supplied for recomputation.
                embedded_reference = case["input"]["reference"][question]
                if reference != embedded_reference:
                    raise ValueError(f"embedded reference differs from fixture: {case_id}/{question}")
                if prediction == reference:
                    matches += 1
                    case_scores[case_id][engine] += 1
                else:
                    mismatches.append(case_id)
            question_metrics[question][engine] = {
                "exact_matches": matches,
                "exact_mismatches": len(archive_by_id) - matches,
                "case_count": len(archive_by_id),
                "mismatch_case_ids": mismatches,
            }

    for case_id, judgment in judgment_by_id.items():
        for engine in ENGINES:
            for question in QUESTIONS:
                j_prediction = judgment["outputs"][engine]["normalized_predictions"][question]
                reference = fixture_by_id[case_id]["reference"][question]
                raw = raw_prediction(archive_by_id[case_id], engine, question, fixture_by_id[case_id])
                if j_prediction != raw:
                    raise ValueError(f"judgment normalized output differs from raw output: {case_id}/{engine}/{question}")
                label = judgment["questions"][question]["assessment"][engine]["correctness"]
                if label == "correct" and j_prediction != reference:
                    assessment_disagreements[engine][question].append(case_id)
                if label == "indeterminate" and j_prediction == reference:
                    matching_but_indeterminate[engine][question].append(case_id)

    case_outcomes = Counter()
    per_case: dict[str, Any] = {}
    for case_id, scores in case_scores.items():
        outcome = "tie" if scores["mlx"] == scores["jev"] else (
            "mlx" if scores["mlx"] > scores["jev"] else "jev"
        )
        case_outcomes[outcome] += 1
        per_case[case_id] = {
            "exact_matches_out_of_3": scores,
            "objective_exact_match_winner": outcome,
        }

    subjective_outcomes = Counter(
        judgment["comparison"]["outcome"] for judgment in judgments
    )
    objective_total = {
        engine: sum(scores[engine] for scores in case_scores.values())
        for engine in ENGINES
    }
    output = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "method": {
            "basis": "raw archived engine responses compared exactly to benchmark fixture references",
            "normalization": {
                "owner": "choice label",
                "urgent": "supported/true iff raw scalar >= 0.5",
                "impact": "label with maximum archived score probability; JEV probability keys are fixture level indices",
            },
            "case_winner": "engine with more exact matches over owner, urgent, impact; equal counts are a tie",
            "subjective_winners": "separately counted from judgment ledger comparison.outcome; not derived from exact-match counts",
        },
        "fixture": str(args.fixture.resolve().relative_to(ROOT)) if args.fixture.resolve().is_relative_to(ROOT) else str(args.fixture.resolve()),
        "case_count": len(archive_by_id),
        "question_exact_matches": question_metrics,
        "overall_exact_matches": {
            engine: {"exact_matches": objective_total[engine], "decisions": len(archive_by_id) * len(QUESTIONS)}
            for engine in ENGINES
        },
        "objective_exact_match_case_winners": dict(case_outcomes),
        "subjective_judgment_case_winners": dict(subjective_outcomes),
        "per_case": per_case,
        "judgment_integrity_checks": {
            "marked_correct_but_prediction_differs_from_reference": assessment_disagreements,
            "marked_indeterminate_but_prediction_matches_reference": matching_but_indeterminate,
        },
        "audit_note": "This derived file does not modify cases.jsonl, judgments.jsonl, or summary.json.",
    }
    output_path = args.output.resolve() if args.output else run_dir / "corrected-metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
