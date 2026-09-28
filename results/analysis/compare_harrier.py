#!/usr/bin/env python3
"""Derive objective exact-match metrics for Harrier, archived Qwen MLX, and JEV.

This is offline-only: it reads the fixture, immutable archived comparison run,
and Harrier output. It does not import the inference package or make API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "benchmarks/cases/decision-cases-100.jsonl"
DEFAULT_ARCHIVE = ROOT / "results/comparison-runs/20260923T082226631235Z/cases.jsonl"
DEFAULT_CORRECTED = ROOT / "results/comparison-runs/20260923T082226631235Z/corrected-metrics.json"
DEFAULT_HARRIER_RUN = ROOT / "results/harrier-runs/20260923T084828012314Z"
ENGINES = ("harrier", "qwen_mlx", "jev")
QUESTION_NAMES = ("owner", "urgent", "impact")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: expected object")
                rows.append(value)
    return rows


def argmax_unique(distribution: dict[str, Any], labels: dict[str, str] | None = None) -> str | None:
    """Return the unique max-probability label (or None for absent/tied data)."""
    numeric = [(str(key), value) for key, value in distribution.items()
               if isinstance(value, (int, float)) and math.isfinite(value)]
    if not numeric:
        return None
    best = max(value for _, value in numeric)
    winners = [key for key, value in numeric if value == best]
    if len(winners) != 1:
        return None
    key = winners[0]
    return labels.get(key, key) if labels is not None else key


def predict_mlx(decisions: dict[str, Any], score_levels: list[dict[str, Any]]) -> dict[str, Any]:
    owner = decisions.get("owner", {})
    urgent = decisions.get("urgent", {})
    impact = decisions.get("impact", {})
    return {
        "owner": owner.get("selected"),
        "urgent": (urgent.get("value") >= 0.5
                   if isinstance(urgent.get("value"), (int, float)) else None),
        "impact": argmax_unique(impact.get("distribution", {})),
    }


def predict_jev(decisions: dict[str, Any], score_levels: list[dict[str, Any]]) -> dict[str, Any]:
    owner = decisions.get("owner", {})
    urgent = decisions.get("urgent", {})
    impact = decisions.get("impact", {})
    label_by_index = {str(index): str(level["label"])
                      for index, level in enumerate(score_levels)}
    return {
        "owner": owner.get("selected"),
        "urgent": (urgent.get("value") >= 0.5
                   if isinstance(urgent.get("value"), (int, float)) else None),
        "impact": argmax_unique(impact.get("probabilities", {}), label_by_index),
    }


def latency_stats(values: list[float]) -> dict[str, Any]:
    values.sort()
    if not values:
        return {"case_count": 0, "timed_count": 0, "milliseconds": None}

    def percentile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * p
        lower = math.floor(rank)
        upper = math.ceil(rank)
        return values[lower] + (values[upper] - values[lower]) * (rank - lower)

    return {
        "case_count": len(values),
        "timed_count": len(values),
        "milliseconds": {
            "min": values[0], "mean": statistics.fmean(values),
            "p50": percentile(0.50), "p95": percentile(0.95), "max": values[-1],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--corrected-metrics", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument("--harrier-run", type=Path, default=DEFAULT_HARRIER_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.harrier_run / "comparison-metrics.json"

    fixture = read_jsonl(args.fixture)
    archive = read_jsonl(args.archive)
    harrier_path = args.harrier_run / "cases.jsonl"
    harrier = read_jsonl(harrier_path)
    corrected = json.loads(args.corrected_metrics.read_text(encoding="utf-8"))
    metadata = json.loads((args.harrier_run / "metadata.json").read_text(encoding="utf-8"))

    def index_by_id(rows: list[dict[str, Any]], getter) -> dict[str, dict[str, Any]]:
        result = {}
        for row in rows:
            case_id = str(getter(row))
            if case_id in result:
                raise ValueError(f"duplicate case ID: {case_id}")
            result[case_id] = row
        return result

    fixture_by_id = index_by_id(fixture, lambda row: row["id"])
    archive_by_id = index_by_id(archive, lambda row: row["case_id"])
    harrier_by_id = index_by_id(harrier, lambda row: row["case_id"])
    case_ids = [str(row["id"]) for row in fixture]
    if len(case_ids) != 100 or len(set(case_ids)) != 100:
        raise ValueError(f"expected exactly 100 unique fixture IDs, found {len(case_ids)}")
    if set(case_ids) != set(archive_by_id) or set(case_ids) != set(harrier_by_id):
        raise ValueError("fixture, comparison archive, and Harrier run IDs do not match")
    for case_id in case_ids:
        if fixture_by_id[case_id] != archive_by_id[case_id].get("input"):
            raise ValueError(f"archived input differs from fixture for {case_id}")
        if harrier_by_id[case_id].get("error") is not None:
            raise ValueError(f"Harrier run has an error for {case_id}")
        if harrier_by_id[case_id].get("original_archive_case_id") != case_id:
            raise ValueError(f"Harrier archive reference mismatch for {case_id}")
        for engine in ("decengine", "jev"):
            if archive_by_id[case_id].get(engine, {}).get("error") is not None:
                raise ValueError(f"archived {engine} run has an error for {case_id}")
    if len(harrier) != 100 or metadata.get("case_count") != 100:
        raise ValueError("Harrier run must contain exactly 100 cases")

    case_results: list[dict[str, Any]] = []
    correct_totals = Counter()
    decision_totals = Counter()
    confusion: dict[str, dict[str, dict[str, Counter[str]]]] = {
        question: {engine: defaultdict(Counter) for engine in ENGINES}
        for question in QUESTION_NAMES
    }
    latencies: dict[str, list[float]] = {engine: [] for engine in ENGINES}
    winner_counts = Counter()
    pairwise = {"harrier_vs_qwen_mlx": Counter(), "harrier_vs_jev": Counter(),
                "qwen_mlx_vs_jev": Counter()}

    for case_id in case_ids:
        input_case = fixture_by_id[case_id]
        reference = input_case["reference"]
        score_levels = input_case["request"]["questions"]["impact"]["levels"]
        archived = archive_by_id[case_id]
        hrow = harrier_by_id[case_id]
        predictions = {
            "harrier": predict_mlx(hrow["normalized_output"], score_levels),
            "qwen_mlx": predict_mlx(archived["decengine"]["normalized_decisions"], score_levels),
            "jev": predict_jev(archived["jev"]["normalized_decisions"], score_levels),
        }
        correctness: dict[str, dict[str, bool | None]] = {}
        for engine in ENGINES:
            correctness[engine] = {}
            for question in QUESTION_NAMES:
                prediction = predictions[engine][question]
                expected = reference[question]
                correct = prediction == expected if prediction is not None else None
                correctness[engine][question] = correct
                if correct is not None:
                    decision_totals[engine] += 1
                    correct_totals[engine] += int(correct)
                confusion[question][engine][str(expected)][
                    "<missing>" if prediction is None else str(prediction)
                ] += 1

        scores = {engine: sum(value is True for value in correctness[engine].values())
                  for engine in ENGINES}
        winner_score = max(scores.values())
        winners = [engine for engine, score in scores.items() if score == winner_score]
        winner = winners[0] if len(winners) == 1 else "tie"
        winner_counts[winner] += 1
        for label, left, right in (
            ("harrier_vs_qwen_mlx", "harrier", "qwen_mlx"),
            ("harrier_vs_jev", "harrier", "jev"),
            ("qwen_mlx_vs_jev", "qwen_mlx", "jev"),
        ):
            if scores[left] == scores[right]:
                pairwise[label]["tie"] += 1
            else:
                pairwise[label][left if scores[left] > scores[right] else right] += 1

        case_results.append({
            "case_id": case_id,
            "reference": {question: reference[question] for question in QUESTION_NAMES},
            "predictions": predictions,
            "correctness": correctness,
            "exact_matches_out_of_3": scores,
            "case_winner": winner,
            "case_winner_ties": winners if len(winners) > 1 else [],
        })
        latencies["harrier"].append(float(hrow["timing_ms"]))
        latencies["qwen_mlx"].append(float(archived["decengine"]["timing_ms"]))
        latencies["jev"].append(float(archived["jev"]["timing_ms"]))

    # The Qwen/JEV output is independently checked against the existing corrected
    # objective metrics before it is used as the comparator.
    existing = corrected["overall_exact_matches"]
    qwen_correct = correct_totals["qwen_mlx"]
    jev_correct = correct_totals["jev"]
    if (existing["mlx"]["exact_matches"] != qwen_correct
            or existing["jev"]["exact_matches"] != jev_correct):
        raise ValueError("recomputed archived Qwen/JEV counts differ from corrected-metrics.json")
    old_winners = corrected["objective_exact_match_case_winners"]
    pairwise_old = Counter()
    for case in case_results:
        q, j = case["exact_matches_out_of_3"]["qwen_mlx"], case["exact_matches_out_of_3"]["jev"]
        pairwise_old["tie" if q == j else "mlx" if q > j else "jev"] += 1
    if any(pairwise_old[k] != old_winners.get(k, 0) for k in ("mlx", "jev", "tie")):
        raise ValueError("recomputed archived Qwen/JEV case winner counts differ from corrected metrics")

    metrics = {
        "schema_version": 1,
        "method": {
            "source": "raw/normalized recorded outputs compared against fixture reference answers",
            "owner": "choice selected label",
            "urgent": "true iff scalar value >= 0.5",
            "impact": "unique argmax label from recorded probability distribution; null on missing/tied max",
            "case_winner": "model(s) with the highest number of exact matches across owner, urgent, impact; equal model totals are ties",
            "confusion_matrix": "rows are reference labels; columns are predictions; <missing> denotes unavailable/tied prediction",
        },
        "case_count": len(case_results),
        "zero_error_validation": {
            "harrier": sum(row.get("error") is not None for row in harrier),
            "qwen_mlx": sum(archive_by_id[cid]["decengine"].get("error") is not None for cid in case_ids),
            "jev": sum(archive_by_id[cid]["jev"].get("error") is not None for cid in case_ids),
        },
        "overall": {
            engine: {"exact_matches": correct_totals[engine], "decisions": decision_totals[engine],
                     "accuracy": correct_totals[engine] / decision_totals[engine]
                     if decision_totals[engine] else None}
            for engine in ENGINES
        },
        "by_question": {
            question: {
                "reference_count": len(case_results),
                "models": {
                    engine: {
                        "exact_matches": sum(
                            row["correctness"][engine][question] is True for row in case_results),
                        "decisions": sum(
                            row["correctness"][engine][question] is not None for row in case_results),
                        "confusion": {reference: dict(counts) for reference, counts in sorted(
                            confusion[question][engine].items())},
                    }
                    for engine in ENGINES
                },
            }
            for question in QUESTION_NAMES
        },
        "case_winners_three_way": dict(winner_counts),
        "pairwise_case_winners": {name: dict(counts) for name, counts in pairwise.items()},
        "latency": {engine: latency_stats(values) for engine, values in latencies.items()},
        "run_references": {
            "harrier_run_id": metadata.get("run_id"),
            "harrier_model": metadata.get("model_id"),
            "harrier_cases": str(harrier_path),
            "harrier_metadata": str(args.harrier_run / "metadata.json"),
            "harrier_profile": metadata.get("profile"),
            "harrier_install_record": metadata.get("install_record"),
            "qwen_jev_run_id": "20260923T082226631235Z",
            "qwen_jev_archive": str(args.archive),
            "existing_corrected_metrics": str(args.corrected_metrics),
            "fixture": str(args.fixture),
            "network_or_api_calls": False,
        },
        "hashes_sha256": {
            "fixture": sha256(args.fixture),
            "qwen_jev_archive": sha256(args.archive),
            "harrier_cases": sha256(harrier_path),
            "harrier_metadata": sha256(args.harrier_run / "metadata.json"),
            "corrected_metrics": sha256(args.corrected_metrics),
        },
        "per_case": case_results,
    }
    if any(metrics["zero_error_validation"].values()):
        raise ValueError(f"expected zero errors in all three runs: {metrics['zero_error_validation']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "overall": metrics["overall"],
                      "three_way_case_winners": metrics["case_winners_three_way"],
                      "pairwise_case_winners": metrics["pairwise_case_winners"],
                      "zero_error_validation": metrics["zero_error_validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
