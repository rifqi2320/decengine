#!/usr/bin/env python3
"""Compare inference-only predictions with saved test predictions by ID, no labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        key = str(item["id"])
        if key in result:
            raise ValueError(f"{path}:{line_number}: duplicate id {key!r}")
        result[key] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True,
                        help="saved training-run predictions.jsonl")
    parser.add_argument("--actual", type=Path, required=True,
                        help="predict_probe.py output JSONL")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    expected, actual = rows(args.expected), rows(args.actual)
    if expected.keys() != actual.keys():
        raise ValueError(f"ID mismatch: missing={sorted(expected.keys()-actual.keys())[:5]}, "
                         f"extra={sorted(actual.keys()-expected.keys())[:5]}")
    tasks = ("owner", "urgent", "impact")
    labels_match = True
    exact_probabilities = True
    max_abs_diff = 0.0
    compared = 0
    for case_id in expected:
        for task in tasks:
            wanted = expected[case_id][task]
            got = actual[case_id][task]
            labels_match &= wanted["label"] == got["label"]
            wanted_probs, got_probs = wanted["probabilities"], got["probabilities"]
            if wanted_probs.keys() != got_probs.keys():
                raise ValueError(f"{case_id}/{task}: class probability keys differ")
            classes = sorted(wanted_probs)
            left = np.asarray([wanted_probs[name] for name in classes], dtype=np.float64)
            right = np.asarray([got_probs[name] for name in classes], dtype=np.float64)
            exact_probabilities &= bool(np.array_equal(left, right))
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(left - right))))
            compared += len(classes)
    report = {
        "expected": str(args.expected), "actual": str(args.actual),
        "case_count": len(expected), "task_predictions_compared": len(expected) * len(tasks),
        "probability_values_compared": compared, "labels_match_exactly": bool(labels_match),
        "probabilities_bitwise_identical": bool(exact_probabilities),
        "maximum_absolute_probability_difference": max_abs_diff,
        "probabilities_within_absolute_tolerance": bool(max_abs_diff <= args.atol),
        "absolute_tolerance": args.atol,
        "pass": bool(labels_match and max_abs_diff <= args.atol),
        "verification_uses_reference_labels": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
