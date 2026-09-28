#!/usr/bin/env python3
"""Validate, append, and summarize case-by-question MLX/JEV judgment JSONL."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

QUESTIONS = ("owner", "urgent", "impact")
BACKENDS = ("mlx", "jev")
CORRECTNESS = ("correct", "partial", "incorrect", "indeterminate")
OUTCOMES = ("mlx", "jev", "tie", "indeterminate")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nonempty(value: Any, where: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{where}: expected non-empty string")


def validate_record(record: Any, where: str) -> None:
    keys = {"schema_version", "judgment_id", "case_id", "industry", "timestamp", "agent", "source_run", "reference", "outputs", "assessment", "comparison", "questions", "notes"}
    require(isinstance(record, dict), f"{where}: record must be an object")
    require(set(record) == keys, f"{where}: keys must be exactly {sorted(keys)}")
    require(record["schema_version"] == 1, f"{where}: unsupported schema_version")
    for key in ("judgment_id", "case_id", "industry"):
        nonempty(record[key], f"{where}.{key}")
    require(isinstance(record["notes"], str), f"{where}.notes: expected string")
    try:
        dt = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        require(dt.tzinfo is not None, f"{where}.timestamp: timezone required")
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{where}.timestamp: expected RFC 3339 date-time with timezone") from None
    for group, fields in (("agent", ("identity", "session_id")), ("source_run", ("run_id", "run_hash"))):
        obj = record[group]
        require(isinstance(obj, dict) and set(obj) == set(fields), f"{where}.{group}: expected keys {fields}")
        for field in fields:
            nonempty(obj[field], f"{where}.{group}.{field}")

    reference = record["reference"]
    require(isinstance(reference, dict) and set(reference) == {"decision", "rationale"}, f"{where}.reference: expected decision and rationale")
    require(isinstance(reference["decision"], dict) and set(reference["decision"]) == set(QUESTIONS), f"{where}.reference.decision: expected {QUESTIONS}")
    nonempty(reference["rationale"], f"{where}.reference.rationale")
    require(isinstance(reference["decision"]["owner"], str), f"{where}: reference owner must be a string")
    require(isinstance(reference["decision"]["urgent"], bool), f"{where}: reference urgent must be boolean")
    require(isinstance(reference["decision"]["impact"], str), f"{where}: reference impact must be a string")

    outputs = record["outputs"]
    require(isinstance(outputs, dict) and set(outputs) == set(BACKENDS), f"{where}.outputs: expected mlx and jev")
    for backend in BACKENDS:
        output = outputs[backend]
        require(isinstance(output, dict) and set(output) == {"archive_ref", "status", "normalized_predictions", "error"}, f"{where}.outputs.{backend}: invalid fields")
        nonempty(output["archive_ref"], f"{where}.outputs.{backend}.archive_ref")
        require(output["status"] in ("ok", "error", "missing"), f"{where}.outputs.{backend}.status: invalid value")
        predictions = output["normalized_predictions"]
        require(isinstance(predictions, dict) and set(predictions) == set(QUESTIONS), f"{where}.outputs.{backend}.normalized_predictions: expected {QUESTIONS}")
        require(all(p is None or isinstance(p, (str, bool, int, float)) for p in predictions.values()), f"{where}.outputs.{backend}.normalized_predictions: answers must be scalar or null")
        require(output["error"] is None or isinstance(output["error"], str), f"{where}.outputs.{backend}.error: expected string or null")
        if output["status"] != "ok":
            require(output["error"] is not None, f"{where}.outputs.{backend}: error text required when status is {output['status']}")

    # Retain a case-level quality/outcome assessment for compatibility and easy
    # headline reporting; detailed correctness is adjudicated per question below.
    _validate_assessments(record.get("assessment"), f"{where}.assessment")
    _validate_comparison(record.get("comparison"), f"{where}.comparison")
    qdata = record["questions"]
    require(isinstance(qdata, dict) and set(qdata) == set(QUESTIONS), f"{where}.questions: expected {QUESTIONS}")
    for question in QUESTIONS:
        item = qdata[question]
        require(isinstance(item, dict) and set(item) == {"assessment", "comparison"}, f"{where}.questions.{question}: expected assessment and comparison")
        _validate_assessments(item["assessment"], f"{where}.questions.{question}.assessment")
        _validate_comparison(item["comparison"], f"{where}.questions.{question}.comparison")
        # A failed/missing response or absent question answer cannot receive a
        # definitive correctness judgment.
        for backend in BACKENDS:
            if outputs[backend]["status"] != "ok" or outputs[backend]["normalized_predictions"][question] is None:
                require(item["assessment"][backend]["correctness"] == "indeterminate", f"{where}: {backend}.{question} must be indeterminate for an error/missing answer")


def _validate_assessments(value: Any, where: str) -> None:
    require(isinstance(value, dict) and set(value) == set(BACKENDS), f"{where}: expected mlx and jev")
    for backend in BACKENDS:
        item = value[backend]
        expected = {"correctness", "confidence", "confidence_basis", "rationale"}
        require(isinstance(item, dict) and set(item) == expected, f"{where}.{backend}: expected {sorted(expected)}")
        require(item["correctness"] in CORRECTNESS, f"{where}.{backend}.correctness: invalid")
        nonempty(item["rationale"], f"{where}.{backend}.rationale")
        basis, conf = item["confidence_basis"], item["confidence"]
        require(basis in ("reported", "unavailable"), f"{where}.{backend}.confidence_basis: invalid")
        if basis == "unavailable":
            require(conf is None, f"{where}.{backend}: unavailable confidence must be null")
        else:
            require(isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf) and 0 <= conf <= 1, f"{where}.{backend}.confidence: expected number [0,1]")


def _validate_comparison(value: Any, where: str) -> None:
    require(isinstance(value, dict) and set(value) == {"outcome", "rationale"}, f"{where}: expected outcome and rationale")
    require(value["outcome"] in OUTCOMES, f"{where}.outcome: invalid")
    nonempty(value["rationale"], f"{where}.rationale")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank lines are not permitted")
            try:
                item = json.loads(line)
                validate_record(item, f"{path}:{line_no}")
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_no}: {error}") from error
            if item["judgment_id"] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate judgment_id {item['judgment_id']!r}")
            seen.add(item["judgment_id"])
            records.append(item)
    return records


def append_record(path: Path, record_path: Path) -> None:
    item = json.loads(record_path.read_text(encoding="utf-8"))
    validate_record(item, str(record_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        if path.stat().st_size:
            existing = read_jsonl(path)
            if any(old["judgment_id"] == item["judgment_id"] for old in existing):
                raise ValueError(f"duplicate judgment_id {item['judgment_id']!r}; use a new ID for a correction")
            stream.seek(0, 2)
            stream.seek(stream.tell() - 1)
            if stream.read(1) != "\n":
                raise ValueError(f"{path}: existing JSONL must end with newline")
        stream.seek(0, 2)
        stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _group_stats(records: list[dict[str, Any]], question: str, backend: str | None = None) -> dict[str, Any]:
    correctness: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    by_reference: dict[str, Counter[str]] = defaultdict(Counter)
    by_prediction: dict[str, Counter[str]] = defaultdict(Counter)
    by_industry: dict[str, Counter[str]] = defaultdict(Counter)
    brier_terms: list[float] = []
    confidence_n = 0
    for record in records:
        assessment = record["questions"][question]
        if backend is None:
            label = assessment["comparison"]["outcome"]
            outcomes[label] += 1
            correctness[label] += 1
        else:
            judgment = assessment["assessment"][backend]
            label = judgment["correctness"]
            correctness[label] += 1
            confidence = judgment["confidence"]
            if judgment["confidence_basis"] == "reported":
                confidence_n += 1
            if judgment["confidence_basis"] == "reported" and label in ("correct", "incorrect"):
                target = 1 if label == "correct" else 0
                brier_terms.append((confidence - target) ** 2)
        reference_value = record["reference"]["decision"][question]
        by_reference[str(reference_value)][label] += 1
        prediction = record["outputs"][backend]["normalized_predictions"][question] if backend else None
        if backend is not None:
            by_prediction[str(prediction)][label] += 1
            by_industry[record["industry"]][label] += 1
    answered = sum(correctness[c] for c in ("correct", "partial", "incorrect")) if backend else len(records)
    correct = correctness["correct"] if backend else correctness["mlx"] + correctness["jev"]
    result: dict[str, Any] = {
        "judgments": len(records), "labels": dict(correctness), "by_reference_value": {k: dict(v) for k, v in sorted(by_reference.items())},
    }
    if backend is None:
        result["outcomes"] = dict(outcomes)
    else:
        result.update({
            "definitive_count": answered,
            "exact_accuracy": correct / answered if answered else None,
            "by_predicted_value": {k: dict(v) for k, v in sorted(by_prediction.items())},
            "by_industry": {k: dict(v) for k, v in sorted(by_industry.items())},
            "confidence_reported_count": confidence_n,
            "binary_scored_count": len(brier_terms),
            "brier_score": sum(brier_terms) / len(brier_terms) if brier_terms else None,
        })
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = {r["case_id"] for r in records}
    expected_ids = {f"case-{n:03}" for n in range(1, 101)}
    return {
        "judgment_count": len(records),
        "case_count": len(case_ids),
        "expected_case_count": 100,
        "missing_case_ids": sorted(expected_ids - case_ids),
        "case_winners": dict(Counter(r["comparison"]["outcome"] for r in records)),
        "case_correctness": {backend: dict(Counter(r["assessment"][backend]["correctness"] for r in records)) for backend in BACKENDS},
        "by_question": {
            question: {
                "relative_outcomes": _group_stats(records, question),
                "mlx": _group_stats(records, question, "mlx"),
                "jev": _group_stats(records, question, "jev"),
            } for question in QUESTIONS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "summary"):
        p = sub.add_parser(name)
        p.add_argument("ledger", type=Path)
    add = sub.add_parser("append")
    add.add_argument("ledger", type=Path)
    add.add_argument("record", type=Path, help="file containing exactly one judgment object")
    args = parser.parse_args()
    try:
        if args.command == "append":
            append_record(args.ledger, args.record)
            print(f"appended {args.record} to {args.ledger}")
        else:
            records = read_jsonl(args.ledger)
            if args.command == "validate":
                print(f"valid: {len(records)} judgment(s), {len({r['case_id'] for r in records})} case(s)")
            else:
                print(json.dumps(summarize(records), indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
