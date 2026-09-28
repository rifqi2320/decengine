#!/usr/bin/env python3
"""Deterministically merge Harrier judgment chunks and cross-check derived data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "results/harrier-runs/20260923T084828012314Z"
DEFAULT_ORIGINAL = ROOT / "results/comparison-runs/20260923T082226631235Z/cases.jsonl"
CHUNK_NAME = re.compile(r"chunk-(\d{3})-(\d{3})\.jsonl$")
QUESTIONS = ("owner", "urgent", "impact")
CHUNK_ENGINE = {"harrier": "harrier", "qwen": "qwen_mlx", "jev": "jev"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--original-archive", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--judgments-path", type=Path,
                        help="validate this derivative instead of rebuilding from source chunks")
    parser.add_argument("--report-path", type=Path,
                        help="write validation report here (default: run-dir/validation-report.json)")
    args = parser.parse_args()

    run_dir = args.run_dir
    judgments_dir = run_dir / "judgments"
    merged_path = args.judgments_path or run_dir / "judgments.jsonl"
    report_path = args.report_path or run_dir / "validation-report.json"
    metrics_path = run_dir / "comparison-metrics.json"
    harrier_cases_path = run_dir / "cases.jsonl"
    metadata_path = run_dir / "metadata.json"

    chunks = sorted(judgments_dir.glob("chunk-*.jsonl"))
    expected_chunk_names = [f"chunk-{start:03d}-{start + 9:03d}.jsonl"
                            for start in range(1, 100, 10)]
    chunk_file_names = [path.name for path in chunks]
    missing_chunks = sorted(set(expected_chunk_names) - set(chunk_file_names))
    unexpected_chunks = sorted(set(chunk_file_names) - set(expected_chunk_names))

    # Copy original line content in deterministic range/name and in-chunk order.
    # For a supplied derivative, validate that file while leaving the original
    # merged artifact and all chunks untouched.
    merged_lines: list[str] = []
    rows: list[dict[str, Any]] = []
    parse_errors = []
    input_paths = [merged_path] if args.judgments_path else chunks
    for path in input_paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(keepends=True), 1):
            if not line.strip():
                continue
            merged_lines.append(line if line.endswith("\n") else line + "\n")
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be an object")
                rows.append(value)
            except (json.JSONDecodeError, ValueError) as error:
                parse_errors.append({"file": path.name, "line": line_no, "error": str(error)})

    # Merge faithfully only in the normal mode. Derivative mode is read-only.
    if not args.judgments_path:
        merged_path.write_text("".join(merged_lines), encoding="utf-8")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual_hashes = {
        "harrier_cases": digest(harrier_cases_path),
        "original_archive": digest(args.original_archive),
    }
    expected_hashes = {
        "harrier_cases": metrics.get("hashes_sha256", {}).get("harrier_cases"),
        "original_archive": metrics.get("hashes_sha256", {}).get("qwen_jev_archive"),
        "run_metadata_original_archive": metadata.get("original_archive_sha256"),
    }
    expected_chunk_hashes = {
        "harrier_cases": "sha256:" + actual_hashes["harrier_cases"],
        "original_archive": "sha256:" + actual_hashes["original_archive"],
    }
    reported_chunk_hashes: dict[str, set[str]] = {key: set() for key in expected_chunk_hashes}
    rows_missing_hash_fields = []
    for row in rows:
        hash_fields = row.get("archive_hashes") or row.get("source_hashes") or {}
        source_archives = row.get("source_archives") or {}
        if not hash_fields and not source_archives:
            rows_missing_hash_fields.append(str(row.get("case_id", "")))
        for key in reported_chunk_hashes:
            source_key = "harrier" if key == "harrier_cases" else "original"
            value = hash_fields.get(source_key, hash_fields.get(key))
            if value is None and source_key in source_archives:
                source = source_archives[source_key]
                value = source.get("sha256") if isinstance(source, dict) else source
            if value is not None:
                reported_chunk_hashes[key].add(str(value))

    ids = [str(row.get("case_id", "")) for row in rows]
    id_counts: dict[str, int] = {}
    for case_id in ids:
        id_counts[case_id] = id_counts.get(case_id, 0) + 1
    duplicate_ids = sorted(case_id for case_id, count in id_counts.items() if count > 1)
    missing_ids = [f"case-{number:03d}" for number in range(1, 101)
                   if f"case-{number:03d}" not in id_counts]
    unexpected_ids = sorted(case_id for case_id in id_counts
                            if not re.fullmatch(r"case-\d{3}", case_id)
                            or not 1 <= int(case_id[-3:]) <= 100)

    metrics_cases = {str(row["case_id"]): row for row in metrics.get("per_case", [])}
    discrepancy_by_case: dict[str, set[str]] = {}
    discrepancy_details: dict[str, dict[str, Any]] = {}

    def discrepancy(case_id: str, field: str, observed: Any = None, expected: Any = None) -> None:
        discrepancy_by_case.setdefault(case_id, set()).add(field)
        if observed is not None or expected is not None:
            discrepancy_details.setdefault(case_id, {})[field] = {
                "chunk_value": observed,
                "metrics_value": expected,
            }

    def chunk_value(row: dict[str, Any], field: str, question: str, engine: str) -> Any:
        values = row.get(field, {})
        # Chunks were independently prepared in both question-major and
        # engine-major orientation; preserve source records but normalize reads.
        if isinstance(values, dict) and engine in values and isinstance(values[engine], dict):
            return values[engine].get(question)
        if isinstance(values, dict) and question in values and isinstance(values[question], dict):
            return values[question].get(engine)
        return None

    for row in rows:
        case_id = str(row.get("case_id", ""))
        metric_case = metrics_cases.get(case_id)
        if metric_case is None:
            discrepancy(case_id, "missing_from_comparison_metrics")
            continue
        chunk_reference = row.get("reference", {})
        if isinstance(chunk_reference.get("decision"), dict):
            chunk_reference = chunk_reference["decision"]
        for question in QUESTIONS:
            if chunk_reference.get(question) != metric_case.get("reference", {}).get(question):
                discrepancy(case_id, f"reference.{question}", chunk_reference.get(question),
                            metric_case.get("reference", {}).get(question))
            for source_engine, metric_engine in CHUNK_ENGINE.items():
                chunk_prediction = chunk_value(row, "predictions", question, source_engine)
                metric_prediction = metric_case.get("predictions", {}).get(metric_engine, {}).get(question)
                if chunk_prediction != metric_prediction:
                    discrepancy(case_id, f"prediction.{question}.{source_engine}", chunk_prediction,
                                metric_prediction)
                chunk_correct = chunk_value(row, "exact_matches", question, source_engine)
                metric_correct = metric_case.get("correctness", {}).get(metric_engine, {}).get(question)
                if chunk_correct != metric_correct:
                    discrepancy(case_id, f"exact_match.{question}.{source_engine}", chunk_correct,
                                metric_correct)
        for source_engine, metric_engine in CHUNK_ENGINE.items():
            totals = row.get("exact_match_totals", row.get("total_exact_matches", {}))
            chunk_total = totals.get(source_engine) if isinstance(totals, dict) else None
            metric_total = metric_case.get("exact_matches_out_of_3", {}).get(metric_engine)
            if chunk_total is not None and chunk_total != metric_total:
                discrepancy(case_id, f"exact_match_total.{source_engine}", chunk_total, metric_total)
        chunk_winners = row.get("winner", row.get("winners"))
        if not isinstance(chunk_winners, list):
            discrepancy(case_id, "winner.not_a_list")
        else:
            normalized_winners = sorted(CHUNK_ENGINE[name] for name in chunk_winners if name in CHUNK_ENGINE)
            if len(normalized_winners) != len(chunk_winners):
                discrepancy(case_id, "winner.unknown_engine")
            expected_winners = metric_case.get("case_winner_ties", [])
            if not expected_winners and metric_case.get("case_winner") != "tie":
                expected_winners = [metric_case.get("case_winner")]
            if normalized_winners != sorted(expected_winners):
                discrepancy(case_id, "case_winner", normalized_winners, sorted(expected_winners))

    expected_case_ids = {f"case-{number:03d}" for number in range(1, 101)}
    if set(metrics_cases) != expected_case_ids or len(metrics.get("per_case", [])) != 100:
        for case_id in sorted(expected_case_ids - set(metrics_cases)):
            discrepancy(case_id, "missing_from_comparison_metrics")
        for case_id in sorted(set(metrics_cases) - expected_case_ids):
            discrepancy(case_id, "unexpected_in_comparison_metrics")

    hash_checks = {
        "harrier_cases_actual_matches_metrics": actual_hashes["harrier_cases"] == expected_hashes["harrier_cases"],
        "original_archive_actual_matches_metrics": actual_hashes["original_archive"] == expected_hashes["original_archive"],
        "original_archive_actual_matches_run_metadata": actual_hashes["original_archive"] == expected_hashes["run_metadata_original_archive"],
        "chunk_reported_harrier_cases_hashes_match": reported_chunk_hashes["harrier_cases"] == {expected_chunk_hashes["harrier_cases"]},
        "chunk_reported_original_archive_hashes_match": reported_chunk_hashes["original_archive"] == {expected_chunk_hashes["original_archive"]},
    }
    discrepancies = [
        {"case_id": case_id, "fields": sorted(fields),
         **({"details": discrepancy_details[case_id]} if case_id in discrepancy_details else {})}
        for case_id, fields in sorted(discrepancy_by_case.items())
    ]
    report = {
        "validation": "passed" if (
            not parse_errors and not missing_chunks and not unexpected_chunks
            and len(rows) == 100 and not duplicate_ids and not missing_ids
            and not unexpected_ids and not discrepancies and all(hash_checks.values())
        ) else "failed",
        "run_id": metadata.get("run_id"),
        "merged_file": str(merged_path),
        "merged_record_count": len(rows),
        "unique_case_id_count": len(set(ids)),
        "chunks": {
            "expected_count": 10,
            "found_count": len(chunks),
            "ordered_files": chunk_file_names,
            "missing": missing_chunks,
            "unexpected": unexpected_chunks,
            "source_chunks_preserved": True,
        },
        "case_ids": {
            "duplicates": duplicate_ids,
            "missing": missing_ids,
            "unexpected": unexpected_ids,
        },
        "archive_hashes": {
            "actual_sha256": actual_hashes,
            "expected_sha256": expected_hashes,
            "chunk_reported_sha256": {
                key: sorted(value.removeprefix("sha256:") for value in values)
                for key, values in reported_chunk_hashes.items()
            },
            "case_ids_without_embedded_hash_fields": rows_missing_hash_fields,
            "checks": hash_checks,
        },
        "metrics_cross_check": {
            "comparison_metrics": str(metrics_path),
            "checked_fields": ["reference", "predictions", "exact_matches", "exact_match_totals", "winner"],
            "discrepancy_case_count": len(discrepancies),
            "discrepancy_case_ids": [item["case_id"] for item in discrepancies],
            "discrepancies": discrepancies,
        },
        "parse_errors": parse_errors,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation": report["validation"],
        "merged_file": str(merged_path),
        "records": len(rows),
        "unique_ids": len(set(ids)),
        "hash_checks": hash_checks,
        "discrepancies": discrepancies,
        "report": str(report_path),
    }, indent=2))
    return 0 if report["validation"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
