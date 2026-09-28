#!/usr/bin/env python3
"""Evaluate only baseline + a locked development-selected profile on holdout.

The inference pass keeps only case IDs and request inputs. It does not access the
holdout reference field. Both complete prediction JSONLs are flushed before a
second pass reads any reference labels for metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCKED_DEV_RUN = ROOT / "results/qwen-development-sweeps/20260923T090036783462Z"
HOLDOUT = ROOT / "benchmarks/cases/decision-cases-holdout.jsonl"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
QUESTION_FIELDS = ("owner", "urgent", "impact")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_requests_only(path: Path) -> list[dict[str, Any]]:
    """Parse cases but retain only ID + request, never read/return reference values."""
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank JSONL line")
            parsed = json.loads(line)
            case_id = str(parsed["id"])
            if not case_id or case_id in seen:
                raise ValueError(f"{path}:{line_no}: invalid/duplicate case id")
            seen.add(case_id)
            # Deliberately discard all other top-level fields (including reference).
            cases.append({"case_id": case_id, "request": parsed["request"]})
    if len(cases) != 60:
        raise ValueError(f"expected 60 holdout cases, got {len(cases)}")
    return cases


def prediction_and_confidence(response: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    predictions: dict[str, Any] = {}
    distributions: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    for key, result in response.results.items():
        confidence[key] = result.confidence
        if hasattr(result, "selected"):
            predictions[key] = result.selected
            distributions[key] = dict(result.probabilities)
        elif key == "urgent":
            predictions[key] = result.value >= 0.5
            distributions[key] = {"unsupported": 1.0 - result.value, "supported": result.value}
        else:
            predictions[key] = max(result.distribution, key=result.distribution.get)
            distributions[key] = dict(result.distribution)
    return predictions, distributions, confidence


def read_raw_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read_references_after_predictions(path: Path, expected_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Read reference labels only after every prediction JSONL has been completed."""
    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            case = json.loads(line)
            case_id = str(case["id"])
            reference_record = case["reference"]
            # Public fixtures may store target fields directly or nest them under
            # `decision` (the earlier 100-case fixture uses the latter).
            targets = reference_record.get("decision", reference_record)
            reference = {key: targets[key] for key in QUESTION_FIELDS}
            if case_id in labels:
                raise ValueError(f"duplicate reference case id {case_id}")
            if set(reference) != set(QUESTION_FIELDS):
                raise ValueError(f"{path}:{line_no}: unexpected reference decision fields")
            labels[case_id] = reference
    if list(labels) != expected_ids:
        raise ValueError("holdout case order/IDs changed between inference and metrics passes")
    return labels


def question_metrics(rows: list[dict[str, Any]], refs: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    correct_ids = {r["case_id"] for r in rows if r.get("prediction", {}).get(field) == refs[r["case_id"]][field]}
    errors = [
        {"case_id": r["case_id"], "predicted": r.get("prediction", {}).get(field), "reference": refs[r["case_id"]][field]}
        for r in rows
        if r.get("prediction", {}).get(field) != refs[r["case_id"]][field]
    ]
    confidence = [r.get("confidence", {}).get(field) for r in rows if isinstance(r.get("confidence", {}).get(field), (int, float))]
    report: dict[str, Any] = {
        "exact_matches": len(correct_ids),
        "total": len(rows),
        "exact_accuracy": len(correct_ids) / len(rows) if rows else None,
        "mean_reported_confidence": statistics.mean(confidence) if confidence else None,
        "mean_confidence_correct": statistics.mean([r["confidence"][field] for r in rows if r["case_id"] in correct_ids]) if correct_ids else None,
        "mean_confidence_incorrect": statistics.mean([r["confidence"][field] for r in rows if r["case_id"] not in correct_ids]) if len(correct_ids) < len(rows) else None,
        "errors": errors,
    }
    if field == "urgent":
        tp = sum(r.get("prediction", {}).get(field) is True and refs[r["case_id"]][field] is True for r in rows)
        tn = sum(r.get("prediction", {}).get(field) is False and refs[r["case_id"]][field] is False for r in rows)
        fp = sum(r.get("prediction", {}).get(field) is True and refs[r["case_id"]][field] is False for r in rows)
        fn = sum(r.get("prediction", {}).get(field) is False and refs[r["case_id"]][field] is True for r in rows)
        positives = sum(refs[r["case_id"]][field] is True for r in rows)
        negatives = len(rows) - positives
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        report["urgent_reference_split"] = {
            "reference_true_count": positives,
            "reference_false_count": negatives,
            "accuracy_on_reference_true": tp / positives if positives else None,
            "accuracy_on_reference_false": tn / negatives if negatives else None,
            "confusion": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None,
        }
    return report


def evaluate(rows: list[dict[str, Any]], refs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    questions = {field: question_metrics(rows, refs, field) for field in QUESTION_FIELDS}
    matches = sum(item["exact_matches"] for item in questions.values())
    total = len(rows) * len(QUESTION_FIELDS)
    latencies = [r["elapsed_ms"] for r in rows]
    return {
        "case_count": len(rows),
        "question_exact_matches": matches,
        "question_total": total,
        "micro_exact_accuracy": matches / total if total else None,
        "per_question": questions,
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "mean": statistics.mean(latencies) if latencies else None,
            "p95": sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        },
        "inference_errors": sum(r.get("error") is not None for r in rows),
    }


def run_inference(cases: list[dict[str, Any]], variant: str, options: dict[str, Any], output: Path, args: argparse.Namespace) -> None:
    from decengine import Engine
    from compare_jev import decode_questions, native_response_dict

    with Engine(MODEL, native_library=args.native_library, options=options) as engine, output.open("w", encoding="utf-8") as stream:
        for case in cases:
            start = time.perf_counter_ns()
            record: dict[str, Any] = {"case_id": case["case_id"], "variant": variant}
            try:
                request = case["request"]
                response = engine.decide(state=request["state"], questions=decode_questions(request["questions"]))
                prediction, distributions, confidence = prediction_and_confidence(response)
                record.update({
                    "prediction": prediction,
                    "distributions": distributions,
                    "confidence": confidence,
                    "raw_response": native_response_dict(response),
                    "error": None,
                })
            except Exception as exc:
                record.update({
                    "prediction": {key: None for key in QUESTION_FIELDS},
                    "distributions": None,
                    "confidence": None,
                    "raw_response": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
            record["elapsed_ms"] = (time.perf_counter_ns() - start) / 1e6
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--locked-dev-run", type=Path, default=LOCKED_DEV_RUN)
    parser.add_argument("--model-store", type=Path, default=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store"))
    parser.add_argument("--native-library", type=Path, default=ROOT / "target/release/libdecengine.dylib")
    parser.add_argument("--output", type=Path, default=ROOT / "results/qwen-holdout")
    args = parser.parse_args()

    dev_meta_path = args.locked_dev_run / "metadata.json"
    dev_meta_bytes = dev_meta_path.read_bytes()
    dev_metadata = json.loads(dev_meta_bytes)
    if dev_metadata.get("evaluation_scope") != "development cases 001-050 only":
        raise ValueError("locked run is not the specified development-only sweep")
    tuned = dev_metadata["variant_configs"]["official_format_task_instruction"]
    baseline = dev_metadata["variant_configs"]["baseline"]
    cases = load_requests_only(args.holdout)
    ids = [case["case_id"] for case in cases]
    if not args.native_library.is_file() or not args.model_store.is_dir():
        raise FileNotFoundError("release MLX library or installed model store is missing")
    install_path = args.model_store / "models/Qwen--Qwen3-Embedding-0.6B/install.json"
    install_bytes = install_path.read_bytes()
    active_manifest = (ROOT / "models/manifests/qwen3-embedding-0.6b.json").read_bytes()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_hashes = {name: sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()) for name, config in (("baseline", baseline), ("official_format_task_instruction", tuned))}
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL,
        "backend": "real MLX inference",
        "network_or_api_calls": False,
        "prediction_pass_completed_before_reference_access": True,
        "holdout_path": str(args.holdout),
        "case_count": len(cases),
        "case_ids": ids,
        "baseline_source": "bundled Qwen manifest defaults, locked explicitly in engine options",
        "selected_variant_source": str(dev_meta_path),
        "selected_variant_name": "official_format_task_instruction",
        "development_metadata_sha256": sha256_bytes(dev_meta_bytes),
        "config_hashes_sha256": config_hashes,
        "configs": {"baseline": baseline, "official_format_task_instruction": tuned},
        "temperature": 0.05,
        "noul_threshold": 0.5,
        "base_manifest_sha256": sha256_bytes(active_manifest),
        "installed_model_record_sha256": sha256_bytes(install_bytes),
        "native_library": str(args.native_library),
        "native_library_sha256": sha256_bytes(args.native_library.read_bytes()),
        "model_store": str(args.model_store),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "reference_metrics_status": "pending: labels are not read until both prediction JSONLs are complete",
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # First stage: do not touch reference labels. Save and flush complete raw predictions
    # for baseline and the locked variant before opening the dataset for metrics.
    options_common = {"engine": "mlx", "model_store": str(args.model_store)}
    run_inference(cases, "baseline", {**options_common, **baseline}, run_dir / "baseline.jsonl", args)
    run_inference(cases, "official_format_task_instruction", {**options_common, **tuned}, run_dir / "official-format-task-instruction.jsonl", args)
    for output in (run_dir / "baseline.jsonl", run_dir / "official-format-task-instruction.jsonl"):
        with output.open("rb") as stream:
            stream.flush()

    # Second stage starts only after the raw prediction artifacts above are complete.
    holdout_bytes = args.holdout.read_bytes()
    refs = read_references_after_predictions(args.holdout, ids)
    baseline_rows = read_raw_predictions(run_dir / "baseline.jsonl")
    tuned_rows = read_raw_predictions(run_dir / "official-format-task-instruction.jsonl")
    baseline_metrics = evaluate(baseline_rows, refs)
    tuned_metrics = evaluate(tuned_rows, refs)
    differences = {
        "micro_exact_accuracy_delta": tuned_metrics["micro_exact_accuracy"] - baseline_metrics["micro_exact_accuracy"],
        "exact_match_count_delta": tuned_metrics["question_exact_matches"] - baseline_metrics["question_exact_matches"],
        "per_question_exact_match_delta": {
            field: tuned_metrics["per_question"][field]["exact_matches"] - baseline_metrics["per_question"][field]["exact_matches"]
            for field in QUESTION_FIELDS
        },
        "urgent_true_class_accuracy_delta": tuned_metrics["per_question"]["urgent"]["urgent_reference_split"]["accuracy_on_reference_true"] - baseline_metrics["per_question"]["urgent"]["urgent_reference_split"]["accuracy_on_reference_true"],
        "urgent_false_class_accuracy_delta": tuned_metrics["per_question"]["urgent"]["urgent_reference_split"]["accuracy_on_reference_false"] - baseline_metrics["per_question"]["urgent"]["urgent_reference_split"]["accuracy_on_reference_false"],
    }
    metadata.update({
        "holdout_sha256": sha256_bytes(holdout_bytes),
        "reference_metrics_status": "computed after both prediction JSONLs were flushed",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "run_id": run_id,
        "case_count": len(cases),
        "baseline": baseline_metrics,
        "locked_variant": tuned_metrics,
        "locked_minus_baseline": differences,
        "metadata": "metadata.json",
        "predictions": {
            "baseline": "baseline.jsonl",
            "locked_variant": "official-format-task-instruction.jsonl",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved locked holdout run: {run_dir}")
    print(json.dumps({"baseline": baseline_metrics, "locked_variant": tuned_metrics, "delta": differences}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
