#!/usr/bin/env python3
"""Run a Qwen prompt/candidate sweep on fixture cases 001-050 only.

This script deliberately reads only the first 50 lines of the development fixture,
archived Qwen results, and archived judgments. It never opens any holdout file or
reads case IDs 051-100. Inference is local MLX only; no API client is used.
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
FIXTURE = ROOT / "benchmarks/cases/decision-cases-100.jsonl"
ARCHIVE = ROOT / "results/comparison-runs/20260923T082226631235Z/cases.jsonl"
JUDGMENTS = ROOT / "results/comparison-runs/20260923T082226631235Z/judgments.jsonl"
MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Variants are fixed before evaluation and contain no reference labels or
# case-specific wording. Existing compiler defaults are retained by the baseline.
GENERIC_TASKS = (
    "choose the single best-matching option from the reported issue and candidate scopes; "
    "do not infer facts absent from the case"
)
URGENCY_TASKS = (
    "decide whether same-day action is needed to prevent likely harm, material interruption, "
    "or a near-term missed deadline; an available safe workaround or no concrete near-term "
    "consequence favors no; emphatic wording alone is not evidence"
)
IMPACT_TASKS = (
    "select the highest impact supported if unresolved; distinguish routine/no concrete near-term "
    "risk, localized disruption or safe workaround, and credible immediate or broad serious risk"
)
TASK_BY_KIND = {
    "choice": GENERIC_TASKS,
    "noul": URGENCY_TASKS,
    "score": IMPACT_TASKS,
}
TASK_INSTRUCTION = "Task: {kind}\nQuestion: {prompt}"

BASELINE = {"query_template": "Instruct: {instruction}\nQuery:{text}", "candidate_template": "{text}"}
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "official_format_task_instruction": {
        "query_template": "Instruct: {instruction}\nQuery:{text}",
        "query_instruction_template": TASK_INSTRUCTION,
    },
    "choice_scope_candidates": {
        "choice_candidate_template": "Function: {label}. Responsibility scope: {criterion}",
    },
    "score_impact_candidates": {
        "score_candidate_template": "Impact level {label}. Supported criterion: {criterion}",
    },
    "urgency_no_risk_workaround": {
        "noul_unsupported": "No: no concrete same-day threat, material interruption, or near-term missed deadline; a safe workaround is available or the issue can wait.",
        "noul_supported": "Yes: same-day action is needed to prevent likely harm, material interruption, or a near-term missed deadline; no safe workaround avoids the consequence.",
    },
    "task_candidates_combined": {
        "query_instruction_template": TASK_INSTRUCTION,
        "choice_candidate_template": "Function: {label}. Responsibility scope: {criterion}",
        "score_candidate_template": "Impact level {label}. Supported criterion: {criterion}",
        "noul_unsupported": "No: no concrete same-day threat, material interruption, or near-term missed deadline; a safe workaround is available or the issue can wait.",
        "noul_supported": "Yes: same-day action is needed to prevent likely harm, material interruption, or a near-term missed deadline; no safe workaround avoids the consequence.",
    },
    "task_candidates_spaced_query": {
        "query_template": "Instruct: {instruction}\nQuery: {text}",
        "query_instruction_template": TASK_INSTRUCTION,
        "choice_candidate_template": "Function: {label}. Responsibility scope: {criterion}",
        "score_candidate_template": "Impact level {label}. Supported criterion: {criterion}",
        "noul_unsupported": "No: no concrete same-day threat, material interruption, or near-term missed deadline; a safe workaround is available or the issue can wait.",
        "noul_supported": "Yes: same-day action is needed to prevent likely harm, material interruption, or a near-term missed deadline; no safe workaround avoids the consequence.",
    },
}


def read_prefix(path: Path, count: int = 50) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for _ in range(count):
            line = f.readline()
            if not line:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prefix_bytes(path: Path, count: int = 50) -> bytes:
    content = bytearray()
    with path.open("rb") as f:
        for _ in range(count):
            line = f.readline()
            if not line:
                break
            content.extend(line)
    return bytes(content)


def summarize(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("owner", "urgent", "impact")
    report: dict[str, Any] = {"cases": len(rows), "fields": {}, "errors": {}}
    for key in fields:
        mismatches = [
            {"case_id": row["case_id"], "predicted": pred.get(key), "reference": row["reference"][key]}
            for row, pred in zip(rows, predictions)
            if pred.get(key) != row["reference"][key]
        ]
        report["fields"][key] = {
            "correct": len(rows) - len(mismatches),
            "total": len(rows),
            "accuracy": (len(rows) - len(mismatches)) / len(rows) if rows else None,
        }
        report["errors"][key] = mismatches
    correct = sum(value["correct"] for value in report["fields"].values())
    report["overall"] = {"correct": correct, "total": len(rows) * 3, "accuracy": correct / (len(rows) * 3) if rows else None}
    return report


def predict(response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions: dict[str, Any] = {}
    distributions: dict[str, Any] = {}
    for key, result in response.results.items():
        if hasattr(result, "selected"):
            predictions[key] = result.selected
            distributions[key] = dict(result.probabilities)
        elif key == "urgent":
            predictions[key] = result.value >= 0.5
            distributions[key] = {"unsupported": 1.0 - result.value, "supported": result.value}
        else:
            predictions[key] = max(result.distribution, key=result.distribution.get)
            distributions[key] = dict(result.distribution)
    return predictions, distributions


def archive_qwen_decision(row: dict[str, Any]) -> dict[str, Any]:
    out = row.get("decengine", {}).get("normalized_decisions", {})
    impacts = out.get("impact", {}).get("distribution", {})
    return {
        "owner": out.get("owner", {}).get("selected"),
        "urgent": out.get("urgent", {}).get("value", 0.0) >= 0.5,
        "impact": max(impacts, key=impacts.get) if impacts else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURE)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--judgments", type=Path, default=JUDGMENTS)
    parser.add_argument("--model-store", type=Path, default=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store"))
    parser.add_argument("--native-library", type=Path, default=ROOT / "target/release/libdecengine.dylib")
    parser.add_argument("--output", type=Path, default=ROOT / "results/qwen-development-sweeps")
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS))
    parser.add_argument("--limit", type=int, help="smoke test using only first N development cases")
    args = parser.parse_args()

    cases = read_prefix(args.fixtures)
    archives = read_prefix(args.archive)
    judgments = read_prefix(args.judgments)
    if len(cases) != 50 or len(archives) != 50 or len(judgments) != 50:
        raise SystemExit("development inputs must provide cases 001-050 as their first 50 records")
    if args.limit is not None:
        if not 1 <= args.limit <= 50:
            raise SystemExit("--limit must be in 1..50")
        cases, archives, judgments = cases[:args.limit], archives[:args.limit], judgments[:args.limit]
    for i, (case, archived, judgment) in enumerate(zip(cases, archives, judgments), 1):
        expected = f"case-{i:03}"
        if str(case["id"]) != expected or archived.get("case_id") != expected or judgment.get("case_id") != expected:
            raise SystemExit(f"development files must align in stable order at {expected}")
        if archived.get("input") != case:
            raise SystemExit(f"archived input mismatch for {expected}")

    from decengine import Engine
    from compare_jev import decode_questions, native_response_dict

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    fixture_hash = hashlib.sha256(prefix_bytes(args.fixtures)).hexdigest()
    archive_prefix_hash = hashlib.sha256("\n".join(json.dumps(x, sort_keys=True) for x in archives).encode()).hexdigest()
    manifest = (ROOT / "models/manifests/qwen3-embedding-0.6b.json").read_bytes()
    install_path = args.model_store / "models/Qwen--Qwen3-Embedding-0.6B/install.json"
    install = install_path.read_bytes()
    selected = args.variant or list(VARIANTS)
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "backend": "real MLX inference",
        "network_calls": False,
        "evaluation_scope": "development cases 001-050 only",
        "holdout_file_accessed": False,
        "dataset": str(args.fixtures),
        "development_fixture_prefix_sha256": fixture_hash,
        "case_count": len(cases),
        "archive": str(args.archive),
        "development_archive_prefix_sha256": archive_prefix_hash,
        "base_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "installed_record_sha256": hashlib.sha256(install).hexdigest(),
        "native_library": str(args.native_library),
        "model_store": str(args.model_store),
        "temperature": 0.05,
        "decision_threshold": 0.5,
        "variant_configs": {name: {**BASELINE, **VARIANTS[name]} for name in selected},
        "task_text": TASK_BY_KIND,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summaries = {}
    for variant in selected:
        options = {"engine": "mlx", "model_store": str(args.model_store), **BASELINE, **VARIANTS[variant]}
        archive_by_id = {r["case_id"]: archive_qwen_decision(r) for r in archives}
        judgment_by_id = {r["case_id"]: r["reference"]["decision"] for r in judgments}
        predictions = []
        baseline_predictions = []
        errors = 0
        latencies = []
        with Engine(MODEL, native_library=args.native_library, options=options) as engine, (run_dir / f"{variant}.jsonl").open("w", encoding="utf-8") as out:
            for case in cases:
                cid = str(case["id"])
                start = time.perf_counter_ns()
                row: dict[str, Any] = {
                    "case_id": cid,
                    "variant": variant,
                    "reference": judgment_by_id[cid],
                    "archived_qwen": archive_by_id[cid],
                }
                try:
                    result = engine.decide(state=case["request"]["state"], questions=decode_questions(case["request"]["questions"]))
                    prediction, distributions = predict(result)
                    row.update(prediction=prediction, distributions=distributions, raw_response=native_response_dict(result), error=None)
                except Exception as exc:
                    errors += 1
                    prediction = {"owner": None, "urgent": None, "impact": None}
                    row.update(prediction=prediction, distributions=None, raw_response=None, error={"type": type(exc).__name__, "message": str(exc)})
                elapsed = (time.perf_counter_ns() - start) / 1e6
                latencies.append(elapsed)
                row["elapsed_ms"] = elapsed
                predictions.append(prediction)
                baseline_predictions.append(archive_by_id[cid])
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
        refs = [{"case_id": str(c["id"]), "reference": judgment_by_id[str(c["id"])]} for c in cases]
        summaries[variant] = {
            "inference_errors": errors,
            "development": summarize(refs, predictions),
            "archived_qwen_baseline_same_cases": summarize(refs, baseline_predictions),
            "timing_ms": {"median": statistics.median(latencies), "mean": statistics.mean(latencies)},
        }
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Development-only sweep saved: {run_dir}")
    print(json.dumps({k: {"development": v["development"]["overall"], "fields": v["development"]["fields"], "errors": v["inference_errors"]} for k, v in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
