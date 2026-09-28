#!/usr/bin/env python3
"""Reproducible, local-only prompt experiment for Qwen3-Embedding-0.6B.

Every candidate is evaluated by the installed MLX model. No request is sent to JEV.
The default split tunes on cases 001-050 and reports the selected variant on 051-100.
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
BASE_QUERY = "Instruct: {instruction}\nQuery:{text}"
TUNED_PROFILE_PATH = ROOT / "models/profiles/qwen3-embedding-0.6b-direct-rubric.experimental.json"
TUNED_PROFILE = json.loads(TUNED_PROFILE_PATH.read_text(encoding="utf-8"))

# Fixed a priori variants. Query-template spacing/formats and candidate wrappers
# are varied independently enough to diagnose the effect. Noul's rubric bags are
# still the compiler's current baseline and are recorded as such in provenance.
VARIANTS = {
    "baseline": {"query_template": BASE_QUERY, "candidate_template": "{text}"},
    "qwen_official_spaced": {
        "query_template": "Instruct: {instruction}\nQuery: {text}",
        "candidate_template": "{text}",
    },
    "binary_direct_rubric": {
        "query_template": TUNED_PROFILE["query_template"],
        "candidate_template": TUNED_PROFILE["candidate_template"],
        "noul_unsupported": TUNED_PROFILE["noul_unsupported"],
        "noul_supported": TUNED_PROFILE["noul_supported"],
    },
    "binary_decision_labels": {
        "query_template": BASE_QUERY, "candidate_template": "{text}",
        "noul_unsupported": "No: this does not require same-day action.",
        "noul_supported": "Yes: this requires same-day action.",
    },
    "passage_candidates": {
        "query_template": BASE_QUERY,
        "candidate_template": "Passage: {text}",
    },
}
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def accuracy(records: list[dict[str, Any]], ids: set[str]) -> dict[str, Any]:
    chosen = [r for r in records if r["case_id"] in ids]
    fields = ("owner", "urgent", "impact")
    summary: dict[str, Any] = {"cases": len(chosen), "fields": {}}
    errors = {}
    for field in fields:
        pairs = [(r["prediction"][field], r["reference"][field]) for r in chosen]
        correct = sum(a == b for a, b in pairs)
        summary["fields"][field] = {"correct": correct, "total": len(pairs), "accuracy": correct / len(pairs) if pairs else None}
        errors[field] = [{"case_id": r["case_id"], "predicted": r["prediction"][field], "reference": r["reference"][field]} for r in chosen if r["prediction"][field] != r["reference"][field]]
    summary["overall"] = {"correct": sum(v["correct"] for v in summary["fields"].values()), "total": len(chosen) * 3, "accuracy": sum(v["correct"] for v in summary["fields"].values()) / (len(chosen) * 3) if chosen else None}
    summary["errors"] = errors
    return summary


def normalize(response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions: dict[str, Any] = {}
    distributions: dict[str, Any] = {}
    for key, result in response.results.items():
        if hasattr(result, "selected"):
            predictions[key] = result.selected
            distributions[key] = dict(result.probabilities)
        elif hasattr(result, "value") and key == "urgent":
            predictions[key] = bool(result.value >= 0.5)
            distributions[key] = {"unsupported": 1.0 - result.value, "supported": result.value}
        else:
            predictions[key] = max(result.distribution, key=result.distribution.get)
            distributions[key] = dict(result.distribution)
    return predictions, distributions


def archive_prediction(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("decengine", {}).get("normalized_decisions") or {}
    return {
        "owner": raw.get("owner", {}).get("selected"),
        "urgent": raw.get("urgent", {}).get("value", 0) >= 0.5,
        "impact": max(raw.get("impact", {}).get("distribution", {"": 0}), key=raw.get("impact", {}).get("distribution", {"": 0}).get),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURE)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--judgments", type=Path, default=JUDGMENTS)
    parser.add_argument("--model-store", type=Path, default=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store"))
    parser.add_argument("--native-library", type=Path, default=ROOT / "target/release/libdecengine.dylib")
    parser.add_argument("--output", type=Path, default=ROOT / "results/qwen-prompt-experiments")
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS), help="repeat; default runs all variants")
    parser.add_argument("--limit", type=int, help="explicit smoke subset; do not use for final metrics")
    args = parser.parse_args()

    cases = read_jsonl(args.fixtures)
    if len(cases) != 100 and args.limit is None:
        raise SystemExit(f"expected 100 cases, got {len(cases)}")
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            raise SystemExit("--limit must be in 1..case_count")
        cases = cases[:args.limit]
    archived = {r["case_id"]: r for r in read_jsonl(args.archive)}
    judgments = {r["case_id"]: r for r in read_jsonl(args.judgments)}
    references = {cid: row["reference"]["decision"] for cid, row in judgments.items()}
    if len(archived) != 100 or set(references) != set(archived):
        raise SystemExit("reference archive/judgments must contain the original 100 cases")
    for case in cases:
        if archived[str(case["id"])].get("input") != case:
            raise SystemExit(f"fixture/archive mismatch: {case['id']}")
    from decengine import Engine
    from compare_jev import decode_questions, native_response_dict

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    fixture_bytes = args.fixtures.read_bytes()
    arch_bytes = args.archive.read_bytes()
    manifest_bytes = (ROOT / "models/manifests/qwen3-embedding-0.6b.json").read_bytes()
    install_path = args.model_store / "models/Qwen--Qwen3-Embedding-0.6B/install.json"
    install_bytes = install_path.read_bytes()
    variants = args.variant or list(VARIANTS)
    metadata = {
        "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "engine": "MLX real inference", "network_calls": False,
        "dataset": str(args.fixtures), "dataset_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "archive": str(args.archive), "archive_sha256": hashlib.sha256(arch_bytes).hexdigest(),
        "active_base_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "installed_model_record_sha256": hashlib.sha256(install_bytes).hexdigest(),
        "case_count": len(cases), "split": "development case-001..050; untouched holdout case-051..100",
        "variants": {name: {**VARIANTS[name], "temperature": "unchanged manifest default 0.05"} for name in variants},
        "baseline_noul_candidate_text": {
            "unsupported": "unsupported false no absent not required can wait",
            "supported": "supported true yes present urgent today immediate required",
        },
        "opt_in_profile": str(TUNED_PROFILE_PATH),
        "opt_in_profile_sha256": hashlib.sha256(TUNED_PROFILE_PATH.read_bytes()).hexdigest(),
        "native_library": str(args.native_library), "model_store": str(args.model_store),
        "python": sys.version.split()[0], "platform": sys.platform,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summaries = {}
    for variant in variants:
        path = run_dir / f"{variant}.jsonl"
        records = []
        errors = 0
        options = {"engine": "mlx", "model_store": str(args.model_store), **VARIANTS[variant]}
        with Engine(MODEL, native_library=args.native_library, options=options) as engine, path.open("w", encoding="utf-8") as out:
            for case in cases:
                cid = str(case["id"])
                start = time.perf_counter_ns()
                row = {"case_id": cid, "variant": variant, "reference": references[cid], "archive_qwen": archive_prediction(archived[cid]), "jev_reference_prediction": judgments[cid]["outputs"]["jev"]["normalized_predictions"]}
                try:
                    response = engine.decide(state=case["request"]["state"], questions=decode_questions(case["request"]["questions"]))
                    pred, dist = normalize(response)
                    row.update(prediction=pred, distributions=dist, raw_response=native_response_dict(response), error=None)
                except Exception as e:
                    errors += 1
                    row.update(prediction={"owner": None, "urgent": None, "impact": None}, distributions=None, raw_response=None, error={"type": type(e).__name__, "message": str(e)})
                row["elapsed_ms"] = (time.perf_counter_ns() - start) / 1e6
                records.append(row)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
        train_ids = {f"case-{i:03}" for i in range(1, 51)}
        hold_ids = {f"case-{i:03}" for i in range(51, 101)}
        archived_records = [dict(r, prediction=r["archive_qwen"]) for r in records]
        jev_records = [dict(r, prediction=r["jev_reference_prediction"]) for r in records]
        summaries[variant] = {
            "errors": errors,
            "development": accuracy(records, train_ids),
            "holdout": accuracy(records, hold_ids),
            "all": accuracy(records, {str(c["id"]) for c in cases}),
            "archived_qwen_same_cases": accuracy(archived_records, {str(c["id"]) for c in cases}),
            "archived_jev_same_cases": accuracy(jev_records, {str(c["id"]) for c in cases}),
            "reference_comparison": {
                "development_archived_qwen": accuracy(archived_records, train_ids),
                "development_archived_jev": accuracy(jev_records, train_ids),
                "holdout_archived_qwen": accuracy(archived_records, hold_ids),
                "holdout_archived_jev": accuracy(jev_records, hold_ids),
            },
            "timing_ms": {"median": statistics.median(r["elapsed_ms"] for r in records), "mean": statistics.mean(r["elapsed_ms"] for r in records)},
        }
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved real MLX predictions and report to {run_dir}")
    print(json.dumps({k: {"development": v["development"]["overall"], "holdout": v["holdout"]["overall"], "errors": v["errors"]} for k, v in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
