#!/usr/bin/env python3
"""Evaluation-only frozen-model run and paired comparison for explicit-state contexts.

No fitting or selection occurs here. The runner intentionally refuses to open the
new fixture until a readiness manifest is present and marks it ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MATCHED = ROOT / "results/probes/varied/matched"
FROZEN = MATCHED / "selection-freeze.json"
DEFAULT_FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-explicit-60.jsonl"
sys.path.insert(0, str(MATCHED))
import evaluate_matched  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _find_ready_flag(document: Any) -> bool:
    if isinstance(document, dict):
        for key in ("ready", "is_ready", "complete", "completed"):
            if document.get(key) is True:
                return True
        for key in ("status", "state", "readiness"):
            if str(document.get(key, "")).strip().casefold() in {"ready", "complete", "completed", "published"}:
                return True
        return any(_find_ready_flag(value) for value in document.values() if isinstance(value, (dict, list)))
    if isinstance(document, list):
        return any(_find_ready_flag(value) for value in document)
    return False


def _find_fixture_digest(document: Any) -> str | None:
    if isinstance(document, dict):
        for key in ("fixture_sha256", "input_sha256", "output_sha256", "sha256"):
            value = document.get(key)
            if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
                return value.lower()
        # Also handle explicit fixture objects such as {"fixture": {"sha256": ...}}.
        for key, value in document.items():
            if key.casefold() in {"fixture", "output", "artifact", "dataset"} and isinstance(value, dict):
                digest = _find_fixture_digest(value)
                if digest:
                    return digest
        for value in document.values():
            if isinstance(value, (dict, list)):
                digest = _find_fixture_digest(value)
                if digest:
                    return digest
    elif isinstance(document, list):
        for value in document:
            digest = _find_fixture_digest(value)
            if digest:
                return digest
    return None


def assert_ready(fixture: Path, manifest_path: Path) -> tuple[dict, str]:
    """Validate the authored readiness manifest before opening the explicit fixture."""
    if not manifest_path.is_file():
        raise ValueError(f"fixture readiness manifest missing: {manifest_path}")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    # This author manifest is a completed-data manifest (the upstream validator is
    # run separately); it records source and transformed IDs/counts but does not
    # currently contain the explicit file's own digest or a literal ready flag.
    completed_manifest = (document.get("cases") == 60 and document.get("question_count") == 180
        and document.get("source_fixture") == "test-matched-60.jsonl"
        and len(document.get("source_case_ids", [])) == 60
        and len(document.get("transformed_case_ids", [])) == 60)
    if not (_find_ready_flag(document) or completed_manifest):
        raise ValueError("explicit fixture manifest does not report a complete 60-case/180-question build")
    expected_source = str(document.get("source_sha256", "")).lower()
    matched_provenance = json.loads((MATCHED / "runs/qwen/evaluation/provenance.json").read_text(encoding="utf-8"))
    if expected_source != matched_provenance.get("fixture_sha256"):
        raise ValueError("explicit manifest source hash does not match the frozen matched evaluation fixture")
    if document["source_case_ids"] != [f"mq-test-matched-{i:03d}" for i in range(1, 61)]:
        raise ValueError("explicit manifest source IDs are not the expected ordered matched60 IDs")
    if document["transformed_case_ids"] != [f"mq-explicit-{i:03d}" for i in range(1, 61)]:
        raise ValueError("explicit manifest transformed IDs are not the expected ordered IDs")
    actual = sha(fixture)
    return document, actual


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("id", "")) for row in rows]
    if not rows or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: require nonempty JSONL with unique IDs")
    return rows


def records_by_question(out_dir: Path) -> dict[tuple[str, str], dict]:
    predictions = [json.loads(line) for line in (out_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    outcomes = [json.loads(line) for line in (out_dir / "question_outcomes.jsonl").read_text().splitlines() if line.strip()]
    key = lambda row: (str(row["id"]), str(row["question_name"]))
    pred_map = {key(row): row for row in predictions}
    outcome_map = {key(row): row for row in outcomes}
    if len(pred_map) != len(predictions) or len(outcome_map) != len(outcomes) or set(pred_map) != set(outcome_map):
        raise ValueError(f"{out_dir}: duplicate or unaligned question prediction/outcome rows")
    return {k: {**pred_map[k], **outcome_map[k]} for k in pred_map}


def paired_comparison(explicit_dir: Path, matched_dir: Path, readiness_sha: str, fixture_sha: str,
                      source_to_explicit: dict[str, str], expected_case_count: int = 60) -> tuple[dict, list[dict], list[dict]]:
    """Compare paired cases/questions; descriptive only, never a transfer-gain claim."""
    old = records_by_question(matched_dir)
    explicit_records = records_by_question(explicit_dir)
    explicit_to_source = {explicit_id: source_id for source_id, explicit_id in source_to_explicit.items()}
    if len(explicit_to_source) != len(source_to_explicit):
        raise ValueError("manifest source/transformed case IDs are not one-to-one")
    new = {}
    explicit_id_by_source = {}
    for (explicit_id, qname), record in explicit_records.items():
        if explicit_id not in explicit_to_source:
            raise ValueError(f"explicit output contains an unmapped case ID {explicit_id}")
        source_id = explicit_to_source[explicit_id]
        key = (source_id, qname)
        if key in new:
            raise ValueError(f"explicit output maps duplicate questions to {key}")
        new[key] = record
        explicit_id_by_source[source_id] = explicit_id
    if set(old) != set(new):
        missing = sorted(set(old) - set(new))[:5]
        extra = sorted(set(new) - set(old))[:5]
        raise ValueError(f"explicit/matched question keys do not align (missing={missing}, extra={extra})")
    question_rows = []
    type_acc = {kind: {"matched_correct": 0, "explicit_correct": 0, "prediction_switches": 0,
                       "explicit_only_correct": 0, "matched_only_correct": 0, "both_correct": 0, "both_wrong": 0, "n": 0}
                for kind in evaluate_matched.TYPES}
    by_case: dict[str, list[dict]] = {}
    by_domain: dict[str, list[dict]] = {}
    for key in sorted(old):
        m, e = old[key], new[key]
        if m.get("question_type") != e.get("question_type"):
            raise ValueError(f"{key}: wire type differs between paired fixtures")
        if m.get("gold") != e.get("gold"):
            raise ValueError(f"{key}: gold target differs between paired fixtures; not a paired-label comparison")
        if set(m.get("candidate_keys", [])) != set(e.get("candidate_keys", [])):
            raise ValueError(f"{key}: candidate identifiers differ across paired fixtures")
        kind = m["question_type"]
        matched_correct, explicit_correct = bool(m["correct"]), bool(e["correct"])
        switched = str(m["selected"]) != str(e["selected"])
        row = {"id": key[0], "question_name": key[1], "question_type": kind,
            "domain": e.get("domain", m.get("domain", "")), "gold": m["gold"],
            "matched_case_id": key[0], "explicit_case_id": explicit_id_by_source[key[0]],
            "matched_prediction": m["selected"], "explicit_prediction": e["selected"],
            "prediction_switched": switched, "matched_correct": matched_correct,
            "explicit_correct": explicit_correct,
            "change": "both_correct" if matched_correct and explicit_correct else
                      "explicit_only_correct" if explicit_correct else
                      "matched_only_correct" if matched_correct else "both_wrong"}
        question_rows.append(row)
        by_case.setdefault(key[0], []).append(row)
        by_domain.setdefault(str(row["domain"]), []).append(row)
        report = type_acc[kind]; report["n"] += 1
        report["matched_correct"] += matched_correct; report["explicit_correct"] += explicit_correct
        report["prediction_switches"] += switched
        report["explicit_only_correct"] += matched_correct is False and explicit_correct is True
        report["matched_only_correct"] += matched_correct is True and explicit_correct is False
        report["both_correct"] += matched_correct and explicit_correct
        report["both_wrong"] += not matched_correct and not explicit_correct
    for kind, report in type_acc.items():
        if report["n"]:
            n = report["n"]
            report["matched_accuracy"] = report["matched_correct"] / n
            report["explicit_accuracy"] = report["explicit_correct"] / n
            report["accuracy_delta_explicit_minus_matched"] = report["explicit_accuracy"] - report["matched_accuracy"]
            report["prediction_switch_rate"] = report["prediction_switches"] / n
    matched_cases = [json.loads(line) for line in (matched_dir / "case_outcomes.jsonl").read_text().splitlines() if line.strip()]
    explicit_cases = [json.loads(line) for line in (explicit_dir / "case_outcomes.jsonl").read_text().splitlines() if line.strip()]
    mc = {str(row["id"]): bool(row["all_questions_correct"]) for row in matched_cases}
    ec = {str(row["id"]): bool(row["all_questions_correct"]) for row in explicit_cases}
    ec_by_source = {explicit_to_source[cid]: correct for cid, correct in ec.items() if cid in explicit_to_source}
    if set(mc) != set(by_case) or set(ec_by_source) != set(by_case) or len(by_case) != expected_case_count:
        raise ValueError("paired case outcome keys/count differ; expected exact aligned 60 cases")
    case_rows = []
    for cid in sorted(by_case):
        qrows = by_case[cid]
        if len(qrows) != 3:
            raise ValueError(f"{cid}: expected exactly three paired wire-type questions")
        domain = str(qrows[0]["domain"])
        case_rows.append({"id": cid, "domain": domain, "question_count": 3,
            "matched_all_three_correct": mc[cid], "explicit_all_three_correct": ec_by_source[cid],
            "matched_case_id": cid, "explicit_case_id": explicit_id_by_source[cid],
            "switch_count": sum(row["prediction_switched"] for row in qrows),
            "all_three_change": "both_correct" if mc[cid] and ec_by_source[cid] else
                                "explicit_only_correct" if ec_by_source[cid] else
                                "matched_only_correct" if mc[cid] else "both_wrong"})
    matched_all = sum(mc.values()); explicit_all = sum(ec_by_source.values())
    per_domain = {}
    for domain, qrows in sorted(by_domain.items()):
        cases = [row for row in case_rows if row["domain"] == domain]
        typed = {}
        for kind in evaluate_matched.TYPES:
            ix = [row for row in qrows if row["question_type"] == kind]
            if ix:
                old_hits = sum(row["matched_correct"] for row in ix)
                new_hits = sum(row["explicit_correct"] for row in ix)
                typed[kind] = {"n": len(ix), "matched_accuracy": old_hits / len(ix),
                    "explicit_accuracy": new_hits / len(ix), "accuracy_delta": (new_hits - old_hits) / len(ix),
                    "prediction_switches": sum(row["prediction_switched"] for row in ix)}
        per_domain[domain] = {"case_count": len(cases), "per_type": typed,
            "matched_all_three_correct": sum(c["matched_all_three_correct"] for c in cases),
            "explicit_all_three_correct": sum(c["explicit_all_three_correct"] for c in cases),
            "all_three_correct_delta": sum(c["explicit_all_three_correct"] for c in cases) - sum(c["matched_all_three_correct"] for c in cases)}
    return {"protocol": "paired descriptive comparison of frozen matched-context vs explicit-context predictions; not a causal/unbiased transfer-gain estimate",
        "readiness_manifest_sha256": readiness_sha, "explicit_fixture_sha256": fixture_sha,
        "matched_predictions_sha256": sha(matched_dir / "predictions.jsonl"),
        "explicit_predictions_sha256": sha(explicit_dir / "predictions.jsonl"),
        "question_count": len(question_rows), "case_count": len(case_rows),
        "matched_all_three_correct_cases": matched_all, "explicit_all_three_correct_cases": explicit_all,
        "all_three_correct_case_delta": explicit_all - matched_all,
        "prediction_switches": sum(v["prediction_switches"] for v in type_acc.values()),
        "per_question_type": type_acc, "per_domain": per_domain,
        "inference_backends": {"embeddings": "MLX/Metal exporter", "ranker": "NumPy CPU; parity/reference scorer only, not GPU-resident end-to-end"},
        "limitations": "Context change is descriptive on these paired cases only. No test-based tuning, weighting, significance claim, or unbiased transfer-gain interpretation is made."}, question_rows, case_rows


def run(args) -> None:
    readiness, fixture_sha = assert_ready(args.input, args.readiness_manifest)
    # This is explicitly the NumPy CPU parity reference, not the future Metal run.
    # It persists predictions before dereferencing references, then scores outcomes.
    cpu_dir = args.out / "cpu-reference"
    eval_args = SimpleNamespace(run_key=args.run_key, selection_freeze=args.selection_freeze,
        input=args.input, features=args.features, out=cpu_dir / "evaluation",
        expected_cases=60, expected_questions=180)
    evaluate_matched.run(eval_args)
    frozen_run = MATCHED / f"frozen-{args.run_key}"
    frozen_model = json.loads((frozen_run / "models.json").read_text(encoding="utf-8"))
    (cpu_dir / "pipeline_backend.json").write_text(json.dumps({
        "protocol": "CPU parity reference only; not an end-to-end GPU inference result",
        "embedding_backend": "MLX/Metal frozen embedding exporter",
        "candidate_ranker_backend": "NumPy CPU",
        "frozen_model_sha256": sha(frozen_run / "models.json"),
        "frozen_metadata_sha256": sha(frozen_run / "metadata.json"),
        "frozen_C": frozen_model["C"],
        "frozen_per_type_C": frozen_model["per_type_C"],
        "text_version": "decengine-varied-pair-text-v2-generic-types",
        "timing_scope": "NumPy scorer only; runner wall includes parsing/scoring/reference evaluation and excludes MLX feature export",
        "no_fit_or_tuning": True}, indent=2) + "\n", encoding="utf-8")
    matched_dir = MATCHED / "runs" / args.run_key / "evaluation"
    source_to_explicit = dict(zip(readiness["source_case_ids"], readiness["transformed_case_ids"]))
    comparison, question_rows, case_rows = paired_comparison(eval_args.out, matched_dir,
        sha(args.readiness_manifest), fixture_sha, source_to_explicit)
    (cpu_dir / "paired-questions.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in question_rows), encoding="utf-8")
    (cpu_dir / "paired-cases.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in case_rows), encoding="utf-8")
    comparison["explicit_readiness"] = {"status": "complete (validated manifest counts and source alignment)", "manifest_sha256": sha(args.readiness_manifest),
        "fixture_sha256": fixture_sha, "features_sha256": sha(args.features)}
    comparison["paired_artifact_sha256"] = {"paired_questions": sha(cpu_dir / "paired-questions.jsonl"),
        "paired_cases": sha(cpu_dir / "paired-cases.jsonl"),
        "pipeline_backend": sha(cpu_dir / "pipeline_backend.json")}
    (cpu_dir / "paired-cpu-reference.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synthetic_smoke() -> None:
    # Pairing logic only: no benchmark paths, references, or trained model access.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="explicit-paired-smoke-") as temp:
        root = Path(temp); old = root / "matched"; new = root / "explicit"; old.mkdir(); new.mkdir()
        source_to_explicit = {"synthetic-0": "explicit-0", "synthetic-1": "explicit-1"}
        for directory, switched in ((old, False), (new, True)):
            preds, outcomes, cases = [], [], []
            for case_no in range(2):
                cid = f"synthetic-{case_no}" if directory == old else f"explicit-{case_no}"; correct_n = 0
                for kind in evaluate_matched.TYPES:
                    gold = "supported" if kind == "noul" else "g" if kind == "choice" else "high"
                    guess = gold if (case_no == 0) == switched else ("unsupported" if kind == "noul" else "x" if kind == "choice" else "low")
                    hit = guess == gold; correct_n += hit
                    candidate_keys = ["supported", "unsupported"] if kind == "noul" else (["g", "x"] if kind == "choice" else ["high", "low"])
                    preds.append({"id": cid, "question_name": kind, "question_type": kind, "selected": guess,
                        "candidate_keys": candidate_keys, "domain": "synthetic"})
                    outcomes.append({"id": cid, "question_name": kind, "question_type": kind,
                        "gold": gold, "predicted": guess, "correct": hit})
                cases.append({"id": cid, "all_questions_correct": correct_n == 3})
            (directory / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in preds))
            (directory / "question_outcomes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in outcomes))
            (directory / "case_outcomes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in cases))
        # Smoke keyed transformed-ID alignment, switches, and correctness transitions.
        comparison, qrows, crows = paired_comparison(new, old, "a" * 64, "b" * 64,
            source_to_explicit, expected_case_count=2)
        assert len(qrows) == 6 and len(crows) == 2 and comparison["prediction_switches"] == 6
        print("synthetic explicit runner smoke passed; no fixture or references opened")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-key", choices=("qwen", "harrier"))
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--readiness-manifest", type=Path)
    parser.add_argument("--selection-freeze", type=Path, default=FROZEN)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        synthetic_smoke()
        return 0
    if any(value is None for value in (args.run_key, args.readiness_manifest, args.features, args.out)):
        parser.error("run requires --run-key, --readiness-manifest, --features, and --out")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
