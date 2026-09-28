#!/usr/bin/env python3
"""Extend the frozen-probe comparison to test-001..300 without retraining.

Default mode exports real MLX task embeddings, applies the locked 20260923 probes,
and runs local cosine + JEV baselines only for cases 091..300. ``--dry-run`` only
validates fixtures and locked artifacts; it never reads the API key or loads models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results"))
sys.path.insert(0, str(ROOT / "python"))

FIXTURE = ROOT / "benchmarks/cases/probe-test-300.jsonl"
BASELINE = ROOT / "results/probes/baselines/20260923T094641685361000Z"
ARTIFACTS = ROOT / "results/probes/runs/20260923T094618Z"
STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
DEFAULT_CLI = ROOT / "target/release/decengine"
MODELS = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "harrier": "microsoft/harrier-oss-v1-0.6b",
}
FEATURE_NAMES = {
    "qwen": "Qwen-Qwen3-Embedding-0-6B",
    "harrier": "microsoft-harrier-oss-v1-0-6b",
}
TASKS = ("owner", "urgent", "impact")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n}: expected object")
            result.append(row)
    return result


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def validate(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.fixture.is_file():
        raise FileNotFoundError(f"fixture not ready: {args.fixture}")
    cases = jsonl(args.fixture)
    ids = [str(c.get("id", "")) for c in cases]
    if len(cases) != 300 or ids != [f"test-{i:03d}" for i in range(1, 301)]:
        raise ValueError("expected exactly ordered unique test-001..test-300 records")
    for c in cases:
        ref = c.get("reference")
        req = c.get("request")
        if not isinstance(ref, dict) or not all(k in ref for k in TASKS):
            raise ValueError(f"{c['id']}: expected owner/urgent/impact reference labels")
        if not isinstance(req, dict) or not isinstance(req.get("state"), dict) or not isinstance(req.get("questions"), dict):
            raise ValueError(f"{c['id']}: missing request state/questions")
    for model in MODELS:
        for path in (args.artifacts / model / "task-specific" / "metadata.json",
                     args.artifacts / model / "task-specific" / "models.json"):
            if not path.is_file():
                raise FileNotFoundError(f"locked fitted probe artifact missing: {path}")
    for name in ("qwen", "harrier", "jev"):
        p = args.baseline / f"{name}-predictions.jsonl"
        if not p.is_file():
            raise FileNotFoundError(f"immutable first-90 baseline missing: {p}")
        old = jsonl(p)
        if [str(x.get("id")) for x in old] != ids[:90]:
            raise ValueError(f"{p} must contain test-001..test-090 in order")
    return cases


def run_exports(cases_file: Path, outdir: Path, cli: Path, store: Path) -> dict[str, Path]:
    outputs = {}
    (outdir / "features").mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        out = outdir / "features" / f"{name}-test300.jsonl"
        cmd = [str(cli), "--home", str(store), "export-features", "--input", str(cases_file), "--output", str(out), "--model", model]
        subprocess.run(cmd, cwd=ROOT, check=True)
        outputs[name] = out
    return outputs


def model_scalar(name: str, decisions: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    owner = decisions.get("owner", {})
    urgent = decisions.get("urgent", {})
    impact = decisions.get("impact", {})
    value = urgent.get("value")
    probs = impact.get("probabilities") if name == "jev" else impact.get("distribution")
    labels = ({str(i): str(level["label"]) for i, level in enumerate(questions["impact"]["levels"])}
              if name == "jev" else {})
    vals = [(str(k), v) for k, v in probs.items() if isinstance(v, (int, float))] if isinstance(probs, dict) else []
    win = [k for k, v in vals if v == max((x[1] for x in vals), default=None)] if vals else []
    return {"owner": owner.get("selected"), "urgent": value >= .5 if isinstance(value, (int, float)) else None,
            "impact": labels.get(win[0], win[0]) if len(win) == 1 else None}


def run_local_and_jev(cases: list[dict[str, Any]], outdir: Path, key_file: Path,
                      lib: Path, store: Path, timeout: float) -> dict[str, list[dict[str, Any]]]:
    from compare_jev import (DEFAULT_URL, decode_questions, jev_questions, native_response_dict,
                             normalize_jev, normalize_native, post_jev, redact)
    from decengine import Engine

    if not lib.is_file():
        raise FileNotFoundError(f"native library missing: {lib}")
    # Deliberately read only after fixture/artifact validation and immediately redact any output.
    key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("JEV key file is empty")
    new = cases[90:]
    raw_path = outdir / "raw-new210.jsonl"
    predictions = {m: [] for m in (*MODELS, "jev")}
    run_info = {"started_at": datetime.now(timezone.utc).isoformat(), "case_count": len(new),
                "scope": "test-091..test-300 only; prior 90 baseline files are read-only"}
    dump_json(outdir / "local-api-metadata.json", run_info)
    with raw_path.open("w", encoding="utf-8") as raw_out:
        with __import__("contextlib").ExitStack() as stack:
            engines = {name: stack.enter_context(Engine(model, native_library=lib,
                        options={"engine": "mlx", "model_store": str(store)})) for name, model in MODELS.items()}
            for i, case in enumerate(new, 91):
                req = case["request"]
                record: dict[str, Any] = {"case_id": str(case["id"]), "input_sha256": hashlib.sha256(
                    json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
                for name, engine in engines.items():
                    start = time.perf_counter_ns()
                    try:
                        response = native_response_dict(engine.decide(state=req["state"], questions=decode_questions(req["questions"])))
                        decisions = normalize_native(response)
                        rec = {"raw_response": response, "normalized_decisions": decisions, "error": None}
                    except Exception as e:
                        decisions = {}
                        rec = {"raw_response": None, "normalized_decisions": None,
                               "error": f"{type(e).__name__}: {str(e).replace(key, '[REDACTED]')}"}
                    rec["timing_ms"] = (time.perf_counter_ns() - start) / 1e6
                    record[name] = rec
                    predictions[name].append({"id": str(case["id"]), **model_scalar(name, decisions, req["questions"])})
                payload = {"state": json.dumps(req["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                           "model": "jev-latest", "questions": jev_questions(req["questions"])}
                start = time.perf_counter_ns()
                try:
                    response, elapsed = post_jev(DEFAULT_URL, key, payload, timeout)
                    body = response["body"]
                    if isinstance(body, dict):
                        body = body.get("result", body)
                    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
                        raise RuntimeError("unexpected JEV response shape")
                    decisions = normalize_jev(body, req["questions"])
                    rec = {"raw_response": response, "normalized_decisions": decisions, "timing_ms": elapsed, "error": None}
                except Exception as e:
                    decisions = {}
                    rec = {"raw_response": None, "normalized_decisions": None,
                           "timing_ms": (time.perf_counter_ns() - start) / 1e6,
                           "error": f"{type(e).__name__}: {str(e).replace(key, '[REDACTED]')}"}
                record["jev"] = rec
                predictions["jev"].append({"id": str(case["id"]), **model_scalar("jev", decisions, req["questions"])})
                record["jev_request_payload"] = payload
                raw_out.write(json.dumps(redact(record, key), ensure_ascii=False, allow_nan=False) + "\n")
                raw_out.flush()
                print(f"[{i}/300] {case['id']} complete", flush=True)
    for name, rows in predictions.items():
        write_jsonl(outdir / f"{name}-predictions-new210.jsonl", rows)
    return predictions


def wilson(correct: int, total: int) -> list[float] | None:
    if not total: return None
    z = 1.959963984540054
    p = correct / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0., mid-half), min(1., mid+half)]


def metrics(cases: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]], outdir: Path,
            baseline: Path) -> dict[str, Any]:
    byid = {str(c["id"]): c for c in cases}
    # Read and preserve immutable baseline predictions; they are never regenerated or rewritten.
    allpred = {}
    for name in ("qwen", "harrier", "jev"):
        oldpath = baseline / f"{name}-predictions.jsonl"
        earlier = jsonl(oldpath)
        later = predictions.get(name)
        if later is None:
            later = jsonl(outdir / f"{name}-predictions-new210.jsonl")
        allpred[name] = (earlier, later)
        write_jsonl(outdir / f"{name}-predictions-pooled300.jsonl", earlier + later)
    for name in ("probe-qwen", "probe-harrier"):
        full = jsonl(outdir / f"{name}-predictions-all300.jsonl")
        # Preserve the inference output with probabilities on disk; score only its labels.
        labels = [{"id": row["id"], **{
            task: (str(row[task]["label"]).lower() == "true" if task == "urgent" else row[task]["label"])
            for task in TASKS}} for row in full]
        allpred[name] = ([], labels)
    result: dict[str, Any] = {"slices": {}, "sources": {"earlier90": "immutable pre-existing baseline",
             "new210": "fresh predictions for test-091..test-300", "pooled300": "earlier90 + new210"}}
    for slice_name, indices in (("new210", range(90, 300)), ("pooled300", range(300))):
        row = {}
        for system, (older, newer) in allpred.items():
            if system.startswith("probe-"):
                predrows = newer[90:] if slice_name == "new210" else newer
            else:
                predrows = newer if slice_name == "new210" else older + newer
            expected_ids = [str(cases[i]["id"]) for i in indices]
            if [str(p.get("id")) for p in predrows] != expected_ids:
                raise ValueError(f"{system}/{slice_name}: prediction ID order/count mismatch")
            tasks = {}
            complete = 0
            for task in TASKS:
                correct = sum(p.get(task) == byid[str(p["id"])] ["reference"].get(task) for p in predrows)
                classes = sorted({str(byid[str(p["id"])] ["reference"][task]) for p in predrows})
                classstats = {}
                for cls in classes:
                    support = sum(str(byid[str(p["id"])] ["reference"][task]) == cls for p in predrows)
                    predicted = sum(str(p.get(task)) == cls for p in predrows)
                    tp = sum(str(p.get(task)) == cls and
                             str(byid[str(p["id"])] ["reference"][task]) == cls for p in predrows)
                    precision = tp/predicted if predicted else None
                    recall = tp/support if support else None
                    classstats[cls] = {"support": support, "precision": precision,
                                       "recall": recall,
                                       "f1": (2 * precision * recall / (precision + recall)
                                              if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall is not None else None),
                                       "accuracy_within_class": tp/support if support else None,
                                       "accuracy_wilson_95_ci": wilson(tp, support)}
                tasks[task] = {"correct": correct, "total": len(predrows), "accuracy": correct/len(predrows),
                               "wilson_95_ci": wilson(correct, len(predrows)), "per_class": classstats}
            for p in predrows:
                ref = byid[str(p["id"])] ["reference"]
                complete += all(p.get(t) == ref.get(t) for t in TASKS)
            row[system] = {"per_question": tasks, "complete_case": {"correct": complete, "total": len(predrows),
                         "accuracy": complete/len(predrows), "wilson_95_ci": wilson(complete, len(predrows))},
                         "overall_question_mean_accuracy": sum(v["accuracy"] for v in tasks.values())/len(TASKS)}
        result["slices"][slice_name] = row
    raw_rows = jsonl(outdir / "raw-new210.jsonl")
    timing = {}
    for name in (*MODELS.keys(), "jev"):
        values = [float(r[name]["timing_ms"]) for r in raw_rows if r.get(name, {}).get("timing_ms") is not None]
        failures = sum(r.get(name, {}).get("error") is not None for r in raw_rows)
        timing[name] = {"n": len(values), "errors": failures,
                        "mean_ms": statistics.fmean(values) if values else None,
                        "median_ms": statistics.median(values) if values else None,
                        "p95_ms": sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None}
    result["new210_latency"] = timing
    dump_json(outdir / "metrics.json", result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    p.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    p.add_argument("--native-library", type=Path, default=ROOT / "target/release/libdecengine.dylib")
    p.add_argument("--model-store", type=Path, default=STORE)
    p.add_argument("--key-file", type=Path, default=Path.home() / "Documents/Code/jev.api.key")
    p.add_argument("--out", type=Path, help="output run directory (default: timestamped under results/probes/comparisons)")
    p.add_argument("--timeout", type=float, default=120.)
    p.add_argument("--dry-run", action="store_true", help="validate only; no key/model/API access")
    args = p.parse_args()
    try:
        cases = validate(args)
        if args.dry_run:
            print(f"Ready: validated {len(cases)} cases and locked artifacts; no model/API calls made.")
            return 0
        if not args.cli.is_file(): raise FileNotFoundError(f"decengine CLI missing: {args.cli}")
        runid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = args.out or ROOT / "results/probes/comparisons" / runid
        outdir = outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=False)
        locked = {}
        for model in MODELS:
            artifact = args.artifacts / model / "task-specific"
            locked[model] = {"metadata_sha256": sha(artifact / "metadata.json"),
                             "models_sha256": sha(artifact / "models.json"),
                             "selected_C": {task: json.loads((artifact / "models.json").read_text())[task]["C"]
                                            for task in TASKS}}
        dump_json(outdir / "run-metadata.json", {"run_id": runid, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(args.fixture), "fixture_sha256": sha(args.fixture), "case_count": len(cases),
            "baseline_directory": str(args.baseline), "baseline_files_sha256": {
                f: sha(args.baseline / f) for f in ("qwen-predictions.jsonl", "harrier-predictions.jsonl", "jev-predictions.jsonl")},
            "probe_artifacts": str(args.artifacts), "locked_probe_parameters": locked,
            "hyperparameters": "loaded from locked models.json; no fitting/tuning",
            "scope": "JEV and local cosine baselines called only for test-091..test-300"})
        features = run_exports(args.fixture, outdir, args.cli, args.model_store)
        from compare_jev import load_cases
        # Run existing inference-only predictor, keeping all 300 embeddings for pooled comparison.
        probe_predictions = {}
        for name in MODELS:
            out = outdir / f"probe-{name}-predictions-all300.jsonl"
            subprocess.run([sys.executable, str(ROOT / "results/probes/predict_probe.py"), "--artifacts",
                str(args.artifacts / name / "task-specific"), "--embeddings", str(features[name]), "--out", str(out)], cwd=ROOT, check=True)
            probe_predictions[name] = jsonl(out)
        # store alias expected by metrics routine
        for name, rows in probe_predictions.items(): write_jsonl(outdir / f"probe-{name}-predictions-all300.jsonl", rows)
        # New210 raw results only; first 90 cosine/JEV predictions are never contacted.
        predictions = run_local_and_jev(cases, outdir, args.key_file, args.native_library, args.model_store, args.timeout)
        # Metrics compares cosine/JEV new predictions plus immutable first 90 and probe predictions.
        result = metrics(cases, predictions, outdir, args.baseline)
        files = {str(x.relative_to(outdir)): sha(x) for x in outdir.rglob("*")
                 if x.is_file() and x.name != "checksums.json"}
        dump_json(outdir / "checksums.json", files)
        print(f"Completed comparison in {outdir}")
        return 0
    except Exception as e:
        print(f"compare_300: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
