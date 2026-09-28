#!/usr/bin/env python3
"""Measure the existing MLX exporter plus saved-classifier latency for locked probes.

The encoder is loaded once by a dedicated `decengine export-features` process for each
model. The current CLI exports four vectors (three task queries plus unused state); this
is explicitly reported and is not equivalent to a three-query GPU-resident probe.
Models run sequentially to avoid concurrent GPU use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results/probes"))
import predict_probe as predictor

FIXTURE = ROOT / "benchmarks/cases/probe-test-300.jsonl"
ARTIFACTS = ROOT / "results/probes/runs/20260923T094618Z"
ARCHIVE = ROOT / "results/probes/comparisons/20260923T104208Z"
CLI = ROOT / "target/release/decengine"
STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
MODEL_IDS = {"qwen": "Qwen/Qwen3-Embedding-0.6B", "harrier": "microsoft/harrier-oss-v1-0.6b"}
TASKS = ("owner", "urgent", "impact")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float:
    return sorted(values)[math.ceil(p * len(values)) - 1]


def bench_model(name: str, out: Path, cli: Path, store: Path, fixture: Path,
                artifacts: Path, archive: Path) -> dict[str, Any]:
    model = MODEL_IDS[name]
    dest = out / name
    dest.mkdir(parents=True)
    embeddings = dest / "features.jsonl"
    predictions_path = dest / "predictions.jsonl"
    cmd = [str(cli), "--home", str(store), "export-features", "--input", str(fixture),
           "--output", str(embeddings), "--model", model]
    started = time.perf_counter_ns()
    subprocess.run(cmd, cwd=ROOT, check=True)
    export_process_ms = (time.perf_counter_ns() - started) / 1e6

    metadata_path = artifacts / name / "task-specific/metadata.json"
    models_path = artifacts / name / "task-specific/models.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    models = json.loads(models_path.read_text(encoding="utf-8"))
    feature_keys = metadata["feature_keys_by_task"]
    expected = metadata["embedding_manifests"]["train"]["contents"]
    manifest_path = Path(str(embeddings) + ".manifest.json")
    manifest = predictor.validate_manifest(manifest_path, model, expected, metadata["embedding_dimensions"])
    records = predictor.read_records(embeddings, model, feature_keys, metadata["embedding_dimensions"])
    if len(records) != 300:
        raise ValueError(f"{name}: expected 300 records")

    inference_ns: list[int] = []
    case_timings = []
    results = []
    for record in records:
        t0 = time.perf_counter_ns()
        row = {"id": record["id"], "model": record["model"]}
        for task in TASKS:
            x = __import__("numpy").asarray(record["embeddings"][feature_keys[task]], dtype=__import__("numpy").float64)[None, :]
            probs = predictor.probabilities(x, models[task])[0]
            classes = models[task]["classes"]
            winner = int(probs.argmax())
            row[task] = {"label": str(classes[winner]),
                         "probabilities": {str(classes[i]): float(probs[i]) for i in range(len(classes))},
                         "confidence": float(probs[winner])}
        inference_ns.append(time.perf_counter_ns() - t0)
        case_timings.append({"id": str(record["id"]), "classifier_ns": inference_ns[-1],
                             "classifier_ms": inference_ns[-1] / 1e6,
                             "encoder_case_timing": None})
        results.append(row)
    predictions_path.write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n" for x in results), encoding="utf-8")
    (dest / "case-timings.jsonl").write_text("".join(json.dumps(x, allow_nan=False, sort_keys=True) + "\n" for x in case_timings), encoding="utf-8")

    expected_path = archive / f"probe-{name}-predictions-all300.jsonl"
    expected_rows = read_jsonl(expected_path)
    if len(expected_rows) != 300 or [x["id"] for x in expected_rows] != [x["id"] for x in results]:
        raise ValueError(f"{name}: archived prediction count/order mismatch")
    diffs = []
    for old, new in zip(expected_rows, results):
        for task in TASKS:
            if old[task]["label"] != new[task]["label"]:
                raise ValueError(f"{name}/{old['id']}/{task}: archived label mismatch")
            for klass, p in old[task]["probabilities"].items():
                diffs.append(abs(float(p) - new[task]["probabilities"][klass]))
    if max(diffs, default=0) > 1e-12:
        raise ValueError(f"{name}: archived probability mismatch max={max(diffs)}")

    classifier_ms = [x / 1e6 for x in inference_ns]
    def stats(xs: list[float]) -> dict[str, float]:
        return {"p50_ms": statistics.median(xs), "p95_ms": percentile(xs, .95), "mean_ms": statistics.fmean(xs)}

    return {"model": model, "case_count": len(records), "embedding_calls_per_case": 4,
            "state_embedding": True, "timings": {"classifier_ms": stats(classifier_ms),
                "encoder_process_wall_ms_including_model_load": export_process_ms,
                "encoder_process_wall_amortized_per_case_ms_including_load": export_process_ms / 300,
                "end_to_end_case_wall_ms": None,
                "limitation": "existing CLI exposes no per-case encoder timers; do not interpret amortized whole-run cost as p50/p95 case latency"},
            "prediction_match": {"archive": str(expected_path), "labels_exact": True,
                "probabilities_max_abs_diff": max(diffs, default=0.0), "compared_probabilities": len(diffs)},
            "input_sha256": sha(fixture), "features_sha256": sha(embeddings),
            "features_manifest_sha256": sha(manifest_path), "manifest": manifest,
            "artifacts_sha256": {"metadata": sha(metadata_path), "models": sha(models_path)},
            "archive_sha256": sha(expected_path), "predictions_sha256": sha(predictions_path),
            "case_timings_sha256": sha(dest / "case-timings.jsonl"),
            "predictor_sha256": sha(Path(predictor.__file__)),
            "export_process_wall_ms_including_model_load_and_warmup": export_process_ms,
            "warmup": "none exposed by existing CLI; model load, first-call warmup, and output serialization included in encoder process wall time",
            "child_process_maxrss_platform_units_cumulative_high_water": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=[*MODEL_IDS, "all"], default="all")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--cli", type=Path, default=CLI)
    p.add_argument("--model-store", type=Path, default=STORE)
    args = p.parse_args()
    if not args.cli.is_file() or not args.fixture.is_file():
        p.error("CLI or fixture missing")
    args.out.mkdir(parents=True, exist_ok=False)
    names = list(MODEL_IDS) if args.model == "all" else [args.model]
    runs = {}
    for name in names:
        runs[name] = bench_model(name, args.out, args.cli, args.model_store, args.fixture, args.artifacts, args.archive)
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "device": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                   "encoder_backend": "MLX/Metal via decengine CLI", "classifier_backend": "NumPy CPU"},
        "fixture": str(args.fixture), "fixture_sha256": sha(args.fixture), "case_count": 300,
        "benchmark_script_sha256": sha(Path(__file__)),
        "reproduction_command": "python results/probes/benchmark_probe.py --model all --out results/probes/benchmarks/<unique-run-id>",
        "classifier_inference": "saved scaler/logistic parameters only; per-case three task classifiers, no fitting",
        "latency_scope": "saved classifier has per-case timings; encoder has whole-process wall only. End-to-end p50/p95 unavailable without exporter instrumentation. Four-vector encoder run includes unused state embedding, model loading and warmup.",
        "runs": runs,
        "comparisons": {"Laya_3_call_p50_ms": {"value": 95.7, "label": "reported reference; not measured in this run"},
            "original_cosine_new210_medians": {"qwen_ms": 84.517521, "harrier_ms": 81.039021,
                "source": "results/probes/comparisons/20260923T104208Z/metrics.json",
                "label": "210-case end-to-end full local cosine-decision medians; different workload, not equivalent to saved-probe inference"}}}
    report_path = args.out / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
