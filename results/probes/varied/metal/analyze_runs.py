#!/usr/bin/env python3
"""Compare real Metal outputs to archived NumPy predictions and record run provenance."""
import hashlib, json, math, pathlib, platform, re, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
BASE = ROOT / "results/probes/varied"
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-matched-60.jsonl"
STORE = pathlib.Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def pct(xs, p):
    xs = sorted(xs); pos = (len(xs)-1)*p; lo=int(pos); hi=min(lo+1,len(xs)-1)
    return xs[lo] + (xs[hi]-xs[lo])*(pos-lo)

for key in ("qwen", "harrier"):
    run = BASE / "metal/runs" / key
    frozen = BASE / "matched" / f"frozen-{key}"
    archived = BASE / "matched/runs" / key / "evaluation/predictions.jsonl"
    preds = read(run / "predictions.jsonl")
    old = read(archived)
    now = {(p["id"], p["question_name"]): p for p in preds}
    if len(preds) != 180 or len(now) != 180 or len(old) != 180:
        raise SystemExit(f"{key}: expected exactly 180 unique predictions")
    max_diff = 0.0; max_score_ev_diff = 0.0; failures = []
    for p in old:
        q = now[(p["id"], p["question_name"])]
        for candidate in p["candidate_keys"]:
            max_diff = max(max_diff, abs(float(p["probabilities"][candidate]) - float(q["probabilities"][candidate])))
        if p["selected"] != q["selected"] or p["label"] != q["label"] or p["value"] != q["value"] or p.get("selected_level") != q.get("selected_level"):
            failures.append((p["id"], p["question_name"], p["selected"], q["selected"]))
        if p.get("expected_value") is not None:
            max_score_ev_diff = max(max_score_ev_diff, abs(p["expected_value"]-q["expected_value"]))
    if failures: raise SystemExit(f"{key}: selected/boolean/score-label parity failures: {failures[:4]}")
    timings = read(run / "predictions.jsonl.timings.jsonl")
    cases = read(run / "predictions.jsonl.case-timings.jsonl")
    if len(timings) != 180 or len(cases) != 60: raise SystemExit(f"{key}: timing count mismatch")
    # Timing rows exclude load and one warm-up per typed scorer (done in the runner).
    stage_names = {"embedding": "embedding_ns", "gpu_scorer": "gpu_scorer_ns", "question_total": "question_total_ns"}
    resource_text = (run/"time-resource.txt").read_text()
    rss = re.search(r"\n\s*(\d+)\s+maximum resident set size", resource_text)
    footprint = re.search(r"\n\s*(\d+)\s+peak memory footprint", resource_text)
    elapsed = re.search(r"([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys", resource_text)
    if not (rss and footprint and elapsed): raise SystemExit(f"{key}: missing macOS resource measurements")
    model_id = "Qwen--Qwen3-Embedding-0.6B" if key == "qwen" else "microsoft--harrier-oss-v1-0.6b"
    install = STORE / "models" / model_id / "install.json"
    install_doc = json.loads(install.read_text())
    summary = {"model_key": key, "backend": "MLX Device(gpu, 0) / Metal; embeddings + 4096d pair features + scaler-folded linear scorer + softmax remain MLX arrays",
      "case_count": 60, "question_count": 180, "warmup": "one full typed inference per type excluded; model load excluded",
      "parity": {"max_probability_absolute_difference": max_diff, "selected_label_boolean_and_score_argmax_exact": True,
                 "max_score_expected_value_absolute_difference_float32": max_score_ev_diff},
      "timing_units": "milliseconds", "case_total_ms": {"p50": pct([x["case_total_ns"]/1e6 for x in cases],.50), "p95": pct([x["case_total_ns"]/1e6 for x in cases],.95)},
      "stages": {name: {"p50": pct([x[field]/1e6 for x in timings],.50), "p95": pct([x[field]/1e6 for x in timings],.95)} for name,field in stage_names.items()},
      "runtime": {"python_not_in_inference": True, "platform": platform.platform(), "rustc_target": "aarch64-apple-darwin", "device": "Device(gpu, 0)",
                  "whole_process_including_load_warmup": {"real_seconds": float(elapsed.group(1)), "user_seconds": float(elapsed.group(2)), "system_seconds": float(elapsed.group(3)),
                    "maximum_resident_set_bytes": int(rss.group(1)), "peak_memory_footprint_bytes": int(footprint.group(1))}, "process_resource_report_sha256": sha(run/"time-resource.txt")},
      "hashes": {"fixture": sha(FIXTURE), "frozen_models": sha(frozen/"models.json"), "frozen_metadata": sha(frozen/"metadata.json"),
                 "selection_freeze": sha(BASE/"matched/selection-freeze.json"), "predictions": sha(run/"predictions.jsonl"),
                 "profile": sha(ROOT/"models/manifests"/("qwen3-embedding-0.6b.json" if key=="qwen" else "harrier-oss-v1-0.6b.json")),
                 "install_record": sha(install), "checkpoint_files": {x["path"]:x["sha256"] for x in install_doc["files"]},
                 "question_timings": sha(run/"predictions.jsonl.timings.jsonl"), "case_timings": sha(run/"predictions.jsonl.case-timings.jsonl")}}
    (run/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(f"{key}: max probability diff {max_diff:.9g}; exact labels/booleans/score argmax; max expected value drift {max_score_ev_diff:.9g}")
