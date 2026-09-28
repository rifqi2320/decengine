#!/usr/bin/env python3
"""Run pinned Laya root checkpoint via real MLX Metal on probe-test-300.

The external laya-mlx runtime is used only for checkpoint/model execution and
tokenization. This harness enforces the original root checkpoint SHA and uses
raw (unclamped) temperatures plus the original Laya response format. Inference
is deliberately single-question per call, matching results/laya/compare_300.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "benchmarks/cases/probe-test-300.jsonl"
COMPARISON_DIR = ROOT / "results/probes/comparisons/20260923T104208Z"
DEFAULT_CHECKPOINT = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/checkpoint")
DEFAULT_PORT = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/laya-mlx-port")
MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"
WEIGHT_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
TASKS = ("owner", "urgent", "impact")
REFERENCE_PRED_FILES = {
    "Qwen": COMPARISON_DIR / "qwen-predictions-pooled300.jsonl",
    "Harrier": COMPARISON_DIR / "harrier-predictions-pooled300.jsonl",
    "JEV": COMPARISON_DIR / "jev-predictions-pooled300.jsonl",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def validate_fixture(path: Path) -> list[dict[str, Any]]:
    cases = read_jsonl(path)
    if [case.get("id") for case in cases] != [f"test-{i:03d}" for i in range(1, 301)]:
        raise ValueError("fixture must contain ordered test-001..test-300")
    for case in cases:
        request = case.get("request", {})
        if not isinstance(request.get("state"), dict) or set(request.get("questions", {})) != set(TASKS):
            raise ValueError(f"{case.get('id')}: expected state and owner/urgent/impact questions")
        if not set(TASKS).issubset(case.get("reference", {})):
            raise ValueError(f"{case.get('id')}: missing reference labels")
    return cases


def laya_question(task: str, q: dict[str, Any]) -> dict[str, Any]:
    if q["type"] == "choice":
        return {"type": "choice", "instructions": q["prompt"], "criteria": q["options"]}
    if q["type"] == "noul":
        return {"type": "noul", "instructions": q["prompt"]}
    if q["type"] == "score":
        return {"type": "score", "instructions": q["prompt"],
                "criteria": [f"{x['label']}: {x['criterion']}" for x in q["levels"]]}
    raise ValueError(f"{task}: unsupported fixture question type {q.get('type')!r}")


def normalize(task: str, answer: dict[str, Any], q: dict[str, Any]) -> Any:
    if task == "owner":
        return answer.get("choice")
    if task == "urgent":
        value = answer.get("noul")
        return bool(value >= 0.5) if isinstance(value, (int, float)) else None
    # Compare the most probable rubric level, not the expected ordinal score.
    dist = answer.get("probabilities", {})
    levels = q.get("levels", [])
    if not isinstance(dist, dict) or not levels:
        return None
    values = [(int(k), float(v)) for k, v in dist.items()]
    if not values:
        return None
    top = max(value for _, value in values)
    winners = [index for index, value in values if value == top]
    winner = winners[0] if len(winners) == 1 else -1
    return levels[winner]["label"] if 0 <= winner < len(levels) else None


def summarize_accuracy(cases: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    refs = {str(c["id"]): c["reference"] for c in cases}
    out: dict[str, Any] = {}
    for task in TASKS:
        good = sum(p.get(task) == refs[p["id"]].get(task) for p in preds)
        out[task] = {"correct": good, "total": len(preds), "accuracy": good / len(preds) if preds else None}
    complete = sum(all(p.get(t) == refs[p["id"]].get(t) for t in TASKS) for p in preds)
    out["complete_case"] = {"correct": complete, "total": len(preds), "accuracy": complete / len(preds) if preds else None}
    return out


def compare_existing(cases: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    ours = {str(row["id"]): row for row in preds}
    result = {}
    for name, path in REFERENCE_PRED_FILES.items():
        if not path.is_file():
            continue
        other = {str(row["id"]): row for row in read_jsonl(path)}
        paired = [key for key in ours.keys() & other.keys()]
        result[name] = {"cases": len(paired), "agreement": {
            task: sum(ours[key].get(task) == other[key].get(task) for key in paired) / len(paired)
            if paired else None for task in TASKS},
            "laya_mlx_accuracy": summarize_accuracy(cases, [ours[k] for k in paired]),
            "baseline_accuracy": summarize_accuracy(cases, [other[k] for k in paired])}
    return result


def rss_bytes() -> int:
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == "darwin" else n * 1024)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT)
    parser.add_argument("--out", type=Path, help="new run directory; defaults to results/laya/mlx/runs/<UTC>")
    parser.add_argument("--case-id", help="run one case for parity smoke; omit to run all 300")
    parser.add_argument("--parity-reference", type=Path, default=ROOT / "results/laya/reference/cpu-reference.json",
                        help="official CPU parity JSON (default: test-001 fixture)")
    parser.add_argument("--check-parity", action="store_true", help="compare normalized outputs for selected case(s)")
    parser.add_argument("--logit-atol", type=float, default=1e-4, help="absolute tolerance for raw option logits")
    parser.add_argument("--action-logit-atol", type=float, default=0.01, help="absolute tolerance for raw act logits")
    parser.add_argument("--dry-run", action="store_true", help="validate the exact 300-case fixture without loading weights")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--device", choices=("gpu", "metal", "cpu"), default="gpu")
    args = parser.parse_args()
    try:
        cases = validate_fixture(args.fixture)
        if args.case_id:
            cases = [c for c in cases if c["id"] == args.case_id]
            if len(cases) != 1:
                raise ValueError(f"unknown case id {args.case_id!r}")
        if args.dry_run:
            print(f"Ready: validated {len(cases)} fixture cases; no MLX load or inference.")
            return 0
        if args.check_parity and (not args.parity_reference.is_file()):
            raise FileNotFoundError(f"official CPU parity fixture missing: {args.parity_reference}")
        weight_path = args.checkpoint / "model.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"pinned checkpoint weights not available yet: {weight_path}")
        actual_weight_sha = sha(weight_path)
        if actual_weight_sha != WEIGHT_SHA256:
            raise ValueError(f"checkpoint SHA mismatch: expected {WEIGHT_SHA256}, got {actual_weight_sha}")

        sys.path.insert(0, str(args.port))
        import mlx
        import mlx.core as mx
        import numpy as np
        import laya_mlx
        from laya_mlx.agent import Agent, collate_items
        from laya_mlx.common import confidence_from_probs, temp_bucket

        load_start = time.perf_counter_ns()
        # The pinned port warns about its normal clamp. This benchmark restores
        # raw values immediately, so suppress only that known misleading warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"laya-mlx: this checkpoint ships temperatures.*",
                                   category=RuntimeWarning)
            agent = Agent(str(args.checkpoint), dtype=args.dtype, device=args.device, batch_size=1)
        # This port clamps upstream temperature values. Override with the exact
        # pinned config values: official root behavior applies raw temperatures.
        agent.temperature = [float(x) for x in agent.temperature_raw]
        agent.temperature_by_options = {k: float(v) for k, v in agent.temperature_by_options_raw.items()}
        load_ms = (time.perf_counter_ns() - load_start) / 1e6
        # One untimed real forward compiles/warms Metal kernels on a fixture case.
        warm_case = cases[0]
        warm_task = "owner"
        warm_q = laya_question(warm_task, warm_case["request"]["questions"][warm_task])
        warm_items, _ = agent.prepare(warm_case["request"]["state"], {warm_task: warm_q})
        mx.eval(agent.forward(collate_items(warm_items, agent.tok.pad_token_id)))
        warm_ms = (time.perf_counter_ns() - load_start) / 1e6 - load_ms

        outdir = (args.out or ROOT / "results/laya/mlx/runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
        outdir.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, Any] = {
            "run_id": outdir.name, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(args.fixture.resolve()), "fixture_sha256": sha(args.fixture),
            "case_count": len(cases), "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "model_weights_sha256": actual_weight_sha, "model_weight_bytes": weight_path.stat().st_size,
            "external_port": str(args.port.resolve()), "external_port_commit": "0a859518634112655cb97c745dbf04f5191aaf13",
            "runtime": "mlx-metal", "mlx_version": importlib.metadata.version("mlx"),
            "numpy_version": np.__version__, "tokenizers_version": __import__("tokenizers").__version__,
            "huggingface_hub_version": __import__("huggingface_hub").__version__,
            "device_requested": args.device, "device_actual": str(mx.default_device()), "dtype": args.dtype,
            "temperature_policy": "exact root config values; no laya-mlx [0.5,5] clamping",
            "inference_mode": "3 independent single-question MLX forward calls per case",
            "score_mapping": "impact scalar label = argmax raw-temperature ordinal probability; rubric label: criterion passed",
            "checkpoint_load_ms": load_ms, "warmup_ms": warm_ms,
        }
        write_json(outdir / "run-metadata.json", metadata)
        raw_rows, predictions, timing_rows = [], [], []
        for index, case in enumerate(cases, 1):
            state = case["request"]["state"]
            raw_case = {"case_id": case["id"], "decisions": {}, "memory": {"peak_rss_before_bytes": rss_bytes()}}
            pred: dict[str, Any] = {"id": case["id"]}
            for task in TASKS:
                fixture_q = case["request"]["questions"][task]
                q = laya_question(task, fixture_q)
                prepare_start = time.perf_counter_ns()
                items, internal = agent.prepare(state, {task: q})
                item = items[0]
                input_tokens = len(item["ids"])
                batch = collate_items(items, agent.tok.pad_token_id)
                prepare_ms = (time.perf_counter_ns() - prepare_start) / 1e6
                start = time.perf_counter_ns()
                try:
                    logits_mx, act_mx = agent.forward(batch)
                    logits = np.asarray(logits_mx)[0, :len(item["markers"])].astype(float)
                    acts = np.asarray(act_mx)[0].astype(float)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    qtype, k = item["qtype"], len(item["markers"])
                    scale = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
                    scaled = logits / float(scale)
                    probs = np.exp(scaled - scaled.max()); probs /= probs.sum()
                    act_probs = np.exp(acts - acts.max()); act_probs /= act_probs.sum()
                    if q["type"] == "choice":
                        labels = list(internal[0]["crit"])
                        answer = {"type": "choice", "choice": labels[int(probs.argmax())],
                                  "probabilities": {key: round(float(v), 4) for key, v in zip(labels, probs)},
                                  "confidence": round(confidence_from_probs(probs, k), 4),
                                  "rl_agent": {"act_probability": float(act_probs[0])}}
                    elif q["type"] == "score":
                        criteria = internal[0]["crit"]
                        answer = {"type": "score", "score": round(float((np.arange(k) * probs).sum()), 4),
                                  "legend": {str(i): v for i, v in enumerate(criteria)},
                                  "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(probs)},
                                  "confidence": round(confidence_from_probs(probs, k), 4),
                                  "rl_agent": {"act_probability": float(act_probs[0])}}
                    else:
                        answer = {"type": "noul", "noul": round(float(probs[1]), 4),
                                  "rl_agent": {"act_probability": float(act_probs[0])}}
                    pred[task] = normalize(task, answer, fixture_q)
                    raw_case["decisions"][task] = {"raw_logits": logits.tolist(), "raw_act_logits": acts.tolist(),
                        "raw_response": {"model": "laya-rl-agent", "answers": {task: answer},
                                         "usage": {"input_tokens": input_tokens, "output_tokens": 0}},
                        "normalized_label": pred[task], "error": None, "inference_ms": elapsed,
                        "preprocess_ms": prepare_ms, "input_tokens": input_tokens, "output_tokens": 0,
                        "temperature": float(scale), "probabilities_unrounded": probs.tolist(),
                        "input_ids": item["ids"], "attention_mask": [1] * input_tokens,
                        "marker_positions": item["markers"], "marker_mask": [1] * len(item["markers"]),
                        "qtype_id": qtype}
                except Exception as exc:
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    pred[task] = None
                    raw_case["decisions"][task] = {"raw_logits": None, "raw_act_logits": None,
                        "raw_response": None, "normalized_label": None,
                        "error": f"{type(exc).__name__}: {exc}", "inference_ms": elapsed,
                        "preprocess_ms": prepare_ms, "input_tokens": input_tokens, "output_tokens": None}
            raw_case["case_inference_ms_sum_of_three_calls"] = sum(raw_case["decisions"][t]["inference_ms"] for t in TASKS)
            raw_case["case_input_tokens_sum"] = sum(raw_case["decisions"][t]["input_tokens"] for t in TASKS)
            raw_case["memory"]["peak_rss_after_bytes"] = rss_bytes()
            raw_rows.append(raw_case); predictions.append(pred)
            timing_rows.append({"id": case["id"], "case_inference_ms_sum_of_three_calls": raw_case["case_inference_ms_sum_of_three_calls"],
                "case_input_tokens_sum": raw_case["case_input_tokens_sum"], **{t: {k: raw_case["decisions"][t][k]
                for k in ("inference_ms", "preprocess_ms", "input_tokens", "error")} for t in TASKS}})
            if index % 10 == 0 or index == len(cases):
                write_jsonl(outdir / "raw-results.jsonl", raw_rows)
                write_jsonl(outdir / "laya-mlx-predictions.jsonl", predictions)
                write_jsonl(outdir / "timings.jsonl", timing_rows)
                print(f"[{index}/{len(cases)}] {case['id']} completed", flush=True)

        if args.check_parity:
            ref = json.loads(args.parity_reference.read_text(encoding="utf-8"))
            parity = {"reference": str(args.parity_reference.resolve()), "case_id": ref["case_id"],
                      "compared": len(predictions), "tolerances": {"raw_option_logits_atol": args.logit_atol,
                      "raw_act_logits_atol": args.action_logit_atol}, "decisions": {}, "passed": False}
            if len(predictions) != 1 or predictions[0]["id"] != ref["case_id"]:
                raise ValueError("raw-logit parity is defined for exactly the official fixture case; use --case-id test-001")
            raw_case = raw_rows[0]
            for task in TASKS:
                expected = ref["decisions"][task]
                got = raw_case["decisions"][task]
                option_diffs = [abs(float(a) - float(b)) for a, b in zip(
                    expected["raw_option_logits_before_calibration"], got["raw_logits"])]
                act_diffs = [abs(float(a) - float(b)) for a, b in zip(
                    expected["raw_action_logits_before_softmax"], got["raw_act_logits"])]
                cpu_answer = expected["official_output"]
                mlx_answer = got["raw_response"]["answers"][task]
                matches_output = all(cpu_answer.get(key) == mlx_answer.get(key)
                                     for key in ("type", "choice", "noul", "score", "probabilities", "confidence", "legend", "rl_agent")
                                     if key in cpu_answer)
                matches_ids = expected["input_ids"] == got["input_ids"]
                parity["decisions"][task] = {"sequence_ids_exact": matches_ids,
                    "cpu_sequence_length": expected["sequence_length"], "mlx_sequence_length": got["input_tokens"],
                    "raw_option_logit_max_abs_diff": max(option_diffs, default=None),
                    "raw_act_logit_max_abs_diff": max(act_diffs, default=None), "sdk_output_exact": matches_output,
                    "normalized_label": predictions[0].get(task),
                    "option_logits_within_tolerance": bool(option_diffs) and max(option_diffs) <= args.logit_atol,
                    "action_logits_within_tolerance": bool(act_diffs) and max(act_diffs) <= args.action_logit_atol}
            parity["passed"] = all(v["sequence_ids_exact"] and v["sdk_output_exact"] and
                                   v["option_logits_within_tolerance"] and v["action_logits_within_tolerance"]
                                   for v in parity["decisions"].values())
            write_json(outdir / "parity.json", parity)
            if not parity["passed"]:
                raise RuntimeError(f"output parity failed; see {outdir / 'parity.json'}")
        metrics = {"laya_mlx": summarize_accuracy(cases, predictions),
                   "comparisons": compare_existing(cases, predictions)}
        elapsed_by_task = {t: [row["decisions"][t]["inference_ms"] for row in raw_rows if not row["decisions"][t]["error"]] for t in TASKS}
        timings = {t: {"n": len(v), "p50_ms": percentile(v, .50), "p95_ms": percentile(v, .95),
                       "mean_ms": statistics.mean(v) if v else None} for t, v in elapsed_by_task.items()}
        all_times = [row["case_inference_ms_sum_of_three_calls"] for row in raw_rows]
        timings["three_call_case"] = {"n": len(all_times), "p50_ms": percentile(all_times, .50), "p95_ms": percentile(all_times, .95),
                                      "mean_ms": statistics.mean(all_times) if all_times else None}
        baseline_metrics_path = COMPARISON_DIR / "metrics.json"
        if baseline_metrics_path.is_file():
            baseline_latency = json.loads(baseline_metrics_path.read_text(encoding="utf-8")).get("new210_latency", {})
            timings["comparison_new210_latency"] = {**baseline_latency,
                "source": str(baseline_metrics_path),
                "note": "Existing Qwen/Harrier/JEV measurements on test-091..test-300, as recorded by the probe comparator; "
                        "not strictly hardware/mode equivalent to these three separate Laya forwards."}
        write_json(outdir / "metrics.json", metrics)
        write_json(outdir / "latency-summary.json", timings)
        token_counts = [d["input_tokens"] for row in raw_rows for d in row["decisions"].values()]
        # Dense-model inference lower-bound estimate: 2 FLOPs per parameter/token.
        # Root config reports 421,293,827 F16 + 3 F32 parameter elements.
        avg_tokens = statistics.mean(token_counts) if token_counts else 0
        write_json(outdir / "flops-estimate.json", {"method": "2 * 421296830 parameters * measured input tokens per independent forward",
            "parameter_count": 421296830, "measured_forward_count": len(token_counts), "measured_token_count_total": sum(token_counts),
            "measured_tokens_per_forward_mean": avg_tokens,
            "estimated_mean_flops_per_forward": 2 * 421296830 * avg_tokens,
            "estimated_mean_gflops_per_forward": 2 * 421296830 * avg_tokens / 1e9,
            "caveat": "Dense projection estimate only; excludes attention quadratic terms, head, nonlinearity, and padding overhead."})
        metadata.update({"finished_at": datetime.now(timezone.utc).isoformat(), "completed_cases": len(raw_rows),
                         "process_peak_rss_bytes": rss_bytes(), "memory_peak_bytes": mx.get_peak_memory()})
        write_json(outdir / "run-metadata.json", metadata)
        write_json(outdir / "checksums.json", {str(p.relative_to(outdir)): sha(p) for p in outdir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"Completed MLX comparison in {outdir}")
        return 0
    except Exception as exc:
        print(f"laya MLX compare: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
