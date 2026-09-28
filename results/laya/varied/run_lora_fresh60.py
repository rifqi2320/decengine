#!/usr/bin/env python3
"""Guarded Laya-only baseline for the frozen LoRA fresh60 fixture.

The default dry-run validates fixture identity/schema without importing MLX or
loading a checkpoint. Do not run inference until the orchestrator authorizes it
and the concurrent LoRA GPU evaluation has finished.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import resource
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-lora-fresh-60.jsonl"
FIXTURE_SHA256 = "2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629"
MLX_RUNNER = ROOT / "results/laya/mlx/compare_300_mlx.py"
CHECKPOINT = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/checkpoint")
PORT = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/laya-mlx-port")
MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"
WEIGHT_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
PORT_COMMIT = "0a859518634112655cb97c745dbf04f5191aaf13"
EARLY_MEMORY = 6 * 1024**3
HARD_MEMORY = 8 * 1024**3
MAX_SWAP_GROWTH = 256 * 1024**2
TASK_ORDER = ("choice", "noul", "score")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def has_forbidden_input_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k in {"reference", "references", "evaluation_metadata"} or has_forbidden_input_field(v)
                   for k, v in value.items())
    if isinstance(value, list):
        return any(has_forbidden_input_field(v) for v in value)
    return False


def load_fixture(path: Path) -> list[dict[str, Any]]:
    if sha(path) != FIXTURE_SHA256:
        raise ValueError("fresh LoRA fixture SHA-256 mismatch; refusing to infer")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(c.get("id", "")) for c in cases]
    if len(cases) != 60 or ids != [f"mq-lora-fresh-{i:03d}" for i in range(1, 61)]:
        raise ValueError("expected ordered mq-lora-fresh-001..mq-lora-fresh-060 records")
    for case in cases:
        req = case.get("request")
        if not isinstance(req, dict) or set(req) != {"state", "questions"} or not isinstance(req.get("state"), dict):
            raise ValueError(f"{case['id']}: malformed state/questions request")
        if has_forbidden_input_field(req):
            raise ValueError(f"{case['id']}: found reference/evaluation metadata under request")
        qs = req.get("questions")
        if not isinstance(qs, dict) or len(qs) != 3 or tuple(q.get("type") for q in qs.values()) != TASK_ORDER:
            raise ValueError(f"{case['id']}: expected exactly choice, noul, score question wires in order")
        for name, q in qs.items():
            if not isinstance(q.get("prompt"), str) or not q["prompt"].strip():
                raise ValueError(f"{case['id']}/{name}: missing prompt")
            if q["type"] == "choice" and (not isinstance(q.get("options"), dict) or not q["options"]):
                raise ValueError(f"{case['id']}/{name}: choice candidates missing")
            if q["type"] == "score" and (not isinstance(q.get("levels"), list) or not q["levels"]):
                raise ValueError(f"{case['id']}/{name}: score rubric missing")
    return cases


def laya_question(q: dict[str, Any]) -> dict[str, Any]:
    """Same typed prompt/candidate translation as the previously run adapter."""
    kind = q["type"]
    if kind == "choice":
        return {"type": "choice", "instructions": q["prompt"], "criteria": q["options"]}
    if kind == "noul":
        return {"type": "noul", "instructions": q["prompt"]}
    if kind == "score":
        return {"type": "score", "instructions": q["prompt"],
                "criteria": [f"{x['label']}: {x['criterion']}" for x in q["levels"]]}
    raise ValueError(f"unsupported typed question {kind!r}")


def normalized_label(q: dict[str, Any], answer: dict[str, Any]) -> Any:
    if q["type"] == "choice":
        return answer.get("choice")
    if q["type"] == "noul":
        value = answer.get("noul")
        return bool(value >= 0.5) if isinstance(value, (int, float)) else None
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    values = [(int(k), float(v)) for k, v in probs.items()]
    best = max(v for _, v in values)
    winners = [i for i, v in values if v == best]
    return q["levels"][winners[0]]["label"] if len(winners) == 1 and 0 <= winners[0] < len(q["levels"]) else None


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def system_swap_bytes() -> int:
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=5)
    import re
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output)
    if not match:
        raise RuntimeError("cannot parse macOS vm.swapusage; refusing to run without swap guard")
    amount = float(match.group(1)); unit = match.group(2)
    return int(amount * {"K": 1024, "M": 1024**2, "G": 1024**3}[unit])


class MemoryGuard(RuntimeError):
    pass


def check_guard(mx: Any, swap_baseline: int, stage: str) -> dict[str, int]:
    process_rss = rss_bytes()
    metal_peak = int(mx.get_peak_memory())
    observed = max(process_rss, metal_peak)
    swap_now = system_swap_bytes()
    sample = {"process_peak_rss_bytes": process_rss, "mlx_peak_memory_bytes": metal_peak,
              "guarded_memory_bytes": observed, "swap_used_bytes": swap_now,
              "swap_growth_from_start_bytes": max(0, swap_now - swap_baseline)}
    if observed >= HARD_MEMORY:
        raise MemoryGuard(f"absolute memory limit reached at {stage}: {observed} bytes >= 8 GiB")
    if observed >= EARLY_MEMORY:
        raise MemoryGuard(f"early memory stop at {stage}: {observed} bytes >= 6 GiB")
    if sample["swap_growth_from_start_bytes"] > MAX_SWAP_GROWTH:
        raise MemoryGuard(f"swap increased by {sample['swap_growth_from_start_bytes']} bytes at {stage}; stop for stability")
    return sample


def reference_label(q: dict[str, Any], ref: dict[str, Any]) -> Any:
    target = ref["target"]
    if q["type"] == "choice":
        return target["selected"]
    if q["type"] == "noul":
        return target["value"]
    return target["label"]


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values); pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--port", type=Path, default=PORT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-inference", action="store_true", help="opt in to model load and Laya inference after orchestrator authorization")
    parser.add_argument("--dry-run", action="store_true", help="validate fixture/hash only; no MLX import or model load")
    args = parser.parse_args()
    run_dir: Path | None = None
    raw_rows: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    active_raw: dict[str, Any] | None = None
    active_pred: dict[str, Any] | None = None
    active_case_start: int | None = None
    errors = 0
    try:
        cases = load_fixture(args.fixture)
        if args.dry_run or not args.run_inference:
            print("Ready: exact 60x3 fixture validated; no MLX import, checkpoint load, or GPU inference.")
            return 0
        weight_path = args.checkpoint / "model.safetensors"
        if not weight_path.is_file() or sha(weight_path) != WEIGHT_SHA256:
            raise ValueError("pinned root checkpoint missing or SHA-256 mismatch")
        # MLX imports occur only after explicit invocation, never in --dry-run.
        sys.path.insert(0, str(args.port.resolve()))
        import mlx
        import mlx.core as mx
        import numpy as np
        import laya_mlx
        from laya_mlx.agent import Agent, collate_items
        from laya_mlx.common import confidence_from_probs, temp_bucket
        if mx.default_device().type != mx.DeviceType.gpu:
            raise RuntimeError(f"expected MLX GPU/Metal device, got {mx.default_device()}")
        swap_baseline = system_swap_bytes()
        check_guard(mx, swap_baseline, "before model load")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"laya-mlx: this checkpoint ships temperatures.*", category=RuntimeWarning)
            agent = Agent(str(args.checkpoint), dtype="float32", device="gpu", batch_size=1)
        agent.temperature = [float(x) for x in agent.temperature_raw]
        agent.temperature_by_options = {k: float(v) for k, v in agent.temperature_by_options_raw.items()}
        check_guard(mx, swap_baseline, "after model load")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = (args.out or ROOT / "results/laya/varied/runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, Any] = {
            "run_id": run_dir.name, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(args.fixture.resolve()), "fixture_sha256": sha(args.fixture), "case_count": 60,
            "question_count": 180, "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "checkpoint_weights_sha256": sha(weight_path), "checkpoint_weight_bytes": weight_path.stat().st_size,
            "verified_runner": str(MLX_RUNNER.resolve()), "external_port": str(args.port.resolve()),
            "external_port_commit": PORT_COMMIT, "runtime": "mlx-metal", "mlx_version": importlib.metadata.version("mlx"),
            "device_actual": str(mx.default_device()), "dtype": "float32",
            "temperature_policy": "exact pinned root config values; no laya-mlx clamp",
            "calls_per_case": 3, "inference_only": "Laya only; no JEV, no prompt tuning, no alternate prompt variants",
            "input_fields": "fixture request.state and request.questions only; reference and metadata excluded",
            "memory_policy_bytes": {"early_stop": EARLY_MEMORY, "absolute_stop": HARD_MEMORY,
                                    "max_swap_growth": MAX_SWAP_GROWTH, "guard_metric": "max(process peak RSS, MLX peak allocation)"},
            "swap_used_at_start_bytes": swap_baseline,
            "reference_scoring": "deferred until raw results and predictions are hash-frozen",
        }
        write_json(run_dir / "run-metadata.json", metadata)
        for case_index, case in enumerate(cases, 1):
            req = case["request"]
            case_decisions: dict[str, Any] = {}
            pred: dict[str, Any] = {"id": str(case["id"])}
            case_start = time.perf_counter_ns()
            active_case_start = case_start
            active_pred = pred
            active_raw = {"case_id": str(case["id"]), "decisions": case_decisions,
                          "case_three_call_inference_ms": None, "memory_guard_samples": []}
            guard_samples = []
            for name, fixture_q in req["questions"].items():
                guard_samples.append(check_guard(mx, swap_baseline, f"{case['id']} before {name}"))
                question = laya_question(fixture_q)
                items, internal = agent.prepare(req["state"], {name: question})
                item = items[0]
                batch = collate_items(items, agent.tok.pad_token_id)
                started = time.perf_counter_ns()
                logits_mx, acts_mx = agent.forward(batch)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                logits = np.asarray(logits_mx)[0, :len(item["markers"])].astype(float)
                acts = np.asarray(acts_mx)[0].astype(float)
                qtype, count = item["qtype"], len(item["markers"])
                temp = agent.temperature_by_options.get(temp_bucket(qtype, count), agent.temperature[qtype])
                scaled = logits / float(temp)
                probs = np.exp(scaled - scaled.max()); probs /= probs.sum()
                act_probs = np.exp(acts - acts.max()); act_probs /= act_probs.sum()
                if fixture_q["type"] == "choice":
                    labels = list(internal[0]["crit"])
                    answer = {"type": "choice", "choice": labels[int(probs.argmax())],
                              "probabilities": {str(k): float(v) for k, v in zip(labels, probs)},
                              "confidence": float(confidence_from_probs(probs, count)),
                              "rl_agent": {"act_probability": float(act_probs[0])}}
                elif fixture_q["type"] == "score":
                    criteria = internal[0]["crit"]
                    answer = {"type": "score", "score": float((np.arange(count) * probs).sum()),
                              "legend": {str(i): v for i, v in enumerate(criteria)},
                              "probabilities": {str(i): float(v) for i, v in enumerate(probs)},
                              "confidence": float(confidence_from_probs(probs, count)),
                              "rl_agent": {"act_probability": float(act_probs[0])}}
                else:
                    answer = {"type": "noul", "noul": float(probs[1]),
                              "rl_agent": {"act_probability": float(act_probs[0])}}
                pred[name] = normalized_label(fixture_q, answer)
                case_decisions[name] = {"raw_logits": logits.tolist(), "raw_act_logits": acts.tolist(),
                    "raw_response": {"model": "laya-rl-agent", "answers": {name: answer},
                                     "usage": {"input_tokens": len(item["ids"]), "output_tokens": 0}},
                    "normalized_typed_outcome": pred[name], "probabilities_unrounded": probs.tolist(),
                    "temperature": float(temp), "input_tokens": len(item["ids"]), "inference_ms": elapsed,
                    "error": None}
                guard_samples.append(check_guard(mx, swap_baseline, f"{case['id']} after {name}"))
            case_elapsed = (time.perf_counter_ns() - case_start) / 1e6
            active_raw["case_three_call_inference_ms"] = case_elapsed
            active_raw["memory_guard_samples"] = guard_samples
            raw_rows.append(active_raw)
            preds.append(pred)
            timing_rows.append({"id": str(case["id"]), "case_three_call_inference_ms": case_elapsed,
                                **{n: {"inference_ms": d["inference_ms"], "error": d["error"]}
                                   for n, d in case_decisions.items()}})
            # Capture a partial run on all normal/guarded paths before continuing.
            write_jsonl(run_dir / "raw-results.jsonl", raw_rows)
            write_jsonl(run_dir / "laya-predictions.jsonl", preds)
            write_jsonl(run_dir / "timings.jsonl", timing_rows)
            print(f"[{case_index}/60] {case['id']}: Laya 3/3 complete; guarded_bytes={max(s['guarded_memory_bytes'] for s in guard_samples)}", flush=True)
            active_raw = None
            active_pred = None
            active_case_start = None

        # Hash all predictions/timings before any reference is used.
        freeze_files = ("raw-results.jsonl", "laya-predictions.jsonl", "timings.jsonl", "run-metadata.json")
        write_json(run_dir / "PRE_REFERENCE_FREEZE.json", {
            "status": "predictions_frozen_before_reference_scoring", "frozen_at": datetime.now(timezone.utc).isoformat(),
            "fixture_sha256": sha(args.fixture), "completed_cases": len(raw_rows),
            "completed_decisions": sum(len(r["decisions"]) for r in raw_rows), "errors": errors,
            "artifact_sha256": {name: sha(run_dir / name) for name in freeze_files},
            "scoring_stage": "not run; no references accessed before freeze"})
        # Reference data is only accessed below, after this freeze artifact exists.
        refs_by_type = {q["type"]: name for name, q in cases[0]["request"]["questions"].items()}
        per_type = {}
        for kind, name in refs_by_type.items():
            good = sum(p.get(name) == reference_label(c["request"]["questions"][name], c["reference"][name])
                       for p, c in zip(preds, cases))
            per_type[kind] = {"correct": good, "total": 60, "accuracy": good / 60}
        all3 = sum(all(p.get(name) == reference_label(c["request"]["questions"][name], c["reference"][name])
                       for name in refs_by_type.values()) for p, c in zip(preds, cases))
        values = [r["case_three_call_inference_ms"] for r in raw_rows]
        metrics = {"per_question_type": per_type,
                   "all_three_correct": {"correct": all3, "total": 60, "accuracy": all3 / 60},
                   "case_latency_ms": {"n": 60, "p50": percentile(values, .50), "p95": percentile(values, .95),
                                       "mean": sum(values) / 60},
                   "error_count": errors, "completed_cases": 60, "completed_decisions": 180,
                   "scored_after_freeze": True}
        write_json(run_dir / "metrics.json", metrics)
        metadata.update({"finished_at": datetime.now(timezone.utc).isoformat(), "completed_cases": 60,
                         "completed_decisions": 180, "error_count": errors})
        write_json(run_dir / "run-metadata.json", metadata)
        write_json(run_dir / "checksums.json", {str(p.relative_to(run_dir)): sha(p)
                   for p in run_dir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"Saved Laya-only run to {run_dir}; errors={errors}")
        return 0
    except MemoryGuard as exc:
        if run_dir is not None:
            if active_raw is not None and active_pred is not None:
                active_raw["case_three_call_inference_ms"] = ((time.perf_counter_ns() - active_case_start) / 1e6
                                                                if active_case_start is not None else None)
                raw_rows.append(active_raw)
                preds.append(active_pred)
                timing_rows.append({"id": active_raw["case_id"],
                                    "case_three_call_inference_ms": active_raw["case_three_call_inference_ms"],
                                    "partial_case": True,
                                    **{n: {"inference_ms": d.get("inference_ms"), "error": d.get("error")}
                                       for n, d in active_raw["decisions"].items()}})
            write_jsonl(run_dir / "raw-results.jsonl", raw_rows)
            write_jsonl(run_dir / "laya-predictions.jsonl", preds)
            write_jsonl(run_dir / "timings.jsonl", timing_rows)
            write_json(run_dir / "guard-stop.json", {"reason": str(exc), "completed_cases": len(raw_rows),
                        "completed_decisions": sum(len(r["decisions"]) for r in raw_rows)})
            write_json(run_dir / "checksums.json", {str(p.relative_to(run_dir)): sha(p)
                       for p in run_dir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"SAFE STOP: {exc}; no further cases will be run", file=sys.stderr)
        return 3
    except Exception as exc:
        if run_dir is not None:
            if active_raw is not None and active_pred is not None:
                active_raw["case_three_call_inference_ms"] = ((time.perf_counter_ns() - active_case_start) / 1e6
                                                                if active_case_start is not None else None)
                raw_rows.append(active_raw)
                preds.append(active_pred)
                timing_rows.append({"id": active_raw["case_id"],
                                    "case_three_call_inference_ms": active_raw["case_three_call_inference_ms"],
                                    "partial_case": True,
                                    **{n: {"inference_ms": d.get("inference_ms"), "error": d.get("error")}
                                       for n, d in active_raw["decisions"].items()}})
            write_jsonl(run_dir / "raw-results.jsonl", raw_rows)
            write_jsonl(run_dir / "laya-predictions.jsonl", preds)
            write_jsonl(run_dir / "timings.jsonl", timing_rows)
            write_json(run_dir / "run-failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                        "completed_cases": len(raw_rows), "completed_decisions": sum(len(r["decisions"]) for r in raw_rows)})
            write_json(run_dir / "checksums.json", {str(p.relative_to(run_dir)): sha(p)
                       for p in run_dir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"Laya preflight/runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
