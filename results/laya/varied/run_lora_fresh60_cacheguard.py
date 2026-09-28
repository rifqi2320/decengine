#!/usr/bin/env python3
"""One-shot cache-cleanup Laya evaluation on the frozen LoRA fresh60 holdout.

The first four cases are a mandatory 12-decision smoke. It compares every
smoke label and probability vector for the first 11 decisions with the preserved
guard-stopped run. A swap/footprint guard or parity failure terminates the run;
only a passing smoke continues in the same process to cases 5-60. No reference
is read until all 180 inference outputs are frozen.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-lora-fresh-60.jsonl"
FIXTURE_SHA256 = "2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629"
PREVIOUS_PARTIAL = ROOT / "results/laya/varied/runs/20260924T052807390692Z/raw-results.jsonl"
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
SMOKE_CASES = 4
PARITY_ATOL = 1e-5
TASKS = ("response_plan", "immediate_intervention", "record_auditability")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def has_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k in {"reference", "references", "evaluation_metadata"} or has_forbidden(v) for k, v in value.items())
    if isinstance(value, list):
        return any(has_forbidden(v) for v in value)
    return False


def load_fixture(path: Path) -> list[dict[str, Any]]:
    if sha(path) != FIXTURE_SHA256:
        raise ValueError("fresh60 fixture SHA mismatch; refusing to run")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 60 or [str(c.get("id")) for c in rows] != [f"mq-lora-fresh-{i:03d}" for i in range(1, 61)]:
        raise ValueError("expected exact ordered 60-case fixture")
    for case in rows:
        req = case.get("request")
        if not isinstance(req, dict) or set(req) != {"state", "questions"} or not isinstance(req.get("state"), dict):
            raise ValueError(f"{case['id']}: malformed state/questions request")
        if has_forbidden(req):
            raise ValueError(f"{case['id']}: evaluation/reference field found under request")
        qs = req["questions"]
        if tuple(qs) != TASKS or tuple(q.get("type") for q in qs.values()) != ("choice", "noul", "score"):
            raise ValueError(f"{case['id']}: typed question wires/order differ from frozen fixture contract")
    return rows


def translate(q: dict[str, Any]) -> dict[str, Any]:
    if q["type"] == "choice":
        return {"type": "choice", "instructions": q["prompt"], "criteria": q["options"]}
    if q["type"] == "noul":
        return {"type": "noul", "instructions": q["prompt"]}
    if q["type"] == "score":
        return {"type": "score", "instructions": q["prompt"],
                "criteria": [f"{x['label']}: {x['criterion']}" for x in q["levels"]]}
    raise ValueError(f"unsupported type {q['type']!r}")


def normalize(q: dict[str, Any], answer: dict[str, Any]) -> Any:
    if q["type"] == "choice":
        return answer.get("choice")
    if q["type"] == "noul":
        v = answer.get("noul")
        return bool(v >= 0.5) if isinstance(v, (int, float)) else None
    dist = answer.get("probabilities")
    if not isinstance(dist, dict) or not dist:
        return None
    vals = [(int(k), float(v)) for k, v in dist.items()]
    top = max(v for _, v in vals); winners = [i for i, v in vals if v == top]
    return q["levels"][winners[0]]["label"] if len(winners) == 1 and 0 <= winners[0] < len(q["levels"]) else None


def footprint_bytes() -> tuple[int, int]:
    proc = subprocess.run(["footprint", "--format", "bytes", "--noCategories", str(__import__("os").getpid())],
                          text=True, capture_output=True, check=True, timeout=10)
    import re
    current = re.search(r"Footprint:\s+([0-9,]+)\s+B", proc.stdout)
    peak = re.search(r"phys_footprint_peak:\s+([0-9,]+)\s+B", proc.stdout)
    if not current or not peak:
        raise RuntimeError(f"unable to read true macOS process footprint: {proc.stdout[:300]}")
    return int(current.group(1).replace(",", "")), int(peak.group(1).replace(",", ""))


def swap_used_bytes() -> int:
    output = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True, timeout=5).stdout
    import re
    match = re.search(r"used\s*=\s*([0-9.]+)M", output)
    if not match:
        raise RuntimeError(f"unable to parse system swap usage: {output}")
    return int(float(match.group(1)) * 1024**2)


class GuardStop(RuntimeError):
    pass


def sample_guard(mx: Any, swap_baseline: int, stage: str) -> dict[str, int]:
    current, peak = footprint_bytes()
    swap = swap_used_bytes()
    mlx_active = int(mx.get_active_memory())
    mlx_peak = int(mx.get_peak_memory())
    sample = {"stage": stage, "process_current_footprint_bytes": current,
              "process_peak_footprint_bytes": peak, "mlx_active_bytes": mlx_active,
              "mlx_peak_bytes_since_reset": mlx_peak, "swap_used_bytes": swap,
              "swap_growth_from_start_bytes": max(0, swap - swap_baseline)}
    combined_peak = max(peak, mlx_peak)
    if combined_peak >= HARD_MEMORY:
        raise GuardStop(f"absolute 8 GiB limit reached at {stage}: {combined_peak} bytes")
    if combined_peak >= EARLY_MEMORY:
        raise GuardStop(f"6 GiB early stop reached at {stage}: {combined_peak} bytes")
    if sample["swap_growth_from_start_bytes"] >= MAX_SWAP_GROWTH:
        raise GuardStop(f"swap rise reached 256 MiB guard at {stage}: {sample['swap_growth_from_start_bytes']} bytes")
    return sample


def clear_call_cache(mx: Any) -> dict[str, int]:
    """Free unused Metal cache and Python-held graph references between calls."""
    mx.clear_cache()
    gc.collect()
    return {"mlx_active_bytes_after_clear": int(mx.get_active_memory()),
            "mlx_cache_bytes_after_clear": int(mx.get_cache_memory())}


def reference_label(q: dict[str, Any], ref: dict[str, Any]) -> Any:
    target = ref["target"]
    return target["selected"] if q["type"] == "choice" else target["value"] if q["type"] == "noul" else target["label"]


def parity_check(new_raw: list[dict[str, Any]]) -> dict[str, Any]:
    old = [json.loads(line) for line in PREVIOUS_PARTIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_by = {r["case_id"]: r for r in old}
    compares = []
    max_diff = 0.0
    for record in new_raw:
        cid = record["case_id"]
        if cid not in old_by:
            continue
        prev = old_by[cid]
        for name, prev_decision in prev["decisions"].items():
            current = record["decisions"].get(name)
            if current is None:
                raise ValueError(f"smoke missing prior decision {cid}/{name}")
            same_label = current["normalized_typed_outcome"] == prev_decision["normalized_typed_outcome"]
            old_probs = prev_decision["probabilities_unrounded"]
            new_probs = current["probabilities_unrounded"]
            if len(old_probs) != len(new_probs):
                raise ValueError(f"smoke candidate count mismatch {cid}/{name}")
            diff = max(abs(float(a) - float(b)) for a, b in zip(old_probs, new_probs))
            max_diff = max(max_diff, diff)
            compares.append({"case_id": cid, "question": name, "label_exact": same_label,
                             "max_probability_abs_diff": diff, "probabilities_within_atol": diff <= PARITY_ATOL})
            if not same_label or diff > PARITY_ATOL:
                raise ValueError(f"smoke parity mismatch for preserved inference {cid}/{name}: label={same_label}, diff={diff}")
    if len(compares) != 11:
        raise ValueError(f"expected parity comparison for 11 preserved decisions, got {len(compares)}")
    return {"passed": True, "compared_decisions": len(compares), "probability_atol": PARITY_ATOL,
            "max_probability_abs_diff": max_diff, "comparisons": compares,
            "partial_run": str(PREVIOUS_PARTIAL)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--port", type=Path, default=PORT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run_dir: Path | None = None
    raw_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    active_raw: dict[str, Any] | None = None
    active_pred: dict[str, Any] | None = None
    active_start: int | None = None
    guard_samples: list[dict[str, Any]] = []
    swap_baseline: int | None = None
    phase = "preflight"
    try:
        cases = load_fixture(args.fixture)
        weight_path = args.checkpoint / "model.safetensors"
        if not weight_path.is_file() or sha(weight_path) != WEIGHT_SHA256:
            raise ValueError("pinned Laya root checkpoint SHA mismatch")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = (args.out or ROOT / "results/laya/varied/runs" / f"{run_id}-cacheguard").resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "attempt.json", {
            "status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
            "runner": str(Path(__file__).resolve()), "fixture": str(args.fixture.resolve()),
            "fixture_sha256": sha(args.fixture), "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_weights_sha256": sha(weight_path), "requested_guard_bytes": {
                "early_stop": EARLY_MEMORY, "absolute_stop": HARD_MEMORY,
                "max_swap_growth_exclusive": MAX_SWAP_GROWTH},
            "attempt_policy": "single attempt; four-case smoke, parity gate, then continuation in same process only if smoke passes",
        })
        sys.path.insert(0, str(args.port.resolve()))
        import mlx
        import mlx.core as mx
        from laya_mlx.agent import Agent, collate_items
        from laya_mlx.common import confidence_from_probs, temp_bucket
        if mx.default_device().type != mx.DeviceType.gpu:
            raise RuntimeError(f"expected MLX Metal GPU, received {mx.default_device()}")
        swap_baseline = swap_used_bytes()
        guard_samples.append(sample_guard(mx, swap_baseline, "before_model_load"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"laya-mlx: this checkpoint ships temperatures.*", category=RuntimeWarning)
            agent = Agent(str(args.checkpoint), dtype="float32", device="gpu", batch_size=1)
        agent.temperature = [float(x) for x in agent.temperature_raw]
        agent.temperature_by_options = {k: float(v) for k, v in agent.temperature_by_options_raw.items()}
        guard_samples.append(sample_guard(mx, swap_baseline, "after_model_load"))
        # Reset only the allocator's per-run peak after preserving post-load guard data.
        mx.reset_peak_memory()

        metadata: dict[str, Any] = {
            "run_id": run_dir.name, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(args.fixture.resolve()), "fixture_sha256": sha(args.fixture),
            "case_count": 60, "question_count": 180, "model": MODEL_ID,
            "model_revision": MODEL_REVISION, "checkpoint_weights_sha256": sha(weight_path),
            "checkpoint_weight_bytes": weight_path.stat().st_size,
            "verified_runner": str(MLX_RUNNER.resolve()), "external_port": str(args.port.resolve()),
            "external_port_commit": PORT_COMMIT, "runtime": "mlx-metal", "mlx_version": importlib.metadata.version("mlx"),
            "device_actual": str(mx.default_device()), "dtype": "float32",
            "temperature_policy": "exact pinned root config values; no clamp",
            "prompt_protocol": "same frozen fixture wires and same prior Laya adapter translation; no prompt tuning",
            "inference_policy": "sequential one-question calls; each of the first 4 cases is smoke; continue to full60 only after parity+guards pass",
            "cache_cleanup": "mx.eval(outputs), convert/copy to host values, delete graph/batch refs, mx.clear_cache(), gc.collect(), sample guard, mx.reset_peak_memory()",
            "guard_bytes": {"early_stop": EARLY_MEMORY, "absolute_stop": HARD_MEMORY,
                            "max_swap_growth_exclusive": MAX_SWAP_GROWTH,
                            "process_monitor": "macOS footprint CLI", "swap_monitor": "sysctl vm.swapusage"},
            "swap_baseline_bytes": swap_baseline,
            "reference_scoring": "not performed until all60 raw predictions are hash-frozen",
        }
        write_json(run_dir / "run-metadata.json", metadata)
        phase = "smoke"
        smoke_passed = False
        for case_index, case in enumerate(cases, 1):
            req = case["request"]
            call_decisions: dict[str, Any] = {}
            pred: dict[str, Any] = {"id": str(case["id"])}
            active_start = time.perf_counter_ns()
            active_pred = pred
            active_raw = {"case_id": str(case["id"]), "decisions": call_decisions,
                          "memory_guard_samples": []}
            per_case_guards = []
            for name, fixture_q in req["questions"].items():
                before = sample_guard(mx, swap_baseline, f"{case['id']} before {name}")
                per_case_guards.append(before)
                question = translate(fixture_q)
                items, internal = agent.prepare(req["state"], {name: question})
                item = items[0]
                batch = collate_items(items, agent.tok.pad_token_id)
                started = time.perf_counter_ns()
                logits_mx, acts_mx = agent.forward(batch)
                mx.eval(logits_mx, acts_mx)
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
                elapsed = (time.perf_counter_ns() - started) / 1e6
                pred[name] = normalize(fixture_q, answer)
                call_decisions[name] = {"raw_logits": logits.tolist(), "raw_act_logits": acts.tolist(),
                    "raw_response": {"model": "laya-rl-agent", "answers": {name: answer},
                                     "usage": {"input_tokens": len(item["ids"]), "output_tokens": 0}},
                    "normalized_typed_outcome": pred[name], "probabilities_unrounded": probs.tolist(),
                    "temperature": float(temp), "input_tokens": len(item["ids"]), "inference_ms": elapsed,
                    "error": None}
                # Release this call's lazy graph / batch before the next typed forward.
                del logits_mx, acts_mx, logits, acts, scaled, probs, act_probs, item, items, internal, batch
                cache_after = clear_call_cache(mx)
                after = sample_guard(mx, swap_baseline, f"{case['id']} after {name} (cache cleared)")
                after.update(cache_after)
                per_case_guards.append(after)
                mx.reset_peak_memory()
                # Incrementally preserve every completed call, including a smoke guard stop.
                active_raw["memory_guard_samples"] = per_case_guards
                write_jsonl(run_dir / "raw-results.jsonl", raw_rows + [active_raw])
                write_jsonl(run_dir / "laya-predictions.jsonl", predictions + [pred])
            case_ms = (time.perf_counter_ns() - active_start) / 1e6
            active_raw["case_three_call_inference_ms"] = case_ms
            active_raw["memory_guard_samples"] = per_case_guards
            raw_rows.append(active_raw); predictions.append(pred)
            timing_rows.append({"id": str(case["id"]), "case_three_call_inference_ms": case_ms,
                                **{n: {"inference_ms": d["inference_ms"], "error": d["error"]}
                                   for n, d in call_decisions.items()}})
            write_jsonl(run_dir / "raw-results.jsonl", raw_rows)
            write_jsonl(run_dir / "laya-predictions.jsonl", predictions)
            write_jsonl(run_dir / "timings.jsonl", timing_rows)
            active_raw = None; active_pred = None; active_start = None
            print(f"[{case_index}/60] {case['id']}: 3/3 complete; footprint_peak={max(s['process_peak_footprint_bytes'] for s in per_case_guards)}; swap_growth={max(s['swap_growth_from_start_bytes'] for s in per_case_guards)}", flush=True)
            if case_index == SMOKE_CASES:
                parity = parity_check(raw_rows)
                write_json(run_dir / "smoke-parity.json", parity)
                smoke_passed = True
                phase = "full_run_after_smoke"
                print(f"Smoke passed: compared 11 persisted decisions, max probability diff={parity['max_probability_abs_diff']:.9g}; continuing cases 5-60 in same process", flush=True)

        if not smoke_passed or len(raw_rows) != 60 or sum(len(r["decisions"]) for r in raw_rows) != 180:
            raise GuardStop("full run did not reach 60 cases/180 decisions")
        # Freeze raw inference outputs before opening reference targets.
        frozen_files = ("raw-results.jsonl", "laya-predictions.jsonl", "timings.jsonl", "run-metadata.json", "smoke-parity.json")
        write_json(run_dir / "PRE_REFERENCE_FREEZE.json", {
            "status": "predictions_frozen_before_reference_scoring", "frozen_at": datetime.now(timezone.utc).isoformat(),
            "fixture_sha256": sha(args.fixture), "completed_cases": 60, "completed_decisions": 180,
            "smoke_cases": SMOKE_CASES, "smoke_parity_passed": True,
            "artifact_sha256": {name: sha(run_dir / name) for name in frozen_files},
            "scoring_stage": "not run before this freeze"})
        # Only now access reference fields for benchmark metrics.
        names_by_type = {q["type"]: name for name, q in cases[0]["request"]["questions"].items()}
        per_type = {}
        for kind, name in names_by_type.items():
            good = sum(p.get(name) == reference_label(c["request"]["questions"][name], c["reference"][name])
                       for p, c in zip(predictions, cases))
            per_type[kind] = {"correct": good, "total": 60, "accuracy": good / 60}
        all_three = sum(all(p.get(name) == reference_label(c["request"]["questions"][name], c["reference"][name])
                            for name in names_by_type.values()) for p, c in zip(predictions, cases))
        lat = [r["case_three_call_inference_ms"] for r in raw_rows]
        metrics = {"per_question_type": per_type,
                   "all_three_correct": {"correct": all_three, "total": 60, "accuracy": all_three / 60},
                   "case_latency_ms": {"scope": "sum of three sequential Laya forward calls per case, including eval synchronization; excludes model load",
                                       "n": 60, "p50": percentile(lat, .50), "p95": percentile(lat, .95),
                                       "mean": sum(lat) / 60},
                   "guard_summary": {"early_stop_bytes": EARLY_MEMORY, "absolute_stop_bytes": HARD_MEMORY,
                                     "max_swap_growth_bytes_exclusive": MAX_SWAP_GROWTH,
                                     "max_process_peak_footprint_bytes": max(s["process_peak_footprint_bytes"] for r in raw_rows for s in r["memory_guard_samples"]),
                                     "max_swap_growth_observed_bytes": max(s["swap_growth_from_start_bytes"] for r in raw_rows for s in r["memory_guard_samples"])},
                   "error_count": 0, "completed_cases": 60, "completed_decisions": 180,
                   "scored_after_freeze": True}
        write_json(run_dir / "metrics.json", metrics)
        metadata.update({"finished_at": datetime.now(timezone.utc).isoformat(), "completed_cases": 60,
                         "completed_decisions": 180, "error_count": 0, "cache_cleanup": True,
                         "max_process_peak_footprint_bytes": metrics["guard_summary"]["max_process_peak_footprint_bytes"],
                         "max_swap_growth_bytes": metrics["guard_summary"]["max_swap_growth_observed_bytes"]})
        write_json(run_dir / "run-metadata.json", metadata)
        write_json(run_dir / "checksums.json", {str(p.relative_to(run_dir)): sha(p)
                   for p in run_dir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"Saved cacheguard full run to {run_dir}; zero errors")
        return 0
    except GuardStop as exc:
        if run_dir is not None:
            if active_raw is not None and active_pred is not None:
                active_raw["case_three_call_inference_ms"] = ((time.perf_counter_ns() - active_start) / 1e6
                                                                if active_start is not None else None)
                raw_rows.append(active_raw); predictions.append(active_pred)
                timing_rows.append({"id": active_raw["case_id"], "partial_case": True,
                                    "case_three_call_inference_ms": active_raw["case_three_call_inference_ms"]})
            write_jsonl(run_dir / "raw-results.jsonl", raw_rows)
            write_jsonl(run_dir / "laya-predictions.jsonl", predictions)
            write_jsonl(run_dir / "timings.jsonl", timing_rows)
            write_json(run_dir / "guard-stop.json", {"phase": phase, "reason": str(exc),
                        "completed_full_cases": len(raw_rows),
                        "completed_calls": sum(len(r["decisions"]) for r in raw_rows),
                        "smoke_only_no_full_run": phase == "smoke"})
            write_json(run_dir / "checksums.json", {str(p.relative_to(run_dir)): sha(p)
                       for p in run_dir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"STOP: {exc}; no retry or further inference", file=sys.stderr)
        return 3
    except Exception as exc:
        if run_dir is not None:
            if active_raw is not None and active_pred is not None:
                active_raw["case_three_call_inference_ms"] = ((time.perf_counter_ns() - active_start) / 1e6
                                                                if active_start is not None else None)
                raw_rows.append(active_raw); predictions.append(active_pred)
                timing_rows.append({"id": active_raw["case_id"], "partial_case": True,
                                    "case_three_call_inference_ms": active_raw["case_three_call_inference_ms"]})
            write_jsonl(run_dir / "raw-results.jsonl", raw_rows)
            write_jsonl(run_dir / "laya-predictions.jsonl", predictions)
            write_jsonl(run_dir / "timings.jsonl", timing_rows)
            write_json(run_dir / "run-failure.json", {"phase": phase, "error_type": type(exc).__name__, "error": str(exc)})
            write_json(run_dir / "checksums.json", {str(p.relative_to(run_dir)): sha(p)
                       for p in run_dir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"cacheguard Laya failed: {type(exc).__name__}: {exc}; no retry", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
