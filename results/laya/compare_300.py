#!/usr/bin/env python3
"""Run standalone Laya inference against the frozen probe-test-300 fixture.

Uses Laya's supported Python SDK (Transformers/PyTorch), not JEV, decengine, or
the fitted probes. Each of owner/urgent/impact is timed as an independent
single-question inference; this makes decision latencies explicit. Laya can
also answer all three in one pass, but that different batching mode is not
used for the per-decision speed figures here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks/cases/probe-test-300.jsonl"
TASKS = ("owner", "urgent", "impact")
MODEL_ID = "convaiinnovations/laya"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            result.append(row)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def validate(path: Path) -> list[dict[str, Any]]:
    cases = read_jsonl(path)
    ids = [str(case.get("id", "")) for case in cases]
    expected = [f"test-{i:03d}" for i in range(1, 301)]
    if ids != expected:
        raise ValueError("expected exactly ordered test-001..test-300 records")
    for case in cases:
        req = case.get("request")
        ref = case.get("reference")
        if not isinstance(req, dict) or not isinstance(req.get("state"), dict):
            raise ValueError(f"{case['id']}: missing request state")
        if not isinstance(ref, dict) or not all(task in ref for task in TASKS):
            raise ValueError(f"{case['id']}: expected owner/urgent/impact reference labels")
        qs = req.get("questions")
        if not isinstance(qs, dict) or set(qs) != set(TASKS):
            raise ValueError(f"{case['id']}: expected exactly owner/urgent/impact questions")
    return cases


def as_laya_question(q: dict[str, Any]) -> dict[str, Any]:
    """Translate this fixture schema to Laya's documented typed-question schema."""
    kind = q.get("type")
    if kind == "choice":
        opts = q.get("options")
        if not isinstance(opts, dict) or not opts:
            raise ValueError("choice requires nonempty options mapping")
        return {"type": "choice", "instructions": q["prompt"], "criteria": opts}
    if kind == "noul":
        return {"type": "noul", "instructions": q["prompt"]}
    if kind == "score":
        levels = q.get("levels")
        if not isinstance(levels, list) or not levels:
            raise ValueError("score requires nonempty levels")
        # Laya's ordinal score ranks descriptions; preserve each full rubric.
        return {"type": "score", "instructions": q["prompt"],
                "criteria": [f"{x['label']}: {x['criterion']}" for x in levels]}
    raise ValueError(f"unsupported fixture question type: {kind!r}")


def rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes (Linux uses KiB).
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def normalize(task: str, answer: dict[str, Any], q: dict[str, Any]) -> Any:
    if task == "owner":
        return answer.get("choice")
    if task == "urgent":
        val = answer.get("noul")
        return bool(val >= 0.5) if isinstance(val, (int, float)) else None
    dist = answer.get("probabilities")
    levels = q.get("levels", [])
    if not isinstance(dist, dict) or not levels:
        return None
    # For an ordinal score, choose the most probable rubric label (not its
    # expected-score bin); this is the closest scalar-label analogue used here.
    winners = [int(k) for k, v in dist.items() if isinstance(v, (int, float)) and
               v == max((x for x in dist.values() if isinstance(x, (int, float))), default=None)]
    return levels[winners[0]]["label"] if len(winners) == 1 and winners[0] < len(levels) else None


def wilson(correct: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0., mid-half), min(1., mid+half)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--out", type=Path, help="run directory (default: results/laya/runs/<UTC timestamp>)")
    p.add_argument("--revision", default="main", help="Hub branch, tag, or commit; resolved SHA is recorded")
    p.add_argument("--device", help="Laya/PyTorch device, e.g. cpu, mps, cuda; default SDK auto-selection")
    p.add_argument("--dry-run", action="store_true", help="validate fixture only; no model load/download")
    args = p.parse_args()
    try:
        cases = validate(args.fixture)
        if args.dry_run:
            print(f"Ready: validated {len(cases)} cases; no Laya import, download, or inference.")
            return 0

        # Resolve a concrete Hub commit and weight path before loading. The model
        # is public; no API/JEV key is read or required.
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(MODEL_ID, revision=args.revision,
            allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]))
        weight_path = snapshot / "model.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Laya checkpoint missing expected weights: {weight_path}")
        revision = snapshot.name
        import laya
        import torch

        agent = laya.load(str(snapshot), device=args.device)
        from laya.common import build_sequence

        runid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = (args.out or ROOT / "results/laya/runs" / runid).resolve()
        outdir.mkdir(parents=True, exist_ok=False)
        start_utc = datetime.now(timezone.utc).isoformat()
        metadata = {
            "run_id": runid, "started_at": start_utc, "fixture": str(args.fixture),
            "fixture_sha256": sha(args.fixture), "case_count": len(cases),
            "model": MODEL_ID, "model_revision": revision,
            "model_weight_file": str(weight_path), "model_weights_sha256": sha(weight_path),
            "model_weight_bytes": weight_path.stat().st_size,
            "laya_version": getattr(laya, "__version__", "unknown"),
            "torch_version": torch.__version__, "transformers_version": None,
            "device_requested": args.device or "SDK automatic selection",
            "inference_device": str(agent.device), "inference_dtype": str(agent.dtype),
            "model_config": {k: agent.cfg.get(k) for k in ("encoder", "max_len", "head_max_len", "amp_dtype")},
            "inference_mode": "three independent typed-question calls per case; no generation; no probe/JEV",
            "score_mapping": "impact label is argmax of Laya ordinal probabilities; rubric label+criterion preserved in criteria",
        }
        try:
            import transformers
            metadata["transformers_version"] = transformers.__version__
        except Exception:
            pass
        write_json(outdir / "run-metadata.json", metadata)

        raw_rows: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for i, case in enumerate(cases, 1):
            req = case["request"]
            raw_case: dict[str, Any] = {"case_id": str(case["id"]), "input_sha256": hashlib.sha256(
                json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "decisions": {}, "memory": {"process_peak_rss_before_bytes": rss_bytes()}}
            pred: dict[str, Any] = {"id": str(case["id"])}
            case_start = time.perf_counter_ns()
            for task in TASKS:
                qraw = req["questions"][task]
                q = as_laya_question(qraw)
                internal = agent._to_internal(q)
                seq, markers = build_sequence(agent.tok, req["state"], internal,
                    agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192))
                token_count = len(seq)
                start = time.perf_counter_ns()
                try:
                    result = agent.predict(req["state"], {task: q})
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    answer = result["answers"][task]
                    pred[task] = normalize(task, answer, qraw)
                    raw_case["decisions"][task] = {
                        "raw_response": result, "normalized_label": pred[task], "error": None,
                        "inference_ms": elapsed, "input_tokens": token_count,
                        "output_tokens": result.get("usage", {}).get("output_tokens", 0),
                        "option_marker_count": len(markers),
                    }
                except Exception as exc:
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    pred[task] = None
                    raw_case["decisions"][task] = {
                        "raw_response": None, "normalized_label": None,
                        "error": f"{type(exc).__name__}: {exc}", "inference_ms": elapsed,
                        "input_tokens": token_count, "output_tokens": None,
                        "option_marker_count": len(markers),
                    }
            raw_case["case_inference_ms_sum_of_three_calls"] = (time.perf_counter_ns() - case_start) / 1e6
            raw_case["case_input_tokens_sum"] = sum(raw_case["decisions"][t]["input_tokens"] for t in TASKS)
            raw_case["memory"]["process_peak_rss_after_bytes"] = rss_bytes()
            if str(agent.device).startswith("cuda") and torch.cuda.is_available():
                raw_case["memory"].update({"cuda_allocated_bytes": torch.cuda.memory_allocated(agent.device),
                                           "cuda_reserved_bytes": torch.cuda.memory_reserved(agent.device)})
            raw_rows.append(raw_case)
            predictions.append(pred)
            if i % 10 == 0 or i == len(cases):
                write_jsonl(outdir / "raw-results.jsonl", raw_rows)
                write_jsonl(outdir / "laya-predictions.jsonl", predictions)
                print(f"[{i}/{len(cases)}] {case['id']} complete", flush=True)

        byid = {str(c["id"]): c for c in cases}
        tasks: dict[str, Any] = {}
        for task in TASKS:
            correct = sum(p.get(task) == byid[p["id"]]["reference"].get(task) for p in predictions)
            classes = sorted({str(byid[p["id"]]["reference"][task]) for p in predictions})
            per_class = {}
            for cls in classes:
                tp = sum(str(p.get(task)) == cls and str(byid[p["id"]]["reference"][task]) == cls for p in predictions)
                support = sum(str(byid[p["id"]]["reference"][task]) == cls for p in predictions)
                predicted = sum(str(p.get(task)) == cls for p in predictions)
                per_class[cls] = {"support": support, "precision": tp / predicted if predicted else None,
                                  "recall": tp / support if support else None,
                                  "accuracy_within_class": tp / support if support else None,
                                  "accuracy_wilson_95_ci": wilson(tp, support)}
            tasks[task] = {"correct": correct, "total": len(predictions), "accuracy": correct / len(predictions),
                           "wilson_95_ci": wilson(correct, len(predictions)), "per_class": per_class}
        complete = sum(all(p.get(t) == byid[p["id"]]["reference"].get(t) for t in TASKS) for p in predictions)
        metrics = {"slices": {"pooled300": {"laya": {
            "per_question": tasks, "complete_case": {"correct": complete, "total": len(predictions),
            "accuracy": complete / len(predictions), "wilson_95_ci": wilson(complete, len(predictions))},
            "overall_question_mean_accuracy": sum(x["accuracy"] for x in tasks.values()) / len(TASKS)}}},
            "sources": {"pooled300": "standalone Laya predictions for all test-001..test-300 cases"}}
        write_json(outdir / "metrics.json", metrics)
        timing_rows = [{"id": row["case_id"], "case_inference_ms_sum_of_three_calls": row["case_inference_ms_sum_of_three_calls"],
                        "case_input_tokens_sum": row["case_input_tokens_sum"],
                        **{t: {"inference_ms": row["decisions"][t]["inference_ms"],
                               "input_tokens": row["decisions"][t]["input_tokens"],
                               "error": row["decisions"][t]["error"]} for t in TASKS}}
                       for row in raw_rows]
        write_jsonl(outdir / "timings.jsonl", timing_rows)
        write_json(outdir / "run-metadata.json", {**metadata, "finished_at": datetime.now(timezone.utc).isoformat(),
            "completed_cases": len(raw_rows), "process_peak_rss_bytes": rss_bytes()})
        checksums = {str(f.relative_to(outdir)): sha(f) for f in outdir.rglob("*")
                     if f.is_file() and f.name != "checksums.json"}
        write_json(outdir / "checksums.json", checksums)
        print(f"Completed Laya comparison in {outdir}")
        return 0
    except Exception as exc:
        print(f"laya compare: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
