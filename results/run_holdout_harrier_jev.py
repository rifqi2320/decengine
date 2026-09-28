#!/usr/bin/env python3
"""Run real Harrier MLX and JEV once each per independently authored holdout case."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_jev import (
    DEFAULT_KEY, DEFAULT_MODEL_STORE, DEFAULT_NATIVE_LIBRARY, DEFAULT_URL,
    decode_questions, jev_questions, load_cases, native_response_dict,
    normalize_jev, normalize_native, post_jev, redact,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks/cases/decision-cases-holdout.jsonl"
QWEN_DIR = ROOT / "results/qwen-holdout/20260923T090609039852Z"
QWEN_BASELINE = QWEN_DIR / "baseline.jsonl"
MODEL = "microsoft/harrier-oss-v1-0.6b"
MANIFEST = ROOT / "models/manifests/harrier-oss-v1-0.6b.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_qwen(cases: list[dict[str, Any]], baseline_path: Path) -> None:
    rows = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = [str(case["id"]) for case in cases]
    actual = [str(row["case_id"]) for row in rows]
    if actual != expected or any(row.get("variant") != "baseline" for row in rows):
        raise ValueError("existing Qwen baseline does not match holdout case IDs/order")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURE)
    parser.add_argument("--qwen-baseline", type=Path, default=QWEN_BASELINE)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/holdout-comparison-runs")
    parser.add_argument("--model-store", type=Path, default=DEFAULT_MODEL_STORE)
    parser.add_argument("--native-library", type=Path, default=DEFAULT_NATIVE_LIBRARY)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--jev-url", default=DEFAULT_URL)
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, help="explicit case limit for live smoke only")
    args = parser.parse_args()
    key = ""
    try:
        all_cases = load_cases(args.fixtures)
        if len(all_cases) != 60:
            raise ValueError(f"expected 60 holdout cases, found {len(all_cases)}")
        validate_qwen(all_cases, args.qwen_baseline)
        cases = all_cases
        if args.limit is not None:
            if args.limit < 1 or args.limit > len(cases):
                raise ValueError("--limit must be between 1 and 60")
            cases = cases[:args.limit]
        install_path = args.model_store / "models/microsoft--harrier-oss-v1-0.6b/install.json"
        install = json.loads(install_path.read_text(encoding="utf-8"))
        manifest_bytes = MANIFEST.read_bytes()
        manifest = json.loads(manifest_bytes)
        if install.get("model_id") != MODEL or install.get("profile_version") != manifest.get("profile_version"):
            raise ValueError("installed Harrier model record/profile mismatch")
        if not args.native_library.is_file():
            raise ValueError(f"native library not found: {args.native_library}")
        if not args.key_file.is_file():
            raise ValueError(f"JEV key file not found: {args.key_file}")
        key = args.key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("JEV key file is empty")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = args.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(args.fixtures), "fixture_sha256": sha256(args.fixtures),
            "case_count": len(cases), "full_holdout_case_count": len(all_cases),
            "qwen_baseline": str(args.qwen_baseline), "qwen_baseline_sha256": sha256(args.qwen_baseline),
            "qwen_baseline_reused_without_inference": True,
            "harrier_model": MODEL, "harrier_engine": "mlx", "jev_model": args.jev_model,
            "jev_url": args.jev_url, "model_store": str(args.model_store), "install_record": install,
            "profile": manifest, "profile_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "native_library": str(args.native_library), "native_library_sha256": sha256(args.native_library),
            "python": sys.version.split()[0], "platform": sys.platform,
            "calls_per_case": {"harrier_mlx": 1, "jev_api": 1},
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        from decengine import Engine

        errors = {"harrier": 0, "jev": 0}
        output = run_dir / "cases.jsonl"
        with ExitStack() as stack, output.open("w", encoding="utf-8") as out:
            engine = stack.enter_context(Engine(MODEL, native_library=args.native_library,
                options={"engine": "mlx", "model_store": str(args.model_store)}))
            for index, case in enumerate(cases, 1):
                request = case["request"]
                questions = decode_questions(request["questions"])
                record: dict[str, Any] = {"case_id": str(case["id"]), "input": case,
                    "metadata": {"index": index, "run_id": run_id}}
                started = time.perf_counter_ns()
                try:
                    response = engine.decide(state=request["state"], questions=questions)
                    raw = native_response_dict(response)
                    record["harrier"] = {"raw_response": raw, "normalized_decisions": normalize_native(raw),
                        "timing_ms": (time.perf_counter_ns() - started) / 1e6, "error": None}
                except Exception as exc:
                    errors["harrier"] += 1
                    record["harrier"] = {"raw_response": None, "normalized_decisions": None,
                        "timing_ms": (time.perf_counter_ns() - started) / 1e6,
                        "error": f"{type(exc).__name__}: {str(exc).replace(key, '[REDACTED]')}"}

                state = json.dumps(request["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                payload = {"state": state, "model": args.jev_model,
                    "questions": jev_questions(request["questions"])}
                record["jev_request_payload"] = payload
                jev_started = time.perf_counter_ns()
                try:
                    raw_jev, elapsed = post_jev(args.jev_url, key, payload, args.timeout)
                    body = raw_jev["body"]
                    if not isinstance(body, dict):
                        raise RuntimeError("JEV response must be a JSON object")
                    body = body.get("result", body)
                    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
                        raise RuntimeError("unexpected JEV response shape")
                    record["jev"] = {"raw_response": raw_jev,
                        "normalized_decisions": normalize_jev(body, request["questions"]),
                        "timing_ms": elapsed, "error": None}
                except Exception as exc:
                    errors["jev"] += 1
                    record["jev"] = {"raw_response": None, "normalized_decisions": None,
                        "timing_ms": (time.perf_counter_ns() - jev_started) / 1e6,
                        "error": f"{type(exc).__name__}: {str(exc).replace(key, '[REDACTED]')}"}
                out.write(json.dumps(redact(record, key), ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
                print(f"[{index}/{len(cases)}] {case['id']}: Harrier={'error' if record['harrier']['error'] else 'ok'}, JEV={'error' if record['jev']['error'] else 'ok'}")
        summary = {"run_id": run_id, "case_count": len(cases), "harrier_errors": errors["harrier"],
            "jev_errors": errors["jev"], "total_errors": sum(errors.values()), "cases_sha256": sha256(output)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(cases)} cases ({summary['total_errors']} failures) to {run_dir}")
        return 0 if summary["total_errors"] == 0 else 2
    except Exception as exc:
        # Do not include exception data that could conceivably contain the credential.
        message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
        print(f"runner failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
