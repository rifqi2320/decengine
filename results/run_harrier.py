#!/usr/bin/env python3
"""Run the local Harrier MLX model against the decision fixture (no network/API)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_jev import DEFAULT_MODEL_STORE, decode_questions, load_cases, native_response_dict


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks/cases/decision-cases-100.jsonl"
ORIGINAL = ROOT / "results/comparison-runs/20260923T082226631235Z/cases.jsonl"
MANIFEST = ROOT / "models/manifests/harrier-oss-v1-0.6b.json"
MODEL = "microsoft/harrier-oss-v1-0.6b"


def original_cases(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row["case_id"])
            if case_id in result:
                raise ValueError(f"{path}:{line_no}: duplicate case_id {case_id}")
            result[case_id] = row
    return result


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.get("results", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURE)
    parser.add_argument("--original-archive", type=Path, default=ORIGINAL)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/harrier-runs")
    parser.add_argument("--model-store", type=Path, default=DEFAULT_MODEL_STORE)
    parser.add_argument("--native-library", type=Path, default=ROOT / "target/release/libdecengine.dylib")
    parser.add_argument("--limit", type=int, help="explicitly limit run, e.g. smoke with --limit 1")
    args = parser.parse_args()

    try:
        cases = load_cases(args.fixtures)
        archived = original_cases(args.original_archive)
        ids = [str(case["id"]) for case in cases]
        if set(ids) != set(archived):
            raise ValueError("fixture and original archive case IDs differ")
        for case in cases:
            case_id = str(case["id"])
            if case != archived[case_id].get("input"):
                raise ValueError(f"fixture/archive input mismatch for {case_id}")
        if len(cases) != 100 and args.limit is None:
            raise ValueError(f"expected 100 fixture cases, found {len(cases)}")
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            cases = cases[:args.limit]

        install_path = args.model_store / "models/microsoft--harrier-oss-v1-0.6b/install.json"
        install = json.loads(install_path.read_text(encoding="utf-8"))
        manifest_bytes = MANIFEST.read_bytes()
        manifest = json.loads(manifest_bytes)
        if install["model_id"] != MODEL or install["profile_version"] != manifest["profile_version"]:
            raise ValueError("installed model record does not match the active Harrier profile")
        if not args.native_library.is_file():
            raise ValueError(f"native library not found: {args.native_library}")

        from decengine import Engine

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = args.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "metadata.json").write_text(json.dumps({
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL,
            "engine": "mlx",
            "fixture": str(args.fixtures),
            "original_archive": str(args.original_archive),
            "original_archive_sha256": hashlib.sha256(args.original_archive.read_bytes()).hexdigest(),
            "case_count": len(cases),
            "model_store": str(args.model_store),
            "install_record": install,
            "profile": manifest,
            "profile_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "native_library": str(args.native_library),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "network_or_api_calls": False,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        output = run_dir / "cases.jsonl"
        errors = 0
        with Engine(MODEL, native_library=args.native_library,
                    options={"engine": "mlx", "model_store": str(args.model_store)}) as engine, \
                output.open("w", encoding="utf-8") as out:
            for case in cases:
                case_id = str(case["id"])
                record: dict[str, Any] = {
                    "case_id": case_id,
                    "source_fixture": str(args.fixtures),
                    "original_archive": str(args.original_archive),
                    "original_archive_case_id": case_id,
                    "harrier_model": MODEL,
                }
                start = time.perf_counter_ns()
                try:
                    response = engine.decide(
                        state=case["request"]["state"],
                        questions=decode_questions(case["request"]["questions"]),
                    )
                    raw = native_response_dict(response)
                    record.update({
                        "raw_output": raw,
                        "normalized_output": normalize(raw),
                        "timing_ms": (time.perf_counter_ns() - start) / 1e6,
                        "error": None,
                    })
                except Exception as exc:  # retain per-case failures and continue
                    errors += 1
                    record.update({
                        "raw_output": None,
                        "normalized_output": None,
                        "timing_ms": (time.perf_counter_ns() - start) / 1e6,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    })
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
        print(f"Harrier run {run_id}: {len(cases)} cases, {errors} errors; {run_dir}")
        return 0 if not errors else 2
    except Exception as exc:
        print(f"Harrier runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
