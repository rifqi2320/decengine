#!/usr/bin/env python3
"""Compare decengine MLX decisions with TypeSafe JEV for JSONL cases.

Fixture format: one JSON object per line with ``id``, ``state`` (JSON object),
and ``questions`` in decengine's public wire format. See ``--help`` and the
comparison README for details. This script never prints or saves credentials.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_MODEL = None
DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_KEY = Path.home() / "Documents/Code/jev.api.key"
DEFAULT_FIXTURE = Path("benchmarks/cases/decision-cases-100.jsonl")
DEFAULT_RESULTS = Path("results/comparison-runs")
DEFAULT_MODEL_STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
DEFAULT_NATIVE_LIBRARY = Path("target/release/libdecengine.dylib")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"{path}:{line_no}: each fixture line must be an object")
            case_id = case.get("id")
            if not isinstance(case_id, (str, int)) or not str(case_id).strip():
                raise ValueError(f"{path}:{line_no}: id must be a non-empty string or integer")
            if str(case_id) in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate id {case_id!r}")
            seen_ids.add(str(case_id))
            request = case.get("request")
            if not isinstance(request, dict):
                raise ValueError(f"{path}:{line_no}: request must be an object")
            if not isinstance(request.get("state"), dict):
                raise ValueError(f"{path}:{line_no}: request.state must be a JSON object")
            questions = request.get("questions")
            if not isinstance(questions, dict) or not questions:
                raise ValueError(f"{path}:{line_no}: request.questions must be a non-empty object")
            model = request.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"{path}:{line_no}: request.model must be a non-empty string")
            # Perform precise public-wire validation via decengine's typed constructors.
            decode_questions(questions)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def decode_questions(wire: dict[str, Any]) -> dict[str, Any]:
    from decengine import Choice, Noul, Score, ScoreLevel

    decoded: dict[str, Any] = {}
    for name, value in wire.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, dict):
            raise ValueError(f"invalid question {name!r}: expected named object")
        kind, prompt = value.get("type"), value.get("prompt")
        try:
            if kind == "choice":
                decoded[name] = Choice(prompt, value["options"])
            elif kind == "noul":
                decoded[name] = Noul(prompt)
            elif kind == "score":
                levels = value["levels"]
                if not isinstance(levels, list):
                    raise ValueError("levels must be an array")
                decoded[name] = Score(prompt, [
                    ScoreLevel(item["label"], item["criterion"], item["value"])
                    for item in levels
                ])
            else:
                raise ValueError(f"unsupported type {kind!r}")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid question {name!r}: {exc}") from exc
    return decoded


def jev_questions(wire: dict[str, Any]) -> dict[str, Any]:
    translated = {}
    for name, question in wire.items():
        kind = question["type"]
        item: dict[str, Any] = {"type": kind, "instructions": question["prompt"]}
        if kind == "choice":
            item["criteria"] = question["options"]
        elif kind == "score":
            # JEV accepts ordered textual criteria, while decengine permits numeric values
            # and custom labels. Encode label + criterion to retain both semantics.
            item["criteria"] = [
                f"{level['label']}: {level['criterion']} (value={level['value']})"
                for level in question["levels"]
            ]
        translated[name] = item
    return translated


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def redact(value: Any, secret: str) -> Any:
    """Defensively scrub an echoed credential from any provider-controlled data."""
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def native_response_dict(response: Any) -> dict[str, Any]:
    return {
        "id": response.id,
        "model": response.model,
        "created": response.created,
        "results": {name: jsonable(result) for name, result in response.results.items()},
        "usage": jsonable(response.usage),
    }


def normalize_native(response: dict[str, Any]) -> dict[str, Any]:
    return response["results"]


def normalize_jev(response: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, answer in response.get("answers", {}).items():
        kind = answer.get("type")
        if kind == "choice":
            result[name] = {"type": kind, "selected": answer.get("choice"),
                            "confidence": answer.get("confidence"),
                            "probabilities": answer.get("probabilities")}
        elif kind == "noul":
            value = answer.get("noul")
            result[name] = {"type": kind, "value": value,
                            "decision": value >= 0.5 if isinstance(value, (int, float)) else None}
        elif kind == "score":
            raw_score = answer.get("score")
            levels = questions[name].get("levels", [])
            mapped_level = levels[int(raw_score)] if (
                isinstance(raw_score, (int, float)) and float(raw_score).is_integer()
                and 0 <= int(raw_score) < len(levels)
            ) else None
            result[name] = {"type": kind, "value": answer.get("score"),
                            "confidence": answer.get("confidence"),
                            "probabilities": answer.get("probabilities"),
                            "legend": answer.get("legend"),
                            "mapped_level": mapped_level}
        else:
            result[name] = {"type": kind, "raw": answer}
    return result


def post_jev(url: str, key: str, payload: dict[str, Any], timeout: float) -> tuple[Any, float]:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "application/json",
    })
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raw = raw.replace(key, "[REDACTED]")
        raise RuntimeError(f"JEV HTTP {exc.code}: {raw[:4000]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"JEV request failed: {type(exc).__name__}: {str(exc).replace(key, '[REDACTED]')}") from None
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"JEV HTTP {status} returned non-JSON response: {raw[:4000]}") from None
    return {"http_status": status, "body": parsed}, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="override fixture decengine model id")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--jev-url", default=DEFAULT_URL)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--native-library", type=Path, default=DEFAULT_NATIVE_LIBRARY,
                        help="decengine shared library (MLX build by default)")
    parser.add_argument("--model-store", type=Path, default=DEFAULT_MODEL_STORE,
                        help="local MLX model store")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--expected-cases", type=int, default=100)
    parser.add_argument("--limit", type=int, help="run only first N fixtures (explicit opt-in)")
    parser.add_argument("--dry-run", action="store_true", help="validate fixtures only; no key/model/network access")
    args = parser.parse_args()
    try:
        cases = load_cases(args.fixtures)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            cases = cases[:args.limit]
        elif len(cases) != args.expected_cases:
            raise ValueError(f"expected {args.expected_cases} cases, found {len(cases)}; use --limit explicitly for a subset")
        parsed_url = urlparse(args.jev_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc or parsed_url.username or parsed_url.password or parsed_url.query:
            raise ValueError("JEV URL must be an absolute HTTPS URL")
        if args.dry_run:
            print(f"Validated {len(cases)} cases from {args.fixtures}; dry run made no model/API calls.")
            return 0
        if not args.key_file.is_file():
            raise ValueError(f"JEV key file not found: {args.key_file}")
        key = args.key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"JEV key file is empty: {args.key_file}")
        # Never print, log, or include the key/header in persisted run metadata.
        from decengine import Engine
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = args.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        output = run_dir / "cases.jsonl"
        manifest = {
            "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture_file": str(args.fixtures), "case_count": len(cases),
            "decengine_models": sorted({args.model or case["request"]["model"] for case in cases}),
            "decengine_engine": "mlx", "jev_model": args.jev_model,
            "jev_url": args.jev_url, "native_library": str(args.native_library) if args.native_library else None,
            "model_store": str(args.model_store),
            "python": sys.version.split()[0], "platform": sys.platform,
        }
        (run_dir / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with ExitStack() as stack, output.open("w", encoding="utf-8") as out:
            engines = {}
            for index, case in enumerate(cases, 1):
                request_data = case["request"]
                model_name = args.model or request_data["model"]
                if model_name not in engines:
                    engines[model_name] = stack.enter_context(Engine(
                        model_name,
                        native_library=args.native_library,
                        options={"engine": "mlx", "model_store": str(args.model_store)},
                    ))
                engine = engines[model_name]
                record: dict[str, Any] = {"case_id": case["id"], "input": case,
                                          "metadata": {"index": index, "run_id": run_id}}
                questions = decode_questions(request_data["questions"])
                started = time.perf_counter_ns()
                try:
                    response = engine.decide(state=request_data["state"], questions=questions)
                    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                    native = native_response_dict(response)
                    record["decengine"] = {"raw_response": native, "normalized_decisions": normalize_native(native),
                                           "timing_ms": elapsed_ms, "error": None}
                except Exception as exc:
                    record["decengine"] = {"raw_response": None, "normalized_decisions": None,
                                           "timing_ms": (time.perf_counter_ns() - started) / 1e6,
                                           "error": f"{type(exc).__name__}: {str(exc).replace(key, '[REDACTED]')}"}
                state = json.dumps(request_data["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                payload = {"state": state, "model": args.jev_model,
                           "questions": jev_questions(request_data["questions"])}
                record["jev_request_payload"] = payload
                jev_raw = None
                jev_elapsed = None
                jev_started = time.perf_counter_ns()
                try:
                    jev_raw, elapsed = post_jev(args.jev_url, key, payload, args.timeout)
                    jev_elapsed = elapsed
                    raw_body = jev_raw["body"]
                    if not isinstance(raw_body, dict):
                        raise RuntimeError("JEV response must be a JSON object")
                    raw_body = raw_body.get("result", raw_body)
                    if not isinstance(raw_body, dict) or not isinstance(raw_body.get("answers"), dict):
                        raise RuntimeError(f"unexpected JEV response shape: {json.dumps(jev_raw['body'])[:4000]}")
                    record["jev"] = {"raw_response": jev_raw, "normalized_decisions": normalize_jev(raw_body, request_data["questions"]),
                                     "timing_ms": elapsed, "error": None}
                except Exception as exc:
                    if jev_elapsed is None:
                        jev_elapsed = (time.perf_counter_ns() - jev_started) / 1e6
                    record["jev"] = {"raw_response": jev_raw, "normalized_decisions": None,
                                     "timing_ms": jev_elapsed,
                                     "error": f"{type(exc).__name__}: {str(exc).replace(key, '[REDACTED]')}"}
                out.write(json.dumps(redact(record, key), ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
                print(f"[{index}/{len(cases)}] {case['id']}: "
                      f"decengine={'ok' if not record['decengine']['error'] else 'error'}, "
                      f"JEV={'ok' if not record['jev']['error'] else 'error'}")
        print(f"Saved comparison results to {run_dir}")
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
