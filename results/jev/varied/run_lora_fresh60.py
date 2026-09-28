#!/usr/bin/env python3
"""JEV-only typed evaluation on the frozen LoRA fresh60 fixture.

Uses the established JEV wire translator and request helper. No Laya/MLX/GPU
runtime is imported or started. Raw API outcomes are saved before references
are used for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-lora-fresh-60.jsonl"
sys.path.insert(0, str(ROOT / "results"))
from compare_jev import DEFAULT_KEY, DEFAULT_URL, jev_questions, post_jev, redact


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


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(c.get("id", "")) for c in cases]
    if len(cases) != 60 or ids != [f"mq-lora-fresh-{i:03d}" for i in range(1, 61)]:
        raise ValueError("expected ordered mq-lora-fresh-001..mq-lora-fresh-060 records")
    for case in cases:
        req = case.get("request")
        if not isinstance(req, dict) or set(req) != {"state", "questions"} or not isinstance(req.get("state"), dict):
            raise ValueError(f"{case['id']}: malformed request state/questions")
        if has_forbidden_input_field(req):
            raise ValueError(f"{case['id']}: reference/evaluation metadata must not occur in request")
        qs = req.get("questions")
        if not isinstance(qs, dict) or len(qs) != 3 or [q.get("type") for q in qs.values()] != ["choice", "noul", "score"]:
            raise ValueError(f"{case['id']}: expected one choice, one noul, and one score question")
        for name, q in qs.items():
            if not isinstance(q.get("prompt"), str) or not q["prompt"].strip():
                raise ValueError(f"{case['id']}/{name}: missing prompt")
            if q["type"] == "choice" and (not isinstance(q.get("options"), dict) or not q["options"]):
                raise ValueError(f"{case['id']}/{name}: choice must have candidates")
            if q["type"] == "score" and (not isinstance(q.get("levels"), list) or not q["levels"]):
                raise ValueError(f"{case['id']}/{name}: score must have ordered levels")
    return cases


def typed_labels(body: dict[str, Any], questions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("JEV answer names do not match the three requested questions")
    typed: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    for name, q in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != q["type"]:
            raise ValueError(f"JEV returned wrong/missing typed answer for {name}")
        if q["type"] == "choice":
            label = answer.get("choice")
            if label not in q["options"]:
                raise ValueError(f"JEV choice outcome is not a candidate for {name}")
            labels[name] = label
            typed[name] = {"type": "choice", "selected": label,
                           "probabilities": answer.get("probabilities"), "confidence": answer.get("confidence")}
        elif q["type"] == "noul":
            val = answer.get("noul")
            if not isinstance(val, (int, float)):
                raise ValueError(f"JEV noul outcome missing numeric value for {name}")
            decision = bool(val >= 0.5)
            labels[name] = decision
            typed[name] = {"type": "noul", "value": val, "decision": decision}
        else:
            probs = answer.get("probabilities")
            label = None
            if isinstance(probs, dict) and probs:
                values = [(int(k), float(v)) for k, v in probs.items()]
                best = max(v for _, v in values)
                winners = [i for i, v in values if v == best]
                if len(winners) == 1 and 0 <= winners[0] < len(q["levels"]):
                    label = q["levels"][winners[0]]["label"]
            else:
                score = answer.get("score")
                if isinstance(score, (int, float)) and float(score).is_integer() and 0 <= int(score) < len(q["levels"]):
                    label = q["levels"][int(score)]["label"]
            if label is None:
                raise ValueError(f"JEV score has no unambiguous rubric label for {name}")
            labels[name] = label
            typed[name] = {"type": "score", "value": answer.get("score"), "label": label,
                           "probabilities": probs, "legend": answer.get("legend"),
                           "confidence": answer.get("confidence")}
    return typed, labels


def reference_label(q: dict[str, Any], ref: dict[str, Any]) -> Any:
    target = ref["target"]
    if q["type"] == "choice":
        return target["selected"]
    if q["type"] == "noul":
        return target["value"]
    return target["label"]


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--jev-url", default=DEFAULT_URL)
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    key = ""
    try:
        cases = load_cases(args.fixture)
        if not args.key_file.is_file():
            raise FileNotFoundError("authorized JEV key file missing")
        key = args.key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("authorized JEV key file empty")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        outdir = (args.out or ROOT / "results/jev/varied/runs" / run_id).resolve()
        outdir.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, Any] = {
            "run_id": outdir.name, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(args.fixture.resolve()), "fixture_sha256": sha(args.fixture), "case_count": 60,
            "question_count": 180, "adapter": str(Path(__file__).resolve()),
            "typed_contract_helpers": str((ROOT / "results/compare_jev.py").resolve()),
            "jev_model": args.jev_model, "jev_url": args.jev_url,
            "requests_per_state": 1, "questions_per_request": 3,
            "input_fields": "request.state and request.questions only; top-level reference and metadata excluded",
            "prompt_policy": "fixture question wires passed through established JEV typed translator unchanged; no prompt tuning",
            "score_mapping": "argmax of score probability distribution maps to rubric label; exact integer scalar index fallback only",
            "other_inference": "JEV only; no Laya, MLX, Metal, GPU, or probe model loaded",
            "key_persisted": False,
        }
        write_json(outdir / "run-metadata.json", metadata)
        raw_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        timings: list[dict[str, Any]] = []
        errors = 0
        for index, case in enumerate(cases, 1):
            req = case["request"]
            payload = {"state": json.dumps(req["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                       "model": args.jev_model, "questions": jev_questions(req["questions"])}
            if set(payload) != {"state", "model", "questions"}:
                raise AssertionError("unexpected fields in JEV payload")
            started = time.perf_counter_ns()
            response = None
            try:
                response, elapsed = post_jev(args.jev_url, key, payload, args.timeout)
                body = response["body"]
                if isinstance(body, dict):
                    body = body.get("result", body)
                if not isinstance(body, dict):
                    raise ValueError("JEV response body must be an object")
                typed, labels = typed_labels(body, req["questions"])
                record = {"case_id": str(case["id"]), "request_payload": payload,
                          "raw_response": response, "normalized_typed_outcomes": typed,
                          "normalized_labels": labels, "timing_ms": elapsed, "error": None}
                prediction_rows.append({"id": str(case["id"]), **labels})
            except Exception as exc:
                errors += 1
                elapsed = (time.perf_counter_ns() - started) / 1e6
                msg = str(exc).replace(key, "[REDACTED]") if key else str(exc)
                record = {"case_id": str(case["id"]), "request_payload": payload,
                          "raw_response": response, "normalized_typed_outcomes": None,
                          "normalized_labels": None, "timing_ms": elapsed,
                          "error": f"{type(exc).__name__}: {msg}"}
                prediction_rows.append({"id": str(case["id"]), **{n: None for n in req["questions"]}})
            raw_rows.append(redact(record, key))
            timings.append({"case_id": str(case["id"]), "jev_single_request_ms": record["timing_ms"], "error": record["error"]})
            # Persist raw bodies/predictions incrementally. References are not included.
            write_jsonl(outdir / "raw-results.jsonl", raw_rows)
            write_jsonl(outdir / "jev-predictions.jsonl", prediction_rows)
            write_jsonl(outdir / "timings.jsonl", timings)
            print(f"[{index}/60] {case['id']}: JEV={'ok' if record['error'] is None else 'error'}", flush=True)

        # Freeze all inference artifacts before references are opened for scoring.
        frozen_paths = ("raw-results.jsonl", "jev-predictions.jsonl", "timings.jsonl", "run-metadata.json")
        freeze = {"status": "predictions_frozen_before_reference_scoring",
                  "frozen_at": datetime.now(timezone.utc).isoformat(),
                  "fixture_sha256": sha(args.fixture), "case_count": len(raw_rows),
                  "request_count": len(raw_rows), "normalized_outcome_count": sum(len(r.get("normalized_labels") or {}) for r in raw_rows),
                  "errors": errors,
                  "artifact_sha256": {name: sha(outdir / name) for name in frozen_paths},
                  "scoring_stage": "not run; no fixture reference values read before this freeze"}
        write_json(outdir / "PRE_REFERENCE_FREEZE.json", freeze)

        # References are opened only after all raw bodies, predictions, timings,
        # provenance, and their hashes have been persisted.
        def metrics(preds: list[dict[str, Any]], indexes: list[int]) -> dict[str, Any]:
            names_by_type = {q["type"]: name for name, q in cases[0]["request"]["questions"].items()}
            per: dict[str, Any] = {}
            for kind, name in names_by_type.items():
                good = sum(preds[i].get(name) == reference_label(cases[i]["request"]["questions"][name], cases[i]["reference"][name]) for i in indexes)
                per[kind] = {"correct": good, "total": len(indexes), "accuracy": good / len(indexes) if indexes else None}
            all3 = sum(all(preds[i].get(name) == reference_label(cases[i]["request"]["questions"][name], cases[i]["reference"][name]) for name in names_by_type.values()) for i in indexes)
            return {"per_question_type": per, "all_three_correct": {"correct": all3, "total": len(indexes), "accuracy": all3 / len(indexes) if indexes else None}}

        all_indexes = list(range(60))
        by_domain = {d: [i for i, c in enumerate(cases) if c.get("domain", "unknown") == d]
                     for d in sorted({c.get("domain", "unknown") for c in cases})}
        elapsed = [t["jev_single_request_ms"] for t in timings]
        scored = {"jev": metrics(prediction_rows, all_indexes),
                  "per_domain": {d: {"case_count": len(ix), **metrics(prediction_rows, ix)} for d, ix in by_domain.items()},
                  "timing_per_case_ms": {"n": 60, "p50": percentile(elapsed, .50), "p95": percentile(elapsed, .95),
                                          "mean": statistics.mean(elapsed), "min": min(elapsed), "max": max(elapsed)},
                  "scored_after_raw_and_predictions_saved": True,
                  "error_count": errors, "completed_states": len(raw_rows), "completed_typed_predictions": sum(len(r.get("normalized_labels") or {}) for r in raw_rows)}
        write_json(outdir / "metrics.json", scored)
        metadata.update({"finished_at": datetime.now(timezone.utc).isoformat(), "completed_states": len(raw_rows),
                         "completed_typed_predictions": scored["completed_typed_predictions"], "error_count": errors})
        write_json(outdir / "run-metadata.json", metadata)
        report = ["# JEV-only LoRA fresh60 evaluation", "", f"Run: `{outdir.name}`",
                  f"Fixture SHA-256: `{metadata['fixture_sha256']}`", "",
                  "JEV only: one typed API request per state with one choice, one noul, and one score question. No Laya/MLX/GPU execution was started.",
                  "", "| Type | Correct / 60 | Accuracy |", "|---|---:|---:|"]
        for kind, value in scored["jev"]["per_question_type"].items():
            report.append(f"| {kind} | {value['correct']}/60 | {value['accuracy']:.1%} |")
        all3 = scored["jev"]["all_three_correct"]
        report += [f"| All three | {all3['correct']}/60 | {all3['accuracy']:.1%} |", "",
                   f"Completed {len(raw_rows)}/60 requests ({scored['completed_typed_predictions']}/180 normalized outcomes), errors {errors}.",
                   f"Per-case single-request latency p50/p95: {scored['timing_per_case_ms']['p50']:.1f}/{scored['timing_per_case_ms']['p95']:.1f} ms.",
                   "Per-domain supports/accuracies are in `metrics.json`; raw bodies are in `raw-results.jsonl`.",
                   "The fixture's reference and metadata fields were excluded from request inputs. Results remain separate from the frozen LoRA trainer and were not used for tuning.", ""]
        (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")
        write_json(outdir / "checksums.json", {str(p.relative_to(outdir)): sha(p) for p in outdir.rglob("*") if p.is_file() and p.name != "checksums.json"})
        print(f"Saved JEV-only results to {outdir}; errors={errors}")
        return 0 if len(raw_rows) == 60 and scored["completed_typed_predictions"] == 180 and errors == 0 else 2
    except Exception as exc:
        msg = str(exc).replace(key, "[REDACTED]") if key else str(exc)
        print(f"JEV-only runner failed: {type(exc).__name__}: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
