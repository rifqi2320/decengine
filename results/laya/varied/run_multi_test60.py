#!/usr/bin/env python3
"""Compare pinned root Laya MLX and JEV on varied multi-question test-60.

Adapter only: keeps the existing verified MLX runner's checkpoint, port,
temperature, candidate mapping, and inference path while accepting this fixture's
three named typed questions. The test fixture is read-only; references are only
scored after raw predictions have been persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-60.jsonl"
MATCHED_FIXTURE_SHA256 = "48a057dd3ebedc6d76b7ca55f0782069f49cd31151576c2bc01932744a6ed614"
EXPLICIT_FIXTURE_SHA256 = "4cb002e1c29f5273ecc23fcdf742b093cb5433d661f5df0b50d209a534fb1409"
PROMPT_FRESH_FIXTURE_SHA256 = "316b3c2e1d0c5cf50356d0e236a5f4adf27fdac9ee6d6c6b56a14f5354ef6a9f"
MATCHED_BASELINE_RUN = ROOT / "results/laya/varied/runs/20260923T151228074560Z"
PROMPT_STUDY_RUN = ROOT / "results/probes/varied/prompt-study/runs/holdout-fresh60-prompt-study-20260924T000000Z"
PROMPT_STUDY_METAL = ROOT / "results/probes/varied/prompt-study/metal/runs/fresh60/shared-v2-baseline"
MLX_RUNNER = ROOT / "results/laya/mlx/compare_300_mlx.py"
sys.path.insert(0, str(ROOT / "results"))
from compare_jev import DEFAULT_KEY, DEFAULT_URL, jev_questions, normalize_jev, post_jev, redact


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_forbidden_input_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"reference", "references", "evaluation_metadata"} or
                   has_forbidden_input_field(item) for key, item in value.items())
    if isinstance(value, list):
        return any(has_forbidden_input_field(item) for item in value)
    return False


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def load_fixture(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c.get("id") for c in cases]
    if ids and ids[0] == "mq-test-001":
        expected_ids = [f"mq-test-{i:03d}" for i in range(1, 61)]
    elif ids and ids[0] == "mq-test-matched-001":
        expected_ids = [f"mq-test-matched-{i:03d}" for i in range(1, 61)]
    elif ids and ids[0] == "mq-explicit-001":
        expected_ids = [f"mq-explicit-{i:03d}" for i in range(1, 61)]
    elif ids and ids[0] == "mq-fresh-001":
        expected_ids = [f"mq-fresh-{i:03d}" for i in range(1, 61)]
    else:
        expected_ids = None
    if len(cases) != 60 or expected_ids is None or ids != expected_ids:
        raise ValueError("expected ordered 60-case multi-question heldout fixture")
    if path.name == "test-matched-60.jsonl" and sha(path) != MATCHED_FIXTURE_SHA256:
        raise ValueError("matched fixture SHA-256 differs from the frozen user-supplied hash")
    if path.name == "test-explicit-60.jsonl" and sha(path) != EXPLICIT_FIXTURE_SHA256:
        raise ValueError("explicit-evidence fixture SHA-256 differs from the frozen user-supplied hash")
    if path.name == "test-prompt-fresh-60.jsonl" and sha(path) != PROMPT_FRESH_FIXTURE_SHA256:
        raise ValueError("prompt-fresh fixture SHA-256 differs from the frozen user-supplied hash")
    for c in cases:
        req = c.get("request", {})
        qs = req.get("questions", {})
        if not isinstance(req.get("state"), dict) or len(qs) != 3:
            raise ValueError(f"{c.get('id')}: malformed three-question request")
        if [q.get("type") for q in qs.values()] != ["choice", "noul", "score"]:
            raise ValueError(f"{c['id']}: expected choice/noul/score typed question order")
    return cases


def laya_question(q: dict[str, Any]) -> dict[str, Any]:
    kind = q["type"]
    if kind == "choice":
        options = q.get("options")
        if not isinstance(options, dict) or not options:
            raise ValueError("choice requires nonempty option mapping")
        return {"type": "choice", "instructions": q["prompt"], "criteria": options}
    if kind == "noul":
        return {"type": "noul", "instructions": q["prompt"]}
    if kind == "score":
        levels = q.get("levels")
        if not isinstance(levels, list) or not levels:
            raise ValueError("score requires ordered rubric levels")
        return {"type": "score", "instructions": q["prompt"],
                "criteria": [f"{x['label']}: {x['criterion']}" for x in levels]}
    raise ValueError(f"unsupported question type: {kind}")


def normalize(q: dict[str, Any], answer: dict[str, Any]) -> Any:
    if q["type"] == "choice":
        return answer.get("choice")
    if q["type"] == "noul":
        v = answer.get("noul")
        return bool(v >= .5) if isinstance(v, (int, float)) else None
    dist, levels = answer.get("probabilities"), q["levels"]
    if not isinstance(dist, dict) or not dist:
        return None
    numeric = [(int(i), float(v)) for i, v in dist.items()]
    top = max(v for _, v in numeric)
    winners = [i for i, v in numeric if v == top]
    # Label argmax, not expected-value binning; ties/invalid indices remain null.
    return levels[winners[0]]["label"] if len(winners) == 1 and 0 <= winners[0] < len(levels) else None


def reference_label(q: dict[str, Any], ref: dict[str, Any]) -> Any:
    kind = q["type"]
    target = ref["target"]
    if kind == "choice":
        return target["selected"]
    if kind == "noul":
        return target["value"]
    return target["label"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--checkpoint", type=Path, default=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root/checkpoint"))
    p.add_argument("--port", type=Path, default=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/laya-mlx-port"))
    p.add_argument("--key-file", type=Path, default=DEFAULT_KEY)
    p.add_argument("--jev-url", default=DEFAULT_URL)
    p.add_argument("--jev-model", default="jev-latest")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    key = ""
    try:
        cases = load_fixture(args.fixture)
        if not args.key_file.is_file():
            raise FileNotFoundError("authorized JEV key file is missing")
        key = args.key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("JEV key file is empty")
        # Import constants and exact pinned-checkpoint safeguards from verified runner.
        import importlib.util
        spec = importlib.util.spec_from_file_location("verified_laya_mlx_runner", MLX_RUNNER)
        verified = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(verified)
        weight_path = args.checkpoint / "model.safetensors"
        if not weight_path.is_file() or sha(weight_path) != verified.WEIGHT_SHA256:
            raise ValueError("pinned root checkpoint missing or SHA-256 mismatch")
        fixture_manifest = args.fixture.with_name("test-matched-60.manifest.json")
        explicit_intervention = args.fixture.name == "test-explicit-60.jsonl"
        if explicit_intervention:
            fixture_manifest = args.fixture.with_name("test-explicit-60.manifest.json")
        prompt_fresh = args.fixture.name == "test-prompt-fresh-60.jsonl"
        if prompt_fresh:
            possible_manifest = args.fixture.with_name("test-prompt-fresh-60.manifest.json")
            fixture_manifest = possible_manifest if possible_manifest.is_file() else None
        sys.path.insert(0, str(args.port.resolve()))
        import mlx
        import mlx.core as mx
        import numpy as np
        import laya_mlx
        from laya_mlx.agent import Agent, collate_items
        from laya_mlx.common import confidence_from_probs, temp_bucket
        if mx.default_device().type != mx.DeviceType.gpu:
            raise RuntimeError(f"expected MLX GPU/Metal device, got {mx.default_device()}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"laya-mlx: this checkpoint ships temperatures.*", category=RuntimeWarning)
            agent = Agent(str(args.checkpoint), dtype="float32", device="gpu", batch_size=1)
        agent.temperature = [float(x) for x in agent.temperature_raw]
        agent.temperature_by_options = {k: float(v) for k, v in agent.temperature_by_options_raw.items()}
        outdir = (args.out or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
        outdir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": outdir.name, "started_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(args.fixture.resolve()), "fixture_sha256": sha(args.fixture), "case_count": len(cases),
            "fixture_manifest": str(fixture_manifest.resolve()) if fixture_manifest and fixture_manifest.is_file() else None,
            "fixture_manifest_sha256": sha(fixture_manifest) if fixture_manifest and fixture_manifest.is_file() else None,
            "adapter": str(Path(__file__).resolve()), "verified_runner": str(MLX_RUNNER.resolve()),
            "model": verified.MODEL_ID, "model_revision": verified.MODEL_REVISION,
            "checkpoint_weights_sha256": sha(weight_path), "checkpoint_weight_bytes": weight_path.stat().st_size,
            "mlx_port_commit": "0a859518634112655cb97c745dbf04f5191aaf13",
            "runtime": "mlx-metal", "mlx_version": importlib.metadata.version("mlx"),
            "device_actual": str(mx.default_device()), "dtype": "float32",
            "temperature_policy": "exact pinned root config values; no port clamp",
            "evaluation_type": ("fresh prompt-study holdout" if prompt_fresh else
                "authored clarity intervention; not an independent holdout" if explicit_intervention else "matched-domain / unseen-question-family holdout"),
            "input_fields": "request.state and request.questions only; top-level reference, metadata, and evaluation_metadata excluded",
            "laya_calls_per_case": 3, "jev_requests_per_case": 1,
            "jev_model": args.jev_model, "jev_url": args.jev_url,
            "score_mapping": "argmax of ordinal probabilities maps to rubric label; never expected-value binning",
            "reference_scoring": "deferred until raw predictions and JEV outcomes persisted",
            "prompt_study_comparison": "frozen v2-baseline shared head only; train-only baseline selection retained; no prompt/head variant selection" if prompt_fresh else None,
        }
        write_json(outdir / "run-metadata.json", metadata)
        raw_rows: list[dict[str, Any]] = []
        laya_preds: list[dict[str, Any]] = []
        jev_preds: list[dict[str, Any]] = []
        errors = {"laya": 0, "jev": 0}
        for index, case in enumerate(cases, 1):
            req = case["request"]
            if has_forbidden_input_field(req):
                raise ValueError(f"{case['id']}: reference/evaluation metadata field found under request")
            laya_row = {"case_id": case["id"], "decisions": {}}
            laya_pred: dict[str, Any] = {"id": case["id"]}
            for name, fixture_q in req["questions"].items():
                q = laya_question(fixture_q)
                items, internal = agent.prepare(req["state"], {name: q})
                item = items[0]
                batch = collate_items(items, agent.tok.pad_token_id)
                started = time.perf_counter_ns()
                try:
                    logits_mx, acts_mx = agent.forward(batch)
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
                        crit = internal[0]["crit"]
                        answer = {"type": "score", "score": float((np.arange(count) * probs).sum()),
                                  "legend": {str(i): v for i, v in enumerate(crit)},
                                  "probabilities": {str(i): float(v) for i, v in enumerate(probs)},
                                  "confidence": float(confidence_from_probs(probs, count)),
                                  "rl_agent": {"act_probability": float(act_probs[0])}}
                    else:
                        answer = {"type": "noul", "noul": float(probs[1]),
                                  "rl_agent": {"act_probability": float(act_probs[0])}}
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    laya_pred[name] = normalize(fixture_q, answer)
                    if laya_pred[name] is None:
                        raise ValueError("Laya returned no unambiguous normalized typed outcome")
                    laya_row["decisions"][name] = {"raw_logits": logits.tolist(), "raw_act_logits": acts.tolist(),
                        "raw_response": {"model": "laya-rl-agent", "answers": {name: answer},
                                         "usage": {"input_tokens": len(item["ids"]), "output_tokens": 0}},
                        "normalized_typed_outcome": laya_pred[name], "error": None, "inference_ms": elapsed,
                        "input_tokens": len(item["ids"]), "temperature": float(temp), "probabilities_unrounded": probs.tolist()}
                except Exception as exc:
                    errors["laya"] += 1
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    laya_pred[name] = None
                    laya_row["decisions"][name] = {"raw_logits": None, "raw_act_logits": None, "raw_response": None,
                        "normalized_typed_outcome": None, "error": f"{type(exc).__name__}: {exc}", "inference_ms": elapsed}
            laya_row["case_inference_ms_sum"] = sum(x["inference_ms"] for x in laya_row["decisions"].values())
            laya_preds.append(laya_pred)

            # Same established JEV typed contract, one request for this shared state.
            payload = {"state": json.dumps(req["state"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                       "model": args.jev_model, "questions": jev_questions(req["questions"])}
            if set(payload) != {"state", "model", "questions"}:
                raise AssertionError("JEV payload contains unexpected fields")
            jev_started = time.perf_counter_ns()
            try:
                raw_jev, elapsed = post_jev(args.jev_url, key, payload, args.timeout)
                body = raw_jev["body"]
                if isinstance(body, dict):
                    body = body.get("result", body)
                if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
                    raise RuntimeError("unexpected JEV response shape")
                typed = normalize_jev(body, req["questions"])
                if set(body["answers"]) != set(req["questions"]):
                    raise RuntimeError("JEV answer names do not match the request question names")
                # Collapse provider schema into the same typed scalar labels for direct scoring.
                pred = {"id": case["id"]}
                for name, q in req["questions"].items():
                    ans = body["answers"].get(name, {})
                    if q["type"] == "choice": pred[name] = ans.get("choice")
                    elif q["type"] == "noul":
                        v = ans.get("noul"); pred[name] = bool(v >= .5) if isinstance(v, (int, float)) else None
                    else:
                        probs = ans.get("probabilities")
                        if isinstance(probs, dict) and probs:
                            labels = [(int(k), float(v)) for k, v in probs.items()]
                            top = max(v for _, v in labels); winners = [i for i, v in labels if v == top]
                            pred[name] = q["levels"][winners[0]]["label"] if len(winners) == 1 and winners[0] < len(q["levels"]) else None
                        else:
                            # Contract may provide scalar ordinal score rather than a distribution.
                            score = ans.get("score")
                            pred[name] = q["levels"][int(score)]["label"] if isinstance(score, (int, float)) and float(score).is_integer() and 0 <= int(score) < len(q["levels"]) else None
                    if pred[name] is None:
                        raise RuntimeError(f"JEV returned no unambiguous normalized outcome for {name}")
                jev_preds.append(pred)
                jev_record = {"raw_response": raw_jev, "normalized_typed_outcomes": typed,
                              "normalized_labels": {k: v for k, v in pred.items() if k != "id"},
                              "timing_ms": elapsed, "error": None}
            except Exception as exc:
                errors["jev"] += 1
                elapsed = (time.perf_counter_ns() - jev_started) / 1e6
                jev_preds.append({"id": case["id"], **{k: None for k in req["questions"]}})
                jev_record = {"raw_response": None, "normalized_typed_outcomes": None, "normalized_labels": None,
                              "timing_ms": elapsed, "error": f"{type(exc).__name__}: {str(exc).replace(key, '[REDACTED]')}"}
            raw_rows.append({"case_id": case["id"], "laya": laya_row, "jev_request_payload": payload,
                             "jev": jev_record})
            # Raw predictions flush incrementally; references are intentionally absent here.
            write_jsonl(outdir / "raw-results.jsonl", raw_rows)
            write_jsonl(outdir / "laya-predictions.jsonl", laya_preds)
            write_jsonl(outdir / "jev-predictions.jsonl", jev_preds)
            print(f"[{index}/60] {case['id']}: Laya={'ok' if not any(v['error'] for v in laya_row['decisions'].values()) else 'error'}, JEV={'ok' if not jev_record['error'] else 'error'}", flush=True)

        # Score only once both complete prediction files have been durably written.
        timing_rows = []
        for row in raw_rows:
            timing_rows.append({"case_id": row["case_id"],
                "laya_case_inference_ms_sum": row["laya"]["case_inference_ms_sum"],
                "laya_questions": {name: {"inference_ms": item["inference_ms"], "error": item["error"]}
                                   for name, item in row["laya"]["decisions"].items()},
                "jev_request_ms": row["jev"]["timing_ms"], "jev_error": row["jev"]["error"]})
        write_jsonl(outdir / "timings.jsonl", timing_rows)
        paired_flips = None
        prompt_baseline_predictions: dict[str, list[dict[str, Any]]] | None = None
        prompt_baseline_provenance = None
        if prompt_fresh:
            # Use only the frozen all-MLX v2-baseline shared-head predictions.
            # This read-only comparison happens after this run's prediction files are saved.
            holdout_path = PROMPT_STUDY_RUN / "holdout-metrics.json"
            freeze_path = PROMPT_STUDY_RUN / "PRE_REFERENCE_FREEZE.json"
            parity_path = PROMPT_STUDY_METAL / "parity-timing-report.json"
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
            study_run_metadata = json.loads((PROMPT_STUDY_RUN / "run-metadata.json").read_text(encoding="utf-8"))
            frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            if (holdout.get("fixture_sha256") != sha(args.fixture) or
                frozen.get("fixture_sha256") != sha(args.fixture) or
                parity.get("fixture_sha256") != sha(args.fixture)):
                raise ValueError("frozen prompt-study baseline artifacts do not match the requested fresh60 fixture hash")
            prompt_baseline_predictions = {}
            prompt_baseline_provenance = {
                "run": str(PROMPT_STUDY_RUN), "fixture_sha256": sha(args.fixture),
                "head_mode": "shared-v2-baseline", "selected_variant": "v2-baseline",
                "train_only_selection_retained": True, "per_variant_head_results_used": False,
                "holdout_metric_stage": holdout.get("metrics_stage"),
                "pre_reference_freeze_sha256": sha(freeze_path), "parity_timing_report_sha256": sha(parity_path),
                "models": {},
                "comparison_note": "Read-only fresh60 baseline predictions; shared v2 head only. Other prompt variants/per-variant heads excluded from the primary comparison.",
            }
            expected_baseline = {"qwen": {"choice": 29, "noul": 49, "score": 12, "all_three": 5},
                                 "harrier": {"choice": 31, "noul": 57, "score": 13, "all_three": 8}}
            for model_name in ("qwen", "harrier"):
                key_name = f"{model_name}/v2-baseline"
                prediction_path = PROMPT_STUDY_METAL / model_name / "v2-baseline" / "predictions.jsonl"
                cpu_reference_path = PROMPT_STUDY_RUN / "predictions" / model_name / "v2-baseline.jsonl"
                wanted_cpu_hash = frozen["prediction_hashes"][key_name]
                model_parity = parity["models"][model_name]["v2-baseline"]
                if (sha(cpu_reference_path) != wanted_cpu_hash or
                    model_parity["hashes"]["cpu_reference_predictions"] != sha(cpu_reference_path) or
                    model_parity["hashes"]["gpu_predictions"] != sha(prediction_path)):
                    raise ValueError(f"{model_name} v2-baseline all-MLX prediction hash mismatch")
                if model_parity.get("head_mode") != "shared-v2-baseline" or model_parity.get("all_three_cases_exact") != 60:
                    raise ValueError(f"{model_name} v2-baseline is not the frozen shared-head exact-parity run")
                if not all(v.get("selected_exact") and (v.get("score_argmax_exact") is not False)
                           for v in model_parity["typed_parity"].values()):
                    raise ValueError(f"{model_name} v2-baseline typed parity did not pass")
                frozen_rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(frozen_rows) != 180:
                    raise ValueError(f"{model_name} frozen v2-baseline has {len(frozen_rows)} prediction rows, expected 180")
                normalized = {str(case["id"]): {} for case in cases}
                for row in frozen_rows:
                    case_id, name, kind = str(row["id"]), row["question_name"], row["question_type"]
                    if case_id not in normalized or name not in cases[0]["request"]["questions"]:
                        raise ValueError(f"{model_name} frozen prediction ID/question is outside this fixture")
                    expected_kind = cases[0]["request"]["questions"][name]["type"]
                    if kind != expected_kind or name in normalized[case_id]:
                        raise ValueError(f"{model_name} frozen prediction typed question mismatch/duplicate")
                    normalized[case_id][name] = (row.get("selected") if kind == "choice" else
                        row.get("value") if kind == "noul" else row.get("selected_level"))
                if any(set(v) != set(cases[0]["request"]["questions"]) for v in normalized.values()):
                    raise ValueError(f"{model_name} frozen v2-baseline missing typed predictions")
                prompt_baseline_predictions[model_name] = [{"id": str(c["id"]), **normalized[str(c["id"])]} for c in cases]
                prompt_baseline_provenance["models"][model_name] = {
                    "model_id": {"qwen": "Qwen/Qwen3-Embedding-0.6B",
                                 "harrier": "microsoft/harrier-oss-v1-0.6b"}[model_name],
                    "C": model_parity["C"], "C_description": "Qwen C=1.0; Harrier C=0.1 from frozen shared-head parity report",
                    "all_mlx_prediction_file": str(prediction_path), "all_mlx_prediction_sha256": sha(prediction_path),
                    "cpu_reference_prediction_file": str(cpu_reference_path), "cpu_reference_prediction_sha256": sha(cpu_reference_path),
                    "cpu_reference_prediction_hash_matches_pre_reference_freeze": True,
                    "all_180_typed_outputs_exact_cpu_reference_parity": model_parity["all_three_cases_exact"] == 60 and all(
                        v["selected_exact"] and (v["score_argmax_exact"] is not False) for v in model_parity["typed_parity"].values()),
                    "frozen_model_hash": study_run_metadata["models"][model_name]["final_fit_sha256"],
                }
            (outdir / "prompt-study-baseline-provenance.json").write_text(
                json.dumps(prompt_baseline_provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if explicit_intervention:
            # Integrity/target equivalence is checked only after all inference outputs
            # have been persisted; the evaluation metadata is never passed to either model.
            import subprocess
            validation_path = ROOT / "benchmarks/cases/varied/multi/validate_explicit.py"
            validation = subprocess.run([sys.executable, str(validation_path)], check=True,
                capture_output=True, text=True)
            (outdir / "fixture-integrity-validation.txt").write_text(validation.stdout, encoding="utf-8")
            manifest = json.loads(fixture_manifest.read_text(encoding="utf-8"))
            base_laya_path = MATCHED_BASELINE_RUN / "laya-predictions.jsonl"
            base_jev_path = MATCHED_BASELINE_RUN / "jev-predictions.jsonl"
            if not base_laya_path.is_file() or not base_jev_path.is_file():
                raise FileNotFoundError("paired matched baseline prediction files are missing")
            base_laya = [json.loads(line) for line in base_laya_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            base_jev = [json.loads(line) for line in base_jev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            baseline_meta = json.loads((MATCHED_BASELINE_RUN / "run-metadata.json").read_text(encoding="utf-8"))
            source_ids = manifest["source_case_ids"]
            transformed_ids = manifest["transformed_case_ids"]
            if len(source_ids) != 60 or transformed_ids != [str(c["id"]) for c in cases]:
                raise ValueError("explicit manifest source/target case-ID mapping does not align")
            if baseline_meta.get("fixture_sha256") != manifest.get("source_sha256") or baseline_meta.get("case_count") != 60:
                raise ValueError("paired baseline run does not reference the explicit manifest's frozen source fixture")
            if [str(x["id"]) for x in base_laya] != source_ids or [str(x["id"]) for x in base_jev] != source_ids:
                raise ValueError("matched baseline outputs do not align with explicit source IDs")
            base_laya_by_id = {str(x["id"]): x for x in base_laya}
            base_jev_by_id = {str(x["id"]): x for x in base_jev}
            flip_rows = []
            paired_flips = {}
            for model_key, baseline_by_id, explicit_preds in (
                ("laya_mlx", base_laya_by_id, laya_preds), ("jev", base_jev_by_id, jev_preds)):
                by_type: dict[str, Any] = {}
                old_all = new_all = correct_to_wrong = wrong_to_correct = tuple_changed = 0
                for qname, q in cases[0]["request"]["questions"].items():
                    changed = same = old_good = new_good = oldgood_newbad = oldbad_newgood = 0
                    for index, (old_id, new_id) in enumerate(zip(source_ids, transformed_ids)):
                        old_pred = baseline_by_id[old_id].get(qname)
                        new_pred = explicit_preds[index].get(qname)
                        old_ok = old_pred == reference_label(cases[index]["request"]["questions"][qname], cases[index]["reference"][qname])
                        new_ok = new_pred == reference_label(cases[index]["request"]["questions"][qname], cases[index]["reference"][qname])
                        did_change = old_pred != new_pred
                        changed += did_change; same += not did_change
                        old_good += old_ok; new_good += new_ok
                        oldgood_newbad += old_ok and not new_ok; oldbad_newgood += not old_ok and new_ok
                        flip_rows.append({"model": model_key, "source_case_id": old_id, "explicit_case_id": new_id,
                            "question": qname, "baseline_prediction": old_pred, "explicit_prediction": new_pred,
                            "prediction_changed": did_change, "baseline_correct": old_ok, "explicit_correct": new_ok})
                    by_type[qname] = {"prediction_flips": changed, "unchanged": same,
                        "baseline_correct": old_good, "explicit_correct": new_good,
                        "correct_to_incorrect": oldgood_newbad, "incorrect_to_correct": oldbad_newgood,
                        "accuracy_delta_cases": new_good - old_good}
                for index, (old_id, new_id) in enumerate(zip(source_ids, transformed_ids)):
                    old = baseline_by_id[old_id]
                    new = explicit_preds[index]
                    names = list(cases[index]["request"]["questions"])
                    old_correct = all(old.get(name) == reference_label(cases[index]["request"]["questions"][name], cases[index]["reference"][name]) for name in names)
                    new_correct = all(new.get(name) == reference_label(cases[index]["request"]["questions"][name], cases[index]["reference"][name]) for name in names)
                    old_all += old_correct; new_all += new_correct
                    correct_to_wrong += old_correct and not new_correct
                    wrong_to_correct += not old_correct and new_correct
                    tuple_changed += any(old.get(name) != new.get(name) for name in names)
                paired_flips[model_key] = {"per_question": by_type,
                    "all_three_correct": {"baseline": old_all, "explicit": new_all,
                        "correct_to_incorrect": correct_to_wrong, "incorrect_to_correct": wrong_to_correct,
                        "net_change_cases": new_all - old_all},
                    "cases_with_any_prediction_flip": tuple_changed, "total_cases": 60}
            write_jsonl(outdir / "paired-flips.jsonl", flip_rows)
            paired_flips["integrity"] = {"same_questions_and_targets": True,
                "source_fixture_sha256": manifest["source_sha256"],
                "paired_baseline_run": MATCHED_BASELINE_RUN.name,
                "paired_baseline_laya_predictions_sha256": sha(base_laya_path),
                "paired_baseline_jev_predictions_sha256": sha(base_jev_path),
                "evaluation_metadata_passed_to_models": False,
                "validation": "benchmarks/cases/varied/multi/validate_explicit.py passed after predictions were saved"}
        def metrics(preds: list[dict[str, Any]], indexes: list[int]) -> dict[str, Any]:
            per: dict[str, Any] = {}
            # The three fixture question names can differ between independent holdouts;
            # report by typed protocol rather than relying on task-name labels.
            names_by_type = {q["type"]: name for name, q in cases[0]["request"]["questions"].items()}
            for kind, name in names_by_type.items():
                correct = sum(preds[i].get(name) == reference_label(
                    cases[i]["request"]["questions"][name], cases[i]["reference"][name]) for i in indexes)
                per[kind] = {"correct": correct, "total": len(indexes), "accuracy": correct / len(indexes) if indexes else None}
            allgood = sum(all(preds[i].get(name) == reference_label(
                cases[i]["request"]["questions"][name], cases[i]["reference"][name]) for name in names_by_type.values())
                for i in indexes)
            return {"per_question_type": per, "all_three_correct": {
                "correct": allgood, "total": len(indexes), "accuracy": allgood / len(indexes) if indexes else None}}

        domains = sorted({case.get("domain", "unknown") for case in cases})
        by_domain = {domain: [i for i, case in enumerate(cases) if case.get("domain", "unknown") == domain]
                     for domain in domains}
        option_counts = sorted({len(case["request"]["questions"][next(
            name for name, q in case["request"]["questions"].items() if q["type"] == "choice")]["options"])
                                for case in cases})
        by_option_count = {str(count): [i for i, case in enumerate(cases)
                            if len(case["request"]["questions"][next(
                                name for name, q in case["request"]["questions"].items() if q["type"] == "choice")]["options"]) == count]
                           for count in option_counts}
        def percentile(values: list[float], p: float) -> float:
            values = sorted(values); pos = (len(values) - 1) * p
            low, high = math.floor(pos), math.ceil(pos)
            return values[low] if low == high else values[low] * (high - pos) + values[high] * (pos - low)
        laya_case_ms = [row["laya"]["case_inference_ms_sum"] for row in raw_rows]
        jev_case_ms = [row["jev"]["timing_ms"] for row in raw_rows]
        all_indexes = list(range(len(cases)))
        prompt_baseline_metrics = ({name: metrics(preds, all_indexes)
                                    for name, preds in prompt_baseline_predictions.items()}
                                   if prompt_baseline_predictions is not None else None)
        if prompt_baseline_metrics is not None:
            expected_baseline = {"qwen": {"choice": 29, "noul": 49, "score": 12, "all_three": 5},
                                 "harrier": {"choice": 31, "noul": 57, "score": 13, "all_three": 8}}
            for model_name, expected in expected_baseline.items():
                got = prompt_baseline_metrics[model_name]
                actual = {k: got["per_question_type"][k]["correct"] for k in ("choice", "noul", "score")}
                actual["all_three"] = got["all_three_correct"]["correct"]
                if actual != expected:
                    raise ValueError(f"{model_name} frozen v2-baseline reference scores differ from the frozen published counts")
        domain_result = {d: {"case_count": len(idx), "laya_mlx": metrics(laya_preds, idx), "jev": metrics(jev_preds, idx)}
                         for d, idx in by_domain.items()}
        option_result = {n: {"case_count": len(idx), "laya_mlx": metrics(laya_preds, idx), "jev": metrics(jev_preds, idx)}
                         for n, idx in by_option_count.items()}
        if prompt_baseline_predictions is not None:
            for d, idx in by_domain.items():
                for name, preds in prompt_baseline_predictions.items():
                    domain_result[d][name] = metrics(preds, idx)
            for n, idx in by_option_count.items():
                for name, preds in prompt_baseline_predictions.items():
                    option_result[n][name] = metrics(preds, idx)
        scored = {"laya_mlx": metrics(laya_preds, list(range(len(cases))),),
                  "jev": metrics(jev_preds, list(range(len(cases))),),
                  "prompt_study_frozen_baselines": prompt_baseline_metrics,
                  "prompt_study_baseline_provenance": prompt_baseline_provenance,
                  "per_domain": domain_result,
                  "by_choice_option_count": option_result,
                  "scored_after_prediction_files": True,
                  "timings": {"laya_decision_ms": {name: {"n": len(vals), "mean": sum(vals) / len(vals),
                                  "min": min(vals), "max": max(vals)} for name in cases[0]["request"]["questions"]
                                  for vals in [[row["laya"]["decisions"][name]["inference_ms"] for row in raw_rows]]},
                               "jev_request_ms": {"n": len(raw_rows), "mean": sum(r["jev"]["timing_ms"] for r in raw_rows) / len(raw_rows),
                                                  "min": min(r["jev"]["timing_ms"] for r in raw_rows),
                                                  "max": max(r["jev"]["timing_ms"] for r in raw_rows)},
                              "case_scoped_latency_ms": {
                                  "laya_three_calls_per_case": {"n": len(laya_case_ms), "p50": percentile(laya_case_ms, .50), "p95": percentile(laya_case_ms, .95)},
                                  "jev_single_request_per_case": {"n": len(jev_case_ms), "p50": percentile(jev_case_ms, .50), "p95": percentile(jev_case_ms, .95)} }},
                   "comparison_context": ({"paired_matched_baseline_run": MATCHED_BASELINE_RUN.name,
                                           "note": "Paired authored clarity intervention against the exact matched source cases; not an independent holdout or generalization claim."}
                                          if explicit_intervention else
                                          {"prior_combined_shift_run": "20260923T132108188977Z",
                                           "prior_combined_shift_accuracy": {
                                               "laya_mlx": {"choice": "29/60", "noul": "39/60", "score": "31/60", "all_three": "13/60"},
                                               "jev": {"choice": "26/60", "noul": "22/60", "score": "20/60", "all_three": "8/60"}},
                                           "note": "Different combined-shift fixture; descriptive only, not a controlled comparison."}),
                   "paired_flips_vs_matched_baseline": paired_flips,
                   "error_counts": errors, "total_errors": sum(errors.values())}
        write_json(outdir / "metrics.json", scored)
        write_json(outdir / "run-metadata.json", {**metadata, "finished_at": datetime.now(timezone.utc).isoformat(),
                   "completed_cases": len(raw_rows), "error_counts": errors})
        # Report is generated only after persisted raw predictions and scored metrics.
        title = "Explicit-evidence authored clarity intervention" if explicit_intervention else "Matched-domain / unseen-question-family holdout"
        lines = [f"# {title}", "",
                 f"Run: `{outdir.name}`", f"Fixture SHA-256: `{metadata['fixture_sha256']}`",
                 f"Pinned checkpoint SHA-256: `{metadata['checkpoint_weights_sha256']}`", "",
                 *( ["This is an authored clarity intervention on the same 60 matched cases, not an independent holdout. Added evidence facts are passed in state; evaluation metadata and references are excluded from model input."] if explicit_intervention else ["Three typed decisions per state; model input contains request state and questions only."]),
                 "", "## Overall accuracy", "",
                 "| Model | Choice | Noul | Score | All three |", "|---|---:|---:|---:|---:|"]
        for model_key, label in (("laya_mlx", "Laya MLX"), ("jev", "JEV")):
            result = scored[model_key]
            vals = [result["per_question_type"][kind]["correct"] for kind in ("choice", "noul", "score")]
            lines.append(f"| {label} | {vals[0]}/60 ({vals[0]/60:.1%}) | {vals[1]}/60 ({vals[1]/60:.1%}) | {vals[2]}/60 ({vals[2]/60:.1%}) | {result['all_three_correct']['correct']}/60 ({result['all_three_correct']['accuracy']:.1%}) |")
        if prompt_baseline_metrics is not None:
            for model_name, label in (("qwen", "Qwen v2-baseline shared head"), ("harrier", "Harrier v2-baseline shared head")):
                result = prompt_baseline_metrics[model_name]
                vals = [result["per_question_type"][kind]["correct"] for kind in ("choice", "noul", "score")]
                lines.append(f"| {label} | {vals[0]}/60 ({vals[0]/60:.1%}) | {vals[1]}/60 ({vals[1]/60:.1%}) | {vals[2]}/60 ({vals[2]/60:.1%}) | {result['all_three_correct']['correct']}/60 ({result['all_three_correct']['accuracy']:.1%}) |")
        lines += ["", "Per-domain and choice-option-count outcomes are in `metrics.json`; per-call and per-case p50/p95 latency is in `timings.jsonl` and `metrics.json`.",
                  "Raw model/API responses and typed outcomes are saved before reference scoring. No prompt/model selection or reference-based inference was performed."]
        if explicit_intervention:
            lines += ["", "## Paired prediction and accuracy flips vs matched native baseline", "",
                      "| Model / question | Prediction flips | Baseline correct | Explicit correct | Correct→incorrect | Incorrect→correct |",
                      "|---|---:|---:|---:|---:|---:|"]
            paired = scored["paired_flips_vs_matched_baseline"]
            for model_key, label in (("laya_mlx", "Laya MLX"), ("jev", "JEV")):
                for qname, values in paired[model_key]["per_question"].items():
                    lines.append(f"| {label} / {qname} | {values['prediction_flips']}/60 | {values['baseline_correct']}/60 | {values['explicit_correct']}/60 | {values['correct_to_incorrect']} | {values['incorrect_to_correct']} |")
                all3 = paired[model_key]["all_three_correct"]
                lines.append(f"| {label} / all three | — | {all3['baseline']}/60 | {all3['explicit']}/60 | {all3['correct_to_incorrect']} | {all3['incorrect_to_correct']} |")
            lines += ["", "This is an authored, label-informed clarity intervention, not an independent holdout and not evidence for a generalization-gain claim."]
            latency = scored["timings"]["case_scoped_latency_ms"]
            lines += [f"", f"Case latency p50/p95 (ms): Laya 3-call sum {latency['laya_three_calls_per_case']['p50']:.1f}/{latency['laya_three_calls_per_case']['p95']:.1f}; JEV single request {latency['jev_single_request_per_case']['p50']:.1f}/{latency['jev_single_request_per_case']['p95']:.1f}."]
        else:
            lines += ["", "The prior combined-shift fixture results in `metrics.json` are descriptive only, since that is a different test fixture."]
        if prompt_fresh:
            timing = scored["timings"]["case_scoped_latency_ms"]
            lines += ["", "Prompt-study baselines are read-only all-MLX predictions with exact CPU prediction parity: Qwen shared v2-baseline C=1.0 and Harrier shared v2-baseline C=0.1. Train-only selection remains the frozen baseline; no per-variant heads or other prompt variants are included in the primary comparison.",
                      f"Case p50/p95 ms: Laya three-call sum {timing['laya_three_calls_per_case']['p50']:.1f}/{timing['laya_three_calls_per_case']['p95']:.1f}; JEV one-request roundtrip {timing['jev_single_request_per_case']['p50']:.1f}/{timing['jev_single_request_per_case']['p95']:.1f}."]
        lines += [f"Errors: Laya {errors['laya']}, JEV {errors['jev']}.", ""]
        (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(outdir / "checksums.json", {str(f.relative_to(outdir)): sha(f) for f in outdir.rglob("*") if f.is_file() and f.name != "checksums.json"})
        print(f"Saved full run to {outdir}; errors={sum(errors.values())}")
        return 0 if len(raw_rows) == 60 and sum(errors.values()) == 0 else 2
    except Exception as exc:
        message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
        print(f"runner failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
