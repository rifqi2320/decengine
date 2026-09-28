#!/usr/bin/env python3
"""Offline same-60 comparison of independently frozen model prediction bundles.

Each bundle directory supplies predictions.jsonl and run-metadata.json. This tool
does no inference/network access; all bundles must attest the exact sealed fixture hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-lora-fresh-60.jsonl"
FIXTURE_SHA256 = "2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--bundle", action="append", required=True,
                   help="NAME=run-directory; include LoRA, full-660 baseline, same-460 head-only, pinned Laya MLX, and JEV for both models")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if sha(args.fixture) != FIXTURE_SHA256:
        raise ValueError("fixture bytes do not match the pinned sealed test-lora-fresh-60 hash")
    cases = read_rows(args.fixture)
    ids = [str(row["id"]) for row in cases]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("expected exactly 60 unique fixture IDs")
    models: dict[str, dict[str, Any]] = {}
    timing: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    expected_values: dict[str, dict[tuple[str, str], float | None]] = {}
    for item in args.bundle:
        if "=" not in item:
            raise ValueError("--bundle must be NAME=run-directory")
        name, raw_dir = item.split("=", 1); directory = Path(raw_dir)
        if not name or name in models:
            raise ValueError(f"duplicate/empty model name: {name}")
        prediction_path, metadata_path = directory / "predictions.jsonl", directory / "run-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fixture_sha256") != FIXTURE_SHA256:
            raise ValueError(f"{name}: run metadata does not attest the exact same fixture hash")
        if metadata.get("predictions_sha256") != sha(prediction_path):
            raise ValueError(f"{name}: metadata prediction hash does not match predictions.jsonl")
        expected_model = ("Qwen/Qwen3-Embedding-0.6B" if name.startswith("qwen_") else
                          "microsoft/harrier-oss-v1-0.6b" if name.startswith("harrier_") else None)
        if expected_model and metadata.get("model") != expected_model:
            raise ValueError(f"{name}: model identity mismatch in run metadata")
        if name.endswith("headonly_same460") and (metadata.get("training_question_count") != 460 or
                metadata.get("C") != 1.0 or metadata.get("dev_labels_used_for_fit") is not False or
                metadata.get("holdout_labels_used_for_fit") is not False):
            raise ValueError(f"{name}: same-460 comparator does not attest exactly the frozen train-only C=1 fit")
        if name.endswith("baseline_full660") and "full 660" not in str(metadata.get("baseline_protocol", "")):
            raise ValueError(f"{name}: historical full-660 baseline is not explicitly distinguished from same-460")
        if name.endswith("baseline_full660"):
            expected_c = 1.0 if name.startswith("qwen_") else 0.1
            if metadata.get("c_value") != expected_c or metadata.get("selected_variant") != "baseline":
                raise ValueError(f"{name}: full-660 frozen baseline C/variant differs from train-only selection")
        rows = read_rows(prediction_path)
        names = set(cases[0]["request"]["questions"])
        kinds = {qname: q["type"] for qname, q in cases[0]["request"]["questions"].items()}
        if len(rows) != 180:
            raise ValueError(f"{name}: expected 180 normalized per-question predictions; found {len(rows)}")
        converted: dict[str, dict[str, Any]] = {case_id: {"id": case_id} for case_id in ids}
        seen: set[tuple[str, str]] = set()
        expected_values[name] = {}
        for row in rows:
            case_id, question = str(row.get("id")), str(row.get("question_name"))
            pair = (case_id, question)
            if case_id not in converted or question not in names or pair in seen:
                raise ValueError(f"{name}: unknown/duplicate case-question key {pair}")
            if row.get("status") not in {"ok", "success"} or row.get("question_type") != kinds[question]:
                raise ValueError(f"{name}/{case_id}/{question}: status/type mismatch")
            value = row.get("selected")
            if value is None or (kinds[question] == "noul" and type(value) is not bool) or (kinds[question] != "noul" and not isinstance(value, str)):
                raise ValueError(f"{name}/{case_id}/{question}: invalid normalized typed outcome")
            converted[case_id][question] = value
            expected_values[name][pair] = row.get("expected_value")
            seen.add(pair)
        expected_pairs = {(case_id, question) for case_id in ids for question in names}
        if seen != expected_pairs:
            raise ValueError(f"{name}: normalized prediction IDs/questions do not cover the same exact 60x3 matrix")
        models[name] = converted
        timing[name] = metadata.get("timing_scopes", metadata.get("timings", "not provided"))
        provenance[name] = {"run_dir": str(directory.resolve()), "predictions_sha256": sha(prediction_path),
                            "metadata_sha256": sha(metadata_path), "runtime": metadata.get("runtime"),
                            "device": metadata.get("device_actual", metadata.get("device")),
                            "dtype": metadata.get("dtype"), "compute_assumptions": metadata.get("compute_assumptions", "not provided"),
                            "resource_guard": metadata.get("resource_guard", "not provided"),
                            "baseline_protocol": metadata.get("baseline_protocol"),
                            "training_question_count": metadata.get("training_question_count"),
                            "inference_policy_id": metadata.get("inference_policy_id"),
                            "inference_policy_sha256": metadata.get("inference_policy_sha256"),
                            "unseen_length_extrapolation": metadata.get("unseen_length_extrapolation")}
    required = {"qwen_lora", "harrier_lora", "qwen_frozen_baseline_full660", "harrier_frozen_baseline_full660",
                "qwen_headonly_same460", "harrier_headonly_same460"}
    if not required.issubset(models):
        raise ValueError(f"missing required same-fixture bundles: {sorted(required - set(models))}")
    questions = list(cases[0]["request"]["questions"])
    kinds = {name: q["type"] for name, q in cases[0]["request"]["questions"].items()}
    correct: dict[str, dict[str, int]] = {model: {kind: 0 for kind in ("choice", "noul", "score")} for model in models}
    all3 = {model: 0 for model in models}
    score_errors: dict[str, list[float]] = {model: [] for model in models}
    ordinal_errors: dict[str, list[float]] = {model: [] for model in models}
    domains: dict[str, dict[str, dict[str, int]]] = {}
    for case in cases:
        case_id = str(case["id"]); domain = str(case["domain"])
        domains.setdefault(domain, {m: {"all_three": 0, **{k: 0 for k in ("choice", "noul", "score")}} for m in models})
        for model, indexed in models.items():
            row = indexed[case_id]; case_good = True
            for question in questions:
                kind = kinds[question]
                target = case["reference"][question]["target"]
                gold = target["selected"] if kind == "choice" else target["value"] if kind == "noul" else target["label"]
                hit = row.get(question) == gold
                correct[model][kind] += int(hit); domains[domain][model][kind] += int(hit)
                if kind == "score":
                    question_wire = case["request"]["questions"][question]
                    levels = question_wire["levels"]
                    labels = [str(level["label"]) for level in levels]
                    target_values = {str(level["label"]): float(level["value"]) for level in levels}
                    expected_value = expected_values[model][(case_id, question)]
                    if expected_value is None:
                        raise ValueError(f"{model}/{case_id}/{question}: missing score expected_value")
                    score_errors[model].append(abs(float(expected_value) - target_values[str(gold)]))
                    predicted = row.get(question)
                    ordinal_errors[model].append(float(abs(labels.index(predicted) - labels.index(str(gold)))
                                                       if predicted in labels else len(labels) - 1))
                case_good &= hit
            all3[model] += int(case_good); domains[domain][model]["all_three"] += int(case_good)
    overall = {}
    for model in models:
        overall[model] = {**{k: {"correct": correct[model][k], "total": 60, "accuracy": correct[model][k] / 60}
                              for k in ("choice", "noul", "score")},
                          "all_three": {"correct": all3[model], "total": 60, "accuracy": all3[model] / 60},
                          "score_expected_value_mae": sum(score_errors[model]) / len(score_errors[model]),
                          "score_ordinal_level_mae": sum(ordinal_errors[model]) / len(ordinal_errors[model])}
    result = {"fixture_sha256": FIXTURE_SHA256, "case_count": 60,
        "overall": overall,
        "per_domain_counts": domains, "timing_scopes_and_compute_assumptions": {"timings": timing, "provenance": provenance},
        "scoring": "strict exact typed label; score expected-value MAE compares probability-weighted rubric values to reference numeric value; ordinal level MAE uses ordered rubric labels",
        "optional_bundles_not_supplied": sorted({"laya_mlx", "jev"} - set(models)),
        "network_or_api_calls": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Frozen model comparison: LoRA fresh-60", "", f"Fixture SHA-256: `{FIXTURE_SHA256}`", "",
             "| Model | Choice | Noul | Score | All three | Score expected-value MAE | Score ordinal MAE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model, values in result["overall"].items():
        fmt = lambda x: f"{x['correct']}/60 ({x['accuracy']:.1%})"
        lines.append(f"| {model} | {fmt(values['choice'])} | {fmt(values['noul'])} | {fmt(values['score'])} | {fmt(values['all_three'])} | {values['score_expected_value_mae']:.3f} | {values['score_ordinal_level_mae']:.3f} |")
    lines += ["", "Latency comparisons are only meaningful with the recorded timing scopes below; do not conflate model-only forward time, per-question/case time, or end-to-end startup/load time.", "",
              "```json", json.dumps(result["timing_scopes_and_compute_assumptions"], indent=2, ensure_ascii=False), "```", "",
              "Compute assumptions are copied from each frozen run's metadata. Missing assumptions/timing scopes are explicitly reported as not provided; no cross-device FLOPs or latency normalization is inferred.", ""]
    args.out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
