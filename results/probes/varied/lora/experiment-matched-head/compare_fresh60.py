#!/usr/bin/env python3
"""Score the frozen corrected matched-head fresh60 outputs with reused same-fixture controls.

All prediction bundles are verified/frozen before this script opens fixture references.
No inference, API calls, model loads, fitting, or tuning occurs here.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-lora-fresh-60.jsonl"
FIXTURE_SHA256 = "2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629"
BASE = ROOT / "results/probes/varied/lora/experiment-matched-head"
FREEZE = BASE / "TRAIN-ONLY-FREEZE.json"
FREEZE_SHA256 = "ecb46f3a952d48a78a9004ad0213e70e57da0e9bd84e183fda31a5b63186eb1f"
POLICIES = {
    "qwen_head_epoch2_matched": (BASE / "MATCHED-INFERENCE-POLICY-512-TOTAL10G-qwen-head-attempt.json",
                                  "1a9414f4647570f1f06de9e9e6ee8710caa0e9a4aff63fbbcfc14c2042b7ab11"),
    "qwen_lora_epoch9_matched": (BASE / "MATCHED-INFERENCE-POLICY-512-TOTAL10G.json",
                                  "a641c64ac7f8fda46d587422f0fe20b94fcd1353246eb9f0c3cd1051313177f7"),
    "harrier_head_epoch9_matched": (BASE / "MATCHED-INFERENCE-POLICY-512-TOTAL10G-FINAL.json",
                                     "57455ae8d88322ea7703d4d67b1ee317ad198b57071b35ebc57be29a01f1a590"),
    "harrier_lora_epoch9_matched": (BASE / "MATCHED-INFERENCE-POLICY-512-TOTAL10G-FINAL.json",
                                     "57455ae8d88322ea7703d4d67b1ee317ad198b57071b35ebc57be29a01f1a590"),
}
SYSTEMS = {
    "qwen_head_epoch2_matched": {"format": "flat", "dir": BASE / "eval/fresh60/qwen-head-epoch02-total10g",
        "model": "Qwen/Qwen3-Embedding-0.6B", "arm": "head", "epoch": 2,
        "checkpoint_sha256": "58bef9ac575faf62344d7e176e5c80910fff7370dd8314295cc2d9a23d0437b8"},
    "qwen_lora_epoch9_matched": {"format": "flat", "dir": BASE / "eval/fresh60/qwen-lora-epoch09-total10g",
        "model": "Qwen/Qwen3-Embedding-0.6B", "arm": "lora", "epoch": 9,
        "checkpoint_sha256": "cc6ceb9cd13819cd162ec494630660c159aa27e2e37cf5b82391b36ef5f6bf06"},
    "harrier_head_epoch9_matched": {"format": "flat", "dir": BASE / "eval/fresh60/harrier-head-epoch09-total10g",
        "model": "microsoft/harrier-oss-v1-0.6b", "arm": "head", "epoch": 9,
        "checkpoint_sha256": "92145ec36c4eeec1d89e70818313f57a758bd696b121ddcd20d8cd1153b59c28"},
    "harrier_lora_epoch9_matched": {"format": "flat", "dir": BASE / "eval/fresh60/harrier-lora-epoch09-total10g",
        "model": "microsoft/harrier-oss-v1-0.6b", "arm": "lora", "epoch": 9,
        "checkpoint_sha256": "06f1f13eb90fb84ce8bfac8a444fad7af2312e53c2c23745342256aa567bc10f"},
    "qwen_frozen_baseline_full660": {"format": "flat", "dir": ROOT / "results/probes/varied/lora/eval/fresh60/predictions/qwen-frozen-baseline-full660",
        "model": "Qwen/Qwen3-Embedding-0.6B", "baseline_protocol": "independent historical frozen scorer trained/selected on full 660 questions"},
    "harrier_frozen_baseline_full660": {"format": "flat", "dir": ROOT / "results/probes/varied/lora/eval/fresh60/predictions/harrier-frozen-baseline-full660",
        "model": "microsoft/harrier-oss-v1-0.6b", "baseline_protocol": "independent historical frozen scorer trained/selected on full 660 questions"},
    "qwen_same460_linear": {"format": "flat", "dir": ROOT / "results/probes/varied/lora/eval/fresh60/predictions/qwen-headonly-same460",
        "model": "Qwen/Qwen3-Embedding-0.6B", "baseline_protocol": "separate C=1 head-only linear comparator fitted on exact 460 training questions"},
    "harrier_same460_linear": {"format": "flat", "dir": ROOT / "results/probes/varied/lora/eval/fresh60/predictions/harrier-headonly-same460",
        "model": "microsoft/harrier-oss-v1-0.6b", "baseline_protocol": "separate C=1 head-only linear comparator fitted on exact 460 training questions"},
    "laya_archived": {"format": "case", "dir": ROOT / "results/laya/varied/runs/20260924T054309737898Z-cacheguard",
        "prediction_name": "laya-predictions.jsonl", "model": "convaiinnovations/laya"},
    "jev_archived": {"format": "case", "dir": ROOT / "results/jev/varied/runs/20260924T042126036050Z",
        "prediction_name": "jev-predictions.jsonl", "model": "jev-latest"},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values); pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def kinds_labels(request: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    types, labels = {}, {}
    for name, q in request["questions"].items():
        kind = q["type"]; types[name] = kind
        labels[name] = (list(q["options"]) if kind == "choice" else
                        ["unsupported", "supported"] if kind == "noul" else
                        [str(level["label"]) for level in q["levels"]])
    return types, labels


def selected_key(kind: str, value: Any) -> str:
    if kind != "noul":
        return str(value)
    if type(value) is bool:
        return "supported" if value else "unsupported"
    return str(value)


def main() -> int:
    if sha(FREEZE) != FREEZE_SHA256:
        raise ValueError("corrected train-only freeze hash mismatch")
    if sha(FIXTURE) != FIXTURE_SHA256:
        raise ValueError("fresh60 fixture hash mismatch")

    # Freeze/check every prediction bundle before opening fixture references.
    flat_rows: dict[str, list[dict[str, Any]]] = {}
    case_rows: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    ordered_ids = [str(json.loads(line)["id"]) for line in FIXTURE.read_text().splitlines() if line.strip()]
    if len(ordered_ids) != 60 or len(set(ordered_ids)) != 60:
        raise ValueError("fixture must have exactly 60 unique IDs")
    expected_pairs = None
    for system, spec in SYSTEMS.items():
        directory = spec["dir"]
        meta_path = directory / "run-metadata.json"
        prediction_path = directory / spec.get("prediction_name", "predictions.jsonl")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checksum_map = json.loads((directory / "checksums.json").read_text()) if (directory / "checksums.json").exists() else {}
        recorded_prediction_sha = meta.get("predictions_sha256", checksum_map.get(prediction_path.name))
        if meta.get("fixture_sha256") != FIXTURE_SHA256 or recorded_prediction_sha != sha(prediction_path):
            raise ValueError(f"{system}: fixture or prediction checksum mismatch")
        if meta.get("model", meta.get("jev_model")) != spec["model"]:
            raise ValueError(f"{system}: model identity mismatch")
        if system in POLICIES:
            policy_path, policy_hash = POLICIES[system]
            if sha(policy_path) != policy_hash or meta.get("matched_policy_sha256", meta.get("inference_policy_sha256")) != policy_hash:
                raise ValueError(f"{system}: policy hash/provenance mismatch")
            arm_spec = SYSTEMS[system]
            if meta.get("arm") != arm_spec["arm"] or meta.get("selected_epoch") != arm_spec["epoch"] or \
                    meta.get("best_checkpoint_sha256") != arm_spec["checkpoint_sha256"] or \
                    meta.get("freeze_manifest_sha256") != FREEZE_SHA256:
                raise ValueError(f"{system}: selected checkpoint/freeze provenance mismatch")
            if (meta.get("reference_access_during_inference", meta.get("reference_access_during_generation", False)) is not False or
                    meta.get("reference_scoring_performed", False) is not False):
                raise ValueError(f"{system}: evaluator metadata does not defer all fixture scoring")
            raw_path = directory / "raw-predictions.jsonl"
            if meta.get("raw_predictions_sha256") != sha(raw_path):
                raise ValueError(f"{system}: raw prediction hash mismatch")
            raw = read_rows(raw_path)
            if len(raw) != 180:
                raise ValueError(f"{system}: expected 180 request-only raw predictions")
        elif spec["format"] == "flat":
            if meta.get("predictions_sha256") != sha(prediction_path):
                raise ValueError(f"{system}: normalized output checksum mismatch")
        rows = read_rows(prediction_path)
        if spec["format"] == "flat":
            if len(rows) != 180:
                raise ValueError(f"{system}: expected 180 per-question rows")
            pairs = [(str(r["id"]), str(r["question_name"])) for r in rows]
            if len(set(pairs)) != 180:
                raise ValueError(f"{system}: duplicate case/question rows")
            if [str(r["id"]) for r in rows][::3] != ordered_ids:
                raise ValueError(f"{system}: case IDs are not in fixture order")
            flat_rows[system] = rows
        else:
            if len(rows) != 60 or [str(r["id"]) for r in rows] != ordered_ids:
                raise ValueError(f"{system}: archived 60 case rows/fixture order mismatch")
            case_rows[system] = rows
        metadata[system] = meta
        provenance[system] = {"run_dir": str(directory), "prediction_path": str(prediction_path),
            "predictions_sha256": sha(prediction_path), "metadata_sha256": sha(meta_path),
            "runtime": meta.get("runtime", meta.get("device_actual", "MLX/Metal" if system in POLICIES else None)),
            "policy_sha256": meta.get("matched_policy_sha256", meta.get("inference_policy_sha256")),
            "model_or_adapter_sha256": meta.get("best_checkpoint_sha256", meta.get("model_or_adapter_sha256")),
            "resource_summary": meta.get("resources", meta.get("guard_summary")),
            "baseline_protocol": spec.get("baseline_protocol"),
            "fixture_reused_previously_scored": True}

    # The request-only view validates candidate order/shape without opening labels.
    sys.path.insert(0, str(ROOT / "results/probes/varied/lora"))
    from evaluate_frozen import request_only_jsonl
    request_cases = list(request_only_jsonl(FIXTURE))
    request_by_id = {str(c["id"]): c for c in request_cases}
    if [str(c["id"]) for c in request_cases] != ordered_ids:
        raise ValueError("request-only fixture pass changed case ordering")
    candidate_structural: dict[str, Any] = {}
    indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    selected_values: dict[str, dict[tuple[str, str], Any]] = {}
    expected_values: dict[str, dict[tuple[str, str], float | None]] = {}
    for system, spec in SYSTEMS.items():
        indexed[system] = {}; selected_values[system] = {}; expected_values[system] = {}
        structural_ok = True
        if spec["format"] == "flat":
            for row in flat_rows[system]:
                cid, name = str(row["id"]), str(row["question_name"])
                q = request_by_id[cid]["request"]["questions"][name]
                names, candidate_labels = kinds_labels(request_by_id[cid]["request"])
                expected = candidate_labels[name]
                if row.get("question_type") != names[name] or row.get("candidate_keys") != expected:
                    structural_ok = False
                probabilities = row.get("probabilities")
                if probabilities is not None and (set(probabilities) != set(expected) or
                        any(not math.isfinite(float(v)) for v in probabilities.values()) or
                        abs(sum(float(v) for v in probabilities.values()) - 1.0) > 1e-5):
                    structural_ok = False
                key = (cid, name); indexed[system][key] = row
                selected_values[system][key] = row["selected"]
                expected_values[system][key] = row.get("expected_value")
        else:
            rows = case_rows[system]
            for case_row in rows:
                cid = str(case_row["id"])
                _, candidate_labels = kinds_labels(request_by_id[cid]["request"])
                for name, prediction in case_row.items():
                    if name == "id": continue
                    if name not in candidate_labels or selected_key(request_by_id[cid]["request"]["questions"][name]["type"], prediction) not in candidate_labels[name]:
                        structural_ok = False
                        continue
                    key = (cid, name)
                    indexed[system][key] = {"id": cid, "question_name": name,
                        "question_type": request_by_id[cid]["request"]["questions"][name]["type"],
                        "candidate_keys": candidate_labels[name], "selected": prediction,
                        "expected_value": None, "probabilities": None}
                    selected_values[system][key] = prediction; expected_values[system][key] = None
        candidate_structural[system] = {"all_rows_match_fixture_candidate_schema": structural_ok,
            "row_count": len(indexed[system]), "first_case_candidate_order_valid": structural_ok}
        if not structural_ok or len(indexed[system]) != 180:
            raise ValueError(f"{system}: structural parity/candidate schema check failed before reference scoring")
    expected_pair_set = {(cid, name) for cid in ordered_ids for name in request_by_id[cid]["request"]["questions"]}
    if any(set(values) != expected_pair_set for values in indexed.values()):
        raise ValueError("one or more systems do not cover the exact same 60x3 case/question matrix")

    # All ten output bundles are now frozen and structurally validated. Only now read references.
    fixture_cases = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    names, _ = kinds_labels(fixture_cases[0]["request"])
    kinds = names
    overall: dict[str, Any] = {}
    per_domain: dict[str, Any] = {}
    candidate_hist: dict[str, dict[str, dict[str, int]]] = {}
    for system in SYSTEMS:
        by_type = {kind: {"correct": 0, "total": 60} for kind in ("choice", "noul", "score")}
        exact_all = 0; ordinal_errors = []; expected_errors = []; selected_value_errors = []
        hist = {kind: Counter() for kind in ("choice", "noul", "score")}
        domain_counts: dict[str, Any] = {}
        for case in fixture_cases:
            cid = str(case["id"]); domain = str(case["domain"]); all_correct = True
            d = domain_counts.setdefault(domain, {kind: 0 for kind in ("choice", "noul", "score")} | {"all_three": 0, "case_count": 0})
            d["case_count"] += 1
            for name, question in case["request"]["questions"].items():
                kind = question["type"]; ref = case["reference"][name]["target"]
                gold = ref["selected"] if kind == "choice" else ref["value"] if kind == "noul" else ref["label"]
                key = (cid, str(name)); row = indexed[system][key]; prediction = selected_values[system][key]
                normalized_prediction = selected_key(kind, prediction)
                if normalized_prediction not in row["candidate_keys"]:
                    raise ValueError(f"{system}/{cid}/{name}: selected outcome absent from candidate list")
                correct = normalized_prediction == (selected_key(kind, gold) if kind == "noul" else str(gold))
                by_type[kind]["correct"] += int(correct); d[kind] += int(correct); all_correct &= correct
                hist[kind][str(row["candidate_keys"].index(normalized_prediction))] += 1
                if kind == "score":
                    levels = question["levels"]; labels = [str(x["label"]) for x in levels]
                    values = {str(x["label"]): float(x["value"]) for x in levels}
                    pred_label = str(normalized_prediction); gold_label = str(gold)
                    ordinal_errors.append(abs(labels.index(pred_label) - labels.index(gold_label)))
                    selected_value_errors.append(abs(values[pred_label] - values[gold_label]))
                    expected = expected_values[system][key]
                    if expected is not None:
                        expected_errors.append(abs(float(expected) - values[gold_label]))
            exact_all += int(all_correct); d["all_three"] += int(all_correct)
        for values in by_type.values():
            values["accuracy"] = values["correct"] / values["total"]
        overall[system] = {**by_type,
            "all_three": {"correct": exact_all, "total": 60, "accuracy": exact_all / 60},
            "score_expected_value_mae": (sum(expected_errors) / len(expected_errors) if len(expected_errors) == 60 else None),
            "score_selected_level_value_mae": sum(selected_value_errors) / len(selected_value_errors),
            "score_ordinal_level_mae": sum(ordinal_errors) / len(ordinal_errors)}
        per_domain[system] = domain_counts
        candidate_hist[system] = {kind: dict(counts) for kind, counts in hist.items()}

    timing: dict[str, Any] = {}
    for system, spec in SYSTEMS.items():
        meta = metadata[system]
        if system in POLICIES:
            raw = read_rows(spec["dir"] / "raw-predictions.jsonl")
            by_case: dict[str, list[float]] = defaultdict(list)
            for row in raw: by_case[str(row["id"])].append(float(row["inference_ms"]))
            case_ms = [sum(xs) for xs in by_case.values()]
            timing[system] = {"scope": "sum of three per-question timings; model load/preflight/reference excluded",
                "n": len(case_ms), "p50_ms": percentile(case_ms, .5), "p95_ms": percentile(case_ms, .95),
                "question_scope": meta.get("timing_scopes", {}).get("question_inference_ms")}
        elif system in {"qwen_frozen_baseline_full660", "harrier_frozen_baseline_full660", "qwen_same460_linear", "harrier_same460_linear"}:
            case = meta.get("timing_scopes", {}).get("case_total_ms", meta.get("timing_scopes", {}).get("three_question_case_sum_ms"))
            timing[system] = case or "not provided"
        elif system == "laya_archived":
            lat = meta.get("case_latency_ms")
            timing[system] = lat or json.loads((spec["dir"] / "metrics.json").read_text()).get("case_latency_ms", "not provided")
        else:
            time_rows = read_rows(spec["dir"] / "timings.jsonl")
            vals = [float(r["jev_single_request_ms"]) for r in time_rows]
            timing[system] = {"scope": "one JEV request containing the three named questions; archived scope",
                "n": len(vals), "p50_ms": percentile(vals, .5), "p95_ms": percentile(vals, .95)}

    result = {"status": "complete_same_fixture_scores_exploratory_reused_holdout",
        "fixture_sha256": FIXTURE_SHA256, "case_count": 60, "question_count": 180,
        "train_only_freeze_sha256": FREEZE_SHA256,
        "reused_seen_holdout_caveat": "This exact fixture was previously scored. New four-arm results are exploratory; no test-driven tuning or selection was performed.",
        "overall": overall, "per_domain": per_domain,
        "candidate_position_distribution": candidate_hist,
        "collapsed_candidate_position_types": {system: [kind for kind,hist in candidate_hist[system].items() if len(hist) == 1]
                                                  for system in candidate_hist},
        "first_question_structural_parity": candidate_structural,
        "timing_scopes": timing, "provenance": provenance,
        "reused_systems": {system: str(spec["dir"]) for system,spec in SYSTEMS.items() if system not in POLICIES},
        "excluded_prior_three_epoch_lora": True,
        "no_inference_or_api_calls_during_comparison": True}
    out = BASE / "eval/fresh60/FINAL-COMPARISON.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Corrected 9-epoch matched-head comparison on reused fresh60", "",
        f"Fixture SHA-256: `{FIXTURE_SHA256}`", "",
        "> **Exploratory / seen holdout:** this exact 60-case fixture was already scored previously. No tuning or selection used these outputs.", "",
        "| System | Choice | Noul | Score | All three | Score expected-value MAE | Score selected-level value MAE | Score ordinal MAE | Case p50/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for system, values in overall.items():
        pct = lambda x: f"{x['correct']}/60 ({x['accuracy']:.1%})"
        t = timing[system]
        if isinstance(t, dict) and "p50_ms" in t: latency = f"{t['p50_ms']:.1f} / {t['p95_ms']:.1f}"
        elif isinstance(t, dict) and "p50" in t: latency = f"{t['p50']:.1f} / {t['p95']:.1f}"
        else: latency = "n/a"
        expected = values["score_expected_value_mae"]
        lines.append(f"| {system} | {pct(values['choice'])} | {pct(values['noul'])} | {pct(values['score'])} | {pct(values['all_three'])} | {expected:.3f}" if expected is not None else
                     f"| {system} | {pct(values['choice'])} | {pct(values['noul'])} | {pct(values['score'])} | {pct(values['all_three'])} | n/a")
        lines[-1] = lines[-1].rstrip() + f" | {values['score_selected_level_value_mae']:.3f} | {values['score_ordinal_level_mae']:.3f} | {latency} |"
    lines += ["", "## Provenance and diagnostics", "", "```json", json.dumps({
        "provenance": provenance, "timing_scopes": timing,
        "candidate_position_distribution": candidate_hist,
        "collapsed_candidate_position_types": result["collapsed_candidate_position_types"],
        "first_question_structural_parity": candidate_structural}, indent=2, ensure_ascii=False), "```", "",
        "Historical full-660 baselines and same-460 linear comparators are separate systems; the old three-epoch LoRA runs are deliberately excluded. Laya and JEV outputs were reused without rerunning either system.", ""]
    (BASE / "eval/fresh60/FINAL-COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "comparison_json": str(out),
        "systems": list(overall), "sha256": sha(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
