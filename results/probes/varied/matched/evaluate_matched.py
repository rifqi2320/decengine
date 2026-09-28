#!/usr/bin/env python3
"""Evaluation-only runner for a train-selection-frozen matched multi-question set.

This script loads saved generic typed scorer parameters; it contains no training or
hyperparameter-selection code. A selection-freeze approval is required before opening
the requested fixture or feature vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FROZEN_RUNS = Path(__file__).resolve().parent
DEFAULT_FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-matched-60.jsonl"
TYPES = ("choice", "noul", "score")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("id", "")) for row in rows]
    if not rows or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: need nonempty rows with unique string IDs")
    return rows


def vectors(path: Path, expected_model: str) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    try:
        document = json.loads(raw)
        records = document.get("embeddings", []) if isinstance(document, dict) else document
        if not isinstance(records, list):
            records = [document]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    models = sorted({str(row.get("model")) for row in records if row.get("model")})
    matches = [model for model in models if model.casefold() == expected_model.casefold()]
    if models and len(matches) != 1:
        raise ValueError(f"{path}: embedding model mismatch/ambiguity: {models}")
    selected = matches[0] if matches else expected_model
    result = {}
    for row in records:
        if row.get("model", selected) != selected:
            continue
        name = str(row.get("question_key", ""))
        key = (str(row["id"]), name)
        if not name or key in result:
            raise ValueError(f"{path}: missing question_key or duplicate {key}")
        query = np.asarray(row.get("query_embedding"), dtype=np.float64)
        candidates = row.get("candidates")
        if isinstance(candidates, list):
            candidates = {str(item["candidate_id"]): item["embedding"] for item in candidates}
        if query.ndim != 1 or not np.isfinite(query).all() or not isinstance(candidates, dict):
            raise ValueError(f"{path}:{key}: invalid query/candidate vectors")
        parsed = {str(k): np.asarray(v, dtype=np.float64) for k, v in candidates.items()}
        if any(v.ndim != 1 or v.shape != query.shape or not np.isfinite(v).all() for v in parsed.values()):
            raise ValueError(f"{path}:{key}: candidate dimension/finiteness mismatch")
        result[key] = (query, parsed)
    return result, selected


def load_scorers(models_path: Path) -> tuple[dict[str, Any], str]:
    document = json.loads(models_path.read_text(encoding="utf-8"))
    typed = document.get("typed_scorers")
    if not isinstance(typed, dict) or set(typed) != set(TYPES):
        raise ValueError("saved model must contain exactly generic choice/noul/score scorers")
    scorers = {}
    for kind in TYPES:
        spec = typed[kind]
        w = np.asarray(spec["coefficients"], dtype=np.float64)
        mean = np.asarray(spec["feature_mean"], dtype=np.float64)
        scale = np.asarray(spec["feature_scale"], dtype=np.float64)
        if w.ndim != 1 or mean.shape != w.shape or scale.shape != w.shape or not all(np.isfinite(x).all() for x in (w, mean, scale)):
            raise ValueError(f"invalid saved {kind} scorer arrays")
        if np.any(scale <= 0):
            raise ValueError(f"saved {kind} feature scale must be positive")
        scorers[kind] = (w, mean, scale)
    return scorers, str(document["model"])


def check_freeze(freeze_path: Path, key: str, run_dir: Path, models_path: Path, metadata_path: Path) -> tuple[dict, dict]:
    """Fail closed unless this exact saved train-only run is explicitly approved."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("selection_status") != "frozen" or not freeze.get("selection_id"):
        raise ValueError("selection freeze must set selection_status='frozen' and a selection_id")
    approval = freeze.get("approved_runs", {}).get(key)
    if not isinstance(approval, dict):
        raise ValueError(f"selection freeze has no approved run entry for {key}")
    metadata_sha = sha(metadata_path)
    models_sha = sha(models_path)
    if Path(approval.get("run_dir", "")).resolve() != run_dir.resolve():
        raise ValueError("approved run directory does not match requested saved run")
    if approval.get("metadata_sha256") != metadata_sha or approval.get("models_sha256") != models_sha:
        raise ValueError("selection freeze hashes do not match saved metadata/models")
    if approval.get("selection_status") != "frozen":
        raise ValueError("selected variant is not marked frozen in approval entry")
    return freeze, approval


def question_candidates(question: dict[str, Any], case_id: str, qname: str) -> tuple[str, dict[str, str], list[float] | None]:
    kind = question.get("type")
    if kind == "choice":
        options = question.get("options")
        if not isinstance(options, dict):
            raise ValueError(f"{case_id}/{qname}: choice options must be an object")
        candidates = {str(k): str(v) for k, v in options.items()}
        values = None
    elif kind == "noul":
        # Stable semantic IDs only; proposition text is constructed by MLX exporter.
        candidates = {"unsupported": "not supported", "supported": "supported"}
        values = None
    elif kind == "score":
        levels = question.get("levels")
        if not isinstance(levels, list):
            raise ValueError(f"{case_id}/{qname}: score levels must be an array")
        candidates = {str(level["label"]): str(level["criterion"]) for level in levels}
        values = [float(level["value"]) for level in levels]
    else:
        raise ValueError(f"{case_id}/{qname}: fail-closed unknown question type {kind!r}")
    max_candidates = 6 if kind == "choice" else 5
    if not 2 <= len(candidates) <= max_candidates:
        raise ValueError(f"{case_id}/{qname}: {kind} candidate count outside 2..{max_candidates}")
    return kind, candidates, values


def pair_features(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.concatenate((query, candidate, query * candidate, np.abs(query - candidate)))


def materialize_inference(rows: list[dict], feature_vectors: dict) -> list[dict]:
    """Dereference request fields only. Deliberately never reads `reference`."""
    cases = []
    expected = set()
    for row in rows:
        request = row.get("request", {})
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError(f"{row['id']}: request.questions must be a nonempty map")
        for qname, question in questions.items():
            qname = str(qname)
            key = (str(row["id"]), qname)
            expected.add(key)
            if not isinstance(question, dict):
                raise ValueError(f"{row['id']}/{qname}: question must be an object")
            kind, options, values = question_candidates(question, str(row["id"]), qname)
            if key not in feature_vectors:
                raise ValueError(f"missing exported features for {key}")
            query, candidates = feature_vectors[key]
            if set(candidates) != set(options):
                raise ValueError(f"{key}: exported candidate IDs differ from question wire")
            cases.append({"id": str(row["id"]), "domain": str(row.get("domain", "")), "name": qname,
                          "type": kind, "keys": list(options), "values": values,
                          "x": np.stack([pair_features(query, candidates[k]) for k in options])})
    if expected != set(feature_vectors) or len(expected) != len(cases):
        raise ValueError("feature/question set mismatch (extra, missing, or duplicate named questions)")
    return cases


def predict(cases: list[dict], scorers: dict, elapsed: list[float]) -> list[dict]:
    predictions = []
    for case in cases:
        w, mean, scale = scorers[case["type"]]
        standardized = (case["x"] - mean) / scale
        t0 = time.perf_counter()
        logits = standardized @ w
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        elapsed.append(time.perf_counter() - t0)
        # Candidate-order parity: pointwise scores/probabilities only permute.
        reversed_logits = (standardized[::-1] @ w)[::-1]
        if not np.allclose(logits, reversed_logits - reversed_logits.max(), rtol=0, atol=1e-12):
            raise AssertionError("candidate permutation changed normalized logits")
        reverse = np.exp((standardized[::-1] @ w) - (standardized[::-1] @ w).max())
        reverse /= reverse.sum()
        if not np.allclose(probs, reverse[::-1], rtol=0, atol=1e-12):
            raise AssertionError("candidate permutation changed probabilities")
        selected_index = int(np.argmax(probs))
        selected = case["keys"][selected_index]
        predictions.append({"id": case["id"], "domain": case["domain"], "question_name": case["name"],
            "question_type": case["type"], "candidate_keys": case["keys"],
            "probabilities": {key: float(probs[i]) for i, key in enumerate(case["keys"])},
            "selected": selected, "label": (selected == "supported" if case["type"] == "noul" else selected),
            "value": (selected == "supported" if case["type"] == "noul" else None),
            "selected_level": selected if case["type"] == "score" else None,
            "confidence": float(probs[selected_index]),
            "expected_value": float(np.dot(probs, case["values"])) if case["values"] is not None else None})
    return predictions


def target_label(reference: dict, kind: str) -> str:
    target = reference["target"]
    if kind == "choice":
        return str(target["selected"])
    if kind == "noul":
        if type(target.get("value")) is not bool:
            raise ValueError("Noul target.value must be boolean")
        return "supported" if target["value"] else "unsupported"
    return str(target["label"])


def calibration(correct: list[bool], confidence: list[float], probabilities: list[list[float]], targets: list[int], bins: int = 10) -> dict:
    n = len(correct)
    if not n:
        return {"n": 0, "ece_10_bins": None, "brier_score": None, "mean_log_loss": None}
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inds = [i for i, c in enumerate(confidence) if lo <= c < hi or (b == bins - 1 and c == hi)]
        if inds:
            ece += len(inds) / n * abs(float(np.mean([confidence[i] for i in inds])) - float(np.mean([correct[i] for i in inds])))
    brier = []
    log_losses = []
    for i, row_probs in enumerate(probabilities):
        p = np.asarray(row_probs, dtype=np.float64)
        onehot = np.zeros_like(p)
        onehot[targets[i]] = 1.0
        brier.append(float(np.sum((p - onehot) ** 2)))
        log_losses.append(-math.log(max(float(p[targets[i]]), 1e-15)))
    return {"n": n, "ece_10_bins": float(ece), "brier_score": float(np.mean(brier)),
            "mean_log_loss": float(np.mean(log_losses)),
            "basis": "question-local candidate distributions; top-label ECE and multiclass Brier with variable candidate counts",
            "calibration_claim": "descriptive point estimate only; no calibration guarantee/interval claimed"}


def evaluate_after_prediction(rows: list[dict], cases: list[dict], predictions: list[dict], training_families: set[str]) -> tuple[dict, list[dict], list[dict]]:
    # First permitted reference access: caller has already durably written predictions.
    refs = []
    by_id = {str(row["id"]): row for row in rows}
    test_families = set()
    for case in cases:
        row = by_id[case["id"]]
        ref_map = row.get("reference", {})
        if set(row["request"]["questions"]) != set(ref_map):
            raise ValueError(f"{case['id']}: reference/question name mismatch")
        ref = ref_map[case["name"]]
        family = str(ref["question_family_id"])
        test_families.add(family)
        refs.append(ref)
    overlap = training_families & test_families
    if overlap:
        raise ValueError(f"matched evaluation family overlaps saved training families: {sorted(overlap)}")
    per_type, question_rows = {}, []
    domains: dict[str, list[int]] = {}
    by_case: dict[str, list[bool]] = {}
    for i, case in enumerate(cases):
        domains.setdefault(case["domain"], []).append(i)
        label = target_label(refs[i], case["type"])
        if label not in case["keys"]:
            raise ValueError(f"{case['id']}/{case['name']}: reference target does not identify a candidate")
        hit = case["keys"].index(label) == case["keys"].index(predictions[i]["selected"])
        by_case.setdefault(case["id"], []).append(hit)
        p = [predictions[i]["probabilities"][key] for key in case["keys"]]
        question_rows.append({"id": case["id"], "domain": case["domain"], "question_name": case["name"],
            "question_type": case["type"], "gold": label, "predicted": predictions[i]["selected"], "correct": hit,
            "gold_noul_value": bool(refs[i]["target"]["value"]) if case["type"] == "noul" else None,
            "predicted_noul_value": predictions[i]["value"] if case["type"] == "noul" else None,
            "score_gold_value": float(refs[i]["target"]["value"]) if case["type"] == "score" else None,
            "score_expected_value": predictions[i]["expected_value"] if case["type"] == "score" else None})
    for kind in TYPES:
        inds = [i for i, c in enumerate(cases) if c["type"] == kind]
        if not inds:
            continue
        correct = [question_rows[i]["correct"] for i in inds]
        targets = [cases[i]["keys"].index(target_label(refs[i], kind)) for i in inds]
        probs = [[predictions[i]["probabilities"][k] for k in cases[i]["keys"]] for i in inds]
        conf = [predictions[i]["confidence"] for i in inds]
        report = {"n": len(inds), "accuracy": float(np.mean(correct)),
                  "candidate_counts": dict(sorted({str(k): sum(len(cases[i]["keys"]) == k for i in inds) for k in {len(cases[i]["keys"]) for i in inds}}.items(), key=lambda x: int(x[0]))),
                  "calibration": calibration(correct, conf, probs, targets)}
        if kind == "score":
            actual = np.asarray([float(refs[i]["target"]["value"]) for i in inds])
            expected = np.asarray([predictions[i]["expected_value"] for i in inds])
            report["label_argmax_accuracy"] = float(np.mean(correct))
            report["expected_value_mae"] = float(np.mean(np.abs(actual - expected)))
        if kind == "noul":
            for i in inds:
                expected_id = "supported" if refs[i]["target"]["value"] else "unsupported"
                if expected_id not in cases[i]["keys"]:
                    raise ValueError(f"{cases[i]['id']}/{cases[i]['name']}: Noul polarity outcomes missing")
        per_type[kind] = report
    case_outcomes = [{"id": cid, "question_count": len(hits), "correct_count": sum(hits),
                      "all_questions_correct": len(hits) == 3 and all(hits)} for cid, hits in sorted(by_case.items())]
    per_domain = {}
    for domain, inds in sorted(domains.items()):
        typed = {}
        for kind in TYPES:
            jj = [i for i in inds if cases[i]["type"] == kind]
            if jj:
                correct = [question_rows[i]["correct"] for i in jj]
                targets = [cases[i]["keys"].index(target_label(refs[i], kind)) for i in jj]
                probabilities = [[predictions[i]["probabilities"][k] for k in cases[i]["keys"]] for i in jj]
                typed[kind] = {"n": len(jj), "accuracy": float(np.mean(correct)),
                               "calibration": calibration(correct, [predictions[i]["confidence"] for i in jj], probabilities, targets)}
                if kind == "score":
                    actual = np.asarray([float(refs[i]["target"]["value"]) for i in jj])
                    expected = np.asarray([predictions[i]["expected_value"] for i in jj])
                    typed[kind]["label_argmax_accuracy"] = float(np.mean(correct))
                    typed[kind]["expected_value_mae"] = float(np.mean(np.abs(actual - expected)))
        case_ids = sorted({cases[i]["id"] for i in inds})
        domain_cases = [item for item in case_outcomes if item["id"] in set(case_ids)]
        per_domain[domain] = {"case_count": len(case_ids), "question_count": len(inds), "per_type": typed,
            "all_three_correct_case_count": sum(x["all_questions_correct"] for x in domain_cases),
            "case_outcomes": domain_cases}
    metrics = {"per_question_type": per_type, "per_domain": per_domain,
        "question_count": len(cases), "case_count": len(by_case),
        "all_three_correct_case_count": sum(item["all_questions_correct"] for item in case_outcomes),
        "all_test_cases_have_three_questions": len(by_case) == 60 and all(len(x) == 3 for x in by_case.values()),
        "family_overlap_count": 0, "calibration_scope": "descriptive on matched held-out fixture only; no interval or guarantee"}
    return metrics, question_rows, case_outcomes


def run(args) -> None:
    started = time.perf_counter()
    run_key = args.run_key
    run_dir = (FROZEN_RUNS / f"frozen-{run_key}").resolve()
    models_path = run_dir / "models.json"
    metadata_path = run_dir / "metadata.json"
    # Gate first: no fixture or feature data is opened until exact variant approval.
    freeze, approval = check_freeze(args.selection_freeze, run_key, run_dir, models_path, metadata_path)
    model_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    models, model_id = load_scorers(models_path)
    if approval.get("model") != model_id:
        raise ValueError("selection freeze model ID does not match the saved scorer")
    if model_meta.get("family_overlap_count") != 0 or model_meta.get("train_question_count") != 660:
        raise ValueError("saved model metadata does not match expected clean300+multi120 train-only run")
    feature_path = args.features
    if not feature_path.is_file():
        raise ValueError("feature file missing; export with the explicit installed model store only after selection freeze")
    feature_manifest_path = Path(str(feature_path) + ".manifest.json")
    manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    reference_manifest = json.loads((run_dir / "train-120.features.jsonl.manifest.json").read_text(encoding="utf-8"))
    identity = ("model", "profile_version", "profile_sha256", "text_version", "embedding_dimensions", "normalization")
    if any(manifest.get(field) != reference_manifest.get(field) for field in identity):
        raise ValueError("matched feature exporter provenance differs from frozen training embeddings")
    rows = jsonl(args.input)
    feature_vectors, exported_model = vectors(feature_path, model_id)
    if exported_model != model_id:
        raise ValueError("matched feature model differs from saved scorer model")
    cases = materialize_inference(rows, feature_vectors)
    if len(rows) != args.expected_cases or len(cases) != args.expected_questions:
        raise ValueError(f"expected {args.expected_cases} cases/{args.expected_questions} questions; got {len(rows)}/{len(cases)}")
    if any(case["type"] not in models for case in cases):
        raise ValueError("unknown question type; refusing matched evaluation")
    latencies = []
    predictions = predict(cases, models, latencies)
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in predictions), encoding="utf-8")
    # Nothing below this line accesses reference targets before prediction persistence.
    training_families = {str(x) for x in model_meta.get("train_families", [])}
    metrics, question_outcomes, case_outcomes = evaluate_after_prediction(rows, cases, predictions, training_families)
    (args.out / "question_outcomes.jsonl").write_text("".join(json.dumps(x, sort_keys=True, allow_nan=False) + "\n" for x in question_outcomes), encoding="utf-8")
    (args.out / "case_outcomes.jsonl").write_text("".join(json.dumps(x, sort_keys=True, allow_nan=False) + "\n" for x in case_outcomes), encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    sorted_latency = sorted(latencies)
    latency = {"n": len(latencies), "unit": "seconds per question", "scope": "numpy frozen-scorer only; excludes MLX export, file IO, feature parsing, reference scoring",
        "mean": statistics.mean(latencies) if latencies else None,
        "p50": float(np.percentile(sorted_latency, 50)) if latencies else None,
        "p95": float(np.percentile(sorted_latency, 95)) if latencies else None}
    provenance = {"protocol": "frozen saved typed scorer; evaluation-only; no fitting/C selection",
        "run_key": run_key, "selection_id": freeze["selection_id"], "approved_variant": approval,
        "model": model_id, "run_metadata_sha256": sha(metadata_path), "models_sha256": sha(models_path),
        "selection_freeze_sha256": sha(args.selection_freeze), "fixture_sha256": sha(args.input),
        "features_sha256": sha(feature_path), "feature_manifest_sha256": sha(feature_manifest_path),
        "feature_manifest_identity": {key: manifest.get(key) for key in identity},
        "raw_predictions_sha256": sha(prediction_path), "train_family_count": len(training_families),
        "family_overlap_count": metrics["family_overlap_count"], "latency": latency,
        "run_wall_seconds": time.perf_counter() - started,
        "outputs_sha256": {name: sha(args.out / name) for name in ("predictions.jsonl", "question_outcomes.jsonl", "case_outcomes.jsonl", "metrics.json")},
        "parity_checks": {"candidate_permutation_logits": True, "candidate_permutation_probabilities": True,
                          "unknown_type_fails_closed": True, "noul_polarity_checked": True,
                          "score_label_argmax_and_expected_value": True},
        "python": sys.version, "numpy": np.__version__, "runner_sha256": sha(Path(__file__).resolve()),
        "reference_policy": "references/labels were not accessed until raw predictions.jsonl had been written"}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synthetic_smoke() -> None:
    """Smoke only in-memory request/features/scorers; never accesses any fixture/reference."""
    def questions(choice_options):
        return {
            "arbitrary_choice_name": {"type": "choice", "prompt": "choose", "options": choice_options},
            "binary_name": {"type": "noul", "prompt": "is it true"},
            "rubric_name": {"type": "score", "prompt": "rate", "levels": [
                {"label": "small", "criterion": "small", "value": 1}, {"label": "large", "criterion": "large", "value": 5}]}
        }
    rows = [{"id": "smoke-1", "domain": "toy-a", "request": {"state": {"x": 1}, "questions": questions({"left": "Left", "right": "Right"})}},
            {"id": "smoke-2", "domain": "toy-b", "request": {"state": {"x": 2}, "questions": questions({"left": "Left", "right": "Right", "middle": "Middle"})}}]
    vectors_by_key = {}
    for row in rows:
        for name, q in row["request"]["questions"].items():
            kind, opts, _ = question_candidates(q, row["id"], name)
            dim = 3
            vector_map = {key: np.full(dim, float(i + 1)) for i, key in enumerate(opts)}
            vectors_by_key[(row["id"], name)] = (np.ones(dim), vector_map)
    cases = materialize_inference(rows, vectors_by_key)
    scorers = {kind: (np.ones(12), np.zeros(12), np.ones(12)) for kind in TYPES}
    latency = []
    preds = predict(cases, scorers, latency)
    assert len(preds) == 6 and len(latency) == 6
    assert {p["question_type"] for p in preds} == set(TYPES)
    assert all(p["selected_level"] is not None for p in preds if p["question_type"] == "score")
    assert all(type(p["value"]) is bool for p in preds if p["question_type"] == "noul")
    # Synthetic labels are attached in memory only; simulate durable raw-prediction
    # write before invoking the reference-scoring function.
    for row in rows:
        qrefs = {}
        for name, q in row["request"]["questions"].items():
            if q["type"] == "choice": target = {"selected": next(iter(q["options"]))}
            elif q["type"] == "noul": target = {"value": True}
            else: target = {"label": "large", "value": 5}
            qrefs[name] = {"target": target, "question_family_id": "synthetic-" + name}
        row["reference"] = qrefs
    with tempfile.TemporaryDirectory(prefix="matched-synthetic-smoke-") as temp:
        raw = Path(temp) / "predictions.jsonl"
        raw.write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
        assert raw.stat().st_size > 0
        metrics, question_rows, case_rows = evaluate_after_prediction(rows, cases, preds, set())
        assert len(question_rows) == 6 and len(case_rows) == 2 and len(metrics["per_domain"]) == 2
        assert metrics["case_count"] == 2 and metrics["per_question_type"]["choice"]["candidate_counts"] == {"2": 1, "3": 1}
    try:
        question_candidates({"type": "unknown"}, "smoke", "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown type must fail closed")
    print("synthetic matched-runner smoke passed; fixture/reference paths were not opened")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run in-memory synthetic smoke only")
    parser.add_argument("--run-key", choices=("qwen", "harrier"))
    parser.add_argument("--selection-freeze", type=Path)
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-cases", type=int, default=60)
    parser.add_argument("--expected-questions", type=int, default=180)
    args = parser.parse_args()
    if args.self_test:
        synthetic_smoke()
        return 0
    required = (args.run_key, args.selection_freeze, args.features, args.out)
    if any(x is None for x in required):
        parser.error("evaluation requires --run-key, --selection-freeze, --features, and --out")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
