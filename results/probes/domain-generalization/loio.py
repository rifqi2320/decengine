#!/usr/bin/env python3
"""Leave-one-industry-out evaluation of frozen task-specific embedding probes.

The fold manifest is deliberately an input: this runner never creates or edits it.
Each industry's records are excluded from fitting and train-only CV in its entirety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results" / "probes"))
import train_probe as probe  # noqa: E402

TASKS = probe.TASKS
MODELS = {
    "qwen": "Qwen-Qwen3-Embedding-0-6B",
    "harrier": "microsoft-harrier-oss-v1-0-6b",
}


def file_hash(path: Path) -> str:
    return probe.sha256(path)


def industry_of(row: dict[str, Any]) -> str:
    scenario = row.get("scenario", {})
    value = scenario.get("industry") if isinstance(scenario, dict) else None
    value = value or row.get("industry") or row.get("metadata", {}).get("industry")
    if value is None:
        raise ValueError(f"{row.get('id')}: no industry field in case record")
    return str(value)


def normalize_folds(document: Any) -> tuple[dict[str, dict[str, set[str]]], dict[str, str]]:
    """Read the frozen manifest's exhaustive assignments, never derive fold ids."""
    if not isinstance(document, dict) or not isinstance(document.get("folds"), dict):
        raise ValueError("fold manifest must contain an explicit folds object")
    result: dict[str, dict[str, set[str]]] = {}
    for industry, fold in document["folds"].items():
        result[str(industry)] = {}
        for field in ("fit_train_ids", "development_validation_ids", "sealed_test_ids"):
            ids = fold.get(field)
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"fold {industry!r} must contain nonempty {field}")
            result[str(industry)][field] = {str(case_id) for case_id in ids}
    case_industries = {str(item["id"]): str(item["industry"]) for item in document.get("cases", [])}
    if not result or not case_industries:
        raise ValueError("fold manifest has no folds or case mapping")
    return result, case_industries


def metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    if len(y) == 0:
        raise ValueError("cannot score an empty evaluation set")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=classes, zero_division=0)
    absent = [name for name in classes if not np.any(y == name)]
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "micro_f1": float(f1_score(y, pred, labels=classes, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "confusion_labels": classes,
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                       for i, name in enumerate(classes)},
        "classes_absent_in_heldout": absent,
        "class_absent_warning": ("Macro-F1 includes absent heldout classes with zero support/F1; "
                                 "interpret with per-class support." if absent else None),
    }


def bootstrap_paired(y: np.ndarray, left: np.ndarray, right: np.ndarray,
                     classes: list[str], seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(2000):
        ix = rng.integers(0, len(y), len(y))
        acc.append(accuracy_score(y[ix], left[ix]) - accuracy_score(y[ix], right[ix]))
        f1.append(f1_score(y[ix], left[ix], labels=classes, average="macro", zero_division=0) -
                  f1_score(y[ix], right[ix], labels=classes, average="macro", zero_division=0))
    return {"qwen_minus_harrier_accuracy_95_percentile": [float(x) for x in np.percentile(acc, [2.5, 97.5])],
            "qwen_minus_harrier_macro_f1_95_percentile": [float(x) for x in np.percentile(f1, [2.5, 97.5])]}


def predict_checked(model: LogisticRegression, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = probe.predict_checked(model, x, probabilities=True)
    pred = probe.predict_checked(model, x)
    return pred, probs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "benchmarks/cases/probe-train-300.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "benchmarks/cases/probe-test-300.jsonl")
    parser.add_argument("--folds", type=Path, default=Path(__file__).with_name("folds.json"))
    parser.add_argument("--features", type=Path, default=Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/probe-features"))
    parser.add_argument("--test-features", type=Path,
                        default=ROOT / "results/probes/comparisons/20260923T104208Z/features")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "loio")
    args = parser.parse_args()
    if not args.folds.exists():
        raise FileNotFoundError(f"fold manifest not ready: {args.folds}")
    train_rows, test_rows = probe.read_jsonl(args.train), probe.read_jsonl(args.test)
    manifest = json.loads(args.folds.read_text(encoding="utf-8"))
    folds, case_industries = normalize_folds(manifest)
    train_ids = [str(r["id"]) for r in train_rows]
    test_ids = [str(r["id"]) for r in test_rows]
    all_train_rows = train_rows
    train_by_id = {str(r["id"]): r for r in train_rows}
    test_by_id = {str(r["id"]): r for r in test_rows}
    if set(train_ids) & set(test_ids):
        raise ValueError("train/test IDs overlap")
    if set(case_industries) != set(train_ids) | set(test_ids):
        raise ValueError("fold manifest case IDs do not exactly match train/test fixture IDs")
    for industry, fold in folds.items():
        if any(case_industries.get(case_id) != industry
               for field in ("development_validation_ids", "sealed_test_ids") for case_id in fold[field]):
            raise ValueError(f"fold {industry!r} includes IDs with a different canonical industry")
        if not fold["fit_train_ids"] <= set(train_ids) or not fold["development_validation_ids"] <= set(train_ids):
            raise ValueError(f"fold {industry!r} fitting/validation IDs must come from train source")
        if not fold["sealed_test_ids"] <= set(test_ids):
            raise ValueError(f"fold {industry!r} sealed IDs must come from test source")
        if fold["fit_train_ids"] & fold["development_validation_ids"] or fold["fit_train_ids"] & fold["sealed_test_ids"]:
            raise ValueError(f"fold {industry!r} has overlapping fit and heldout sets")
    if set().union(*(fold["fit_train_ids"] | fold["development_validation_ids"] for fold in folds.values())) != set(train_ids):
        raise ValueError("fold manifest train-source assignments do not cover all train records")
    if set().union(*(fold["sealed_test_ids"] for fold in folds.values())) != set(test_ids):
        raise ValueError("fold manifest sealed assignments do not cover all test records")
    for split_field in ("development_validation_ids", "sealed_test_ids"):
        assigned = [case_id for fold in folds.values() for case_id in fold[split_field]]
        if len(assigned) != len(set(assigned)):
            raise ValueError(f"fold manifest assigns IDs more than once in {split_field}")
    args.out.mkdir(parents=True, exist_ok=True)
    # Explicit input validation; never load saved models or prediction artifacts.
    vector_paths = {(model, split): ((args.features / f"{name}-train300.jsonl") if split == "train" else
                                     (args.test_features / f"{model}-test300.jsonl"))
                    for model, name in MODELS.items() for split in ("train", "test")}
    vectors: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {name: {} for name in MODELS}
    model_ids: dict[str, str] = {}
    for (model, split), path in vector_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing exported feature file for {model}: {path}")
        vectors[model][split], dimensions, resolved_model_id = probe.embedding_map(path, model)
        if model in model_ids and model_ids[model] != resolved_model_id:
            raise ValueError(f"{model}: train/test export model IDs differ")
        model_ids[model] = resolved_model_id
        expected_ids = set(train_ids if split == "train" else test_ids)
        if set(vectors[model][split]) != expected_ids:
            raise ValueError(f"{model}/{split} feature ID mismatch: missing={len(expected_ids-set(vectors[model][split]))}, "
                             f"extra={len(set(vectors[model][split])-expected_ids)}")
        for task in TASKS:
            if dimensions.get(task) != 1024 or any(task not in vectors[model][split][case_id]
                                                   for case_id in expected_ids):
                raise ValueError(f"{model}: require finite 1024-D task-specific {task} vectors")
    results: dict[str, Any] = {"folds": {}, "pooled": {"train_outer_validation": {}, "sealed_test": {}},
                               "paired_differences": {"train_outer_validation": {}, "sealed_test": {}},
                               "worst_industry": {"train_outer_validation": {}, "sealed_test": {}}}
    prediction_by_id: dict[str, dict[str, Any]] = {
        key: {"id": key, "industry": case_industries[key], "source_split": "train"}
        for key in train_ids}
    prediction_by_id.update({key: {"id": key, "industry": case_industries[key], "source_split": "test"}
                             for key in test_ids})
    failures: list[dict[str, str]] = []
    for fold_index, (industry, fold) in enumerate(folds.items()):
        fit_ids = [key for key in train_ids if key in fold["fit_train_ids"]]
        validation_ids = [key for key in train_ids if key in fold["development_validation_ids"]]
        sealed_ids = [key for key in test_ids if key in fold["sealed_test_ids"]]
        fit_rows = [train_by_id[key] for key in fit_ids]
        targets = {
            "train_outer_validation": (validation_ids, train_by_id),
            "sealed_test": (sealed_ids, test_by_id),
        }
        fold_result: dict[str, Any] = {"n_fit_train": len(fit_ids), "n_train_outer_validation": len(validation_ids),
                                      "n_sealed_test": len(sealed_ids), "tasks": {}}
        for task in TASKS:
            y_train_text = np.asarray([probe.label(row, task) for row in fit_rows])
            y_test_by_split = {split: np.asarray([probe.label(row_map[key], task) for key in target_ids])
                               for split, (target_ids, row_map) in targets.items()}
            classes = sorted(set(probe.label(row, task) for row in all_train_rows))
            fold_majority = Counter(y_train_text).most_common(1)[0][0]
            missing_train = sorted(set(classes) - set(y_train_text))
            if missing_train:
                failure = {"industry": industry, "task": task, "error": f"training fold lacks classes {missing_train}"}
                failures.append(failure)
                fold_result["tasks"][task] = {"failure": failure["error"]}
                continue
            task_output: dict[str, Any] = {"classes": classes, "models": {}}
            outputs: dict[str, dict[str, np.ndarray]] = {split: {} for split in targets}
            for model_name in MODELS:
                try:
                    x_train = np.stack([vectors[model_name]["train"][key][task] for key in fit_ids])
                    encoder = probe.LabelEncoder().fit(y_train_text)
                    y_train = encoder.transform(y_train_text)
                    groups = probe.groups_for(fit_rows)
                    selected_c, cv = probe.choose_c(x_train, y_train, groups)
                    scaler = StandardScaler().fit(x_train)
                    model = LogisticRegression(C=selected_c, penalty="l2", solver="lbfgs",
                                               max_iter=probe.MAX_ITER, random_state=probe.SEED)
                    probe.fit_checked(model, scaler.transform(x_train), y_train)
                    model_result: dict[str, Any] = {
                        "selected_C": selected_c, "train_cv": cv,
                        "coefficient_sha256": hashlib.sha256(
                            np.asarray(model.coef_, dtype="<f8").tobytes() +
                            np.asarray(model.intercept_, dtype="<f8").tobytes()).hexdigest(),
                    }
                    for split, (target_ids, row_map) in targets.items():
                        x_eval = np.stack([vectors[model_name]["test" if split == "sealed_test" else "train"][key][task]
                                           for key in target_ids])
                        pred_num, prob = predict_checked(model, scaler.transform(x_eval))
                        pred = encoder.inverse_transform(pred_num).astype(str)
                        class_probs = np.zeros((len(target_ids), len(classes)))
                        for j, name in enumerate(encoder.classes_):
                            class_probs[:, classes.index(str(name))] = prob[:, j]
                        outputs[split][model_name] = pred
                        y_eval = y_test_by_split[split]
                        model_result[split] = metrics(y_eval, pred, classes)
                        for index, case_id in enumerate(target_ids):
                            prediction_by_id[case_id].setdefault(task, {
                                "truth": str(y_eval[index]), "majority_baseline": str(fold_majority), "models": {}})
                            prediction_by_id[case_id][task]["models"][model_name] = {
                                "prediction": str(pred[index]),
                                "probabilities": {name: float(class_probs[index, k]) for k, name in enumerate(classes)}}
                    task_output["models"][model_name] = model_result
                except Exception as exc:
                    failure = {"industry": industry, "task": task, "model": model_name,
                               "error": f"{type(exc).__name__}: {exc}"}
                    failures.append(failure)
                    task_output["models"][model_name] = {"failure": failure["error"]}
                    traceback.print_exc()
            for split, (target_ids, _) in targets.items():
                if len(outputs[split]) == 2:
                    y_eval = y_test_by_split[split]
                    q, h = outputs[split]["qwen"], outputs[split]["harrier"]
                    task_output.setdefault("paired_qwen_minus_harrier", {})[split] = {
                        "complete_case_n": int(len(y_eval)),
                        "accuracy_difference": float(accuracy_score(y_eval, q) - accuracy_score(y_eval, h)),
                        "macro_f1_difference": float(f1_score(y_eval, q, labels=classes, average="macro", zero_division=0) -
                                                     f1_score(y_eval, h, labels=classes, average="macro", zero_division=0)),
                        "bootstrap_95_percentile": bootstrap_paired(y_eval, q, h, classes,
                                                                      probe.SEED + fold_index + (100 if split == "sealed_test" else 0)),
                    }
                    task_output.setdefault("majority_baseline", {})[split] = metrics(
                        y_eval, np.repeat(fold_majority, len(y_eval)), classes)
            fold_result["tasks"][task] = task_output
        results["folds"][industry] = fold_result
        

    # Pooled complete-case comparisons include only cases with successful predictions
    # from both models for the task, never a model-specific denominator.
    for split in ("train_outer_validation", "sealed_test"):
        split_ids = set(train_ids if split == "train_outer_validation" else test_ids)
        for task in TASKS:
            case_rows = [prediction_by_id[key] for key in (train_ids if split == "train_outer_validation" else test_ids)
                         if task in prediction_by_id[key] and
                         "qwen" in prediction_by_id[key][task]["models"] and
                         "harrier" in prediction_by_id[key][task]["models"]]
            if not case_rows:
                continue
            y = np.asarray([r[task]["truth"] for r in case_rows])
            classes = sorted(set(probe.label(row, task) for row in all_train_rows))
            q = np.asarray([r[task]["models"]["qwen"]["prediction"] for r in case_rows])
            h = np.asarray([r[task]["models"]["harrier"]["prediction"] for r in case_rows])
            results["pooled"][split][task] = {"complete_case_n": len(case_rows),
                                               "qwen": metrics(y, q, classes), "harrier": metrics(y, h, classes),
                                               "majority_baseline": metrics(y, np.asarray([r[task]["majority_baseline"] for r in case_rows]), classes)}
            for model_name, prediction in (("qwen", q), ("harrier", h)):
                results["pooled"][split][task][model_name]["bootstrap_ci"] = probe.bootstrap_ci(
                    y, prediction, seed=probe.SEED + (100 if split == "sealed_test" else 0))
            results["paired_differences"][split][task] = {
                "complete_case_n": len(case_rows),
                "qwen_minus_harrier_accuracy": float(accuracy_score(y, q) - accuracy_score(y, h)),
                "qwen_minus_harrier_macro_f1": float(f1_score(y, q, labels=classes, average="macro", zero_division=0) -
                                                     f1_score(y, h, labels=classes, average="macro", zero_division=0)),
                "bootstrap_95_percentile": bootstrap_paired(y, q, h, classes, probe.SEED + (100 if split == "sealed_test" else 0)),
            }
            per_industry = [(name, data["tasks"].get(task, {}).get("models", {}))
                            for name, data in results["folds"].items()]
            results["worst_industry"][split][task] = {
                model: min(((name, info[model][split]["macro_f1"]) for name, info in per_industry
                            if model in info and split in info[model]), key=lambda item: item[1], default=None)
                for model in MODELS
            }
    results["uncertainty_note"] = (
        "Percentile bootstrap intervals resample heldout cases (paired for model differences); "
        "they do not capture sampling of industries. Industry count and synthetic case/source-style "
        "limitations preclude broad real-world generalization claims.")
    results["caveat"] = ("Cases are synthetic benchmark cases and may encode template/source-style artifacts; "
                         "LOIO tests only the declared industries, not arbitrary domains or deployment populations.")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.out / "predictions.jsonl").write_text("".join(json.dumps(prediction_by_id[key], sort_keys=True) + "\n"
                                                               for key in train_ids + test_ids), encoding="utf-8")
    metadata = {
        "inputs": {str(path): file_hash(path) for path in [args.train, args.test, args.folds, *vector_paths.values()]},
        "embedding_manifests": {
            str(Path(str(path) + ".manifest.json")): file_hash(Path(str(path) + ".manifest.json"))
            for path in vector_paths.values() if Path(str(path) + ".manifest.json").exists()},
        "fold_manifest_sha256": file_hash(args.folds), "fold_id_sha256": {
            name: {part: hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest()
                   for part, case_ids in fold.items()} for name, fold in folds.items()},
        "model_ids": model_ids, "feature_dimensions": {name: {task: 1024 for task in TASKS} for name in MODELS},
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "runner_sha256": file_hash(Path(__file__)),
        "seed": probe.SEED, "C_grid": probe.C_VALUES, "selection": "mean train-only grouped stratified CV macro-F1; tie favors smaller C",
        "transform": "StandardScaler fitted within each CV fit split and again on the full non-heldout train fold",
        "failures": failures,
        "heldout_leakage_control": "heldout industry's train validation and sealed test IDs excluded from fitting and CV; outer C selected using fit_train_ids only",
        "saved_probe_weights_used": False, "heldout_labels_used_for_tuning": False,
        "caveat": results["caveat"],
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
