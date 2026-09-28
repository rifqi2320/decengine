#!/usr/bin/env python3
"""Train and evaluate regularized linear probes on frozen exported embeddings.

The test split is only read for final predictions and metrics. All tuning is confined
to train-only cross-validation. See README.md for input schemas and leakage controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

TASKS = ("owner", "urgent", "impact")
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 271828
MAX_ITER = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, model_id: str) -> dict[str, Any] | None:
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != model_id:
        raise ValueError(f"{manifest_path}: manifest model does not match embeddings ({model_id})")
    return {"path": str(manifest_path), "sha256": sha256(manifest_path), "contents": manifest}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: every record must have a unique non-empty id")
    return rows


def embedding_map(path: Path, model: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int], str]:
    """Accept JSONL records or a JSON object with an embeddings array."""
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict) and isinstance(document.get("embeddings"), list):
        records = document["embeddings"]
    elif isinstance(document, list):
        records = document
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    model_ids = sorted({str(item["model"]) for item in records
                        if isinstance(item, dict) and item.get("model") is not None})
    exact = [candidate for candidate in model_ids if candidate.casefold() == model.casefold()]
    if not model_ids:
        selected_model = model
    elif exact:
        selected_model = exact[0]
    else:
        # Aliases match a complete token or prefix of one token in the exporter model
        # ID. This supports e.g. "qwen" for "Qwen3" but refuses ambiguous matches.
        alias = re.sub(r"[^a-z0-9]+", " ", model.casefold()).split()
        matching = [candidate for candidate in model_ids
                    if all(any(part.startswith(token) for part in
                               re.sub(r"[^a-z0-9]+", " ", candidate.casefold()).split())
                           for token in alias)] if alias else []
        if len(matching) != 1:
            raise ValueError(f"--model {model!r} did not identify one exporter model; available: {model_ids}")
        selected_model = matching[0]
    vectors: dict[str, dict[str, np.ndarray]] = {}
    dimensions: dict[str, set[int]] = {}
    for item in records:
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError(f"{path}: each embedding record must contain id")
        if item.get("model") not in (None, selected_model):
            continue
        key = str(item["id"])
        if key in vectors:
            raise ValueError(f"{path}: duplicate embedding id {key!r}")
        if "embeddings" in item:
            encoded = item["embeddings"]
            if not isinstance(encoded, dict) or not encoded:
                raise ValueError(f"{path}: {key}: embeddings must be a non-empty task/vector object")
            entries = encoded
        elif "embedding" in item:
            # Backward compatibility: one legacy vector is shared across tasks.
            entries = {task: item["embedding"] for task in (*TASKS, "state")}
        else:
            raise ValueError(f"{path}: {key}: expected embeddings object or legacy embedding vector")
        vectors[key] = {}
        for task, raw in entries.items():
            if task not in (*TASKS, "state"):
                continue
            vector = np.asarray(raw, dtype=np.float64)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError(f"{path}: {key}: {task} embedding must be a finite 1-D vector")
            dimensions.setdefault(task, set()).add(vector.size)
            vectors[key][task] = vector
    dimensions_by_task = {}
    for task, sizes in dimensions.items():
        if len(sizes) != 1:
            raise ValueError(f"{path}: inconsistent {task} embedding dimensions: {sorted(sizes)}")
        dimensions_by_task[task] = next(iter(sizes))
    return vectors, dimensions_by_task, selected_model


def label(record: dict[str, Any], task: str) -> str:
    value = record.get("reference", {}).get(task)
    if task == "urgent":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower()
        raise ValueError(f"{record.get('id')}: urgent must be a boolean")
    if value is None or isinstance(value, (dict, list)):
        raise ValueError(f"{record.get('id')}: missing/invalid reference.{task}")
    return str(value)


def request_text(record: dict[str, Any]) -> str:
    """Build a stable source-text representation without reference labels/rationale."""
    parts: list[str] = []
    scenario = record.get("scenario")
    if isinstance(scenario, str):
        parts.append(scenario)
    elif isinstance(scenario, dict):
        parts.extend(str(v) for k, v in scenario.items() if k not in ("industry", "tag"))
    request = record.get("request", {})
    if isinstance(request, dict):
        parts.extend(str(v) for v in request.values() if isinstance(v, (str, int, float)))
        questions = request.get("questions", [])
        if isinstance(questions, list):
            parts.extend(json.dumps(q, sort_keys=True, ensure_ascii=False) for q in questions)
    return re.sub(r"\W+", " ", " ".join(parts).lower()).strip()


def declared_group(record: dict[str, Any]) -> str | None:
    for source in (record, record.get("metadata", {})):
        if isinstance(source, dict):
            value = source.get("group_id", source.get("group"))
            if value is not None:
                return str(value)
    return None


def near_duplicate_components(rows: list[dict[str, Any]], threshold: float = 0.88) -> list[int]:
    """Connected components by exact normalized text or high token-set Jaccard."""
    texts = [request_text(row) for row in rows]
    tokens = [set(text.split()) for text in texts]
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i):
            if texts[i] == texts[j] and texts[i]:
                score = 1.0
            else:
                union = tokens[i] | tokens[j]
                score = len(tokens[i] & tokens[j]) / len(union) if union else 0.0
            if score >= threshold:
                parent[root(i)] = root(j)
    return [root(i) for i in range(len(rows))]


def groups_for(rows: list[dict[str, Any]]) -> np.ndarray:
    near_components = near_duplicate_components(rows)
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[root(left)] = root(right)

    representatives: dict[int, int] = {}
    for index, component in enumerate(near_components):
        if component in representatives:
            union(index, representatives[component])
        else:
            representatives[component] = index
    declared: dict[str, int] = {}
    for index, row in enumerate(rows):
        key = declared_group(row)
        if key is not None:
            if key in declared:
                union(index, declared[key])
            else:
                declared[key] = index
    return np.asarray([root(index) for index in range(len(rows))])


def assert_no_split_leakage(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    joined = train + test
    groups = groups_for(joined)
    n = len(train)
    overlap = set(groups[:n]) & set(groups[n:])
    if overlap:
        raise ValueError("train/test contains matching declared/near-duplicate groups; refusing leakage")


def choose_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    counts = Counter(y)
    splits = min(5, min(counts.values()))
    if splits < 2:
        raise ValueError("each class needs at least two train examples for stratified CV")
    if len(set(groups)) < splits:
        raise ValueError("too few independent groups for train-only CV")
    try:
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=SEED)
        folds = list(splitter.split(x, y, groups))
        if any(len(set(y[valid])) < 2 for _, valid in folds):
            raise ValueError("group folds missing class")
    except ValueError:
        # Only fall back when the source has no repeated/near-duplicate groups.
        if len(set(groups)) != len(groups):
            raise ValueError("cannot form stratified group folds; provide enough examples per group/class")
        folds = list(StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED).split(x, y))
    results = []
    fold_supports = [{str(int(cls)): int(np.sum(y[valid] == cls)) for cls in np.unique(y)}
                     for _, valid in folds]
    fold_group_counts = [int(len(set(groups[valid]))) for _, valid in folds]
    for c in C_VALUES:
        scores = []
        for fit, valid in folds:
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(x[fit])
            x_valid = scaler.transform(x[valid])
            model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=MAX_ITER,
                                       random_state=SEED)
            fit_checked(model, x_fit, y[fit])
            scores.append(f1_score(y[valid], predict_checked(model, x_valid),
                                   average="macro", zero_division=0))
        results.append({"C": c, "fold_macro_f1": scores, "mean_macro_f1": float(np.mean(scores)),
                        "validation_class_support_by_fold": fold_supports,
                        "validation_group_count_by_fold": fold_group_counts})
    # Explicit deterministic tie-break: stronger regularization (smaller C).
    best = max(results, key=lambda item: (item["mean_macro_f1"], -item["C"]))
    return float(best["C"]), results


def fit_checked(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> None:
    """Fit while silencing this host's spurious BLAS matmul FPE warnings only.

    On the available macOS OpenBLAS build, even multiplying a finite matrix by an
    all-zero coefficient matrix returns finite zeros but emits divide/overflow/invalid
    RuntimeWarnings. Do not hide convergence warnings; validate every fitted parameter.
    """
    if not np.isfinite(x).all():
        raise FloatingPointError("non-finite design matrix before classifier fit")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
        model.fit(x, y)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise FloatingPointError("logistic regression produced non-finite fitted parameters")
    if int(np.max(model.n_iter_)) >= MAX_ITER:
        raise RuntimeError("logistic regression reached max_iter without confirmed convergence")


def predict_checked(model: LogisticRegression, x: np.ndarray,
                    probabilities: bool = False) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
        result = model.predict_proba(x) if probabilities else model.predict(x)
    if not np.isfinite(result).all():
        raise FloatingPointError("logistic regression produced non-finite predictions")
    if probabilities and (np.any(result < 0) or np.any(result > 1) or
                          not np.allclose(result.sum(axis=1), 1.0, atol=1e-8)):
        raise FloatingPointError("logistic regression returned invalid probability rows")
    return result


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, seed: int = SEED, repeats: int = 2000) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values = []
    accuracies = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[indices], pred[indices], average="macro", zero_division=0))
        accuracies.append(accuracy_score(y[indices], pred[indices]))
    return {"macro_f1_95_percentile": [float(v) for v in np.percentile(values, [2.5, 97.5])],
            "accuracy_95_percentile": [float(v) for v in np.percentile(accuracies, [2.5, 97.5])]}


def evaluate(y: np.ndarray, pred: np.ndarray, probabilities: np.ndarray,
             classes: list[str], task: str) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=classes, zero_division=0)
    report: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "confusion_labels": classes,
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                       for i, name in enumerate(classes)},
        "bootstrap_ci": bootstrap_ci(y, pred),
    }
    if task == "urgent":
        positive_index = classes.index("true")
        positive = (y == "true").astype(float)
        report["brier_score"] = float(np.mean((probabilities[:, positive_index] - positive) ** 2))
        report["sensitivity"] = float(np.sum((y == "true") & (pred == "true")) /
                                       max(1, np.sum(y == "true")))
        report["specificity"] = float(np.sum((y == "false") & (pred == "false")) /
                                       max(1, np.sum(y == "false")))
        rng = np.random.default_rng(SEED)
        sens, spec = [], []
        for _ in range(2000):
            indices = rng.integers(0, len(y), len(y))
            positive_mask, negative_mask = y[indices] == "true", y[indices] == "false"
            if positive_mask.any():
                sens.append(float(np.mean(pred[indices][positive_mask] == "true")))
            if negative_mask.any():
                spec.append(float(np.mean(pred[indices][negative_mask] == "false")))
        report["bootstrap_ci"].update({
            "sensitivity_95_percentile": [float(v) for v in np.percentile(sens, [2.5, 97.5])],
            "specificity_95_percentile": [float(v) for v in np.percentile(spec, [2.5, 97.5])],
        })
        report["calibration_bins"] = []
        confidence = probabilities[:, positive_index]
        for low, high in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
            mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
            if mask.any():
                report["calibration_bins"].append({"lower": float(low), "upper": float(high),
                                                    "count": int(mask.sum()),
                                                    "mean_probability": float(confidence[mask].mean()),
                                                    "observed_rate": float(positive[mask].mean())})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-embeddings", type=Path,
                        help="model-specific train-split embeddings JSONL/JSON")
    parser.add_argument("--test-embeddings", type=Path,
                        help="model-specific test-split embeddings JSONL/JSON")
    parser.add_argument("--embeddings", type=Path,
                        help="deprecated combined export; prefer separate split-specific files")
    parser.add_argument("--model", required=True, help="exact exporter model ID or unique alias")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-only", action="store_true",
                        help="control run: use embeddings.state for all three classifiers")
    parser.add_argument("--previous-predictions", type=Path,
                        help="optional test JSONL with id and owner/urgent/impact predictions")
    parser.add_argument("--expected-counts", action="store_true",
                        help="require 300 train and 90 test examples")
    args = parser.parse_args()
    if args.embeddings:
        if args.train_embeddings or args.test_embeddings:
            parser.error("use either --embeddings or the pair --train-embeddings/--test-embeddings")
        train_embedding_path = test_embedding_path = args.embeddings
    elif args.train_embeddings and args.test_embeddings:
        train_embedding_path, test_embedding_path = args.train_embeddings, args.test_embeddings
    else:
        parser.error("provide both --train-embeddings and --test-embeddings")
    train, test = read_jsonl(args.train), read_jsonl(args.test)
    if args.expected_counts and (len(train), len(test)) != (300, 90):
        raise ValueError(f"expected 300 train/90 test, got {len(train)}/{len(test)}")
    train_ids, test_ids = [str(r["id"]) for r in train], [str(r["id"]) for r in test]
    if set(train_ids) & set(test_ids):
        raise ValueError("train and test dataset IDs overlap")
    assert_no_split_leakage(train, test)
    train_vectors, train_dimensions, train_model_id = embedding_map(train_embedding_path, args.model)
    test_vectors, test_dimensions, test_model_id = embedding_map(test_embedding_path, args.model)
    if train_model_id != test_model_id:
        raise ValueError(f"train/test embedding model mismatch: {train_model_id!r} vs {test_model_id!r}")
    model_id = train_model_id
    for split_name, expected_ids, vectors in (("train", set(train_ids), train_vectors),
                                               ("test", set(test_ids), test_vectors)):
        actual_ids = set(vectors)
        if actual_ids != expected_ids:
            raise ValueError(f"{split_name} embedding ID mismatch: missing={sorted(expected_ids-actual_ids)[:5]}, "
                             f"extra={sorted(actual_ids-expected_ids)[:5]}")
    if train_dimensions != test_dimensions:
        raise ValueError(f"train/test embedding dimensions differ: {train_dimensions} vs {test_dimensions}")
    dimensions = train_dimensions
    train_manifest = read_manifest(train_embedding_path, model_id)
    test_manifest = read_manifest(test_embedding_path, model_id)
    if train_manifest and test_manifest:
        provenance_keys = ("model", "profile_version", "profile_sha256", "text_version",
                           "embedding_dimensions")
        train_provenance = train_manifest["contents"]
        test_provenance = test_manifest["contents"]
        mismatches = [key for key in provenance_keys
                      if train_provenance.get(key) != test_provenance.get(key)]
        if mismatches:
            raise ValueError(f"train/test exporter manifest provenance differs: {mismatches}")
    feature_tasks = {task: ("state" if args.state_only else task) for task in TASKS}
    for task, feature_task in feature_tasks.items():
        missing_train = [key for key in train_ids if feature_task not in train_vectors[key]]
        missing_test = [key for key in test_ids if feature_task not in test_vectors[key]]
        if missing_train or missing_test:
            raise ValueError(f"missing {feature_task!r} embeddings: train={len(missing_train)}, "
                             f"test={len(missing_test)}")
        if dimensions.get(feature_task) != 1024:
            raise ValueError(f"expected genuine 1024-D {feature_task!r} embeddings, "
                             f"got {dimensions.get(feature_task)}")
    args.out.mkdir(parents=True, exist_ok=True)
    train_group_ids = groups_for(train)
    all_predictions: dict[str, dict[str, Any]] = {key: {"id": key, "model": model_id} for key in test_ids}
    summary: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}
    for task in TASKS:
        feature_task = feature_tasks[task]
        x_train = np.stack([train_vectors[key][feature_task] for key in train_ids])
        x_test = np.stack([test_vectors[key][feature_task] for key in test_ids])
        encoder = LabelEncoder()
        y_train = encoder.fit_transform([label(row, task) for row in train])
        y_test_names = np.asarray([label(row, task) for row in test])
        class_names = [str(item) for item in encoder.classes_]
        y_test = encoder.transform(y_test_names)
        c, cv = choose_c(x_train, y_train, train_group_ids)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=MAX_ITER,
                                   random_state=SEED)
        fit_checked(model, x_train_scaled, y_train)
        probs = predict_checked(model, x_test_scaled, probabilities=True)
        preds = predict_checked(model, x_test_scaled)
        names = np.asarray(class_names)[preds]
        task_report = evaluate(y_test_names, names, probs, class_names, task)
        majority = Counter(y_train).most_common(1)[0][0]
        baseline_pred = np.repeat(encoder.inverse_transform([majority])[0], len(test))
        baseline_probs = np.zeros((len(test), len(class_names)))
        baseline_probs[:, majority] = 1.0
        task_report["majority_baseline"] = evaluate(y_test_names, baseline_pred,
                                                     baseline_probs,
                                                     class_names, task)
        task_report.update({"selected_C": c, "cv": cv, "classes": class_names})
        summary[task] = task_report
        fitted_models[task] = {"classes": class_names, "C": c,
                               "embedding_key": feature_task,
                               "intercept": model.intercept_.tolist(),
                               "coefficients": model.coef_.tolist(),
                               "feature_scaler_mean": scaler.mean_.tolist(),
                               "feature_scaler_scale": scaler.scale_.tolist(),
                               "cv": cv, "selection_metric": "mean stratified train-only CV macro-F1"}
        for i, key in enumerate(test_ids):
            all_predictions[key][task] = {"label": str(names[i]),
                                          "probabilities": {class_names[j]: float(probs[i, j])
                                                                for j in range(len(class_names))}}
    previous = None
    if args.previous_predictions:
        previous_rows = read_jsonl(args.previous_predictions)
        previous = {str(row["id"]): row for row in previous_rows}
        if set(previous) != set(test_ids):
            raise ValueError("previous predictions must contain exactly the test IDs")
        for task in TASKS:
            pred = np.asarray([str(previous[key][task]).lower() if task == "urgent"
                               else str(previous[key][task]) for key in test_ids])
            truth = np.asarray([label(row, task) for row in test])
            # No probabilities are implied by the prior hard-label cosine output;
            # mark calibration as unavailable rather than fabricating scores.
            summary[task]["previous_cosine_baseline"] = evaluate(
                truth, pred, np.zeros((len(test), len(summary[task]["classes"]))),
                summary[task]["classes"], task)
            if task == "urgent":
                summary[task]["previous_cosine_baseline"].pop("brier_score", None)
                summary[task]["previous_cosine_baseline"].pop("calibration_bins", None)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "predictions.jsonl").write_text("".join(
        json.dumps(all_predictions[key], sort_keys=True) + "\n" for key in test_ids), encoding="utf-8")
    (args.out / "models.json").write_text(json.dumps(fitted_models, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "model": model_id, "model_argument": args.model,
        "embedding_dimensions": dimensions, "feature_mode": "state-only" if args.state_only else "task-specific",
        "feature_keys_by_task": feature_tasks, "seed": SEED,
        "train_count": len(train), "test_count": len(test), "train_ids_sha256": hashlib.sha256(
            "\n".join(train_ids).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "train_test_ids_overlap": False,
        "training_group_count": int(len(set(train_group_ids))),
        "training_records_in_repeated_groups": int(len(train) - len(set(train_group_ids))),
        "inputs": {str(path): sha256(path) for path in
                   [args.train, args.test, train_embedding_path, test_embedding_path] + ([args.previous_predictions]
                                                                if args.previous_predictions else [])},
        "embedding_manifests": {"train": train_manifest, "test": test_manifest},
        "python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "trainer_script_sha256": sha256(Path(__file__)),
        "requirements_sha256": sha256(Path(__file__).with_name("requirements.txt")),
        "argv": sys.argv,
        "features": "frozen exported embeddings; train-fitted StandardScaler per classifier, no encoder tuning",
        "leakage_audit": "train/test exact and >=0.88 token-Jaccard duplicates rejected; grouped CV",
        "numerical_validation": "all fitted coefficients/intercepts, probabilities and CV scores checked finite; "
                                "finite-result BLAS matmul runtime warnings suppressed during sklearn calls",
        "hyperparameter_search": {"C_values": C_VALUES, "penalty": "l2", "solver": "lbfgs",
                                  "max_iter": MAX_ITER,
                                  "metric": "mean stratified train-only CV macro-F1"},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
