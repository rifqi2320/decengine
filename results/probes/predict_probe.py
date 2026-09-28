#!/usr/bin/env python3
"""Apply a saved frozen-embedding probe to unlabeled exported JSONL cases.

This inference-only command loads serialized scaler and logistic-regression parameters.
It never reads a ``reference`` field and never fits, tunes, or normalizes on input cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("owner", "urgent", "impact")
MANIFEST_KEYS = ("model", "profile_version", "profile_sha256", "text_version",
                 "embedding_dimensions")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path, model_id: str, feature_keys: dict[str, str],
                 dimensions: dict[str, int]) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict) or "id" not in record or "model" not in record:
            raise ValueError(f"{path}:{line_number}: expected id and model")
        if str(record["model"]) != model_id:
            raise ValueError(f"{path}:{line_number}: expected model {model_id!r}, "
                             f"got {record['model']!r}")
        case_id = str(record["id"])
        if not case_id or case_id in seen:
            raise ValueError(f"{path}:{line_number}: empty or duplicate id {case_id!r}")
        seen.add(case_id)
        embedding_obj = record.get("embeddings")
        if not isinstance(embedding_obj, dict):
            raise ValueError(f"{path}:{line_number}: expected nested embeddings object")
        for task, key in feature_keys.items():
            if key not in embedding_obj:
                raise ValueError(f"{path}:{line_number}: missing embeddings.{key} for {task}")
            vector = np.asarray(embedding_obj[key], dtype=np.float64)
            dimension = dimensions.get(key)
            if vector.ndim != 1 or vector.size != dimension or not np.isfinite(vector).all():
                raise ValueError(f"{path}:{line_number}: embeddings.{key} must be a finite "
                                 f"{dimension}-D vector")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no embedding records")
    return records


def validate_manifest(path: Path, model_id: str, expected: dict[str, Any],
                      expected_dimensions: dict[str, int]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required exporter manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"{path}: missing manifest fields {missing}")
    if manifest["model"] != model_id:
        raise ValueError(f"{path}: manifest model {manifest['model']!r} != artifact model {model_id!r}")
    mismatches = [key for key in MANIFEST_KEYS
                  if manifest.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(f"{path}: exporter/trained-probe provenance differs in {mismatches}")
    if manifest["embedding_dimensions"] != 1024 or any(
        value != manifest["embedding_dimensions"] for value in expected_dimensions.values()
    ):
        raise ValueError(f"{path}: embedding dimension does not match trained probe artifacts")
    return manifest


def probabilities(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Recreate sklearn LogisticRegression predict_proba from exported parameters."""
    mean = np.asarray(model["feature_scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scaler_scale"], dtype=np.float64)
    coef = np.asarray(model["coefficients"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    if mean.ndim != 1 or scale.shape != mean.shape or coef.ndim != 2:
        raise ValueError("invalid saved scaler/coefficient shape")
    if coef.shape[1] != mean.size or intercept.shape != (coef.shape[0],):
        raise ValueError("saved coefficient/intercept dimensions disagree with scaler")
    scaled = (x - mean) / scale
    if not np.isfinite(scaled).all():
        raise FloatingPointError("scaled feature matrix contains non-finite values")
    # The macOS OpenBLAS build used for the locked run spuriously warns on otherwise
    # finite matrix products; mirror the trainer's narrow suppression and validate output.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                                category=RuntimeWarning)
        scores = scaled @ coef.T + intercept
    if not np.isfinite(scores).all():
        raise FloatingPointError("inference produced non-finite classifier scores")
    classes = model["classes"]
    if len(classes) == 2 and scores.shape[1] == 1:
        # sklearn encodes binary logistic regression as [negative_score, positive_score].
        positive = np.exp(-np.logaddexp(0.0, -scores[:, 0]))
        result = np.column_stack((1.0 - positive, positive))
    else:
        if scores.shape[1] != len(classes):
            raise ValueError("saved multinomial coefficient count does not match classes")
        exponentials = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        result = exponentials / exponentials.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-12):
        raise FloatingPointError("inference returned invalid probability rows")
    return result


def infer(records: list[dict[str, Any]], models: dict[str, Any],
          feature_keys: dict[str, str]) -> list[dict[str, Any]]:
    result = [{"id": str(record["id"]), "model": str(record["model"])} for record in records]
    for task in TASKS:
        model = models[task]
        key = feature_keys[task]
        x = np.stack([np.asarray(record["embeddings"][key], dtype=np.float64) for record in records])
        probs = probabilities(x, model)
        classes = model["classes"]
        for index, row in enumerate(result):
            class_index = int(np.argmax(probs[index]))
            row[task] = {
                "label": str(classes[class_index]),
                "probabilities": {str(classes[i]): float(probs[index, i])
                                  for i in range(len(classes))},
                "confidence": float(probs[index, class_index]),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="one trained model/mode directory containing metadata.json and models.json")
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="new unlabeled exporter JSONL with nested task embeddings")
    parser.add_argument("--manifest", type=Path,
                        help="exporter sidecar manifest; defaults to EMBEDDINGS.manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="output predictions JSONL")
    args = parser.parse_args()

    metadata_path = args.artifacts / "metadata.json"
    models_path = args.artifacts / "models.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    models = json.loads(models_path.read_text(encoding="utf-8"))
    model_id = metadata["model"]
    expected_manifests = metadata.get("embedding_manifests", {})
    expected = expected_manifests.get("train", {}).get("contents")
    if not isinstance(expected, dict):
        raise ValueError("trained metadata does not include the training exporter manifest")
    manifest_path = args.manifest or Path(str(args.embeddings) + ".manifest.json")
    manifest = validate_manifest(manifest_path, model_id, expected,
                                 metadata["embedding_dimensions"])
    feature_keys = metadata["feature_keys_by_task"]
    if set(feature_keys) != set(TASKS) or set(models) != set(TASKS):
        raise ValueError("trained artifacts must contain owner, urgent, and impact classifiers")
    for task in TASKS:
        if models[task].get("embedding_key") != feature_keys[task]:
            raise ValueError(f"{task}: model and metadata feature keys disagree")
        c_value = float(models[task].get("C", 0))
        if not np.isfinite(c_value) or c_value <= 0:
            raise ValueError(f"{task}: invalid saved regularization value")
        if not models[task].get("classes") or not models[task].get("coefficients"):
            raise ValueError(f"{task}: incomplete fitted model parameters")
    records = read_records(args.embeddings, model_id, feature_keys,
                           metadata["embedding_dimensions"])
    predictions = infer(records, models, feature_keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                         for row in predictions), encoding="utf-8")
    output_metadata = {
        "artifact_directory": str(args.artifacts), "model": model_id,
        "feature_mode": metadata["feature_mode"], "feature_keys_by_task": feature_keys,
        "case_count": len(records), "input_sha256": sha256(args.embeddings),
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "manifest": manifest, "artifacts_sha256": {
            str(metadata_path): sha256(metadata_path), str(models_path): sha256(models_path)},
        "predictor_script_sha256": sha256(Path(__file__)),
        "output_sha256": sha256(args.out),
        "inference": "saved coefficients/scaler only; no fitting, labels, references, or tuning",
    }
    Path(str(args.out) + ".metadata.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
