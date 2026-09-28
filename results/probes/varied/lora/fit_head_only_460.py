#!/usr/bin/env python3
"""Fit and hash the train-only C=1 typed head on the LoRA run's exact 460 indices.

CPU-only: consumes cached training embeddings and references from clean300/multi120
training fixtures. It never reads dev targets, the sealed holdout, or any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "results/probes/varied/lora/SELECTION-FREEZE.json"
FREEZE_SHA256 = "5f7dd19638df1f7bdcb56ecd44a9371ae5a0ac32a6314538146e324ac8a2e48f"
TRAINER_DIR = Path(__file__).resolve().parent
MODELS = {
    "qwen": ("Qwen/Qwen3-Embedding-0.6B", "multi-qwen"),
    "harrier": ("microsoft/harrier-oss-v1-0.6b", "multi-harrier"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_index(record: dict[str, Any], name: str, question: dict[str, Any], multi: bool,
                 labels: list[str]) -> int:
    ref = record["reference"][name] if multi else record["reference"]
    target = ref["target"]
    kind = question["type"]
    if kind == "choice":
        gold = str(target["selected"])
    elif kind == "noul":
        if type(target.get("value")) is not bool: raise ValueError("noul train target must be bool")
        gold = "supported" if target["value"] else "unsupported"
    else:
        gold = str(target["label"])
    if gold not in labels:
        raise ValueError(f"{record['id']}/{name}: target does not match candidate keys")
    return labels.index(gold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/probes/varied/lora/eval/fresh60/headonly-same460")
    args = parser.parse_args()
    freeze_hash = sha(FREEZE)
    if freeze_hash != FREEZE_SHA256:
        raise ValueError("train/dev selection-freeze hash mismatch")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "TRAIN_DEV_SELECTION_FROZEN" or freeze["shared"]["holdout"].get("labels_read") is not False:
        raise ValueError("freeze manifest does not establish train-only selection and sealed holdout")
    model_id, export_name = MODELS[args.model]
    record = freeze["models"][args.model]
    if record.get("model_id") != model_id or record.get("status") != "train_dev_complete":
        raise ValueError("model run is not the hash-frozen completed run")
    run_dir = ROOT / record["run_dir"]
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha(config_path) != record["artifacts"]["config.json"]["sha256"]:
        raise ValueError("frozen trainer config changed")
    train_indices = [int(i) for i in config["train_indices"]]
    dev_indices = [int(i) for i in config["dev_indices"]]
    if (len(train_indices) != 460 or len(dev_indices) != 200 or set(train_indices) & set(dev_indices) or
            set(train_indices) | set(dev_indices) != set(range(660))):
        raise ValueError("expected the exact grouped 460-train/200-dev split from frozen config")
    train_sources = freeze["shared"]["training_fixtures"]
    clean_path = Path(train_sources["clean300"]["path"])
    multi_path = ROOT / train_sources["multi_train120"]["path"]
    if sha(clean_path) != config["source_sha256"]["clean"] or sha(multi_path) != config["source_sha256"]["multi_train"]:
        raise ValueError("training fixture hash differs from frozen trainer source hashes")
    feature_dir = ROOT / "results/probes/varied/runs" / export_name
    feature_paths = [feature_dir / "train-clean-300.features.jsonl", feature_dir / "train-120.features.jsonl"]
    for feature_path, hash_key in zip(feature_paths, ("clean_feature_text_source", "multi_feature_text_source")):
        if sha(feature_path) != config["source_sha256"][hash_key]:
            raise ValueError(f"training feature source hash differs: {feature_path}")

    # Recreate only the ordered target/candidate metadata. Reference targets are
    # dereferenced solely for indices in train_indices; no dev targets enter the fit.
    records = read_jsonl(clean_path) + read_jsonl(multi_path)
    multi_ids = {str(row["id"]) for row in records[300:]}
    ordered: list[dict[str, Any]] = []
    for row in records:
        is_multi = str(row["id"]) in multi_ids
        for name, question in row["request"]["questions"].items():
            kind = question["type"]
            labels = (list(question["options"]) if kind == "choice" else
                      ["unsupported", "supported"] if kind == "noul" else
                      [str(level["label"]) for level in question["levels"]])
            ordered.append({"id": str(row["id"]), "name": str(name), "kind": kind,
                            "labels": [str(label) for label in labels], "question": question,
                            "record": row, "multi": is_multi})
    if len(ordered) != 660:
        raise ValueError(f"expected 660 ordered training questions, got {len(ordered)}")

    import numpy as np
    sys.path.insert(0, str(ROOT / "results/probes/varied"))
    import multi_train_pairwise as pairwise
    train_set = set(train_indices)
    selected_by_key = {(ordered[index]["id"], ordered[index]["name"]): index for index in train_indices}
    features_by_index: dict[int, dict[str, Any]] = {}
    seen_feature_keys: set[tuple[str, str]] = set()
    feature_count = 0
    for feature_path in feature_paths:
        with feature_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("id")), str(row.get("question_key")))
                if key in seen_feature_keys:
                    raise ValueError(f"duplicate feature key {key}")
                seen_feature_keys.add(key); feature_count += 1
                if key in selected_by_key:
                    train_index = selected_by_key[key]
                    wanted = ordered[train_index]
                    query = np.asarray(row["query_embedding"], dtype=np.float64)
                    candidates = {str(c["candidate_id"]): np.asarray(c["embedding"], dtype=np.float64)
                                  for c in row["candidates"]}
                    if set(candidates) != set(wanted["labels"]):
                        raise ValueError(f"candidate IDs differ for frozen train index {global_index}")
                    gold_index = target_index(wanted["record"], wanted["name"], wanted["question"],
                                              wanted["multi"], wanted["labels"])
                    case = {"id": wanted["id"], "question_name": wanted["name"],
                            "kind": wanted["kind"], "keys": wanted["labels"], "target": gold_index,
                            "x": np.stack([pairwise.pair_features(query, candidates[k]) for k in wanted["labels"]])}
                    features_by_index[train_index] = case
    if feature_count != 660 or set(features_by_index) != train_set:
        raise ValueError(f"feature files do not exactly cover frozen 660 rows / 460 fit indices: {feature_count}")

    type_counts = Counter(features_by_index[i]["kind"] for i in train_indices)
    scorer_docs: dict[str, Any] = {}
    for kind in ("choice", "noul", "score"):
        fit_cases = [features_by_index[i] for i in train_indices if features_by_index[i]["kind"] == kind]
        targets = [case["target"] for case in fit_cases]
        weights, mean, aux = pairwise.fit_bounded(fit_cases, targets, 1.0)
        scorer_docs[kind] = {"coefficients": weights.tolist(), "feature_mean": mean.tolist(),
            "feature_scale": aux["scale"].tolist(),
            "fit_diagnostics": {key: value for key, value in aux.items() if key != "scale"}}
    if sum(type_counts.values()) != 460:
        raise AssertionError("fit did not consume exactly 460 train questions")

    out_root = args.out_root.resolve(); out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / args.model; run_dir.mkdir(parents=True, exist_ok=False)
    models_doc = {"model": model_id, "C": 1.0, "per_type_C": {kind: 1.0 for kind in scorer_docs},
                  "typed_scorers": scorer_docs,
                  "provenance": {"split": "frozen exact 460 train indices", "dev_indices_used": False,
                      "holdout_used": False, "C": 1.0}}
    models_path = run_dir / "models.json"
    models_path.write_text(json.dumps(models_doc, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    feature_manifests = [Path(str(p) + ".manifest.json") for p in feature_paths]
    text_manifest = json.loads(feature_manifests[1].read_text(encoding="utf-8"))
    metadata = {"model": model_id, "selection_id": "lora-same460-head-only-c1-train-only",
        "training_scope": "exact 460 config.train_indices only", "dev_targets_used": False,
        "holdout_accessed": False, "selection_or_sweep": "none; C=1.0 predeclared comparator",
        "C": 1.0, "training_question_count": 460, "training_questions_by_type": dict(type_counts),
        "train_indices_sha256": hashlib.sha256(json.dumps(train_indices, separators=(",", ":")).encode()).hexdigest(),
        "trainer_config_sha256": sha(config_path), "trainer_sha256": config["trainer_sha256"],
        "training_feature_sha256": [sha(p) for p in feature_paths],
        "training_manifest_sha256": [sha(p) for p in feature_manifests],
        "train_fixture_sha256": [sha(clean_path), sha(multi_path)],
        "profile_sha256": text_manifest["profile_sha256"], "profile_version": text_manifest["profile_version"],
        "text_version": text_manifest["text_version"], "fit_method": "bounded per-type pairwise softmax fit, same implementation as train-only comparator",
        "metadata_note": "No dev targets, sealed holdout, or test predictions were read/used.",
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    frozen_config = {"selection_id": metadata["selection_id"], "model": model_id, "C": 1.0,
                     "train_indices_sha256": metadata["train_indices_sha256"], "train_question_count": 460,
                     "dev_indices_used": False, "holdout_used": False, "profile_sha256": text_manifest["profile_sha256"]}
    frozen_config_path = run_dir / "frozen-config.json"
    frozen_config_path.write_text(json.dumps(frozen_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection = out_root / "selection-freeze.json"
    doc = json.loads(selection.read_text()) if selection.exists() else {"selection_status": "frozen", "approved_runs": {}}
    key = args.model
    doc["approved_runs"][key] = {"model": model_id, "run_dir": str(run_dir.resolve()),
        "models_sha256": sha(models_path), "metadata_sha256": sha(metadata_path),
        "frozen_config_sha256": sha(frozen_config_path), "selection_status": "frozen", "C": 1.0}
    selection.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model": model_id, "run_dir": str(run_dir), "selected_train_questions": 460,
        "type_counts": dict(type_counts), "C": 1.0, "models_sha256": sha(models_path),
        "metadata_sha256": sha(metadata_path), "selection_freeze": str(selection)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
