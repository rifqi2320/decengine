#!/usr/bin/env python3
"""Deferred, hash-gated evaluation of a frozen LoRA typed scorer.

Importing this module and --self-test are CPU-only and use synthetic arrays. The
benchmark fixture and MLX are touched only by the normal CLI after a freeze
attestation has been validated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "benchmarks/cases/varied/multi/test-lora-fresh-60.jsonl"
FIXTURE_SHA256 = "2848d28c79120c2fcf3ab49b755c402386f60ff96d0ac69dce801a606df44629"
ADAPTER = Path(__file__).with_name("checkpoint_adapter.py")
TRAINER = Path(__file__).with_name("train.py")
TEXT_COMPILER = ROOT / "results/probes/varied/exporter.rs"
TYPES = ("choice", "noul", "score")
TEXT_VERSION = "decengine-varied-pair-text-v2-generic-types"
MODEL_FEATURE_MANIFESTS = {
    "Qwen/Qwen3-Embedding-0.6B": ROOT / "results/probes/varied/runs/multi-qwen/train-120.features.jsonl",
    "microsoft/harrier-oss-v1-0.6b": ROOT / "results/probes/varied/runs/multi-harrier/train-120.features.jsonl",
}
MODEL_PROFILE_MANIFESTS = {
    "qwen": ROOT / "models/manifests/qwen3-embedding-0.6b.json",
    "harrier": ROOT / "models/manifests/harrier-oss-v1-0.6b.json",
}
FREEZE_DEFAULT = Path(__file__).with_name("SELECTION-FREEZE.json")
FREEZE_SHA256 = "5f7dd19638df1f7bdcb56ecd44a9371ae5a0ac32a6314538146e324ac8a2e48f"
TRAINING_TOKEN_LIMIT = 288
AUTHORIZED_INFERENCE_TOKEN_LIMIT = 512
MATCHED_FREEZE_DEFAULT = Path(__file__).with_name("experiment-matched-head") / "TRAIN-ONLY-FREEZE.json"
MATCHED_FREEZE_SHA256 = "ecb46f3a952d48a78a9004ad0213e70e57da0e9bd84e183fda31a5b63186eb1f"
MATCHED_ARMS = {
    "qwen-head": {"model_key": "qwen", "arm": "head", "model": "Qwen/Qwen3-Embedding-0.6B", "epoch": 2},
    "qwen-lora": {"model_key": "qwen", "arm": "lora", "model": "Qwen/Qwen3-Embedding-0.6B", "epoch": 9},
    "harrier-head": {"model_key": "harrier", "arm": "head", "model": "microsoft/harrier-oss-v1-0.6b", "epoch": 9},
    "harrier-lora": {"model_key": "harrier", "arm": "lora", "model": "microsoft/harrier-oss-v1-0.6b", "epoch": 9},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile_from_bundled_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    source = json.loads(path.read_text(encoding="utf-8"))
    # Match ModelProfile serde field order and defaults from decengine-models.
    source.update({
        "query_instruction_template": source.get("query_instruction_template", "{prompt}"),
        "choice_candidate_template": source.get("choice_candidate_template", "{label}: {criterion}"),
        "score_candidate_template": source.get("score_candidate_template", "{label}: {criterion}"),
        "noul_unsupported": source.get("noul_unsupported", "unsupported false no absent not required can wait"),
        "noul_supported": source.get("noul_supported", "supported true yes present urgent today immediate required"),
    })
    order = ("schema_version", "id", "family", "architecture", "pooling", "normalize", "padding_side",
             "query_template", "candidate_template", "query_instruction_template", "choice_candidate_template",
             "score_candidate_template", "noul_unsupported", "noul_supported", "temperature", "max_length",
             "profile_version", "required_files")
    profile = {key: source[key] for key in order}
    raw = json.dumps(profile, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return profile, raw


def verify_freeze_manifest(path: Path, expected_sha256: str = FREEZE_SHA256) -> tuple[dict[str, Any], str]:
    actual = digest(path)
    if actual != expected_sha256:
        raise ValueError(f"selection freeze manifest hash mismatch: {actual}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    holdout = doc.get("shared", {}).get("holdout", {})
    if (doc.get("schema_version") != 1 or doc.get("status") != "TRAIN_DEV_SELECTION_FROZEN" or
            doc.get("scope") != "train/dev only; no heldout/test labels, predictions, or hashes accessed" or
            holdout.get("status") != "SEALED_NOT_ACCESSED" or holdout.get("hash_read") is not False or
            holdout.get("labels_read") is not False or holdout.get("predictions_read") is not False):
        raise ValueError("selection freeze manifest schema/sealed holdout status is invalid")
    return doc, actual


def prepare_freeze_artifacts(freeze_path: Path, prepared_dir: Path) -> list[dict[str, str]]:
    freeze, freeze_sha = verify_freeze_manifest(freeze_path)
    fixture_manifest_path = FIXTURE.with_name("test-lora-fresh-60.manifest.json")
    fixture_manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    if fixture_manifest.get("sha256") != FIXTURE_SHA256:
        raise ValueError("sealed fixture manifest hash does not equal evaluator's pinned case hash")
    out = prepared_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    created = []
    for key, expected_model in (("qwen", "Qwen/Qwen3-Embedding-0.6B"),
                                ("harrier", "microsoft/harrier-oss-v1-0.6b")):
        record = freeze.get("models", {}).get(key)
        if not isinstance(record, dict) or record.get("model_id") != expected_model or record.get("status") != "train_dev_complete":
            raise ValueError(f"freeze record missing completed {key} train/dev selection")
        run_dir = ROOT / record["run_dir"]
        artifacts: dict[str, str] = {}
        for name in ("config.json", "best.safetensors", "best.json", "summary.json"):
            item = record.get("artifacts", {}).get(name)
            if not isinstance(item, dict):
                raise ValueError(f"freeze record missing {key} artifact {name}")
            artifact_path = ROOT / item["path"]
            actual = digest(artifact_path)
            if actual != item["sha256"] or artifact_path.stat().st_size != item["size_bytes"]:
                raise ValueError(f"{key} frozen artifact hash/size mismatch: {name}")
            artifacts[{"best.safetensors": "best_weights", "best.json": "best_record"}.get(name, name.removesuffix(".json"))] = actual
        # Freeze provenance must agree with the trainer files before an attestation is emitted.
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if (config.get("model") != expected_model or summary.get("status") != "train_dev_complete" or
                summary.get("steps") != 1380 or summary.get("best_epoch") != record.get("best_epoch") or
                best.get("epoch") != record.get("best_epoch") or best.get("dev", {}).get("loss") != record.get("best_dev", {}).get("loss") or
                record.get("selected_weight_hashes", {}).get("best.safetensors") != artifacts["best_weights"]):
            raise ValueError(f"{key} train summary/best/config differs from frozen selection manifest")
        for name, path_str in freeze.get("shared", {}).get("code_sha256", {}).items():
            if name in {"results/probes/varied/lora/checkpoint_adapter.py", "results/probes/varied/lora/train.py"}:
                if digest(ROOT / name) != path_str:
                    raise ValueError(f"frozen training source differs at {name}")
        base_manifest = record.get("base_model_manifest", {})
        base_manifest_path = ROOT / base_manifest.get("path", "")
        if digest(base_manifest_path) != base_manifest.get("sha256"):
            raise ValueError(f"{key} bundled model profile source differs from freeze record")
        profile, profile_bytes = profile_from_bundled_manifest(base_manifest_path)
        text_profile = record.get("text_profile", {})
        feature = record.get("feature_exports", {}).get("multi120", {})
        profile_sha = hashlib.sha256(profile_bytes).hexdigest()
        profile_identity = {"model": expected_model, "profile_version": profile["profile_version"],
                            "profile_sha256": profile_sha, "text_version": TEXT_VERSION}
        if (profile_sha != text_profile.get("profile_sha256") or profile_sha != feature.get("profile_sha256") or
                str(profile["profile_version"]) != str(text_profile.get("profile_version")) or
                feature.get("text_version") != TEXT_VERSION):
            raise ValueError(f"{key} resolved profile does not reproduce the actual train-export manifest")
        profile_path = out / f"{key}-profile.json"
        profile_path.write_bytes(profile_bytes)
        att = {
            "schema_version": 1, "model": expected_model, "run_id": run_dir.name,
            "train_only_selection_frozen": True, "reference_accessed": False,
            "fixture_sha256": FIXTURE_SHA256, "source_freeze_manifest_sha256": freeze_sha,
            "text_compiler_sha256": digest(TEXT_COMPILER),
            "evaluator_sha256": digest(Path(__file__).resolve()),
            "comparison_script_sha256": digest(Path(__file__).with_name("compare_frozen.py")),
            "train_only_selection": {
                "method": record["selection_rule"], "scope": freeze["scope"], "status": record["status"],
                "heldout_accessed": False, "completed_steps": summary["steps"],
                "selected_epoch": record["best_epoch"], "best_dev_loss": record["best_dev"]["loss"],
            },
            "training_profile": profile_identity,
            "sha256": {
                "config": artifacts["config"], "best_weights": artifacts["best_weights"],
                "best_record": artifacts["best_record"], "summary": artifacts["summary"],
            },
        }
        att_path = out / f"{key}-freeze-attestation.json"
        att_path.write_text(json.dumps(att, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        created.append({"model": key, "profile": str(profile_path), "attestation": str(att_path)})
    return created


def prepare_inference_policy(freeze_path: Path, policy_path: Path, attestations_dir: Path) -> dict[str, Any]:
    freeze, freeze_sha = verify_freeze_manifest(freeze_path)
    fixture_manifest_path = FIXTURE.with_name("test-lora-fresh-60.manifest.json")
    fixture_manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    if fixture_manifest.get("sha256") != FIXTURE_SHA256:
        raise ValueError("sealed fixture manifest hash mismatch")
    policy = {
        "schema_version": 1,
        "policy_id": "lora-fresh60-inference-extrapolation-512-v1",
        "source_freeze_manifest_sha256": freeze_sha,
        "fixture_sha256": FIXTURE_SHA256,
        "training_policy": {"max_tokens_per_text": TRAINING_TOKEN_LIMIT,
            "unchanged": True, "trainer_config_unchanged": True, "selected_weights_unchanged": True},
        "inference_policy": {"max_tokens_per_text": AUTHORIZED_INFERENCE_TOKEN_LIMIT,
            "max_profile_length": 32768, "truncate": False, "reject_above_limit": True,
            "authorize_unseen_length_extrapolation": True,
            "reason": "Explicit user instruction: preserve complete fresh60 inputs; permit inference-only limit 512, with separate hash-bound policy; do not alter training guard, profile, or weights."},
        "profile_identities": {key: {"model": rec["model_id"], "profile_version": rec["text_profile"]["profile_version"],
            "profile_sha256": rec["text_profile"]["profile_sha256"], "text_version": rec["text_profile"]["text_version"]}
            for key, rec in freeze["models"].items()},
        "best_weights_sha256": {key: rec["selected_weight_hashes"]["best.safetensors"]
            for key, rec in freeze["models"].items()},
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    policy_sha = digest(policy_path)
    for key in ("qwen", "harrier"):
        att_path = attestations_dir / f"{key}-freeze-attestation.json"
        att = json.loads(att_path.read_text(encoding="utf-8"))
        if att.get("source_freeze_manifest_sha256") != freeze_sha:
            raise ValueError(f"{key}: evaluator attestation is not bound to verified train freeze")
        att["evaluation_policy_sha256"] = policy_sha
        att["evaluation_policy_id"] = policy["policy_id"]
        att_path.write_text(json.dumps(att, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"policy_path": str(policy_path), "policy_sha256": policy_sha,
            "attestations": [str(attestations_dir / f"{k}-freeze-attestation.json") for k in ("qwen", "harrier")]}


def require_hash(path: Path, expected: str, label: str) -> str:
    if not expected or digest(path) != expected.removeprefix("sha256:"):
        raise ValueError(f"{label} SHA-256 missing or mismatch: {path}")
    return digest(path)


def resource_snapshot(mx: Any = None) -> dict[str, Any]:
    proc = subprocess.run(["footprint", "--format", "bytes", "--noCategories", str(os.getpid())],
                         text=True, capture_output=True, check=True)
    current = re.search(r"Footprint:\s+([0-9,]+)\s+B", proc.stdout)
    peak = re.search(r"phys_footprint_peak:\s+([0-9,]+)\s+B", proc.stdout)
    if not current or not peak:
        raise RuntimeError(f"unable to read macOS process footprint: {proc.stdout[:300]}")
    swap_text = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout
    swap = re.search(r"used\s*=\s*([0-9.]+)M", swap_text)
    if not swap:
        raise RuntimeError(f"unable to read system swap usage: {swap_text}")
    return {"pid": os.getpid(), "process_footprint_bytes": int(current.group(1).replace(",", "")),
            "process_peak_footprint_bytes": int(peak.group(1).replace(",", "")),
            "system_swap_used_bytes": int(float(swap.group(1)) * 1024 ** 2),
            "mlx_active_bytes": int(mx.get_active_memory()) if mx is not None else None,
            "mlx_peak_bytes": int(mx.get_peak_memory()) if mx is not None else None}


def guard_resources(snapshot: dict[str, Any], baseline_swap: int, outdir: Path) -> None:
    limit6, limit8 = 6 * 1024 ** 3, 8 * 1024 ** 3
    violation = None
    if snapshot["process_peak_footprint_bytes"] >= limit8:
        violation = "absolute 8 GiB process peak cap reached"
    elif snapshot["process_peak_footprint_bytes"] >= limit6:
        violation = "6 GiB process-footprint early-stop threshold reached"
    elif snapshot["system_swap_used_bytes"] > baseline_swap:
        violation = "system swap increased above evaluation baseline"
    if violation:
        marker = {"status": "aborted_resource_guard", "reason": violation,
                  "baseline_swap_bytes": baseline_swap, **snapshot}
        (outdir / "STOPPED_RESOURCE_GUARD.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        raise MemoryError(f"{violation}: {snapshot}")


def guard_matched_resources(snapshot: dict[str, Any], baseline_swap: int, outdir: Path,
                             policy: dict[str, Any]) -> None:
    """Apply the new, explicit matched-eval budget without relaxing training/legacy guards."""
    limits = policy.get("resource_guard", {})
    growth = max(0, int(snapshot["system_swap_used_bytes"]) - int(baseline_swap))
    process_peak = int(snapshot["process_peak_footprint_bytes"])
    combined = process_peak + growth
    reason = None
    if process_peak >= int(limits.get("absolute_process_stop_bytes", 8 * 1024**3)):
        reason = "absolute 8 GiB process cap reached"
    elif process_peak >= int(limits.get("process_early_stop_bytes", 6 * 1024**3)):
        reason = "6 GiB process early-stop threshold reached"
    elif combined >= int(limits.get("process_peak_plus_positive_swap_growth_stop_bytes", 10 * 1024**3)):
        reason = "10 GiB process-peak plus positive swap-growth hard ceiling reached"
    if reason:
        marker = {"status": "aborted_resource_guard", "reason": reason,
                  "baseline_swap_bytes": int(baseline_swap), "positive_swap_growth_bytes": growth,
                  "process_peak_plus_swap_growth_bytes": combined, **snapshot}
        (outdir / "STOPPED_RESOURCE_GUARD.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        raise MemoryError(f"{reason}: {marker}")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def softmax(values: Any) -> list[float]:
    vals = [float(x) for x in values]
    if not vals or not all(math.isfinite(x) for x in vals):
        raise ValueError("logits must be a nonempty finite vector")
    top = max(vals)
    exp = [math.exp(x - top) for x in vals]
    total = sum(exp)
    return [x / total for x in exp]


def typed_prediction(kind: str, labels: list[str], logits: Any) -> tuple[Any, list[float]]:
    probs = softmax(logits)
    if len(labels) != len(probs):
        raise ValueError("typed head output width differs from ordered candidates")
    winner = max(range(len(probs)), key=probs.__getitem__)
    if sum(x == probs[winner] for x in probs) != 1:
        raise ValueError("tied maximum; refusing ambiguous typed prediction")
    return (bool(winner) if kind == "noul" else labels[winner]), probs


KIND_INSTRUCTIONS = {
    "choice": "select the single best-matching option from the reported issue and candidate scopes; do not infer facts absent from the case",
    "noul": "decide whether the proposition asked about is supported by the reported state; evaluate the complete proposition from evidence and do not infer facts absent from the case",
    "score": "select the rubric level whose criterion best matches the reported state and requested rating; use the level definitions and do not infer facts absent from the case",
}


def replace(template: str, key: str, value: str) -> str:
    return template.replace(key, value)


def compile_training_text(profile: dict[str, Any], state: dict[str, Any], question: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Match results/probes/varied/exporter.rs v2-generic text used by the LoRA train features."""
    kind, prompt = question["type"], question["prompt"]
    if kind not in KIND_INSTRUCTIONS:
        raise ValueError(f"unsupported wire type {kind!r}")
    state_text = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    raw = f"Decision: {prompt}\nState: {state_text}"
    instruction = replace(replace(profile["query_instruction_template"], "{prompt}", prompt),
                          "{kind}", KIND_INSTRUCTIONS[kind])
    query = replace(replace(profile["query_template"], "{instruction}", instruction), "{text}", raw)
    labels: list[str] = []
    candidate_texts: list[str] = []
    if kind == "choice":
        candidates = [(str(label), replace(replace(profile["choice_candidate_template"], "{label}", str(label)),
                                           "{criterion}", str(criterion)))
                      for label, criterion in question["options"].items()]
    elif kind == "score":
        candidates = [(str(level["label"]), replace(replace(profile["score_candidate_template"], "{label}", str(level["label"])),
                                                      "{criterion}", str(level["criterion"])))
                      for level in question["levels"]]
    else:
        candidates = [
            ("unsupported", f"For the question ‘{prompt}’, the proposition is not supported by the reported state; the evidence does not establish that this outcome applies."),
            ("supported", f"For the question ‘{prompt}’, the proposition is supported by the reported state; the evidence establishes that this outcome applies."),
        ]
    for label, text in candidates:
        labels.append(label)
        candidate_texts.append(replace(profile["candidate_template"], "{text}", text))
    if not query or not labels or any(not x for x in candidate_texts):
        raise ValueError("profile compiler generated an empty text or candidate list")
    return query, labels, candidate_texts


def validate_checkpoint_header(tensors: dict[str, dict[str, Any]], layer: int,
                               hidden: int, q_width: int, v_width: int, head_width: int = 192) -> None:
    """Validate the trainer's exact LoRA+TypedScorer state dict before MLX loading."""
    prefix = f"backbone.model.layers.{layer}.self_attn."
    expected = {
        prefix + "q_proj.lora_a": (2, hidden), prefix + "q_proj.lora_b": (q_width, 2),
        prefix + "v_proj.lora_a": (2, hidden), prefix + "v_proj.lora_b": (v_width, 2),
        "head.proj.weight": (head_width, hidden * 4), "head.proj.bias": (head_width,),
        "head.choice.weight": (1, head_width), "head.choice.bias": (1,),
        "head.noul.weight": (1, head_width), "head.noul.bias": (1,),
        "head.score.weight": (1, head_width), "head.score.bias": (1,),
    }
    if set(tensors) != set(expected):
        raise ValueError(f"best.safetensors key set differs from trainer LoRA+typed-head schema: {set(tensors) ^ set(expected)}")
    for name, shape in expected.items():
        tensor = tensors[name]
        if tensor.get("dtype") != "F32" or tuple(tensor.get("shape", ())) != shape:
            raise ValueError(f"best.safetensors tensor schema mismatch at {name}: {tensor}")


def synthetic_self_test() -> None:
    # No paths, fixture, MLX, tokenizer, checkpoint, or benchmark labels are read.
    assert typed_prediction("choice", ["left", "right"], [0.0, 2.0])[0] == "right"
    assert typed_prediction("noul", ["unsupported", "supported"], [2.0, 0.0])[0] is False
    assert typed_prediction("noul", ["unsupported", "supported"], [0.0, 2.0])[0] is True
    assert typed_prediction("score", ["low", "mid", "high"], [0.0, 1.0, 0.0])[0] == "mid"
    try:
        typed_prediction("score", ["a", "b"], [1.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("tie must be rejected")
    profile = {"query_instruction_template": "{prompt} :: {kind}", "query_template": "Q[{instruction}] {text}",
               "candidate_template": "C[{text}]", "choice_candidate_template": "{label}={criterion}",
               "score_candidate_template": "{label}={criterion}"}
    query, labels, texts = compile_training_text(profile, {"z": 1, "a": True},
        {"type": "choice", "prompt": "pick", "options": {"b": "B", "a": "A"}})
    assert query.startswith("Q[pick :: select the single best-matching") and labels == ["b", "a"]
    assert texts == ["C[b=B]", "C[a=A]"] and '"a":true,"z":1' in query
    _, score_labels, score_texts = compile_training_text(profile, {}, {"type": "score", "prompt": "rate",
        "levels": [{"label": "low", "criterion": "L"}, {"label": "high", "criterion": "H"}]})
    assert score_labels == ["low", "high"] and score_texts == ["C[low=L]", "C[high=H]"]
    _, bool_labels, bool_texts = compile_training_text(profile, {}, {"type": "noul", "prompt": "act now?"})
    assert bool_labels == ["unsupported", "supported"] and "act now?" in bool_texts[0] and "act now?" in bool_texts[1]
    sample = {"backbone.model.layers.27.self_attn.q_proj.lora_a": {"dtype": "F32", "shape": [2, 1024]},
              "backbone.model.layers.27.self_attn.q_proj.lora_b": {"dtype": "F32", "shape": [2048, 2]},
              "backbone.model.layers.27.self_attn.v_proj.lora_a": {"dtype": "F32", "shape": [2, 1024]},
              "backbone.model.layers.27.self_attn.v_proj.lora_b": {"dtype": "F32", "shape": [1024, 2]},
              "head.proj.weight": {"dtype": "F32", "shape": [192, 4096]},
              "head.proj.bias": {"dtype": "F32", "shape": [192]}}
    for layer_name in ("choice", "noul", "score"):
        sample[f"head.{layer_name}.weight"] = {"dtype": "F32", "shape": [1, 192]}
        sample[f"head.{layer_name}.bias"] = {"dtype": "F32", "shape": [1]}
    validate_checkpoint_header(sample, 27, 1024, 2048, 1024)
    invalid_header = dict(sample)
    invalid_header["head.score.weight"] = {"dtype": "F16", "shape": [1, 192]}
    try:
        validate_checkpoint_header(invalid_header, 27, 1024, 2048, 1024)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid safetensors dtype must be rejected")
    print("synthetic typed-array self-test passed; no fixture/model/MLX access")


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected object")
                yield row


def request_only_jsonl(path: Path):
    """Parse ID/domain/request while lexically skipping references and all other fields."""
    decoder = json.JSONDecoder()

    def ws(text: str, pos: int) -> int:
        while pos < len(text) and text[pos] in " \t\r\n": pos += 1
        return pos

    def skip_value(text: str, pos: int) -> int:
        pos = ws(text, pos)
        if pos >= len(text): raise ValueError("truncated JSON value")
        if text[pos] == '"':
            i = pos + 1; escaped = False
            while i < len(text):
                ch = text[i]
                if escaped: escaped = False
                elif ch == "\\": escaped = True
                elif ch == '"': return i + 1
                i += 1
            raise ValueError("unterminated skipped JSON string")
        if text[pos] in "[{":
            stack = ["]" if text[pos] == "[" else "}"]
            i = pos + 1
            while i < len(text) and stack:
                ch = text[i]
                if ch == '"':
                    i = skip_value(text, i); continue
                if ch == "[": stack.append("]")
                elif ch == "{": stack.append("}")
                elif ch in "]}":
                    if not stack or ch != stack[-1]: raise ValueError("mismatched JSON delimiter")
                    stack.pop()
                i += 1
            if stack: raise ValueError("unterminated skipped JSON container")
            return i
        i = pos
        while i < len(text) and text[i] not in ",} \t\r\n": i += 1
        if i == pos: raise ValueError("empty primitive JSON value")
        return i

    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip(): continue
            i = ws(line, 0)
            if i >= len(line) or line[i] != "{": raise ValueError(f"{path}:{line_no}: expected top-level object")
            i += 1; fields: dict[str, Any] = {}
            while True:
                i = ws(line, i)
                if i < len(line) and line[i] == "}":
                    i += 1; break
                key, after_key = decoder.raw_decode(line, i)
                if not isinstance(key, str): raise ValueError(f"{path}:{line_no}: non-string object key")
                i = ws(line, after_key)
                if i >= len(line) or line[i] != ":": raise ValueError(f"{path}:{line_no}: missing colon")
                i = ws(line, i + 1)
                if key in {"id", "domain", "request"}:
                    fields[key], i = decoder.raw_decode(line, i)
                else:
                    i = skip_value(line, i)
                i = ws(line, i)
                if i < len(line) and line[i] == ",": i += 1; continue
                if i < len(line) and line[i] == "}": i += 1; break
                raise ValueError(f"{path}:{line_no}: expected comma or close brace")
            if line[i:].strip(): raise ValueError(f"{path}:{line_no}: trailing content")
            if set(fields) != {"id", "domain", "request"}:
                raise ValueError(f"{path}:{line_no}: missing request-only fields")
            yield fields


def verify_matched_freeze(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = digest(path)
    if actual != expected_sha256:
        raise ValueError(f"matched-head train-only freeze SHA-256 mismatch: {actual}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema_version") != 1 or doc.get("status") != "TRAIN_ONLY_SELECTION_FROZEN" or
            doc.get("scope") != "train/dev only; no heldout/test labels, predictions, or hashes accessed" or
            doc.get("holdout") != {"hashes_read": False, "labels_read": False,
                                  "predictions_read": False, "status": "SEALED_NOT_ACCESSED"}):
        raise ValueError("matched-head freeze schema/status/holdout fields are invalid")
    if set(doc.get("selected_checkpoints", {})) != {"qwen", "harrier"}:
        raise ValueError("matched-head freeze has unexpected model selections")
    return doc, actual


def prepare_matched_inference_policy(freeze_path: Path, policy_path: Path,
                                     expected_sha256: str = MATCHED_FREEZE_SHA256) -> dict[str, Any]:
    freeze, freeze_sha = verify_matched_freeze(freeze_path, expected_sha256)
    policy = {
        "schema_version": 1,
        "policy_id": "matched-head-fresh60-inference-extrapolation-512-total10g-v3",
        "freeze_manifest_sha256": freeze_sha,
        "fixture_path": str(FIXTURE.relative_to(ROOT)),
        "fixture_sha256": FIXTURE_SHA256,
        "evaluator_sha256": digest(Path(__file__).resolve()),
        "text_compiler_sha256": digest(TEXT_COMPILER),
        "matched_trainer_sha256": digest(MATCHED_FREEZE_DEFAULT.with_name("train_matched.py")),
        "shared_trainer_sha256": digest(TRAINER),
        "adapter_sha256": digest(ADAPTER),
        "training": {"max_tokens_per_text_guard": 288, "observed_train_max_tokens": 266,
                     "unchanged": True, "no_training_artifacts_changed": True},
        "inference": {"max_tokens_per_text": AUTHORIZED_INFERENCE_TOKEN_LIMIT,
                      "truncate": False, "reject_above_limit": True,
                      "unseen_length_extrapolation": True,
                      "fixture_max_tokens": 375,
                      "caveat": "Previously accessed/scored fixture reused; table is exploratory, with no test-driven tuning."},
        "resource_guard": {
            "process_early_stop_bytes": 6 * 1024**3,
            "absolute_process_stop_bytes": 8 * 1024**3,
            "process_peak_plus_positive_swap_growth_stop_bytes": 10 * 1024**3,
            "positive_swap_growth_only": True,
            "independent_swap_growth_ceiling": None,
            "sampling": "macOS process footprint and vm.swapusage; per question/case plus external 0.5s monitor",
        },
        "parity_baseline": {"run_key": "qwen-head-epoch02-swap512", "case_prefix_count": 7,
            "normalized_predictions_sha256": digest(MATCHED_FREEZE_DEFAULT.parent / "eval/fresh60/qwen-head-epoch02-swap512/predictions.jsonl"),
            "raw_predictions_sha256": digest(MATCHED_FREEZE_DEFAULT.parent / "eval/fresh60/qwen-head-epoch02-swap512/raw-predictions.jsonl"),
            "comparison": "exact candidate schema, selected values, logits and probabilities; timing fields excluded"},
        "selected": {run_key: {"model": MATCHED_ARMS[run_key]["model"],
                               "arm": MATCHED_ARMS[run_key]["arm"],
                               "epoch": MATCHED_ARMS[run_key]["epoch"],
                               "best_checkpoint_sha256": freeze["selected_checkpoints"][spec["model_key"]][
                                   "frozen_head" if spec["arm"] == "head" else "lora"]["checkpoint"]["sha256"]}
                     for run_key, spec in MATCHED_ARMS.items()},
        "selection_was_train_dev_only": True,
        "references_used_for_selection": False,
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return {"policy_path": str(policy_path), "policy_sha256": digest(policy_path),
            "freeze_manifest_sha256": freeze_sha}


def _matched_run_artifacts(freeze: dict[str, Any], run_key: str) -> dict[str, Any]:
    spec = MATCHED_ARMS[run_key]
    selection_key = "frozen_head" if spec["arm"] == "head" else "lora"
    selected = freeze["selected_checkpoints"][spec["model_key"]][selection_key]
    record = freeze["runs"][f"{spec['model_key']}_{spec['arm']}"]
    if (record.get("status") != "complete" or record.get("arm") != spec["arm"] or
            record.get("model") != spec["model"] or record.get("best_epoch") != spec["epoch"] or
            selected.get("epoch") != spec["epoch"] or record.get("artifacts", {}).get("best.safetensors", {}).get("sha256") !=
            selected.get("checkpoint", {}).get("sha256")):
        raise ValueError(f"{run_key}: completed run/selected checkpoint differs from explicit frozen selection")
    run_dir = ROOT / record["run_dir"]
    for name, item in record.get("shared_source_hashes", {}).items():
        source_path = Path(item["path"])
        if digest(source_path) != item["sha256"] or source_path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"{run_key}: shared frozen source hash/size mismatch: {name}")
    for name, path in (("config.json", run_dir / "config.json"),
                       ("best.json", run_dir / "best.json"),
                       ("summary.json", run_dir / "summary.json")):
        expected = record.get("config_sha256") if name == "config.json" else (
            record.get("best_json_sha256") if name == "best.json" else record.get("summary_sha256"))
        if not expected or digest(path) != expected:
            raise ValueError(f"{run_key}: frozen {name} hash mismatch")
    for name, item in record.get("artifacts", {}).items():
        artifact = ROOT / item["path"]
        if digest(artifact) != item["sha256"] or artifact.stat().st_size != item["size_bytes"]:
            raise ValueError(f"{run_key}: frozen artifact hash/size mismatch: {name}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if (config.get("model") != spec["model"] or config.get("arm") != spec["arm"] or
            config.get("profile_sha256") != record.get("profile", {}).get("profile_sha256") or
            config.get("max_tokens") != TRAINING_TOKEN_LIMIT or summary.get("status") != "complete" or
            summary.get("epochs") != 9 or summary.get("steps") != 4140 or
            summary.get("best_epoch") != spec["epoch"] or best.get("epoch") != spec["epoch"]):
        raise ValueError(f"{run_key}: config/summary/best checkpoint schema differs from selected run")
    return {"spec": spec, "record": record, "run_dir": run_dir, "config": config,
            "best": best, "summary": summary, "checkpoint": run_dir / "best.safetensors",
            "checkpoint_sha256": selected["checkpoint"]["sha256"]}


def verify_matched_parity_prefix(prefix_dir: Path, raw_rows: list[dict[str, Any]],
                                 prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact reference-free parity against the preserved first seven Qwen-head cases."""
    prior_raw = [json.loads(line) for line in (prefix_dir / "raw-predictions.jsonl").read_text().splitlines() if line.strip()]
    prior_predictions = [json.loads(line) for line in (prefix_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if len(prior_raw) != 21 or len(prior_predictions) != 21 or len(raw_rows) != 21 or len(prediction_rows) != 21:
        raise ValueError("Qwen-head parity check requires exactly the preserved 7x3 prefix")
    raw_fields = ("id", "question_name", "question_type", "candidate_keys", "token_lengths",
                  "raw_logits", "probabilities", "selected", "selected_candidate_index")
    pred_fields = ("id", "question_name", "question_type", "candidate_keys", "probabilities",
                   "selected", "expected_value", "selected_candidate_index", "run_config_sha256",
                   "model_or_adapter_sha256")
    mismatches = []
    for index, (old, new) in enumerate(zip(prior_raw, raw_rows)):
        for field in raw_fields:
            if old.get(field) != new.get(field):
                mismatches.append({"row": index, "view": "raw", "field": field})
    for index, (old, new) in enumerate(zip(prior_predictions, prediction_rows)):
        for field in pred_fields:
            if old.get(field) != new.get(field):
                mismatches.append({"row": index, "view": "normalized", "field": field})
    if mismatches:
        raise ValueError(f"Qwen-head 7-case parity mismatch against preserved output: {mismatches[:8]}")
    return {"status": "exact_first_7_cases_parity", "case_count": 7, "question_count": 21,
            "references_read": False, "raw_sha256": digest(prefix_dir / "raw-predictions.jsonl"),
            "predictions_sha256": digest(prefix_dir / "predictions.jsonl"),
            "comparison": "exact raw logits/probabilities/schema and normalized selections/probabilities/config/checkpoint provenance; timing ignored"}


def run_matched_frozen(args: argparse.Namespace) -> int:
    if args.matched_run_key not in MATCHED_ARMS:
        raise ValueError(f"unsupported matched run key: {args.matched_run_key}")
    freeze, freeze_sha = verify_matched_freeze(args.matched_freeze_manifest.resolve(), args.matched_freeze_sha256)
    if args.fixture.resolve() != FIXTURE.resolve() or digest(args.fixture) != FIXTURE_SHA256:
        raise ValueError("matched evaluation permits only the exact hash-pinned fresh60 fixture")
    manifest_path = args.fixture.with_name("test-lora-fresh-60.manifest.json")
    if json.loads(manifest_path.read_text(encoding="utf-8")).get("sha256") != FIXTURE_SHA256:
        raise ValueError("fresh60 fixture sidecar does not pin expected fixture hash")
    policy_path = args.matched_policy.resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_sha = digest(policy_path)
    if (policy.get("policy_id") != "matched-head-fresh60-inference-extrapolation-512-total10g-v3" or
            policy.get("freeze_manifest_sha256") != freeze_sha or
            policy.get("fixture_sha256") != FIXTURE_SHA256 or
            policy.get("evaluator_sha256") != digest(Path(__file__).resolve()) or
            policy.get("text_compiler_sha256") != digest(TEXT_COMPILER) or
            policy.get("matched_trainer_sha256") != digest(MATCHED_FREEZE_DEFAULT.with_name("train_matched.py")) or
            policy.get("shared_trainer_sha256") != digest(TRAINER) or
            policy.get("adapter_sha256") != digest(ADAPTER) or
            policy.get("training", {}).get("max_tokens_per_text_guard") != TRAINING_TOKEN_LIMIT or
            policy.get("training", {}).get("observed_train_max_tokens") != 266 or
            policy.get("inference", {}).get("max_tokens_per_text") != AUTHORIZED_INFERENCE_TOKEN_LIMIT or
            policy.get("inference", {}).get("truncate") is not False or
            policy.get("inference", {}).get("fixture_max_tokens") != 375 or
            policy.get("resource_guard") != {
                "process_early_stop_bytes": 6 * 1024**3,
                "absolute_process_stop_bytes": 8 * 1024**3,
                "process_peak_plus_positive_swap_growth_stop_bytes": 10 * 1024**3,
                "positive_swap_growth_only": True,
                "independent_swap_growth_ceiling": None,
                "sampling": "macOS process footprint and vm.swapusage; per question/case plus external 0.5s monitor"}):
        raise ValueError("matched inference policy is invalid or not tied to exact freeze/fixture")
    prior_prefix = policy.get("parity_baseline", {})
    if (prior_prefix.get("run_key") != "qwen-head-epoch02-swap512" or prior_prefix.get("case_prefix_count") != 7 or
            prior_prefix.get("normalized_predictions_sha256") != digest(MATCHED_FREEZE_DEFAULT.parent / "eval/fresh60/qwen-head-epoch02-swap512/predictions.jsonl") or
            prior_prefix.get("raw_predictions_sha256") != digest(MATCHED_FREEZE_DEFAULT.parent / "eval/fresh60/qwen-head-epoch02-swap512/raw-predictions.jsonl")):
        raise ValueError("swap512 policy does not pin preserved Qwen-head parity baseline")
    bundle = _matched_run_artifacts(freeze, args.matched_run_key)
    spec, record, run_dir = bundle["spec"], bundle["record"], bundle["run_dir"]
    config, best, summary = bundle["config"], bundle["best"], bundle["summary"]
    policy_selection = policy.get("selected", {}).get(args.matched_run_key, {})
    if (policy_selection.get("model") != spec["model"] or policy_selection.get("arm") != spec["arm"] or
            policy_selection.get("epoch") != spec["epoch"] or
            policy_selection.get("best_checkpoint_sha256") != bundle["checkpoint_sha256"]):
        raise ValueError("matched policy does not pin this selected model/arm/epoch/checkpoint")
    parity_baseline = policy.get("parity_baseline") if args.matched_run_key == "qwen-head" else None
    if args.matched_run_key == "qwen-head":
        if args.parity_prefix is None:
            raise ValueError("Qwen-head evaluation requires --parity-prefix to verify the preserved first-7-case outputs")
        if (digest(args.parity_prefix / "predictions.jsonl") != parity_baseline["normalized_predictions_sha256"] or
                digest(args.parity_prefix / "raw-predictions.jsonl") != parity_baseline["raw_predictions_sha256"]):
            raise ValueError("Qwen-head parity prefix differs from policy-pinned preserved partial output")
    if args.profile_json is None or args.model_store is None or args.out is None:
        raise ValueError("matched evaluation requires --profile-json --model-store --out")
    profile_bytes = args.profile_json.read_bytes()
    profile = json.loads(profile_bytes)
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    if (profile_sha != config.get("profile_sha256") or profile_sha != record["profile"]["profile_sha256"] or
            profile.get("profile_version") != record["profile"]["profile_version"] or
            record["profile"]["text_version"] != TEXT_VERSION):
        raise ValueError("matched profile bytes/version/text serialization differ from frozen training manifest")
    model_id = spec["model"]
    model_leaf = {"Qwen/Qwen3-Embedding-0.6B": "Qwen--Qwen3-Embedding-0.6B",
                  "microsoft/harrier-oss-v1-0.6b": "microsoft--harrier-oss-v1-0.6b"}[model_id]
    model_dir = args.model_store.resolve() / "models" / model_leaf
    ckpt_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_path = model_dir / "tokenizer.json"
    if digest(tokenizer_path) != record["profile"].get("tokenizer_sha256"):
        raise ValueError("matched tokenizer hash differs from train-only freeze profile identity")
    if digest(model_dir / "model.safetensors") != config["source_sha256"]["base_checkpoint"]:
        raise ValueError("installed base checkpoint differs from the train-only frozen provenance")
    max_positions = int(ckpt_config.get("max_position_embeddings", ckpt_config.get("max_sequence_length", 0)))
    if max_positions < AUTHORIZED_INFERENCE_TOKEN_LIMIT:
        raise ValueError(f"512-token inference policy exceeds base-model context {max_positions}")

    from tokenizers import Tokenizer
    cpu_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    plans: list[dict[str, Any]] = []
    lengths: list[int] = []
    overlength: list[dict[str, Any]] = []
    text_hash = hashlib.sha256()
    ids: list[str] = []
    for case_row in request_only_jsonl(args.fixture):
        case_id = str(case_row["id"]); ids.append(case_id)
        request = case_row["request"]
        if not isinstance(request.get("state"), dict) or not isinstance(request.get("questions"), dict):
            raise ValueError(f"{case_id}: malformed request-only schema")
        case = {"id": case_id, "domain": str(case_row["domain"]), "questions": []}
        for name, q in request["questions"].items():
            query, labels, candidates = compile_training_text(profile, request["state"], q)
            all_texts = [query] + candidates
            token_ids = []
            for text_ix, text in enumerate(all_texts):
                encoded = [int(x) for x in cpu_tokenizer.encode(text, add_special_tokens=True).ids]
                lengths.append(len(encoded)); token_ids.append(encoded)
                raw = text.encode("utf-8"); text_hash.update(len(raw).to_bytes(8, "little")); text_hash.update(raw)
                if len(encoded) < 1 or len(encoded) > AUTHORIZED_INFERENCE_TOKEN_LIMIT:
                    overlength.append({"id": case_id, "question": str(name),
                        "role": "query" if text_ix == 0 else f"candidate:{labels[text_ix - 1]}",
                        "tokens": len(encoded)})
            case["questions"].append({"name": str(name), "type": q["type"], "labels": labels,
                "token_ids": token_ids, "token_lengths": [len(x) for x in token_ids],
                "score_levels": ([{"label": str(x["label"]), "value": int(x["value"])} for x in q["levels"]]
                                 if q["type"] == "score" else None)})
        plans.append(case)
    expected_ids = [f"mq-lora-fresh-{i:03d}" for i in range(1, 61)]
    if ids != expected_ids or sum(len(c["questions"]) for c in plans) != 180 or len(lengths) != 750:
        raise ValueError("matched preflight requires the ordered complete 60x3/750-text fixture")
    preflight = {"model": model_id, "run_key": args.matched_run_key, "arm": spec["arm"],
        "case_count": 60, "question_count": 180, "compiled_text_count": len(lengths),
        "training_token_limit": TRAINING_TOKEN_LIMIT, "observed_training_max_tokens": 266,
        "inference_token_limit": AUTHORIZED_INFERENCE_TOKEN_LIMIT, "compiled_text_max_tokens": max(lengths),
        "compiled_text_min_tokens": min(lengths), "over_training_limit_text_count": sum(n > TRAINING_TOKEN_LIMIT for n in lengths),
        "over_inference_limit": overlength, "compiled_text_sha256": text_hash.hexdigest(),
        "tokenizer_sha256": digest(tokenizer_path), "references_or_targets_read": False}
    if args.matched_preflight_only:
        print(json.dumps({"freeze_manifest_sha256": freeze_sha, "policy_sha256": policy_sha,
                          "preflight": preflight}, indent=2))
        return 2 if overlength else 0
    if overlength:
        raise ValueError(f"{len(overlength)} compiled texts exceed the authorized 512-token cap; no model loaded")

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "token-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    resource_path = args.out / "resource-guard.jsonl"
    baseline = resource_snapshot()
    swap_baseline = baseline["system_swap_used_bytes"]
    append_jsonl(resource_path, {"event": "pre_model_load_baseline", **baseline})
    guard_matched_resources(baseline, swap_baseline, args.out, policy)

    import importlib.util
    import mlx.core as mx
    import numpy as np
    import mlx.nn as nn  # noqa: F401
    from mlx.utils import tree_flatten
    trainer_spec = importlib.util.spec_from_file_location("matched_frozen_shared_trainer", TRAINER)
    trainer = importlib.util.module_from_spec(trainer_spec)
    assert trainer_spec and trainer_spec.loader
    sys.modules[trainer_spec.name] = trainer
    trainer_spec.loader.exec_module(trainer)
    snap = resource_snapshot(mx); append_jsonl(resource_path, {"event": "after_runtime_imports", **snap})
    guard_matched_resources(snap, swap_baseline, args.out, policy)
    adapter_spec = importlib.util.spec_from_file_location("matched_frozen_adapter", ADAPTER)
    adapter = importlib.util.module_from_spec(adapter_spec)
    assert adapter_spec and adapter_spec.loader
    sys.modules[adapter_spec.name] = adapter
    adapter_spec.loader.exec_module(adapter)
    loaded = adapter.load_checkpoint(model_dir)
    if loaded.checkpoint_dtype != mx.bfloat16:
        raise TypeError("matched base checkpoint is not BF16")
    snap = resource_snapshot(mx); append_jsonl(resource_path, {"event": "after_backbone_load", **snap})
    guard_matched_resources(snap, swap_baseline, args.out, policy)
    backbone = loaded.model
    if spec["arm"] == "lora":
        adapter.inject_upper_lora(loaded, layers=1, projections=("q_proj", "v_proj"), rank=2, alpha=4.0)
    else:
        # train_matched freezes the complete base before exposing the shared typed head.
        backbone.freeze()
    head = trainer.TypedScorer(); head.set_dtype(mx.float32)
    model = trainer.PreferenceModel(backbone, head)
    expected_shapes = {
        "head.choice.bias": [1], "head.choice.weight": [1, 192],
        "head.noul.bias": [1], "head.noul.weight": [1, 192],
        "head.proj.bias": [192], "head.proj.weight": [192, 4096],
        "head.score.bias": [1], "head.score.weight": [1, 192],
    }
    if spec["arm"] == "lora":
        layer = int(config["lora"]["targets"][0].split(".")[1])
        expected_shapes.update({
            f"backbone.model.layers.{layer}.self_attn.q_proj.lora_a": [2, 1024],
            f"backbone.model.layers.{layer}.self_attn.q_proj.lora_b": [2048, 2],
            f"backbone.model.layers.{layer}.self_attn.v_proj.lora_a": [2, 1024],
            f"backbone.model.layers.{layer}.self_attn.v_proj.lora_b": [1024, 2],
        })
    weights_path = bundle["checkpoint"]
    with weights_path.open("rb") as stream:
        header_length = int.from_bytes(stream.read(8), "little")
        if not 0 < header_length <= 16 * 1024 * 1024: raise ValueError("invalid matched safetensors header")
        header = json.loads(stream.read(header_length))
    header_tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if set(header_tensors) != set(expected_shapes) or any(
            header_tensors[k].get("shape") != expected_shapes[k] or header_tensors[k].get("dtype") != "F32"
            for k in expected_shapes):
        raise ValueError(f"{args.matched_run_key}: selected checkpoint header does not match exact frozen trainable schema")
    trainable_names = {str(k) for k, _ in tree_flatten(model.trainable_parameters())}
    if trainable_names != set(expected_shapes):
        raise ValueError(f"{args.matched_run_key}: instantiated trainable tensor names differ from selected checkpoint header; "
                         f"actual={sorted(trainable_names)} expected={sorted(expected_shapes)}")
    loaded_weights = mx.load(str(weights_path))
    if set(loaded_weights) != set(expected_shapes):
        raise ValueError("matched checkpoint payload keys differ from prevalidated header")
    if any(value.dtype != mx.float32 or tuple(value.shape) != tuple(expected_shapes[key]) for key, value in loaded_weights.items()):
        raise ValueError("matched checkpoint payload dtype/shape mismatch")
    model.load_weights(list(loaded_weights.items()), strict=False)
    snap = resource_snapshot(mx); append_jsonl(resource_path, {"event": "after_selected_checkpoint_load", **snap})
    guard_matched_resources(snap, swap_baseline, args.out, policy)
    raw_path, pred_path = args.out / "raw-predictions.jsonl", args.out / "predictions.jsonl"
    raw_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    position_hist: dict[str, dict[str, int]] = {kind: {} for kind in TYPES}
    per_case_ms: list[float] = []
    for case_ix, case in enumerate(plans, 1):
        case_start = time.perf_counter_ns()
        for question in case["questions"]:
            started = time.perf_counter_ns(); vectors = []
            for ids_one in question["token_ids"]:
                tok = mx.array([ids_one], dtype=mx.int32)
                hidden = backbone.model(tok)
                pooled = hidden[0, len(ids_one) - 1, :].astype(mx.float32)
                pooled = pooled / mx.maximum(mx.sqrt(mx.sum(pooled * pooled)), 1e-12)
                mx.eval(pooled)
                vector = np.asarray(pooled, dtype=np.float32)
                if vector.shape != (1024,) or not np.isfinite(vector).all():
                    raise ValueError(f"{case['id']}/{question['name']}: nonfinite/non-1024 embedding")
                vectors.append(vector); del hidden, tok
            logits_mx = model.head(mx.array(vectors[0].tolist()),
                                  mx.array([x.tolist() for x in vectors[1:]]), question["type"])
            mx.eval(logits_mx)
            logits = np.asarray(logits_mx, dtype=np.float32).reshape(-1)
            if logits.size != len(question["labels"]) or not np.isfinite(logits).all():
                raise ValueError(f"{case['id']}/{question['name']}: invalid typed head logits")
            selected, probs = typed_prediction(question["type"], question["labels"], logits)
            if not all(math.isfinite(x) for x in probs) or abs(sum(probs) - 1.0) > 1e-5:
                raise ValueError(f"{case['id']}/{question['name']}: invalid probabilities")
            selected_key = "supported" if question["type"] == "noul" and selected else "unsupported" if question["type"] == "noul" else selected
            candidate_ix = question["labels"].index(selected_key)
            kind_hist = position_hist[question["type"]]
            kind_hist[str(candidate_ix)] = kind_hist.get(str(candidate_ix), 0) + 1
            probabilities = {label: float(prob) for label, prob in zip(question["labels"], probs)}
            expected_value = (sum(float(level["value"]) * float(probs[i])
                                  for i, level in enumerate(question["score_levels"]))
                              if question["type"] == "score" else None)
            inference_ms = (time.perf_counter_ns() - started) / 1e6
            row = {"system_id": args.matched_run_key.replace("-", "_"), "id": case["id"],
                "question_name": question["name"], "question_type": question["type"], "status": "ok",
                "candidate_keys": question["labels"], "probabilities": probabilities,
                "selected": selected, "expected_value": expected_value,
                "selected_candidate_index": candidate_ix,
                "run_config_sha256": digest(run_dir / "config.json"),
                "model_or_adapter_sha256": bundle["checkpoint_sha256"]}
            raw = {"id": case["id"], "question_name": question["name"],
                "question_type": question["type"], "candidate_keys": question["labels"],
                "token_lengths": question["token_lengths"], "raw_logits": [float(x) for x in logits],
                "probabilities": probabilities, "selected": selected,
                "selected_candidate_index": candidate_ix, "inference_ms": inference_ms}
            raw_rows.append(raw); pred_rows.append(row)
            append_jsonl(raw_path, raw); append_jsonl(pred_path, row)
            snap = resource_snapshot(mx)
            append_jsonl(resource_path, {"event": "question_complete", "case_id": case["id"],
                "question_name": question["name"], "baseline_swap_bytes": swap_baseline, **snap})
            guard_matched_resources(snap, swap_baseline, args.out, policy)
        mx.clear_cache()
        gc.collect()
        post_case = resource_snapshot(mx)
        append_jsonl(resource_path, {"event": "after_case_cache_clear", "case_id": case["id"],
            "baseline_swap_bytes": swap_baseline, **post_case})
        guard_matched_resources(post_case, swap_baseline, args.out, policy)
        per_case_ms.append((time.perf_counter_ns() - case_start) / 1e6)
        if args.matched_run_key == "qwen-head" and case_ix == 7:
            parity_summary = verify_matched_parity_prefix(args.parity_prefix, raw_rows[:21], pred_rows[:21])
            (args.out / "parity-prefix-check.json").write_text(json.dumps(parity_summary, indent=2) + "\n", encoding="utf-8")
        print(f"[{case_ix}/60] {case['id']}: raw three-question predictions flushed", flush=True)
    if len(raw_rows) != 180 or len(pred_rows) != 180:
        raise ValueError("matched output did not write exactly 180 raw and normalized predictions")
    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values); pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    question_ms = [r["inference_ms"] for r in raw_rows]
    resource_rows = list(jsonl(resource_path))
    process_peak = max(int(row["process_peak_footprint_bytes"]) for row in resource_rows)
    maximum_swap_growth = max(max(0, int(row["system_swap_used_bytes"]) - swap_baseline)
                              for row in resource_rows)
    maximum_combined = max(int(row["process_peak_footprint_bytes"]) +
                           max(0, int(row["system_swap_used_bytes"]) - swap_baseline)
                           for row in resource_rows)
    run_metadata = {
        "schema_version": 1, "system_id": args.matched_run_key.replace("-", "_"),
        "model": model_id, "arm": spec["arm"], "selected_epoch": spec["epoch"],
        "fixture_sha256": FIXTURE_SHA256, "fixture_manifest_sha256": digest(manifest_path),
        "freeze_manifest_sha256": freeze_sha, "freeze_manifest_path": str(args.matched_freeze_manifest),
        "inference_policy_sha256": policy_sha, "inference_policy_path": str(policy_path),
        "predictions_sha256": digest(pred_path), "raw_predictions_sha256": digest(raw_path),
        "run_config_sha256": digest(run_dir / "config.json"), "best_checkpoint_sha256": bundle["checkpoint_sha256"],
        "best_record_sha256": digest(run_dir / "best.json"), "training_summary_sha256": digest(run_dir / "summary.json"),
        "matched_trainer_sha256": digest(MATCHED_FREEZE_DEFAULT.with_name("train_matched.py")),
        "shared_trainer_sha256": digest(TRAINER), "adapter_sha256": digest(ADAPTER),
        "base_weights_sha256": digest(model_dir / "model.safetensors"),
        "base_config_sha256": digest(model_dir / "config.json"), "backbone_max_position_embeddings": max_positions,
        "profile_sha256": profile_sha, "profile_version": record["profile"]["profile_version"],
        "text_version": TEXT_VERSION, "text_compiler_sha256": digest(TEXT_COMPILER),
        "training_token_guard": TRAINING_TOKEN_LIMIT, "observed_train_max_tokens": 266,
        "inference_token_limit": AUTHORIZED_INFERENCE_TOKEN_LIMIT, "input_truncation": False,
        "test_max_compiled_tokens": max(lengths), "over_training_limit_text_count": preflight["over_training_limit_text_count"],
        "candidate_position_distribution": position_hist,
        "collapsed_prediction_warning": {kind: len(position_hist[kind]) == 1 for kind in TYPES},
        "runtime": "MLX/Metal", "mlx_version": importlib.metadata.version("mlx"),
        "device_actual": str(mx.default_device()),
        "timing_scopes": {
            "question_inference_ms": {"scope": "sequential individual text forwards plus typed scorer; excludes model load, preflight and reference scoring",
                "n": len(question_ms), "p50": pct(question_ms, .5), "p95": pct(question_ms, .95)},
            "three_question_case_sum_ms": {"scope": "sum of 3 question forwards per case; excludes model load, preflight and reference scoring",
                "n": len(per_case_ms), "p50": pct(per_case_ms, .5), "p95": pct(per_case_ms, .95)}},
        "compute_assumptions": {"batching": "no batching; one query/candidate at a time", "flops": "not estimated"},
        "resources": {"baseline_swap_bytes": swap_baseline, "max_process_peak_bytes": process_peak,
                      "max_positive_swap_growth_observed_bytes": maximum_swap_growth,
                      "max_process_peak_plus_swap_growth_bytes": maximum_combined,
                      "early_stop_bytes": policy["resource_guard"]["process_early_stop_bytes"],
                      "absolute_stop_bytes": policy["resource_guard"]["absolute_process_stop_bytes"],
                      "standalone_swap_growth_stop_bytes": None,
                      "combined_stop_bytes": policy["resource_guard"]["process_peak_plus_positive_swap_growth_stop_bytes"],
                      "swap_zero_growth_required": False},
        "reference_access_during_inference": False,
        "fixture_reused_previously_scored": True,
        "interpretation": "exploratory reused holdout; no test-driven tuning",
    }
    (args.out / "run-metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "predictions_frozen_before_reference_scoring", "run_key": args.matched_run_key,
        "predictions_sha256": run_metadata["predictions_sha256"], "raw_sha256": run_metadata["raw_predictions_sha256"],
        "timing_scopes": run_metadata["timing_scopes"], "candidate_position_distribution": position_hist,
        "resources": run_metadata["resources"]}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", help="synthetic arrays only; no fixture or MLX")
    p.add_argument("--preflight-only", action="store_true", help="verify hashes and compile/tokenize the sealed requests, without loading MLX/model")
    p.add_argument("--matched-preflight-only", action="store_true", help="matched-head mode: preflight exact frozen selection, request-only fixture, and 512-token cap")
    p.add_argument("--matched-freeze-manifest", type=Path)
    p.add_argument("--matched-freeze-sha256", default=MATCHED_FREEZE_SHA256)
    p.add_argument("--matched-policy", type=Path)
    p.add_argument("--matched-run-key", choices=tuple(MATCHED_ARMS))
    p.add_argument("--parity-prefix", type=Path, help="preserved partial Qwen-head run to compare first 7 cases before scoring")
    p.add_argument("--prepare-freeze", type=Path, help="verify freeze manifest and emit per-run attestations/profiles only")
    p.add_argument("--prepare-inference-policy", action="store_true", help="write explicit inference-only 512-token extrapolation policy and bind evaluator attestations")
    p.add_argument("--prepare-matched-inference-policy", action="store_true", help="write policy hash-bound to the corrected 9-epoch matched-head train-only freeze")
    p.add_argument("--policy-path", type=Path)
    p.add_argument("--freeze-sha256", default=FREEZE_SHA256)
    p.add_argument("--prepared-dir", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--freeze-manifest", type=Path)
    p.add_argument("--freeze-attestation", type=Path)
    p.add_argument("--profile-json", type=Path, help="exact compact JSON serialization of the training-resolved ModelProfile")
    p.add_argument("--model-store", type=Path)
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--out", type=Path)
    p.add_argument("--smoke-longest-only", action="store_true", help="run one longest compiled question; request-only input and no scoring")
    args = p.parse_args()
    if args.self_test:
        synthetic_self_test()
        return 0
    if args.prepare_freeze:
        if args.prepared_dir is None:
            p.error("--prepare-freeze requires --prepared-dir")
        verify_freeze_manifest(args.prepare_freeze, args.freeze_sha256)
        created = prepare_freeze_artifacts(args.prepare_freeze, args.prepared_dir)
        print(json.dumps({"freeze_manifest_sha256": digest(args.prepare_freeze), "attestations": created}, indent=2))
        return 0
    if args.prepare_inference_policy:
        if args.policy_path is None or args.prepared_dir is None:
            p.error("--prepare-inference-policy requires --policy-path and --prepared-dir (attestation directory)")
        result = prepare_inference_policy(args.freeze_manifest or FREEZE_DEFAULT, args.policy_path, args.prepared_dir)
        print(json.dumps(result, indent=2))
        return 0
    if args.prepare_matched_inference_policy:
        freeze_path = args.matched_freeze_manifest or MATCHED_FREEZE_DEFAULT
        if args.policy_path is None:
            p.error("--prepare-matched-inference-policy requires --policy-path")
        result = prepare_matched_inference_policy(freeze_path, args.policy_path, args.matched_freeze_sha256)
        print(json.dumps(result, indent=2))
        return 0
    if args.matched_freeze_manifest is not None:
        if not args.matched_policy or not args.matched_run_key:
            p.error("matched evaluation requires --matched-policy and --matched-run-key")
        return run_matched_frozen(args)
    needed = (args.run_dir, args.freeze_manifest, args.freeze_attestation, args.profile_json, args.model_store, args.out)
    if any(x is None for x in needed):
        p.error("normal evaluation requires --run-dir --freeze-manifest --freeze-attestation --profile-json --model-store --out")

    # Gate first. No fixture read, MLX import, model load, or inference before this succeeds.
    freeze, freeze_hash = verify_freeze_manifest(args.freeze_manifest.resolve())
    att_path = args.freeze_attestation.resolve()
    att = json.loads(att_path.read_text(encoding="utf-8"))
    if (att.get("train_only_selection_frozen") is not True or att.get("reference_accessed") is not False or
            att.get("source_freeze_manifest_sha256") != freeze_hash):
        raise ValueError("orchestrator attestation must explicitly certify frozen train-only selection and no reference access")
    if (att.get("evaluator_sha256") != digest(Path(__file__).resolve()) or
            att.get("comparison_script_sha256") != digest(Path(__file__).with_name("compare_frozen.py"))):
        raise ValueError("evaluator/comparison code changed after hash-bound freeze attestation")
    if not args.policy_path:
        raise ValueError("normal inference requires a separately hash-bound --policy-path")
    policy = json.loads(args.policy_path.read_text(encoding="utf-8"))
    policy_sha = digest(args.policy_path)
    if (att.get("evaluation_policy_sha256") != policy_sha or
            att.get("evaluation_policy_id") != policy.get("policy_id") or
            policy.get("source_freeze_manifest_sha256") != freeze_hash or
            policy.get("fixture_sha256") != FIXTURE_SHA256 or
            policy.get("training_policy", {}).get("max_tokens_per_text") != TRAINING_TOKEN_LIMIT or
            policy.get("inference_policy", {}).get("max_tokens_per_text") != AUTHORIZED_INFERENCE_TOKEN_LIMIT or
            policy.get("inference_policy", {}).get("truncate") is not False or
            policy.get("inference_policy", {}).get("authorize_unseen_length_extrapolation") is not True):
        raise ValueError("hash-bound inference-only policy is missing, mismatched, or does not authorize the stated non-truncating extrapolation")
    if att.get("fixture_sha256") != FIXTURE_SHA256:
        raise ValueError("freeze attestation does not name this exact sealed fixture hash")
    if att.get("text_compiler_sha256") != digest(TEXT_COMPILER):
        raise ValueError("freeze attestation does not pin the exact generic v2 text compiler source")
    train_selection = att.get("train_only_selection")
    if (not isinstance(train_selection, dict) or train_selection.get("status") != "train_dev_complete" or
            train_selection.get("heldout_accessed") is not False):
        raise ValueError("attestation must record completed grouped-dev-only best-checkpoint selection")
    manifest_path = args.fixture.with_name("test-lora-fresh-60.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != FIXTURE_SHA256 or args.fixture.resolve() != FIXTURE.resolve():
        raise ValueError("sealed fixture path/manifest hash mismatch")
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    weights_path = run_dir / "best.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model") not in {"Qwen/Qwen3-Embedding-0.6B", "microsoft/harrier-oss-v1-0.6b"}:
        raise ValueError("unknown training model in config")
    if att.get("model") != config.get("model") or att.get("run_id") != args.run_dir.name:
        raise ValueError("freeze attestation model/run ID does not match the supplied trainer artifacts")
    freeze_key = "qwen" if config["model"] == "Qwen/Qwen3-Embedding-0.6B" else "harrier"
    freeze_record = freeze.get("models", {}).get(freeze_key, {})
    if (freeze_record.get("model_id") != config["model"] or
            train_selection.get("method") != freeze_record.get("selection_rule") or
            train_selection.get("scope") != freeze.get("scope") or
            att.get("text_compiler_sha256") != digest(TEXT_COMPILER)):
        raise ValueError("run attestation selection/compiler provenance differs from the hash-verified freeze manifest")
    required_config = {"schema_version": 1, "base_dtype": "bfloat16", "trainable_dtype": "float32",
                       "lora_rank": 2, "lora_alpha": 4.0, "max_tokens_per_text": 288,
                       "batch_size_questions": 1, "gradient_accumulation_steps": 1,
                       "max_epochs": 3, "max_steps": 1380}
    if any(config.get(key) != value for key, value in required_config.items()):
        raise ValueError("training config is inconsistent with the declared frozen LoRA profile")
    targets = config.get("lora_targets")
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("expected exactly two upper-layer LoRA targets")
    parsed_targets = [str(target).rsplit(".", 2) for target in targets]
    if (any(len(parts) != 3 or parts[1] != "self_attn" or parts[2] not in {"q_proj", "v_proj"}
            for parts in parsed_targets) or
            {parts[2] for parts in parsed_targets} != {"q_proj", "v_proj"} or
            len({parts[0] for parts in parsed_targets}) != 1 or
            not parsed_targets[0][0].startswith("layers.")):
        raise ValueError("LoRA targets must be q_proj/v_proj in the same final decoder layer")
    lora_layer = int(parsed_targets[0][0].split(".")[1])
    frozen_hashes = att.get("sha256", {})
    recorded_artifacts = freeze_record.get("artifacts", {})
    frozen_to_attested = {"config.json": "config", "best.safetensors": "best_weights",
                          "best.json": "best_record", "summary.json": "summary"}
    if any(recorded_artifacts.get(name, {}).get("sha256") != frozen_hashes.get(attested)
           for name, attested in frozen_to_attested.items()):
        raise ValueError("per-run attestation hashes are not copied from the verified freeze manifest")
    require_hash(config_path, frozen_hashes.get("config", ""), "config")
    require_hash(weights_path, frozen_hashes.get("best_weights", ""), "best weights")
    require_hash(run_dir / "best.json", frozen_hashes.get("best_record", ""), "best record")
    require_hash(run_dir / "summary.json", frozen_hashes.get("summary", ""), "train/dev summary")
    adapter_hash = require_hash(ADAPTER, config.get("source_sha256", {}).get("adapter", ""), "checkpoint adapter")
    require_hash(TRAINER, config.get("trainer_sha256", ""), "trainer source")
    best_record = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
    if best_record.get("epoch") != train_selection.get("selected_epoch"):
        raise ValueError("freeze attestation selected epoch differs from train-only best checkpoint record")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "train_dev_complete" or summary.get("steps") != config.get("max_steps"):
        raise ValueError("training run is not complete at its declared full step budget")
    if summary.get("best_epoch") != best_record.get("epoch"):
        raise ValueError("train/dev summary best epoch differs from best checkpoint record")
    if (train_selection.get("completed_steps") != summary["steps"] or
            train_selection.get("selected_epoch") != best_record["epoch"] or
            train_selection.get("best_dev_loss") != best_record["dev"]["loss"]):
        raise ValueError("attested grouped-dev selection does not match completed trainer artifacts")
    ckpt_dir = args.model_store / "models" / {
        "Qwen/Qwen3-Embedding-0.6B": "Qwen--Qwen3-Embedding-0.6B",
        "microsoft/harrier-oss-v1-0.6b": "microsoft--harrier-oss-v1-0.6b",
    }[config["model"]]
    base_weights = ckpt_dir / "model.safetensors"
    require_hash(base_weights, config.get("source_sha256", {}).get("checkpoint", ""), "BF16 backbone")
    base_config = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    max_positions = int(base_config.get("max_position_embeddings", base_config.get("max_sequence_length", 0)))
    if max_positions < AUTHORIZED_INFERENCE_TOKEN_LIMIT:
        raise ValueError(f"authorized inference limit {AUTHORIZED_INFERENCE_TOKEN_LIMIT} exceeds backbone config max {max_positions}")
    if lora_layer != int(base_config["num_hidden_layers"]) - 1:
        raise ValueError("frozen LoRA targets are not the final decoder layer")
    hidden = int(base_config["hidden_size"])
    head_dim = int(base_config["head_dim"])
    q_width = int(base_config["num_attention_heads"]) * head_dim
    v_width = int(base_config["num_key_value_heads"]) * head_dim
    with weights_path.open("rb") as stream:
        header_len_raw = stream.read(8)
        if len(header_len_raw) != 8:
            raise ValueError("truncated best.safetensors header")
        header_len = int.from_bytes(header_len_raw, "little")
        if not 0 < header_len <= 16 * 1024 * 1024:
            raise ValueError("invalid/oversized best.safetensors header")
        header = json.loads(stream.read(header_len))
    validate_checkpoint_header({k: v for k, v in header.items() if k != "__metadata__"},
                               lora_layer, hidden, q_width, v_width)
    feature_path = MODEL_FEATURE_MANIFESTS[config["model"]]
    feature_manifest_path = Path(str(feature_path) + ".manifest.json")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if (digest(feature_path) != config.get("source_sha256", {}).get("multi_feature_text_source") or
            feature_manifest.get("input_sha256") != config.get("source_sha256", {}).get("multi_train") or
            feature_manifest.get("text_version") != TEXT_VERSION or
            feature_manifest.get("model") != config["model"]):
        raise ValueError("recorded training feature/profile provenance is inconsistent")
    profile_raw = args.profile_json.read_bytes()
    if hashlib.sha256(profile_raw).hexdigest() != feature_manifest.get("profile_sha256"):
        raise ValueError("resolved profile JSON bytes do not match training feature profile hash")
    profile = json.loads(profile_raw)
    profile_identity = {"model": config["model"], "profile_version": feature_manifest["profile_version"],
                        "profile_sha256": feature_manifest["profile_sha256"], "text_version": TEXT_VERSION}
    if config.get("parity_report", {}).get("profile_version") != str(profile_identity["profile_version"]):
        raise ValueError("training parity report profile version differs from feature manifest")
    if att.get("training_profile") != profile_identity:
        raise ValueError("freeze attestation training_profile does not match the actual train-export manifest/profile")
    if (policy.get("profile_identities", {}).get(freeze_key) != {
            "model": config["model"], "profile_version": profile_identity["profile_version"],
            "profile_sha256": profile_identity["profile_sha256"], "text_version": TEXT_VERSION} or
            policy.get("best_weights_sha256", {}).get(freeze_key) != frozen_hashes.get("best_weights")):
        raise ValueError("inference policy does not pin this model's frozen profile and best weights")

    # Once-and-only-after freeze and inference-policy verification, compile/tokenize
    # the sealed requests on CPU. Reject above 512; never truncate.
    if args.fixture.resolve() != FIXTURE.resolve():
        raise ValueError("only the hash-pinned sealed LoRA fresh-60 fixture is permitted")
    if digest(args.fixture) != FIXTURE_SHA256:
        raise ValueError("sealed fixture bytes do not match its pinned manifest hash")
    tokenizers_path = ckpt_dir / "tokenizer.json"
    tokenizer_sha = digest(tokenizers_path)
    if tokenizer_sha != feature_manifest.get("tokenizer_sha256"):
        raise ValueError("base tokenizer hash differs from train profile feature manifest")
    from tokenizers import Tokenizer
    cpu_tokenizer = Tokenizer.from_file(str(tokenizers_path))
    plans: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    overlength: list[dict[str, Any]] = []
    compiled_digest = hashlib.sha256()
    seen_ids: list[str] = []
    for source_case in request_only_jsonl(args.fixture):
        case_id = str(source_case.get("id", "")); seen_ids.append(case_id)
        request = source_case.get("request")
        if not isinstance(request, dict) or not isinstance(request.get("state"), dict) or not isinstance(request.get("questions"), dict):
            raise ValueError(f"{case_id}: malformed request-only evaluation input")
        case_plan = {"id": case_id, "domain": str(source_case.get("domain", "unknown")), "questions": []}
        for name, q in request["questions"].items():
            query, labels, candidates = compile_training_text(profile, request["state"], q)
            ids_by_text = []
            for text_index, text in enumerate([query] + candidates):
                token_ids = [int(x) for x in cpu_tokenizer.encode(text, add_special_tokens=True).ids]
                token_count = len(token_ids)
                token_lengths.append(token_count)
                compiled_digest.update(len(text.encode("utf-8")).to_bytes(8, "little"))
                compiled_digest.update(text.encode("utf-8"))
                if token_count < 1 or token_count > AUTHORIZED_INFERENCE_TOKEN_LIMIT:
                    overlength.append({"id": case_id, "question_name": str(name),
                        "role": "query" if text_index == 0 else f"candidate:{labels[text_index - 1]}",
                        "token_count": token_count})
                ids_by_text.append(token_ids)
            case_plan["questions"].append({"name": str(name), "type": q["type"], "labels": labels,
                "token_ids": ids_by_text, "token_lengths": [len(x) for x in ids_by_text],
                "score_levels": ([{"label": str(level["label"]), "value": int(level["value"])} for level in q["levels"]]
                                 if q["type"] == "score" else None)})
        plans.append(case_plan)
    if seen_ids != [f"mq-lora-fresh-{i:03d}" for i in range(1, 61)] or sum(len(c["questions"]) for c in plans) != 180:
        raise ValueError("compiled fixture must contain the exact ordered 60 cases and 180 questions")
    if sorted(q["type"] for case in plans for q in case["questions"]) != sorted(["choice", "noul", "score"] * 60):
        raise ValueError("expected one compiled choice/noul/score question per case")
    preflight = {"case_count": len(plans), "question_count": 180, "compiled_text_count": len(token_lengths),
                 "compiled_text_max_tokens": max(token_lengths), "compiled_text_min_tokens": min(token_lengths),
                 "training_token_limit": TRAINING_TOKEN_LIMIT,
                 "inference_token_limit": AUTHORIZED_INFERENCE_TOKEN_LIMIT, "all_compiled_text_within_limit": True,
                 "compiled_text_sha256": compiled_digest.hexdigest(), "tokenizer_sha256": tokenizer_sha,
                 "overflow_count": len(overlength), "overflow_samples": overlength[:20],
                 "overflow_count_by_role": {role: sum(item["role"] == role for item in overlength)
                                            for role in sorted({item["role"] for item in overlength})},
                 "input_fields": "request.state and request.questions only; references not used"}
    preflight["all_compiled_text_within_limit"] = not overlength
    del cpu_tokenizer
    if args.preflight_only:
        print(json.dumps({"freeze_manifest_sha256": freeze_hash, "model": config["model"],
                          "profile_sha256": profile_identity["profile_sha256"], "preflight": preflight}, indent=2))
        return 2 if overlength else 0
    if overlength:
        raise ValueError(f"{len(overlength)} compiled texts exceed the separately authorized 512-token inference limit; no truncation or model load performed")

    args.out.mkdir(parents=True, exist_ok=False)
    resource_path = args.out / "resource-guard.jsonl"
    baseline = resource_snapshot()
    swap_baseline = baseline["system_swap_used_bytes"]
    append_jsonl(resource_path, {"event": "pre_model_load_baseline", **baseline})
    guard_resources(baseline, swap_baseline, args.out)
    (args.out / "token-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    # Import heavyweight MLX dependencies only beyond the explicit freeze/policy gates
    # and after all 180 questions/candidate strings pass the 512-token inference guard.
    import importlib.util
    import mlx.core as mx
    import mlx.nn as nn  # noqa: F401 - trainer's typed head dependency
    import numpy as np
    from mlx.utils import tree_flatten

    spec = importlib.util.spec_from_file_location("frozen_lora_trainer", TRAINER)
    trainer = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    snapshot = resource_snapshot(mx)
    append_jsonl(resource_path, {"event": "after_runtime_imports", **snapshot})
    guard_resources(snapshot, swap_baseline, args.out)
    adapter_spec = importlib.util.spec_from_file_location("frozen_lora_adapter", ADAPTER)
    adapter = importlib.util.module_from_spec(adapter_spec)
    assert adapter_spec and adapter_spec.loader
    sys.modules[adapter_spec.name] = adapter
    adapter_spec.loader.exec_module(adapter)
    loaded = adapter.load_checkpoint(ckpt_dir)
    snapshot = resource_snapshot(mx)
    append_jsonl(resource_path, {"event": "after_backbone_load", **snapshot})
    guard_resources(snapshot, swap_baseline, args.out)
    if loaded.checkpoint_dtype != mx.bfloat16:
        raise TypeError("base checkpoint must be exact BF16")
    adapter.inject_upper_lora(loaded, layers=1, projections=("q_proj", "v_proj"), rank=2, alpha=4.0)
    backbone = loaded.model  # adapter freezes base tensors and leaves injected LoRA trainable
    head = trainer.TypedScorer(); head.set_dtype(mx.float32)
    model = trainer.PreferenceModel(backbone, head)
    # The checkpoint intentionally contains only trainable LoRA/head tensors; the
    # static safetensors-header validator above requires the exact complete key set.
    # Non-strict load here is solely to permit absent frozen backbone weights.
    model.load_weights(list(mx.load(str(weights_path)).items()), strict=False)
    snapshot = resource_snapshot(mx)
    append_jsonl(resource_path, {"event": "after_best_adapter_head_load", **snapshot})
    guard_resources(snapshot, swap_baseline, args.out)
    if any(v.dtype != mx.bfloat16 for n, v in tree_flatten(backbone.parameters()) if ".lora_" not in str(n)):
        raise TypeError("non-LoRA backbone parameters are not BF16")

    raw_path, pred_path = args.out / "raw-predictions.jsonl", args.out / "predictions.jsonl"
    if args.smoke_longest_only:
        longest_case, longest_question = max(
            ((case, question) for case in plans for question in case["questions"]),
            key=lambda pair: max(pair[1]["token_lengths"]))
        question = longest_question
        started = time.perf_counter_ns()
        vectors = []
        for ids_one in question["token_ids"]:
            token_ids = mx.array([ids_one], dtype=mx.int32)
            hidden_states = backbone.model(token_ids)
            pooled = hidden_states[0, len(ids_one) - 1, :].astype(mx.float32)
            pooled = pooled / mx.maximum(mx.sqrt(mx.sum(pooled * pooled)), 1e-12)
            mx.eval(pooled)
            vector = np.asarray(pooled, dtype=np.float32)
            if vector.shape != (1024,) or not np.isfinite(vector).all():
                raise ValueError(f"smoke embedding invalid: shape={vector.shape} finite={np.isfinite(vector).all()}")
            vectors.append(vector)
            del hidden_states, token_ids
        logits_mx = model.head(mx.array(vectors[0].tolist()), mx.array([v.tolist() for v in vectors[1:]]), question["type"])
        mx.eval(logits_mx)
        logits = np.asarray(logits_mx, dtype=np.float32).reshape(-1)
        if not np.isfinite(logits).all():
            raise ValueError("smoke typed-head logits are non-finite")
        selected, probs = typed_prediction(question["type"], question["labels"], logits)
        if not all(math.isfinite(x) for x in probs) or abs(sum(probs) - 1.0) > 1e-5:
            raise ValueError("smoke typed-head probabilities are non-finite or fail normalization")
        raw_smoke = {"case_id": longest_case["id"], "question_name": question["name"],
            "question_type": question["type"], "candidate_keys": question["labels"],
            "token_lengths": question["token_lengths"], "max_token_length": max(question["token_lengths"]),
            "embedding_dimensions": [int(v.shape[0]) for v in vectors], "embeddings_finite": True,
            "raw_logits": [float(x) for x in logits], "probabilities": {k: float(v) for k, v in zip(question["labels"], probs)},
            "selected": selected, "inference_ms": (time.perf_counter_ns() - started) / 1e6,
            "references_read": False, "smoke_only": True}
        append_jsonl(raw_path, raw_smoke)
        snapshot = resource_snapshot(mx)
        append_jsonl(resource_path, {"event": "smoke_longest_complete", "case_id": longest_case["id"],
            "question_name": question["name"], "baseline_swap_bytes": swap_baseline, **snapshot})
        guard_resources(snapshot, swap_baseline, args.out)
        smoke_meta = {"status": "smoke_passed", "model": config["model"], "case_id": longest_case["id"],
            "question_name": question["name"], "inference_policy_sha256": policy_sha,
            "max_text_tokens": max(question["token_lengths"]), "training_guard_tokens": TRAINING_TOKEN_LIMIT,
            "inference_guard_tokens": AUTHORIZED_INFERENCE_TOKEN_LIMIT,
            "raw_prediction_sha256": digest(raw_path), "references_read": False, "scoring_performed": False,
            "resource_snapshot": snapshot}
        (args.out / "smoke-metadata.json").write_text(json.dumps(smoke_meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke_meta, indent=2))
        return 0

    raw_rows: list[dict[str, Any]] = []
    flat_predictions: list[dict[str, Any]] = []
    case_predictions: list[dict[str, Any]] = []
    for case_index, case in enumerate(plans, 1):
        case_pred: dict[str, Any] = {"id": case["id"]}
        case_raw = []
        for question in case["questions"]:
            started = time.perf_counter_ns()
            vectors = []
            # Single-text decoder forwards; no batching and no runtime truncation.
            for ids_one in question["token_ids"]:
                token_ids = mx.array([ids_one], dtype=mx.int32)
                hidden = backbone.model(token_ids)
                pooled = hidden[0, len(ids_one) - 1, :].astype(mx.float32)
                pooled = pooled / mx.maximum(mx.sqrt(mx.sum(pooled * pooled)), 1e-12)
                mx.eval(pooled)
                vectors.append(np.asarray(pooled, dtype=np.float32))
                del hidden, token_ids
            logits_mx = model.head(mx.array(vectors[0].tolist()), mx.array([v.tolist() for v in vectors[1:]]), question["type"])
            mx.eval(logits_mx)
            logits = np.asarray(logits_mx).reshape(-1)
            label, probs = typed_prediction(question["type"], question["labels"], logits)
            inference_ms = (time.perf_counter_ns() - started) / 1e6
            case_pred[question["name"]] = label
            probabilities = {key: float(probability) for key, probability in zip(question["labels"], probs)}
            score_levels = question.get("score_levels")
            expected_value = (sum(float(level["value"]) * probs[index] for index, level in enumerate(score_levels))
                              if question["type"] == "score" else None)
            row = {"system_id": f"{freeze_key}_lora", "id": case["id"],
                   "question_name": question["name"], "question_type": question["type"], "status": "ok",
                   "candidate_keys": question["labels"], "probabilities": probabilities,
                   "selected": label, "expected_value": expected_value,
                   "run_config_sha256": digest(config_path), "model_or_adapter_sha256": digest(weights_path)}
            flat_predictions.append(row)
            case_raw.append({"id": case["id"], "question_name": question["name"],
                "question_type": question["type"], "candidate_keys": question["labels"],
                "token_lengths": question["token_lengths"], "raw_logits": [float(x) for x in logits],
                "probabilities": probabilities, "selected": label, "inference_ms": inference_ms})
            raw_rows.append(case_raw[-1])
            append_jsonl(raw_path, case_raw[-1])
            append_jsonl(pred_path, row)
            snapshot = resource_snapshot(mx)
            append_jsonl(resource_path, {"event": "question_complete", "case_id": case["id"],
                "question_name": question["name"], "baseline_swap_bytes": swap_baseline, **snapshot})
            guard_resources(snapshot, swap_baseline, args.out)
        case_predictions.append(case_pred)
        print(f"[{case_index}/60] {case['id']}: three raw predictions flushed", flush=True)

    # Only now reopen reference-bearing rows, after durable raw and normalized outputs.
    scoring_cases = list(jsonl(args.fixture))
    metrics = score(scoring_cases, case_predictions)
    expected_by_key = {(row["id"], row["question_name"]): row.get("expected_value") for row in flat_predictions}
    score_value_errors = []
    for case in scoring_cases:
        for name, question in case["request"]["questions"].items():
            if question["type"] != "score":
                continue
            expected = expected_by_key[(str(case["id"]), str(name))]
            gold_label = case["reference"][str(name)]["target"]["label"]
            levels = {str(level["label"]): float(level["value"]) for level in question["levels"]}
            score_value_errors.append(abs(float(expected) - levels[str(gold_label)]))
    metrics["score_expected_value_mae"] = (sum(score_value_errors) / len(score_value_errors)
        if score_value_errors else None)
    question_times = [row["inference_ms"] for row in raw_rows]
    by_id_times: dict[str, list[float]] = defaultdict(list)
    for row in raw_rows:
        by_id_times[row["id"]].append(row["inference_ms"])
    case_times = [sum(times) for times in by_id_times.values()]
    def percentile(values: list[float], pctl: float) -> float:
        ordered = sorted(values); pos = (len(ordered) - 1) * pctl
        low, high = math.floor(pos), math.ceil(pos)
        return ordered[low] if low == high else ordered[low] * (high - pos) + ordered[high] * (pos - low)
    timing_scopes = {
        "question_inference_ms": {"scope": "sequential individual text backbone forwards + typed scorer; tokenizer/compilation and load excluded",
            "n": len(question_times), "p50": percentile(question_times, .5), "p95": percentile(question_times, .95)},
        "three_question_case_sum_ms": {"scope": "sum of three question inference scopes; tokenizer/compilation and model load excluded",
            "n": len(case_times), "p50": percentile(case_times, .5), "p95": percentile(case_times, .95)},
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "run-metadata.json").write_text(json.dumps({
        "fixture_sha256": FIXTURE_SHA256, "fixture_manifest_sha256": digest(manifest_path),
        "inference_policy_id": policy["policy_id"], "inference_policy_sha256": policy_sha,
        "training_max_tokens_per_text": TRAINING_TOKEN_LIMIT,
        "inference_max_tokens_per_text": AUTHORIZED_INFERENCE_TOKEN_LIMIT,
        "unseen_length_extrapolation": {"authorized": True,
            "inputs_over_training_limit": sum(n > TRAINING_TOKEN_LIMIT for n in token_lengths),
            "maximum_compiled_token_count": max(token_lengths), "truncation": False},
        "predictions_sha256": digest(pred_path), "raw_predictions_sha256": digest(raw_path),
        "freeze_attestation_sha256": digest(att_path), "config_sha256": digest(config_path),
        "best_weights_sha256": digest(weights_path), "best_record_sha256": digest(run_dir / "best.json"),
        "training_summary_sha256": digest(summary_path), "trainer_sha256": digest(TRAINER),
        "base_weights_sha256": digest(base_weights), "training_feature_manifest_sha256": digest(feature_manifest_path),
        "adapter_sha256": adapter_hash, "compiled_profile_hash": profile_identity["profile_sha256"],
        "training_profile": profile_identity, "text_compiler_sha256": digest(TEXT_COMPILER),
        "text_compiler": "results/probes/varied/exporter.rs v2-generic text serialization (same profile templates; no embeddings consumed)",
        "model": config["model"],
        "runtime": "MLX/Metal", "mlx_version": importlib.metadata.version("mlx"),
        "device_actual": str(mx.default_device()),
        "timing_scopes": timing_scopes,
        "compute_assumptions": {"forward_policy": "one query/candidate text forward individually; no batching",
            "candidate_count": "fixture supplied 2-5 per question", "flops": "not estimated; no hardware-normalized compute claim"},
        "reference_scoring_after_raw_prediction_flush": True,
        "inference_granularity": "one query/candidate text forward at a time", "metrics": metrics,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


def score(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[int]] = defaultdict(list)
    for i, case in enumerate(cases): by_domain[str(case["domain"])].append(i)
    def summarize(ix: list[int]) -> dict[str, Any]:
        per_type = {}; all3 = 0
        for kind in TYPES:
            pairs = []
            for i in ix:
                case = cases[i]
                name, q = next((n, q) for n, q in case["request"]["questions"].items() if q["type"] == kind)
                ref = case["reference"][name]["target"]
                target = ref["selected"] if kind == "choice" else ref["value"] if kind == "noul" else ref["label"]
                pairs.append((predictions[i].get(name), target))
            per_type[kind] = {"correct": sum(p == r for p, r in pairs), "total": len(ix), "accuracy": sum(p == r for p, r in pairs) / len(ix) if ix else None}
            if kind == "score":
                distances, normalized_value_errors = [], []
                for i in ix:
                    case = cases[i]
                    name, q = next((n, q) for n, q in case["request"]["questions"].items() if q["type"] == "score")
                    levels = q["levels"]
                    labels = [str(level["label"]) for level in levels]
                    predicted_label = predictions[i].get(name)
                    gold_label = case["reference"][name]["target"]["label"]
                    distances.append(abs(labels.index(predicted_label) - labels.index(gold_label)) if predicted_label in labels else len(labels) - 1)
                    by_label = {str(level["label"]): int(level["value"]) for level in levels}
                    span = max(by_label.values()) - min(by_label.values())
                    normalized_value_errors.append(abs(by_label.get(predicted_label, by_label[labels[0]]) - by_label[gold_label]) / span if span else 0.0)
                per_type[kind]["ordinal_level_mae"] = sum(distances) / len(distances) if distances else None
                per_type[kind]["within_one_level"] = sum(d <= 1 for d in distances)
                per_type[kind]["normalized_value_mae"] = sum(normalized_value_errors) / len(normalized_value_errors) if normalized_value_errors else None
        for i in ix:
            case = cases[i]
            good = True
            for name, q in case["request"]["questions"].items():
                target = case["reference"][name]["target"]
                gold = target["selected"] if q["type"] == "choice" else target["value"] if q["type"] == "noul" else target["label"]
                good &= predictions[i].get(name) == gold
            all3 += int(good)
        return {"per_type": per_type, "all_three": {"correct": all3, "total": len(ix), "accuracy": all3 / len(ix) if ix else None}}
    per_case = []
    for i, case in enumerate(cases):
        decisions = {}
        for name, q in case["request"]["questions"].items():
            target = case["reference"][name]["target"]
            gold = target["selected"] if q["type"] == "choice" else target["value"] if q["type"] == "noul" else target["label"]
            decisions[name] = {"type": q["type"], "prediction": predictions[i].get(name), "reference": gold,
                               "correct": predictions[i].get(name) == gold}
        per_case.append({"id": case["id"], "domain": case["domain"], "decisions": decisions,
                         "all_three_correct": all(x["correct"] for x in decisions.values())})
    return {"overall": summarize(list(range(len(cases))),),
            "per_domain": {d: {"case_count": len(ix), **summarize(ix)} for d, ix in sorted(by_domain.items())},
            "score_ordinal": "exact ordered level-label match; ordinal level MAE and within-one-level; normalized numeric-value MAE",
            "per_case": per_case}


if __name__ == "__main__":
    raise SystemExit(main())
