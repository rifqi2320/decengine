#!/usr/bin/env python3
"""Train end-to-end LoRA + generic typed preference scorers on train-only data.

Only clean300 and multi train120 are accepted. The feature JSONL files are used
solely as a source of the already compiled query/candidate *texts*; their embedding
arrays are never used. The differentiable MLX-LM decoder re-embeds every text on
every step, including both the query and candidates.

Adapter contract (provided by checkpoint_adapter.py):
    load_checkpoint(checkpoint_dir) -> loaded checkpoint wrapper
The wrapper exposes `.model` (MLX-LM Qwen3 model), `.tokenizer`, and
`.checkpoint_dtype`; the adapter provides `inject_upper_lora` and must attest Rust
parity before training is allowed to start.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

ROOT = Path(__file__).resolve().parents[4]
VARIED = ROOT / "results/probes/varied"
DEFAULT_STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store")
DEFAULT_CLEAN = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/varied-loqfo-clean300/train-clean-300.jsonl")
DEFAULT_MULTI = ROOT / "benchmarks/cases/varied/multi/train-120.jsonl"
MODEL_CONFIGS = {
    "Qwen/Qwen3-Embedding-0.6B": ("Qwen--Qwen3-Embedding-0.6B", VARIED / "runs/multi-qwen/train-clean-300.features.jsonl", VARIED / "runs/multi-qwen/train-120.features.jsonl"),
    "microsoft/harrier-oss-v1-0.6b": ("microsoft--harrier-oss-v1-0.6b", VARIED / "runs/multi-harrier/train-clean-300.features.jsonl", VARIED / "runs/multi-harrier/train-120.features.jsonl"),
}

# Predeclared, deliberately small resource budget. Select epoch on grouped dev loss;
# no sweep of these values and no test-set inputs are accepted by this CLI.
SEED = 271828
DEV_FRACTION = 0.20
MAX_EPOCHS = 3
MAX_STEPS = 1380  # exact 3 x 460 grouped-training questions on this fixed split
MAX_TOKENS = 288  # rounded up from the train-only observed max (266); reject, never truncate
LORA_LAST_LAYERS = 1
LORA_RANK = 2
LORA_ALPHA = 4.0
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
SCORE_ORDINAL_WEIGHT = 0.75
SCORE_VALUE_TEMPERATURE = 0.20
MEMORY_CHECK_INTERVAL = 1
PROCESS_FOOTPRINT_LIMIT = 6 * 1024 ** 3  # stop with >=2 GiB margin under hard 8 GiB cap
SYSTEM_SWAP_LIMIT = 4 * 1024 ** 3
SEQUENCE_BUCKETS = (64, 128, 192, 288)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def train_rows(clean_path: Path, multi_path: Path, clean_features: Path, multi_features: Path):
    clean, multi = read_jsonl(clean_path), read_jsonl(multi_path)
    if len(clean) != 300 or len(multi) != 120:
        raise ValueError(f"expected clean300 + multi120, got {len(clean)} + {len(multi)}")
    rows, feature_files = clean + multi, (clean_features, multi_features)
    cases = {}
    for source, is_multi in ((clean, False), (multi, True)):
        for record in source:
            rid = str(record["id"])
            questions = record["request"]["questions"]
            references = record.get("reference", {})
            if is_multi and set(questions) != set(references):
                raise ValueError(f"{rid}: train questions/references differ")
            for name, q in questions.items():
                kind = q.get("type")
                if kind == "choice":
                    keys = list(q["options"])
                    gold = str(references[name]["target"]["selected"] if is_multi else record["reference"]["target"]["selected"])
                    values = None
                elif kind == "noul":
                    keys = ["unsupported", "supported"]
                    target = references[name]["target"] if is_multi else record["reference"]["target"]
                    if type(target["value"]) is not bool:
                        raise ValueError(f"{rid}/{name}: noul target must be bool")
                    gold = "supported" if target["value"] else "unsupported"
                    values = None
                elif kind == "score":
                    levels = q["levels"]
                    keys = [str(x["label"]) for x in levels]
                    values = [float(x["value"]) for x in levels]
                    target = references[name]["target"] if is_multi else record["reference"]["target"]
                    gold = str(target["label"])
                    if len(set(values)) != len(values):
                        raise ValueError(f"{rid}/{name}: score rubric values must be distinct")
                else:
                    raise ValueError(f"{rid}/{name}: unsupported question type {kind!r}")
                if gold not in keys or len(keys) < 2:
                    raise ValueError(f"{rid}/{name}: invalid target/candidates")
                ref = references[name] if is_multi else record
                family = str(ref["question_family_id"])
                cases[(rid, str(name))] = {
                    "id": rid, "name": str(name), "family": family, "kind": kind,
                    "keys": keys, "target": keys.index(gold), "values": values,
                }

    # Read text only; never retain or consume query_embedding/candidate embedding.
    text_map = {}
    model_names = set()
    for feature_path in feature_files:
        for line in feature_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            # Strip cached float vectors before decoding; they are neither retained
            # in memory nor consumed as model inputs.
            text_only, removed = re.subn(r'"(?:query_embedding|embedding)":\[[^\]]*\],', "", line)
            if removed < 3:
                raise ValueError(f"{feature_path}: expected query + candidate cached vectors")
            row = json.loads(text_only)
            model_names.add(str(row.get("model", "")))
            key = (str(row["id"]), str(row["question_key"]))
            if key not in cases or key in text_map:
                raise ValueError(f"feature text key unexpected/duplicate: {key}")
            query = row.get("query_text")
            candidates = row.get("candidates")
            if not query or not isinstance(candidates, list):
                raise ValueError(f"{key}: missing compiled text")
            candidate_map = {str(x["candidate_id"]): str(x["text"]) for x in candidates}
            if set(candidate_map) != set(cases[key]["keys"]):
                raise ValueError(f"{key}: compiled text candidate IDs differ")
            text_map[key] = (str(query), candidate_map)
            cases[key]["compiled_token_lengths"] = [int(row["query_token_length"])] + [
                int(next(x["token_length"] for x in candidates if str(x["candidate_id"]) == k))
                for k in cases[key]["keys"]]
    if len(text_map) != 660 or len(model_names) != 1:
        raise ValueError(f"feature text manifest mismatch: {len(text_map)} records, models={model_names}")
    result = []
    for key, case in cases.items():
        query, candidate_map = text_map[key]
        case["query"] = query
        case["candidate_texts"] = [candidate_map[k] for k in case["keys"]]
        case["case_key"] = case["id"]
        result.append(case)
    return rows, result


def grouped_split(cases):
    """Deterministic 80/20 component split; any shared family or case stays together."""
    parent = list(range(len(cases)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, c in enumerate(cases):
        for field in ("family", "case_key"):
            value = (field, c[field])
            if value in seen:
                parent[root(i)] = root(seen[value])
            else:
                seen[value] = i
    groups = {}
    for i in range(len(cases)):
        groups.setdefault(root(i), []).append(i)
    unique = list(groups)
    rng = random.Random(SEED)
    rng.shuffle(unique)
    ndev = max(1, round(len(unique) * DEV_FRACTION))
    dev_group_ids = set(unique[:ndev])
    train_ix = [i for i in range(len(cases)) if root(i) not in dev_group_ids]
    dev_ix = [i for i in range(len(cases)) if root(i) in dev_group_ids]
    if not train_ix or not dev_ix:
        raise ValueError("grouped split produced empty train or dev")
    train_fam = {cases[i]["family"] for i in train_ix}
    dev_fam = {cases[i]["family"] for i in dev_ix}
    train_ids = {cases[i]["id"] for i in train_ix}
    dev_ids = {cases[i]["id"] for i in dev_ix}
    if train_fam & dev_fam or train_ids & dev_ids:
        raise AssertionError("case/family leakage across train/dev")
    return train_ix, dev_ix, {"component_count": len(unique), "dev_components": ndev,
                              "train_case_ids": sorted(train_ids), "dev_case_ids": sorted(dev_ids),
                              "train_families": sorted(train_fam), "dev_families": sorted(dev_fam)}


def head_only_baseline(cases, train_ix, dev_ix, clean_features, multi_features, model_name):
    """Explicit cached-embedding linear baseline, separate from LoRA training."""
    import sys
    sys.path.insert(0, str(VARIED))
    import multi_train_pairwise as pairwise
    by_key = {(c["id"], c["name"]): c for c in cases}
    vectors = {}
    for path in (clean_features, multi_features):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") != model_name:
                raise ValueError(f"head-only baseline model mismatch in {path}")
            key = (str(row["id"]), str(row["question_key"]))
            case = by_key.get(key)
            if case is None or key in vectors:
                raise ValueError(f"baseline vector key missing/duplicate: {key}")
            q = np.asarray(row["query_embedding"], dtype=np.float64)
            candidates = {str(c["candidate_id"]): np.asarray(c["embedding"], dtype=np.float64)
                          for c in row["candidates"]}
            if set(candidates) != set(case["keys"]):
                raise ValueError(f"baseline candidate vectors differ at {key}")
            case_copy = dict(case)
            case_copy["type"] = case["kind"]
            case_copy["domain"] = ""
            case_copy["question_name"] = case["name"]
            case_copy["x"] = np.stack([pairwise.pair_features(q, candidates[k]) for k in case["keys"]])
            vectors[key] = case_copy
    if len(vectors) != len(cases):
        raise ValueError(f"head-only baseline missing vectors for {len(cases)-len(vectors)} train questions")
    models = {}
    for kind in ("choice", "noul", "score"):
        fit_cases = [vectors[(cases[i]["id"], cases[i]["name"])] for i in train_ix if cases[i]["kind"] == kind]
        targets = [c["target"] for c in fit_cases]
        if fit_cases:
            w, mean, aux = pairwise.fit_bounded(fit_cases, targets, 1.0)
            models[kind] = (w, mean, aux["scale"])
    predicted_rows = pairwise.pred([vectors[(cases[i]["id"], cases[i]["name"])] for i in dev_ix], models)
    prediction_by_key = {(str(p["id"]), str(p["question_name"])): p for p in predicted_rows}
    metrics = {}
    for kind in ("choice", "noul", "score"):
        ix = [i for i in dev_ix if cases[i]["kind"] == kind]
        if not ix:
            continue
        rows = [prediction_by_key[(cases[i]["id"], cases[i]["name"])] for i in ix]
        hits = [pred["selected"] == cases[i]["keys"][cases[i]["target"]] for i, pred in zip(ix, rows)]
        losses = []
        mae = []
        for i, pred in zip(ix, rows):
            p_gold = max(float(pred["probabilities"][cases[i]["keys"][cases[i]["target"]]]), 1e-12)
            losses.append(-math.log(p_gold))
            if kind == "score":
                values = cases[i]["values"]
                span = max(values) - min(values)
                if span > 0:
                    mae.append(abs(float(pred["expected_value"]) - values[cases[i]["target"]]) / span)
        metrics[kind] = {"n": len(ix), "accuracy": float(np.mean(hits)), "cross_entropy": float(np.mean(losses))}
        if kind == "score":
            metrics[kind]["normalized_rubric_expected_value_mae"] = float(np.mean(mae)) if mae else None
    return {"baseline": "explicit_cached_embedding_pairwise_linear_C1", "split": "same grouped train/dev",
            "training_n": len(train_ix), "dev_n": len(dev_ix), "per_type": metrics,
            "macro_accuracy_by_type": float(np.mean([m["accuracy"] for m in metrics.values()])),
            "scope": "head-only comparator; cached embeddings are not used by LoRA training"}


def compiled_text_parity(loaded, train_features: Path, model_name: str):
    """Compare compiled profile text against its Rust-exported train embedding."""
    manifest_path = Path(str(train_features) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != model_name or manifest.get("backend") != "MLX/Metal":
        raise ValueError("compiled-text parity source must be the matching Rust MLX train feature export")
    # Choose one train-side row, and explicitly compare the exact generic typed
    # compiled input strings consumed by training (not raw uncompiled question text).
    row = json.loads(train_features.open(encoding="utf-8").readline())
    candidates = row["candidates"]
    if not candidates:
        raise ValueError("compiled-text parity row has no candidates")
    checks = [("query", row["query_text"], row["query_embedding"]),
              ("candidate", candidates[0]["text"], candidates[0]["embedding"])]
    result = []
    for role, text, reference in checks:
        actual = loaded.embed(text, pooling_dtype="float32")
        mx.eval(actual)
        actual = np.asarray(actual).reshape(-1).astype(np.float64)
        expected = np.asarray(reference, dtype=np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"{role} compiled-text parity shape/nonfinite mismatch")
        max_abs = float(np.max(np.abs(actual - expected)))
        cosine = float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        if cosine < 0.9995 or max_abs > 0.01:
            raise RuntimeError(f"{role} compiled-text parity outside tolerance: cosine={cosine}, max_abs={max_abs}")
        result.append({"role": role, "max_abs_diff": max_abs, "cosine": cosine,
                       "rust_norm": float(np.linalg.norm(expected)), "mlx_norm": float(np.linalg.norm(actual))})
    return {"passed": True, "kind": "compiled profile text vs Rust MLX train export",
            "model": model_name, "profile_version": manifest.get("profile_version"),
            "case_id": row["id"], "question_key": row["question_key"],
            "thresholds": {"min_cosine": 0.9995, "max_abs_diff": 0.01}, "checks": result}


class TypedScorer(nn.Module):
    """Shared generic pair encoder with separate scalar output heads by wire type."""
    def __init__(self, hidden=1024, width=192):
        super().__init__()
        self.proj = nn.Linear(hidden * 4, width, bias=True)
        self.choice = nn.Linear(width, 1, bias=True)
        self.noul = nn.Linear(width, 1, bias=True)
        self.score = nn.Linear(width, 1, bias=True)

    def __call__(self, query, candidates, kind):
        q = mx.broadcast_to(query[None, :], candidates.shape)
        features = mx.concatenate([q, candidates, q * candidates, mx.abs(q - candidates)], axis=-1).astype(mx.float32)
        h = nn.gelu(self.proj(features))
        return getattr(self, kind)(h).squeeze(-1)


class PreferenceModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, token_batch, lengths, candidate_count, kind):
        # Right padding is safe for causal attention; gather each text's actual last token.
        hidden = self.backbone.model(token_batch)
        row_ix = mx.arange(token_batch.shape[0])
        pooled = hidden[row_ix, lengths - 1, :].astype(mx.float32)
        norms = mx.maximum(mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True)), 1e-12)
        pooled = pooled / norms
        query = pooled[0]
        candidates = pooled[1:1 + candidate_count]
        return self.head(query, candidates, kind)


def encode(tokenizer, text: str):
    # Rust tokenizer encode(..., add_special_tokens=True) semantics.
    encoded = tokenizer.encode(text)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    ids = [int(x) for x in ids]
    if not ids:
        raise ValueError("empty tokenized training string")
    if len(ids) > MAX_TOKENS:
        raise ValueError(f"token guard: {len(ids)} > {MAX_TOKENS}")
    return ids


def batch_inputs(case, tokenizer):
    seqs = [encode(tokenizer, case["query"])] + [encode(tokenizer, x) for x in case["candidate_texts"]]
    lengths = [len(x) for x in seqs]
    max_length = max(lengths)
    width = next((bucket for bucket in SEQUENCE_BUCKETS if bucket >= max_length), None)
    if width is None:
        raise ValueError(f"sequence length {max_length} exceeds final evidence-safe bucket {SEQUENCE_BUCKETS[-1]}")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        # For right padding this position is causally invisible to every gathered
        # final real token. ID 0 is used only to fill unused batch tail positions.
        pad_id = 0
    ids = [seq + [int(pad_id)] * (width - len(seq)) for seq in seqs]
    return mx.array(ids, dtype=mx.int32), mx.array(lengths, dtype=mx.int32)


def enable_upper_layer_checkpointing(backbone, layer_count):
    """Checkpoint only LoRA-bearing Qwen blocks using MLX-LM's parameter-pytree pattern."""
    decoder_layers = backbone.model.layers
    if not 1 <= layer_count <= len(decoder_layers):
        raise ValueError(f"invalid checkpoint layer count {layer_count}")
    layer_type = type(decoder_layers[-1])
    original_call = layer_type.__call__
    selected = {id(layer) for layer in decoder_layers[-layer_count:]}

    def checkpointed_call(layer, *args, **kwargs):
        if id(layer) not in selected:
            return original_call(layer, *args, **kwargs)

        def recompute(params, *inner_args, **inner_kwargs):
            layer.update(params)
            return original_call(layer, *inner_args, **inner_kwargs)

        return mx.checkpoint(recompute)(layer.trainable_parameters(), *args, **kwargs)

    layer_type.__call__ = checkpointed_call

    def restore():
        layer_type.__call__ = original_call

    return restore


def preference_loss(logits, case):
    # Always compute local softmax in FP32. Choice/Noul use exact local CE.
    logits = logits.astype(mx.float32)
    target = int(case["target"])
    log_probs = logits - mx.logsumexp(logits, axis=0)
    if case["kind"] != "score":
        return -log_probs[target]
    values = mx.array(case["values"], dtype=mx.float32)
    span = mx.maximum(mx.max(values) - mx.min(values), 1e-6)
    distance = mx.abs(values - values[target]) / span
    soft_target = mx.softmax(-distance / SCORE_VALUE_TEMPERATURE, axis=0)
    ordinal_ce = -mx.sum(soft_target * log_probs)
    probs = mx.softmax(logits, axis=0)
    expected = mx.sum(probs * values)
    target_value = values[target]
    normalized_error = (expected - target_value) / span
    abs_error = mx.abs(normalized_error)
    huber = mx.where(abs_error <= 1.0, 0.5 * normalized_error * normalized_error, abs_error - 0.5)
    return (1.0 - SCORE_ORDINAL_WEIGHT) * ordinal_ce + SCORE_ORDINAL_WEIGHT * huber


def trainable_count(model):
    return sum(int(v.size) for _, v in tree_flatten(model.trainable_parameters()))


def save_safetensors_atomic(path: Path, arrays):
    tmp = path.with_name(path.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), dict(tree_flatten(arrays)))
    os.replace(tmp, path)


def memory_snapshot():
    """Capture MLX allocator counters and macOS process physical footprint."""
    active = int(mx.get_active_memory())
    peak = int(mx.get_peak_memory())
    proc = subprocess.run(["footprint", "--format", "bytes", "--noCategories", str(os.getpid())],
                          text=True, capture_output=True, check=True)
    current_match = re.search(r"Footprint:\s+([0-9,]+)\s+B", proc.stdout)
    peak_match = re.search(r"phys_footprint_peak:\s+([0-9,]+)\s+B", proc.stdout)
    if not current_match or not peak_match:
        raise RuntimeError(f"could not parse `footprint` output: {proc.stdout[:300]}")
    return {"pid": os.getpid(), "mlx_active_bytes": active, "mlx_peak_bytes": peak,
            "process_footprint_bytes": int(current_match.group(1).replace(",", "")),
            "process_peak_footprint_bytes": int(peak_match.group(1).replace(",", "")),
            "system_swap_used_bytes": system_swap_used_bytes()}


def system_swap_used_bytes():
    swap = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, check=True).stdout
    swap_match = re.search(r"used\s*=\s*([0-9.]+)M", swap)
    if not swap_match:
        raise RuntimeError(f"could not parse system swap usage: {swap}")
    return int(float(swap_match.group(1)) * 1024 ** 2)


def restore_optimizer_state(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"missing resumable optimizer state: {path}")
    return tree_unflatten(list(mx.load(str(path)).items()))


def eval_cases(model, cases, indices, tokenizer):
    losses, hits = [], []
    for i in indices:
        case = cases[i]
        tokens, lengths = batch_inputs(case, tokenizer)
        logits = model(tokens, lengths, len(case["keys"]), case["kind"])
        loss = preference_loss(logits, case)
        mx.eval(loss)
        arr = np.asarray(loss).item()
        if not math.isfinite(arr):
            raise FloatingPointError("nonfinite dev loss")
        losses.append(float(arr))
        hits.append(int(int(np.argmax(np.asarray(logits))) == case["target"]))
    return {"n": len(indices), "loss": float(np.mean(losses)), "exact_accuracy": float(np.mean(hits))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=MODEL_CONFIGS)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    p.add_argument("--multi", type=Path, default=DEFAULT_MULTI)
    p.add_argument("--adapter", type=Path, help="checkpoint_adapter.py path; default sibling")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-only", action="store_true", help="run one real train minibatch and stop")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help="bounded optimizer-step limit; partial runs do not emit an epoch checkpoint")
    args = p.parse_args()
    if not 1 <= args.max_steps <= MAX_STEPS:
        raise ValueError(f"--max-steps must be between 1 and {MAX_STEPS}")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite nonempty run directory {args.out}")
    config_dir, clean_features, multi_features = MODEL_CONFIGS[args.model]
    adapter_path = args.adapter or Path(__file__).with_name("checkpoint_adapter.py")
    if not adapter_path.is_file():
        raise FileNotFoundError(f"checkpoint adapter is not ready: {adapter_path}")
    spec = importlib.util.spec_from_file_location("varied_lora_checkpoint_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    if not hasattr(adapter, "load_checkpoint"):
        raise AttributeError("checkpoint_adapter.py must expose load_checkpoint(path) -> (model, tokenizer)")

    random.seed(SEED); np.random.seed(SEED); mx.random.seed(SEED)
    raw_rows, cases = train_rows(args.clean, args.multi, clean_features, multi_features)
    train_ix, dev_ix, split = grouped_split(cases)
    token_lengths = [n for case in cases for n in case["compiled_token_lengths"]]
    token_distribution = {"n_texts": len(token_lengths), "p50": float(np.percentile(token_lengths, 50)),
                          "p90": float(np.percentile(token_lengths, 90)),
                          "p95": float(np.percentile(token_lengths, 95)),
                          "max": int(max(token_lengths)), "guard": MAX_TOKENS,
                          "overflow_policy": "reject; no truncation"}
    if token_distribution["max"] > MAX_TOKENS:
        raise ValueError(f"train text max {token_distribution['max']} exceeds {MAX_TOKENS} token guard")
    baseline = head_only_baseline(cases, train_ix, dev_ix, clean_features, multi_features, args.model)
    model_dir = args.store / "models" / config_dir
    start = time.time()
    swap_before_model_load = system_swap_used_bytes()
    loaded = adapter.load_checkpoint(model_dir)
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        backbone, tokenizer = loaded[:2]
        parity_report = loaded[2] if len(loaded) > 2 else None
        wrapped_checkpoint = None
    else:
        wrapped_checkpoint = loaded
        backbone, tokenizer = loaded.model, loaded.tokenizer
        parity_report = compiled_text_parity(loaded, clean_features, args.model)
    if not parity_report or parity_report.get("passed") is not True:
        raise RuntimeError("adapter must supply a successful initial Rust-vs-MLX embedding parity report")
    if wrapped_checkpoint is not None and wrapped_checkpoint.checkpoint_dtype != mx.bfloat16:
        raise TypeError(f"expected frozen BF16 checkpoint, got {wrapped_checkpoint.checkpoint_dtype}")
    if any(value.dtype != mx.bfloat16 for _, value in tree_flatten(backbone.parameters())):
        raise TypeError("every loaded backbone parameter must remain BF16 before adapter injection")
    # Frozen BF16 backbone; replace only final-layer q/v with trainable FP32 LoRA.
    inject_lora = getattr(adapter, "inject_upper_lora", None)
    if inject_lora is not None and wrapped_checkpoint is not None:
        lora_modules = inject_lora(wrapped_checkpoint, layers=LORA_LAST_LAYERS,
                                   projections=("q_proj", "v_proj"), rank=LORA_RANK, alpha=LORA_ALPHA)
    else:
        backbone.freeze()
        raise RuntimeError("checkpoint adapter must provide inject_upper_lora for its validated LoRA modules")
    restore_checkpointing = enable_upper_layer_checkpointing(backbone, LORA_LAST_LAYERS)
    for name, value in tree_flatten(backbone.trainable_parameters()):
        if "lora_" not in str(name):
            raise RuntimeError(f"unexpected unfrozen backbone parameter: {name}")
        if value.dtype != mx.float32:
            raise TypeError(f"LoRA parameter must be FP32: {name} is {value.dtype}")
    head = TypedScorer()
    head.set_dtype(mx.float32)
    if any(value.dtype != mx.float32 for _, value in tree_flatten(head.parameters())):
        raise TypeError("typed scorer parameters must be FP32")
    model = PreferenceModel(backbone, head)
    if trainable_count(model) == 0:
        raise RuntimeError("no trainable parameters: expected LoRA and typed heads")

    ckpt = model_dir / "model.safetensors"
    source_hashes = {"checkpoint": sha(ckpt), "clean": sha(args.clean), "multi_train": sha(args.multi),
                     "clean_feature_text_source": sha(clean_features), "multi_feature_text_source": sha(multi_features),
                     "adapter": sha(adapter_path)}
    config = {
        "schema_version": 1, "model": args.model, "seed": SEED, "split": split,
        "train_question_count": len(cases), "train_indices": train_ix, "dev_indices": dev_ix,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP, "max_epochs": MAX_EPOCHS, "max_steps": args.max_steps,
        "batch_size_questions": 1, "gradient_accumulation_steps": 1, "max_tokens_per_text": MAX_TOKENS,
        "lora_targets": lora_modules, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "base_dtype": "bfloat16", "trainable_dtype": "float32", "score_ordinal_weight": SCORE_ORDINAL_WEIGHT,
        "score_value_temperature": SCORE_VALUE_TEMPERATURE, "source_sha256": source_hashes,
        "trainer_sha256": sha(Path(__file__).resolve()), "parity_report": parity_report,
        "mlx_device": str(mx.default_device()), "mlx_device_info": mx.metal.device_info(),
        "token_distribution": token_distribution,
        "memory_budget": {"host_bytes": 24 * 1024 ** 3, "hard_process_peak_cap_bytes": 8 * 1024 ** 3,
                          "stop_threshold_bytes": PROCESS_FOOTPRINT_LIMIT,
                          "system_swap_stop_bytes": SYSTEM_SWAP_LIMIT,
                          "memory_check_interval_steps": MEMORY_CHECK_INTERVAL,
                          "micro_batch_questions": 1, "gradient_accumulation_steps": 1,
                          "activation_checkpointing": "MLX checkpoint on LoRA-bearing upper block(s)",
                          "cache_clear_interval_steps": MEMORY_CHECK_INTERVAL,
                          "swap_must_not_increase_from_run_start": True},
    }
    config_path = args.out / "config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise ValueError("resume config does not exactly match the frozen run config")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.out / "head-only-baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    logs = args.out / "steps.jsonl"
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_dev = float("inf"); best_epoch = -1; best_path = args.out / "best.safetensors"
    step = 0
    start_epoch = 0
    if args.resume:
        latest = json.loads((args.out / "latest.json").read_text(encoding="utf-8"))
        model.load_weights(list(mx.load(str(args.out / "latest.safetensors")).items()), strict=False)
        if int(latest["completed_epoch"]) > 0:
            optimizer.state = restore_optimizer_state(args.out / "optimizer.safetensors")
        start_epoch = int(latest["completed_epoch"])
        step = int(latest["step"])
        if start_epoch < 0 or start_epoch > MAX_EPOCHS:
            raise ValueError("invalid resumable epoch marker")
        # Drop partial epoch entries: the deterministic epoch is replayed from its
        # last fully committed checkpoint, so every retained row is reproducible.
        kept = []
        if logs.exists():
            for line in logs.read_text(encoding="utf-8").splitlines():
                if line.strip() and int(json.loads(line)["step"]) <= step:
                    kept.append(line)
        logs.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        epochs_path = args.out / "epochs.jsonl"
        if epochs_path.exists():
            rows = [line for line in epochs_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and int(json.loads(line)["epoch"]) <= start_epoch]
            epochs_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        if (args.out / "best.json").exists():
            best_record = json.loads((args.out / "best.json").read_text())
            best_epoch = int(best_record["epoch"])
            best_dev = float(best_record["dev"]["loss"])
    else:
        save_safetensors_atomic(args.out / "latest.safetensors", model.trainable_parameters())
        (args.out / "latest.json").write_text(json.dumps({"completed_epoch": 0, "step": 0}, indent=2) + "\n")
    mx.reset_peak_memory()
    startup_memory = memory_snapshot()
    with (args.out / "memory.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "startup_or_resume", "swap_baseline_before_model_load_bytes": swap_before_model_load,
                            **startup_memory}, sort_keys=True) + "\n")
    if startup_memory["process_peak_footprint_bytes"] > PROCESS_FOOTPRINT_LIMIT:
        raise MemoryError(f"startup footprint exceeds safe 6 GiB stop threshold: {startup_memory}")
    if startup_memory["system_swap_used_bytes"] > SYSTEM_SWAP_LIMIT or startup_memory["system_swap_used_bytes"] > swap_before_model_load:
        raise MemoryError(f"system swap increased/exceeded 4 GiB during model load: {startup_memory}")
    for epoch in range(start_epoch, MAX_EPOCHS):
        order = list(train_ix)
        random.Random(SEED + epoch).shuffle(order)
        epoch_losses = []
        for i in order:
            case = cases[i]
            tokens, lengths = batch_inputs(case, tokenizer)
            batch_width = int(tokens.shape[1])
            token_sum = int(np.asarray(mx.sum(lengths)))
            token_max = int(np.asarray(mx.max(lengths)))
            def loss_fn(m):
                logits = m(tokens, lengths, len(case["keys"]), case["kind"])
                return preference_loss(logits, case)
            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model)
            mx.eval(loss, grads)
            loss_value = float(np.asarray(loss))
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"nonfinite loss at step={step}")
            flat = tree_flatten(grads)
            norm = math.sqrt(sum(float(np.asarray(mx.sum(v.astype(mx.float32) ** 2))) for _, v in flat))
            adapter_norm = math.sqrt(sum(float(np.asarray(mx.sum(v.astype(mx.float32) ** 2)))
                                          for name, v in flat if "lora_" in str(name)))
            if not math.isfinite(norm):
                raise FloatingPointError(f"nonfinite gradient norm at step={step}")
            if step == 0 and norm <= 0:
                raise RuntimeError("smoke check failed: real batch has zero parameter gradient")
            if step == 0 and adapter_norm <= 0:
                raise RuntimeError("smoke check failed: real batch has zero LoRA gradient")
            scale = min(1.0, GRAD_CLIP / max(norm, 1e-12))
            if scale != 1.0:
                grads = tree_map(lambda x: x * scale, grads)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            if step == 0:
                after = loss_fn(model)
                mx.eval(after)
                after_value = float(np.asarray(after))
                if not math.isfinite(after_value) or after_value >= loss_value:
                    raise RuntimeError(f"real minibatch smoke loss did not decrease: {loss_value} -> {after_value}")
            else:
                after_value = None
            epoch_losses.append(loss_value)
            step += 1
            with logs.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "epoch": epoch + 1, "case": case["id"],
                                    "question": case["name"], "type": case["kind"],
                                    "candidate_count": len(case["keys"]), "batch_rows": len(case["keys"]) + 1,
                                    "batch_width": batch_width, "max_text_tokens": token_max,
                                    "sum_text_tokens": token_sum,
                                    "loss": loss_value, "grad_norm": norm, "lora_grad_norm": adapter_norm,
                                    "post_update_loss": after_value, "clip_scale": scale,
                                    "elapsed_seconds": time.time() - start}, sort_keys=True) + "\n")
            # Drop all references into the just-evaluated graph before asking MLX
            # to return inactive buffers. Keep only Python scalar log metrics.
            del loss, grads, flat, loss_and_grad, loss_fn, tokens, lengths
            if "after" in locals():
                del after
            gc.collect()
            mx.clear_cache()
            mem = memory_snapshot()
            with (args.out / "memory.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "step", "step": step, "candidate_count": len(case["keys"]),
                                    "batch_rows": len(case["keys"]) + 1, "batch_width": batch_width,
                                    "max_text_tokens": token_max, "sum_text_tokens": token_sum,
                                    "swap_baseline_before_model_load_bytes": swap_before_model_load,
                                    **mem}, sort_keys=True) + "\n")
            if mem["process_peak_footprint_bytes"] >= 8 * 1024 ** 3:
                raise MemoryError(f"absolute 8 GiB process peak cap reached: {mem}")
            if mem["process_peak_footprint_bytes"] >= PROCESS_FOOTPRINT_LIMIT:
                raise MemoryError(f"process peak reached 6 GiB stop threshold: {mem}")
            if mem["system_swap_used_bytes"] > swap_before_model_load:
                raise MemoryError(f"system swap increased from run start: {mem}")
            if mem["system_swap_used_bytes"] > SYSTEM_SWAP_LIMIT:
                raise MemoryError(f"system swap exceeded 4 GiB threshold: {mem}")
            if args.smoke_only and step >= 1:
                save_safetensors_atomic(args.out / "smoke.safetensors", model.trainable_parameters())
                smoke = {"model": args.model, "case": case["id"], "question": case["name"],
                         "type": case["kind"], "loss_before": loss_value, "loss_after": after_value,
                         "gradient_norm": norm, "lora_gradient_norm": adapter_norm,
                         "parity": parity_report, "status": "real_minibatch_gradient_and_decrease_passed"}
                (args.out / "smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
                print(json.dumps(smoke, sort_keys=True))
                return
            if step >= args.max_steps:
                break
        if len(epoch_losses) < len(order):
            summary = {"model": args.model, "steps": step, "status": "pilot_stopped_in_epoch",
                       "best_epoch": best_epoch, "best_dev_loss": best_dev,
                       "note": "No epoch checkpoint or model selection emitted for a partial epoch."}
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            restore_checkpointing()
            print(json.dumps(summary, sort_keys=True))
            return
        # Train-only diagnostics plus grouped dev selection.
        dev = eval_cases(model, cases, dev_ix, tokenizer)
        mx.clear_cache()
        mem = memory_snapshot()
        with (args.out / "memory.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "epoch_boundary", "epoch": epoch + 1, **mem}, sort_keys=True) + "\n")
        if mem["process_peak_footprint_bytes"] > PROCESS_FOOTPRINT_LIMIT:
            raise MemoryError(f"epoch-boundary process peak exceeded 6 GiB stop threshold: {mem}")
        if mem["system_swap_used_bytes"] > SYSTEM_SWAP_LIMIT:
            raise MemoryError(f"epoch-boundary system swap exceeded 4 GiB threshold: {mem}")
        train_loss = float(np.mean(epoch_losses))
        record = {"epoch": epoch + 1, "steps": step, "train_loss": train_loss, "dev": dev}
        with (args.out / "epochs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        if epoch == 0 and (not math.isfinite(train_loss) or train_loss <= 0):
            raise RuntimeError("smoke check failed: invalid real-minibatch loss")
        if dev["loss"] < best_dev:
            best_dev, best_epoch = dev["loss"], epoch + 1
            save_safetensors_atomic(best_path, model.trainable_parameters())
            (args.out / "best.json").write_text(json.dumps({"epoch": best_epoch, "dev": dev,
                                                              "train_loss": train_loss}, indent=2) + "\n")
        # Commit a restartable checkpoint at each epoch boundary. An interrupted
        # epoch is replayed deterministically on --resume; its partial logs are cut.
        save_safetensors_atomic(args.out / "latest.safetensors", model.trainable_parameters())
        optimizer_arrays = dict(tree_flatten(optimizer.state))
        save_safetensors_atomic(args.out / "optimizer.safetensors", optimizer_arrays)
        (args.out / "latest.json").write_text(json.dumps({"completed_epoch": epoch + 1, "step": step}, indent=2) + "\n")
        if step >= args.max_steps:
            break
    restore_checkpointing()
    summary = {"model": args.model, "steps": step, "best_epoch": best_epoch,
               "best_dev_loss": best_dev, "status": "train_dev_complete",
               "scope": "train+grouped-dev only; no heldout accessed"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
