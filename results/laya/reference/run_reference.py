#!/usr/bin/env python3
"""Capture one official Laya CPU inference as a parity fixture.

The model implementation is imported from the pinned Hugging Face snapshot;
this runner only adapts probe-test-300's question schema and records the
official RLAgent.system_one forward pass, including its pre-calibration logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "convaiinnovations/laya"
REVISION = "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"
WEIGHT_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
DEFAULT_STORE = Path("/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/laya-root")
TASKS = ("owner", "urgent", "impact")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_laya_question(q: dict[str, Any]) -> dict[str, Any]:
    if q["type"] == "choice":
        criteria = q["options"]
    elif q["type"] == "score":
        criteria = [f"{level['label']}: {level['criterion']}" for level in q["levels"]]
    elif q["type"] == "noul":
        criteria = None
    else:
        raise ValueError(f"unsupported question type: {q['type']!r}")
    return {"type": q["type"], "instructions": q["prompt"], "criteria": criteria}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="test-001")
    parser.add_argument("--fixture", type=Path, default=ROOT / "benchmarks/cases/probe-test-300.jsonl")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "cpu-reference.json")
    args = parser.parse_args()

    store = args.store.resolve()
    checkpoint = store / "checkpoint"
    # Import the Hub-pinned, published inference implementation, not the pip
    # SDK or the experimental MLX helper.
    sys.path.insert(0, str(checkpoint))
    import numpy as np
    import torch
    import transformers
    from rl_agent_api import RLAgent
    from rl_common import QTYPES, build_sequence, render_options, temp_bucket

    if transformers.__version__ != "5.0.0":
        raise RuntimeError(f"expected Transformers 5.0.0 for this checkpoint, got {transformers.__version__}")
    weight_path = checkpoint / "model.safetensors"
    actual_weight_hash = sha256(weight_path)
    if actual_weight_hash != WEIGHT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {actual_weight_hash}")

    rows = [json.loads(line) for line in args.fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if row.get("id") == args.case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {args.case_id} in {args.fixture}")
    case = matches[0]
    state = case["request"]["state"]
    questions = {task: as_laya_question(case["request"]["questions"][task]) for task in TASKS}

    # Ensure this is strictly a CPU reference and avoid occupying accelerators.
    torch.set_num_threads(int(os.environ.get("LAYA_CPU_THREADS", "4")))
    agent = RLAgent(str(checkpoint), device="cpu")
    if str(agent.device) != "cpu":
        raise RuntimeError(f"unexpected inference device: {agent.device}")

    decisions: dict[str, Any] = {}
    for task in TASKS:
        q = agent._to_internal(questions[task])
        ids, marker_positions = build_sequence(
            agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"]
        )
        if len(marker_positions) != len(render_options(q)):
            raise RuntimeError(f"{task}: official sequence truncated answer markers")

        captured: dict[str, Any] = {}

        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
            for name, tensor in zip(names, inputs):
                captured[name] = tensor.detach().cpu().tolist()

        def post_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            logits, act_logits = output
            captured["option_logits_before_calibration"] = logits.detach().float().cpu().tolist()
            captured["action_logits_before_softmax"] = act_logits.detach().float().cpu().tolist()

        pre = agent.model.register_forward_pre_hook(pre_hook)
        post = agent.model.register_forward_hook(post_hook)
        try:
            # This is the public, pinned Hub implementation's CPU prediction
            # path. Hooks record the exact tensors and raw model outputs.
            result = agent.system_one(state, {task: questions[task]})
        finally:
            pre.remove()
            post.remove()

        sequences = captured["input_ids"][0]
        attention = captured["attention_mask"][0]
        if sequences != ids or len(sequences) != len(attention):
            raise RuntimeError(f"{task}: recorded sequence differs from official build_sequence")
        raw_option_logits = captured["option_logits_before_calibration"][0]
        raw_action_logits = captured["action_logits_before_softmax"][0]
        option_count = len(marker_positions)
        temperature_key = temp_bucket(QTYPES[q["t"]], option_count)
        temperature = agent.temperature_by_options.get(temperature_key, agent.temperature[QTYPES[q["t"]]])
        scaled = np.asarray(raw_option_logits[:option_count], dtype=np.float32) / temperature
        option_exp = np.exp(scaled - scaled.max())
        calibrated = option_exp / option_exp.sum()
        options = render_options(q)
        if q["t"] == "choice":
            option_keys = list(q["crit"].keys())
        elif q["t"] == "score":
            option_keys = [str(i) for i in range(option_count)]
        else:
            option_keys = ["false", "true"]
        action_probabilities = torch.softmax(
            torch.tensor(raw_action_logits, dtype=torch.float32), dim=-1
        ).tolist()
        calibrated_distribution = {
            key: round(float(probability), 4)
            for key, probability in zip(option_keys, calibrated)
        }
        decisions[task] = {
            "question": questions[task],
            "sequence_length": len(sequences),
            "input_ids": sequences,
            "attention_mask": attention,
            "marker_positions": captured["marker_pos"][0],
            "marker_mask": captured["marker_mask"][0],
            "qtype_id": captured["qtype"][0],
            "raw_option_logits_before_calibration": raw_option_logits,
            "raw_action_logits_before_softmax": raw_action_logits,
            "raw_action_probabilities": action_probabilities,
            "option_temperature_bucket": temperature_key,
            "option_temperature": float(temperature),
            "calibrated_option_distribution": calibrated_distribution,
            "calibrated_probabilities": result["answers"][task].get("probabilities", calibrated_distribution),
            "official_output": result["answers"][task],
            "official_usage": result.get("usage"),
        }

    if min(x["sequence_length"] for x in decisions.values()) >= 128 or not any(
        x["sequence_length"] > 128 for x in decisions.values()
    ):
        raise RuntimeError("fixture must contain both a <128-token and a >128-token input")

    try:
        import laya
        laya_version = getattr(laya, "__version__", "unknown")
    except Exception:
        laya_version = None
    output = {
        "schema_version": 1,
        "model": MODEL_ID,
        "model_revision": REVISION,
        "checkpoint_sha256": actual_weight_hash,
        "checkpoint_files_sha256": {
            str(p.relative_to(checkpoint)): sha256(p)
            for p in sorted(checkpoint.rglob("*"))
            if p.is_file() and ".cache" not in p.parts and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
        },
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": sha256(args.fixture),
        "case_id": case["id"],
        "case_state": state,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "laya_sdk_package_version_installed_but_not_used_for_inference": laya_version,
            "official_implementation": "rl_agent_api.py + rl_common.py from pinned Hub snapshot",
            "device": str(agent.device),
            "configured_amp_dtype_not_used_on_cpu": str(agent.dtype),
            "actual_model_parameter_dtype": str(next(agent.model.parameters()).dtype),
            "threads": torch.get_num_threads(),
            "autocast": False,
            "head_max_len": agent.cfg["head_max_len"],
            "max_len": agent.cfg["max_len"],
        },
        "decisions": decisions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved official CPU parity reference for {case['id']} -> {args.out}")
    for task, item in decisions.items():
        print(f"  {task}: {item['sequence_length']} tokens; {item['official_output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
