"""Faithful Laya root-checkpoint input/output helpers for a future MLX backend.

This module intentionally does not provide a PyTorch/CPU fallback.  The
ModernBERT encoder and decision-head forward pass are not implemented here yet;
``backend`` must be supplied by a genuine MLX implementation before inference
can be claimed.
"""
from __future__ import annotations

import json
import math
from typing import Any, Protocol


class Tokenizer(Protocol):
    mask_token: str
    mask_token_id: int
    cls_token_id: int
    sep_token_id: int

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, Any]: ...


class MissingMLXBackend(RuntimeError):
    """Raised rather than quietly running a different inference backend."""


def serialize_state(state: Any) -> str:
    """Match ``rl_common.serialize_state`` including insertion order/UTF-8."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_options(question: dict[str, Any]) -> list[str]:
    """Render options exactly as the official root-checkpoint implementation."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = {value: None for value in criteria}
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a nonempty mapping or list")
        return [str(key) if not value else f"{key}: {value}" for key, value in criteria.items()]
    if kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("score criteria must be a nonempty list")
        return [f"level {i}: {criterion}" for i, criterion in enumerate(criteria)]
    if kind == "noul":
        criteria = criteria or {}
        return ["false: " + (criteria.get("false") or "no, the statement does not hold"),
                "true: " + (criteria.get("true") or "yes, the statement holds")]
    raise ValueError(f"unsupported Laya question type: {kind!r}")


def build_sequence(tokenizer: Tokenizer, state: Any, question: dict[str, Any], *,
                   max_len: int = 512, head_max_len: int = 192,
                   truncate_left: bool = False) -> tuple[list[int], list[int]]:
    """Exact official ``build_sequence`` rendering; return IDs and mask positions."""
    mask = tokenizer.mask_token
    options = render_options(question)
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions)
    instructions = instructions.replace(mask, " ")
    kind = question["type"]
    head_ids = tokenizer(f"{kind} question: {instructions}", add_special_tokens=False)["input_ids"]
    option_ids = [
        [tokenizer.mask_token_id]
        + tokenizer(" " + option.replace(mask, " "), add_special_tokens=False)["input_ids"][:48]
        for option in options
    ]
    budget = head_max_len - sum(map(len, option_ids))
    if budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [ids[:per_option] for ids in option_ids]
        budget = head_max_len - sum(map(len, option_ids))
    head_ids = head_ids[:max(8, budget)]
    ids = [tokenizer.cls_token_id, *head_ids, tokenizer.sep_token_id]
    markers: list[int] = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    state_ids = tokenizer(serialize_state(state).replace(mask, " "),
                          add_special_tokens=False)["input_ids"]
    state_ids = state_ids[-room:] if truncate_left else state_ids[:room]
    ids = (ids + state_ids + [tokenizer.sep_token_id])[:max_len]
    return ids, [pos for pos in markers if pos < max_len]


def temperature_bucket(kind: str, option_count: int) -> str:
    bucket = "2" if option_count <= 2 else "3-5" if option_count <= 5 else (
        "6-10" if option_count <= 10 else "11+")
    return f"{kind}:{bucket}"


def postprocess(question: dict[str, Any], logits: list[float], act_logits: list[float],
                config: dict[str, Any]) -> dict[str, Any]:
    """Official Laya calibrated choice/noul/score formatting from raw head logits."""
    options = render_options(question)
    if len(logits) != len(options):
        raise ValueError(f"expected {len(options)} option logits, received {len(logits)}")
    if not act_logits:
        raise ValueError("act head must return at least one logit")
    kind = question["type"]
    qt = {"choice": 0, "score": 1, "noul": 2}[kind]
    temperatures = config.get("temperature_by_options", {})
    temperature = temperatures.get(temperature_bucket(kind, len(options)),
                                   config.get("temperature", [1., 1., 1.])[qt])
    scaled = [float(z) / float(temperature) for z in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    probs = [x / total for x in exps]
    act_peak = max(act_logits)
    act_exp = [math.exp(float(x) - act_peak) for x in act_logits]
    act_probability = act_exp[0] / sum(act_exp)
    answer: dict[str, Any] = {"type": kind, "rl_agent": {"act_probability": float(act_probability)}}
    if kind == "choice":
        criteria = question["criteria"]
        keys = list(criteria) if isinstance(criteria, dict) else list(criteria)
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        answer.update(choice=keys[max(range(len(probs)), key=probs.__getitem__)],
                      probabilities={key: round(p, 4) for key, p in zip(keys, probs)},
                      confidence=round(1 - entropy / math.log(len(probs)), 4))
    elif kind == "score":
        criteria = question["criteria"]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        answer.update(score=round(sum(i * p for i, p in enumerate(probs)), 4),
                      legend={str(i): value for i, value in enumerate(criteria)},
                      probabilities={str(i): round(p, 4) for i, p in enumerate(probs)},
                      confidence=round(1 - entropy / math.log(len(probs)), 4))
    else:
        answer["noul"] = round(probs[1], 4)
    return answer


def require_mlx_backend(backend: Any | None) -> Any:
    """Fail closed until the genuine encoder+head implementation is available."""
    if backend is None or getattr(backend, "backend_name", None) != "mlx-metal":
        raise MissingMLXBackend(
            "No genuine Laya MLX Metal backend is installed; refusing CPU/PyTorch fallback. "
            "Only exact tokenization/postprocessing helpers are currently available."
        )
    return backend
