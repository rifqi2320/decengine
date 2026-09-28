"""Strict MLX-LM adapter for decengine's native Qwen3 embedding checkpoints.

The Rust checkpoints contain decoder weights without a ``model.`` prefix, whereas
``mlx_lm.models.qwen3.Model`` registers its decoder under ``model.*``.  This file
is deliberately inference/training glue only; it does not perform model downloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs
from tokenizers import Tokenizer


DEFAULT_MODEL_STORE = Path(
    "/private/var/folders/_c/j5jmxy5967sf4vqn028g6zj80000gq/T/opencode/decengine-model-store"
)


def _model_args(config: dict[str, Any]) -> ModelArgs:
    """Build mlx-lm Qwen3 args from either installed embedding checkpoint config."""
    required = (
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "rms_norm_eps",
        "vocab_size",
        "num_key_value_heads",
        "max_position_embeddings",
        "rope_theta",
        "head_dim",
        "tie_word_embeddings",
    )
    missing = [key for key in required if key not in config]
    if config.get("model_type") != "qwen3" or missing:
        raise ValueError(
            f"expected Qwen3 config; missing={missing}, "
            f"model_type={config.get('model_type')!r}"
        )
    values = {key: config[key] for key in required}
    return ModelArgs(model_type="qwen3", rope_scaling=config.get("rope_scaling"), **values)


def map_decoder_weights(weights: dict[str, mx.array], model: Model) -> list[tuple[str, mx.array]]:
    """Return a validated one-to-one source -> mlx-lm model parameter mapping.

    Accepts the unprefixed decoder namespace used by Rust or the already nested
    ``model.*`` namespace.  It rejects mixed/duplicate aliases, missing or extra
    tensors (including lm_head), shape drift, and non-BF16/FP32 checkpoints before
    delegating to MLX's own strict loader.
    """
    expected = dict(tree_flatten(model.parameters()))
    if not expected:
        raise ValueError("mlx-lm Qwen3 model unexpectedly has no parameters")
    source_names = set(weights)
    nested = {name.startswith("model.") for name in source_names}
    if len(nested) != 1:
        raise ValueError("checkpoint has mixed decoder namespaces")
    is_nested = nested.pop()
    mapped: dict[str, mx.array] = {}
    reverse: dict[str, str] = {}
    for source_name, value in weights.items():
        target_name = source_name if is_nested else f"model.{source_name}"
        if target_name in mapped:
            raise ValueError(f"multiple source tensors map to {target_name!r}")
        mapped[target_name] = value
        reverse[target_name] = source_name

    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    if missing or unexpected or len(mapped) != len(weights):
        raise ValueError(
            "checkpoint/model tensor count mismatch: "
            f"source={len(weights)} mapped={len(mapped)} expected={len(expected)} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for name, target in expected.items():
        value = mapped[name]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for {reverse[name]} -> {name}: "
                f"checkpoint={tuple(value.shape)} model={tuple(target.shape)}"
            )
        if value.dtype not in (mx.bfloat16, mx.float32):
            raise ValueError(f"unsupported dtype for {reverse[name]}: {value.dtype}")
    dtypes = {value.dtype for value in mapped.values()}
    if len(dtypes) != 1:
        raise ValueError(f"checkpoint tensors have mixed dtypes: {dtypes}")
    return sorted(mapped.items())


@dataclass
class Qwen3Embedding:
    """Loaded decoder and tokenizer. Use ``model.model`` for hidden states, not logits."""

    model: Model
    tokenizer: Tokenizer
    max_length: int
    checkpoint_dtype: Any

    def tokenize(self, text: str) -> mx.array:
        ids = self.tokenizer.encode(text, add_special_tokens=True).ids
        if not ids:
            raise ValueError("tokenizer produced an empty input")
        if len(ids) > self.max_length:
            raise ValueError(f"input has {len(ids)} tokens; maximum is {self.max_length}")
        return mx.array([ids], dtype=mx.int32)

    def forward_hidden(self, input_ids: mx.array) -> mx.array:
        """Decoder last hidden states with shape ``(batch, sequence, hidden)``."""
        return self.model.model(input_ids)

    def embed(self, text_or_ids: str | mx.array, pooling_dtype: str = "float32") -> mx.array:
        """Return last-token, L2-normalized embeddings as an MLX Array.

        ``float32`` matches Rust's pooling path. ``bfloat16`` is provided for the
        explicitly comparable lower-precision pooling probe; the returned array
        retains the selected pooling dtype.
        """
        ids = self.tokenize(text_or_ids) if isinstance(text_or_ids, str) else text_or_ids
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError(f"expected nonempty token IDs shaped (1, sequence), got {ids.shape}")
        pooled = self.forward_hidden(ids)[:, -1, :]
        if pooling_dtype == "float32":
            pooled = pooled.astype(mx.float32)
        elif pooling_dtype == "bfloat16":
            pooled = pooled.astype(mx.bfloat16)
        else:
            raise ValueError("pooling_dtype must be 'float32' or 'bfloat16'")
        return pooled / mx.sqrt(mx.sum(mx.square(pooled), axis=-1, keepdims=True))


def load_checkpoint(model_dir: str | Path, max_length: int = 32768) -> Qwen3Embedding:
    """Load Qwen3 or Harrier checkpoint locally with complete strict validation."""
    root = Path(model_dir)
    with (root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    model = Model(_model_args(config))
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint in {root}")
    weights: dict[str, mx.array] = {}
    for path in files:
        for name, value in mx.load(path).items():
            if name in weights:
                raise ValueError(f"duplicate checkpoint tensor {name!r}")
            weights[name] = value
    mapped = map_decoder_weights(weights, model)
    dtype = mapped[0][1].dtype
    model.load_weights(mapped, strict=True)
    model.eval()
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    return Qwen3Embedding(
        model=model,
        tokenizer=tokenizer,
        max_length=min(max_length, int(config["max_position_embeddings"])),
        checkpoint_dtype=dtype,
    )


class LoRALinear(nn.Module):
    """Frozen linear projection plus trainable FP32 low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.freeze()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = mx.random.normal((rank, base.weight.shape[1])) / (base.weight.shape[1] ** 0.5)
        self.lora_b = mx.zeros((base.weight.shape[0], rank), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        residual = (x.astype(mx.float32) @ self.lora_a.T @ self.lora_b.T) * self.scale
        return self.base(x) + residual.astype(self.base.weight.dtype)


def inject_upper_lora(
    loaded: Qwen3Embedding,
    layers: int = 4,
    projections: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
) -> list[str]:
    """Freeze the backbone and inject LoRA into selected upper decoder projections."""
    decoder_layers = loaded.model.model.layers
    if layers < 1 or layers > len(decoder_layers):
        raise ValueError(f"layers must be between 1 and {len(decoder_layers)}")
    if not projections:
        raise ValueError("at least one projection is required")
    loaded.model.freeze()
    injected = []
    for layer_index in range(len(decoder_layers) - layers, len(decoder_layers)):
        attention = decoder_layers[layer_index].self_attn
        for projection in projections:
            base = getattr(attention, projection, None)
            if not isinstance(base, nn.Linear):
                raise ValueError(f"unknown Qwen3 attention projection {projection!r}")
            setattr(attention, projection, LoRALinear(base, rank=rank, alpha=alpha))
            injected.append(f"layers.{layer_index}.self_attn.{projection}")
    return injected
