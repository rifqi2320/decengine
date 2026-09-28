"""Inference-only ModernBERT encoder implemented with MLX.

Architecture follows the pinned Laya root encoder config and the public
Transformers ModernBERT implementation (Apache-2.0).  This module deliberately
has no Torch import/fallback.  Weights are read from safetensors as NumPy arrays
and copied to MLX; unexpected or missing encoder tensors are errors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    import mlx.core as mx
except ImportError as exc:  # fail clearly on non-MLX hosts
    raise ImportError("ModernBERT MLX encoder requires mlx.core (Metal backend)") from exc


class ModernBertConfig:
    """Small config adapter; accepts the Hub JSON fields without Transformers."""

    def __init__(self, data: Mapping[str, Any]):
        self.hidden_size = int(data["hidden_size"])
        self.num_hidden_layers = int(data["num_hidden_layers"])
        self.num_attention_heads = int(data["num_attention_heads"])
        self.intermediate_size = int(data["intermediate_size"])
        self.vocab_size = int(data["vocab_size"])
        self.pad_token_id = int(data["pad_token_id"])
        self.norm_eps = float(data.get("norm_eps", data.get("layer_norm_eps", 1e-5)))
        self.attention_bias = bool(data.get("attention_bias", False))
        self.mlp_bias = bool(data.get("mlp_bias", False))
        self.norm_bias = bool(data.get("norm_bias", False))
        self.local_attention = int(data.get("local_attention", 128))
        self.layer_types = tuple(data.get("layer_types", ()))
        rope = data.get("rope_parameters", {})
        self.rope_theta = {
            "full_attention": float(rope.get("full_attention", {}).get("rope_theta", 10000.0)),
            "sliding_attention": float(rope.get("sliding_attention", {}).get("rope_theta", 10000.0)),
        }
        for kind in ("full_attention", "sliding_attention"):
            params = rope.get(kind, {})
            if params and params.get("rope_type", "default") != "default":
                raise ValueError(f"unsupported {kind} RoPE type: {params['rope_type']!r}")
        self.validate()

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("ModernBERT RoPE requires an even attention head dimension")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must specify every encoder layer")
        if any(t not in ("full_attention", "sliding_attention") for t in self.layer_types):
            raise ValueError(f"unsupported ModernBERT layer types: {self.layer_types}")
        if self.local_attention < 1:
            raise ValueError("local_attention must be positive")
        # The public ModernBERT encoder only defines default, unscaled RoPE for
        # this checkpoint architecture; reject configs we cannot reproduce.

    @classmethod
    def from_json(cls, path: str | Path) -> "ModernBertConfig":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _linear(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


def _layer_norm(x, w, b, eps):
    # PyTorch LayerNorm accumulates in fp32 under fp16 autocast and returns the
    # original dtype. Mirror that explicitly rather than depending on backend
    # reduction promotion rules.
    dtype = x.dtype
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    y = (xf - mean) * mx.rsqrt(var + eps)
    y = y * w.astype(mx.float32)
    if b is not None:
        y = y + b.astype(mx.float32)
    return y.astype(dtype)


def _gelu(x):
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


class ModernBertEncoder:
    """ModernBERT base encoder (B,S -> B,S,H), entirely executed by MLX."""

    backend_name = "mlx-metal"

    def __init__(self, config: ModernBertConfig | Mapping[str, Any], weights: Mapping[str, Any] | None = None):
        self.config = config if isinstance(config, ModernBertConfig) else ModernBertConfig(config)
        self.weights: dict[str, Any] = {}
        self._expected = self._expected_shapes()
        if weights is not None:
            self.load_weights(weights)

    def _expected_shapes(self) -> dict[str, tuple[int, ...]]:
        c, h, i = self.config, self.config.hidden_size, self.config.intermediate_size
        shapes: dict[str, tuple[int, ...]] = {
            "embeddings.tok_embeddings.weight": (c.vocab_size, h),
            "embeddings.norm.weight": (h,),
            "final_norm.weight": (h,),
        }
        if c.norm_bias:
            shapes.update({"embeddings.norm.bias": (h,), "final_norm.bias": (h,)})
        if c.attention_bias:
            for n in range(c.num_hidden_layers):
                shapes[f"layers.{n}.attn.Wqkv.bias"] = (3 * h,)
                shapes[f"layers.{n}.attn.Wo.bias"] = (h,)
        if c.mlp_bias:
            for n in range(c.num_hidden_layers):
                shapes[f"layers.{n}.mlp.Wi.bias"] = (2 * i,)
                shapes[f"layers.{n}.mlp.Wo.bias"] = (h,)
        for n in range(c.num_hidden_layers):
            prefix = f"layers.{n}."
            shapes[prefix + "attn.Wqkv.weight"] = (3 * h, h)
            shapes[prefix + "attn.Wo.weight"] = (h, h)
            shapes[prefix + "mlp.Wi.weight"] = (2 * i, h)
            shapes[prefix + "mlp.Wo.weight"] = (h, i)
            shapes[prefix + "mlp_norm.weight"] = (h,)
            if c.norm_bias:
                shapes[prefix + "mlp_norm.bias"] = (h,)
            # ModernBERT layer zero uses Identity instead of attn_norm.
            if n:
                shapes[prefix + "attn_norm.weight"] = (h,)
                if c.norm_bias:
                    shapes[prefix + "attn_norm.bias"] = (h,)
        return shapes

    def load_weights(self, weights: Mapping[str, Any]) -> None:
        """Load encoder tensors; accept full DecisionModel keys or encoder-only keys.

        No fuzzy key rewriting: allowed keys are either precisely the encoder
        state-dict paths or those paths with one leading ``encoder.`` prefix.
        Head tensors in a full checkpoint are ignored, but encoder omissions,
        duplicates, unexpected encoder-prefixed keys, and shape mismatches fail.
        """
        normalized: dict[str, Any] = {}
        unexpected = []
        for original, value in weights.items():
            key = original[8:] if original.startswith("encoder.") else original
            if key in self._expected:
                if key in normalized:
                    raise ValueError(f"duplicate encoder tensor after prefix normalization: {key}")
                normalized[key] = value
            elif original.startswith("encoder.") or key.startswith(("embeddings.", "layers.", "final_norm.")):
                unexpected.append(original)
        missing = sorted(set(self._expected) - set(normalized))
        if missing or unexpected:
            raise ValueError(f"encoder state keys mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        converted = {}
        for key, shape in self._expected.items():
            value = weights_value = normalized[key]
            # MLX arrays, NumPy arrays, and array-protocol values are supported.
            actual = tuple(getattr(value, "shape", ()))
            if actual != shape:
                raise ValueError(f"{key}: expected shape {shape}, got {actual}")
            converted[key] = mx.array(weights_value)
        self.weights = converted

    @classmethod
    def from_safetensors(cls, config_path: str | Path, weights_path: str | Path) -> "ModernBertEncoder":
        """Load safetensors without importing PyTorch (requires safetensors.numpy)."""
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError("loading ModernBERT weights requires safetensors") from exc
        model = cls(ModernBertConfig.from_json(config_path))
        model.load_weights(load_file(str(weights_path)))
        return model

    def __call__(self, input_ids, attention_mask=None):
        return self.forward(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask=None):
        if not self.weights:
            raise RuntimeError("ModernBERT weights have not been loaded")
        ids = mx.array(input_ids, dtype=mx.int32)
        if ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {ids.shape}")
        batch, seq = ids.shape
        c, w = self.config, self.weights
        if attention_mask is None:
            valid = ids != c.pad_token_id
        else:
            valid = mx.array(attention_mask).astype(mx.bool_)
            if valid.shape != ids.shape:
                raise ValueError(f"attention_mask shape {valid.shape} does not match input_ids {ids.shape}")
        x = w["embeddings.tok_embeddings.weight"][ids]
        x = _layer_norm(x, w["embeddings.norm.weight"], w.get("embeddings.norm.bias"), c.norm_eps)
        dim = c.head_dim
        positions = mx.arange(seq, dtype=mx.float32)
        q_positions = positions[:, None]
        half = dim // 2
        # Masks are [1,1,S,S]; keys at padding slots are excluded, queries are
        # not (matching Transformers' bidirectional mask behavior).
        key_valid = valid[:, None, None, :]
        distances = mx.abs(positions[:, None] - positions[None, :])
        for layer, layer_type in enumerate(c.layer_types):
            p = f"layers.{layer}."
            residual = x
            if layer:
                x = _layer_norm(x, w[p + "attn_norm.weight"], w.get(p + "attn_norm.bias"), c.norm_eps)
            qkv = _linear(x, w[p + "attn.Wqkv.weight"], w.get(p + "attn.Wqkv.bias"))
            qkv = qkv.reshape(batch, seq, 3, c.num_attention_heads, dim)
            q, k, v = (qkv[:, :, j, :, :].transpose(0, 2, 1, 3) for j in range(3))
            theta = c.rope_theta[layer_type]
            inv = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
            angles = positions[:, None] * inv[None, :]
            emb = mx.concatenate((angles, angles), axis=-1)
            cos, sin = mx.cos(emb)[None, None], mx.sin(emb)[None, None]
            rotary_dtype = q.dtype
            q = (q.astype(mx.float32) * cos + _rotate_half(q.astype(mx.float32)) * sin).astype(rotary_dtype)
            k = (k.astype(mx.float32) * cos + _rotate_half(k.astype(mx.float32)) * sin).astype(rotary_dtype)
            scores = ((q @ k.transpose(0, 1, 3, 2)) * (dim ** -0.5)).astype(mx.float32)
            allowed = key_valid
            if layer_type == "sliding_attention":
                # Hub config's local_attention=128 means +/-64 tokens. This is
                # the window used by ModernBERT's sliding-window mask.
                window = c.local_attention // 2
                allowed = allowed & (distances[None, None, :, :] <= window)
            scores = mx.where(allowed, scores, mx.array(-1e30, dtype=mx.float32))
            probs = mx.softmax(scores, axis=-1).astype(q.dtype)
            attn = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, seq, c.hidden_size)
            x = residual + _linear(attn, w[p + "attn.Wo.weight"], w.get(p + "attn.Wo.bias"))
            residual = x
            x = _layer_norm(x, w[p + "mlp_norm.weight"], w.get(p + "mlp_norm.bias"), c.norm_eps)
            glu = _linear(x, w[p + "mlp.Wi.weight"], w.get(p + "mlp.Wi.bias"))
            value, gate = mx.split(glu, 2, axis=-1)
            x = residual + _linear(_gelu(value) * gate, w[p + "mlp.Wo.weight"], w.get(p + "mlp.Wo.bias"))
        return _layer_norm(x, w["final_norm.weight"], w.get("final_norm.bias"), c.norm_eps)


__all__ = ["ModernBertConfig", "ModernBertEncoder"]
