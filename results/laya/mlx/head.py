"""MLX implementation of the root Laya decision transformer head.

The input is the ModernBERT ``last_hidden_state``; the encoder itself is
intentionally outside this module.  Parameters are loaded from the published
PyTorch ``model.safetensors`` state dict using the exact ``rl_common.py`` names.
No host/PyTorch fallback is provided.
"""
from __future__ import annotations

from typing import Any, Mapping

import mlx.core as mx


class HeadWeightsError(ValueError):
    """Checkpoint decision-head tensors do not match the published architecture."""


class DecisionHead:
    """Two full-sequence, pre-norm transformer layers and Laya output heads.

    ``forward`` accepts an encoded sequence ``[B,L,D]``, a 1=valid attention
    mask ``[B,L]``, marker positions/mask ``[B,K]``, and integer question types
    (choice=0, score=1, noul=2) ``[B]``. It returns uncalibrated option logits
    and act logits, matching ``rl_common.DecisionModel.forward``.
    """

    backend_name = "mlx-metal"
    REQUIRED_LAYERS = 2

    def __init__(self, weights: Mapping[str, Any], *, hidden_size: int = 1024,
                 nhead: int | None = None, n_act: int = 2):
        self.hidden_size = int(hidden_size)
        self.nhead = int(nhead or max(1, hidden_size // 64))
        self.n_act = int(n_act)
        if self.hidden_size <= 0 or self.nhead <= 0 or self.hidden_size % self.nhead:
            raise ValueError("hidden_size must be positive and divisible by nhead")
        self.head_dim = self.hidden_size // self.nhead
        self._p: dict[str, mx.array] = {}

        # PyTorch TransformerEncoderLayer uses packed Q/K/V projection and the
        # default 4*D feed-forward width. Reject shape drift rather than silently
        # reshaping/interpreting another checkpoint architecture.
        expected: dict[str, tuple[int, ...]] = {
            "type_emb.weight": (3, self.hidden_size),
            "scorer.0.weight": (self.hidden_size,),
            "scorer.0.bias": (self.hidden_size,),
            "scorer.1.weight": (self.hidden_size, self.hidden_size),
            "scorer.1.bias": (self.hidden_size,),
            "scorer.3.weight": (1, self.hidden_size),
            "scorer.3.bias": (1,),
            "act_head.0.weight": (256, self.hidden_size + 4),
            "act_head.0.bias": (256,),
            "act_head.2.weight": (self.n_act, 256),
            "act_head.2.bias": (self.n_act,),
            "temperature": (3,),
        }
        ff = 4 * self.hidden_size
        for i in range(self.REQUIRED_LAYERS):
            prefix = f"head.layers.{i}."
            expected.update({
                prefix + "self_attn.in_proj_weight": (3 * self.hidden_size, self.hidden_size),
                prefix + "self_attn.in_proj_bias": (3 * self.hidden_size,),
                prefix + "self_attn.out_proj.weight": (self.hidden_size, self.hidden_size),
                prefix + "self_attn.out_proj.bias": (self.hidden_size,),
                prefix + "linear1.weight": (ff, self.hidden_size),
                prefix + "linear1.bias": (ff,),
                prefix + "linear2.weight": (self.hidden_size, ff),
                prefix + "linear2.bias": (self.hidden_size,),
                prefix + "norm1.weight": (self.hidden_size,),
                prefix + "norm1.bias": (self.hidden_size,),
                prefix + "norm2.weight": (self.hidden_size,),
                prefix + "norm2.bias": (self.hidden_size,),
            })

        # The official checkpoint may also contain a full ModernBERT encoder.
        # Consume only decision-head names while demanding every one of them.
        missing = sorted(set(expected) - set(weights))
        if missing:
            raise HeadWeightsError("missing decision-head checkpoint tensors: " + ", ".join(missing))
        bad = [(key, tuple(getattr(weights[key], "shape", ())), shape)
               for key, shape in expected.items()
               if tuple(getattr(weights[key], "shape", ())) != shape]
        if bad:
            details = "; ".join(f"{k}: got {got}, expected {want}" for k, got, want in bad)
            raise HeadWeightsError("decision-head checkpoint shape mismatch: " + details)
        for key in expected:
            self._p[key] = mx.array(weights[key])

    @classmethod
    def from_safetensors(cls, path: str, **kwargs: Any) -> "DecisionHead":
        """Load the decision head directly from the official safetensors file."""
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("install safetensors to load Laya checkpoint weights") from exc
        with safe_open(path, framework="numpy") as f:
            names = set(f.keys())
            head_names = {name for name in names if name.startswith((
                "head.layers.", "type_emb.", "scorer.", "act_head.", "temperature"))}
            tensors = {name: f.get_tensor(name) for name in head_names}
        return cls(tensors, **kwargs)

    def _linear(self, x: mx.array, key: str) -> mx.array:
        return x @ self._p[key + ".weight"].T + self._p[key + ".bias"]

    def _layer_norm(self, x: mx.array, key: str) -> mx.array:
        # PyTorch LayerNorm uses epsilon=1e-5, including the scorer's norm.
        return mx.fast.layer_norm(x, self._p[key + ".weight"], self._p[key + ".bias"], eps=1e-5)

    @staticmethod
    def _gelu(x: mx.array) -> mx.array:
        # Exact PyTorch nn.GELU (approximate="none"); MLX 0.29 has no gelu op.
        return 0.5 * x * (1.0 + mx.erf(x * (2.0 ** -0.5)))

    def _transformer_layer(self, x: mx.array, valid: mx.array, i: int) -> mx.array:
        p = f"head.layers.{i}."
        b, length, dim = x.shape
        normed = self._layer_norm(x, p + "norm1")
        qkv = normed @ self._p[p + "self_attn.in_proj_weight"].T + self._p[p + "self_attn.in_proj_bias"]
        q, k, v = mx.split(qkv, 3, axis=-1)
        # [B,L,D] -> [B,H,L,Dh], same head ordering as torch.nn.MultiheadAttention.
        q = q.reshape(b, length, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, length, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, length, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)
        # key_padding_mask only masks keys in PyTorch MHA; padded query rows may
        # be evaluated because their outputs are never pooled/scored.
        scores = mx.where(valid[:, None, None, :].astype(mx.bool_), scores, -1e30)
        probs = mx.softmax(scores, axis=-1)
        context = (probs @ v).transpose(0, 2, 1, 3).reshape(b, length, dim)
        x = x + self._linear(context, p + "self_attn.out_proj")
        normed = self._layer_norm(x, p + "norm2")
        ff = self._linear(normed, p + "linear1")
        # torch.nn.TransformerEncoderLayer defaults to ReLU (not GELU).
        ff = mx.maximum(ff, 0)
        return x + self._linear(ff, p + "linear2")

    def forward(self, encoded: mx.array, attention_mask: mx.array,
                marker_pos: mx.array, marker_mask: mx.array,
                qtype: mx.array) -> tuple[mx.array, mx.array]:
        if len(encoded.shape) != 3 or encoded.shape[-1] != self.hidden_size:
            raise ValueError(f"encoded must have shape [B,L,{self.hidden_size}]")
        b, length, _ = encoded.shape
        if tuple(attention_mask.shape) != (b, length):
            raise ValueError("attention_mask must have shape [B,L]")
        if len(marker_pos.shape) != 2 or marker_pos.shape[0] != b or marker_mask.shape != marker_pos.shape:
            raise ValueError("marker_pos and marker_mask must have matching [B,K] shapes")
        if tuple(qtype.shape) != (b,):
            raise ValueError("qtype must have shape [B]")

        valid = attention_mask.astype(mx.bool_)
        x = encoded + self._p["type_emb.weight"][qtype.astype(mx.int32)][:, None, :]
        for i in range(self.REQUIRED_LAYERS):
            x = self._transformer_layer(x, valid, i)

        positions = mx.clip(marker_pos.astype(mx.int32), 0, length - 1)
        gather_idx = mx.broadcast_to(positions[:, :, None], (*positions.shape, self.hidden_size))
        markers = mx.take_along_axis(x, gather_idx, axis=1)
        scored = self._gelu(self._linear(self._layer_norm(markers, "scorer.0"), "scorer.1"))
        logits = self._linear(scored, "scorer.3").squeeze(-1).astype(mx.float32)
        logits = mx.where(marker_mask.astype(mx.bool_), logits, -1e4)

        # The act head intentionally sees a detached option distribution. In
        # inference there is no gradient graph, so stop_gradient is explicit.
        p = mx.stop_gradient(mx.softmax(logits, axis=-1))
        k_count = mx.maximum(marker_mask.astype(mx.float32).sum(axis=-1), 2.0)
        entropy = -(p * mx.log(mx.maximum(p, 1e-9))).sum(axis=-1) / mx.log(k_count)
        # The maintained MLX port pads marker tensors to at least two slots;
        # mirror that behavior here so a one-option question remains defined.
        p_for_top = mx.concatenate([p, mx.zeros((b, 2 - p.shape[1]), dtype=p.dtype)], axis=-1) \
            if p.shape[1] < 2 else p
        top2 = mx.sort(p_for_top, axis=-1)[:, -2:][:, ::-1]
        features = mx.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy,
                             k_count / 255.0], axis=-1)
        pooled = x[:, 0, :].astype(mx.float32)
        act_hidden = self._gelu(self._linear(mx.concatenate([pooled, features], axis=-1), "act_head.0"))
        act_logits = self._linear(act_hidden, "act_head.2").astype(mx.float32)
        return logits, act_logits

    __call__ = forward

    def calibrated(self, logits: mx.array, qtype: int, option_count: int,
                   temperature_by_options: Mapping[str, float] | None = None,
                   base_temperatures: list[float] | tuple[float, ...] | None = None) -> mx.array:
        """Return option probabilities using published per-type/cardinality temperatures."""
        kind = ("choice", "score", "noul")[qtype]
        bucket = "2" if option_count <= 2 else "3-5" if option_count <= 5 else (
            "6-10" if option_count <= 10 else "11+")
        key = f"{kind}:{bucket}"
        table = temperature_by_options or {}
        temp = table.get(key)
        if temp is None:
            # The upstream runtime takes both calibration values from
            # rl_agent_config.json. Its model buffer is initialized to ones and
            # is not the calibrated fallback. Preserve that distinction.
            if base_temperatures is None or len(base_temperatures) != 3:
                raise ValueError("pass rl_agent_config.json temperature values for the fallback")
            temp = float(base_temperatures[qtype])
        if not float(temp) > 0:
            raise ValueError("calibration temperature must be positive")
        return mx.softmax(logits / float(temp), axis=-1)


def synthetic_forward_smoke() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Tiny deterministic-shape smoke test; requires MLX but no model weights."""
    d, heads, ff, acts = 16, 2, 64, 2
    rng = mx.random.key(0)
    weights: dict[str, mx.array] = {}

    def put(name: str, shape: tuple[int, ...]) -> None:
        nonlocal rng
        rng, subkey = mx.random.split(rng)
        weights[name] = mx.random.normal(shape, key=subkey) * 0.02

    for name, shape in {
        "type_emb.weight": (3, d), "scorer.0.weight": (d,), "scorer.0.bias": (d,),
        "scorer.1.weight": (d, d), "scorer.1.bias": (d,), "scorer.3.weight": (1, d),
        "scorer.3.bias": (1,), "act_head.0.weight": (256, d + 4), "act_head.0.bias": (256,),
        "act_head.2.weight": (acts, 256), "act_head.2.bias": (acts,), "temperature": (3,),
    }.items():
        put(name, shape)
    for i in range(2):
        p = f"head.layers.{i}."
        for name, shape in {
            "self_attn.in_proj_weight": (3 * d, d), "self_attn.in_proj_bias": (3 * d,),
            "self_attn.out_proj.weight": (d, d), "self_attn.out_proj.bias": (d,),
            "linear1.weight": (ff, d), "linear1.bias": (ff,), "linear2.weight": (d, ff),
            "linear2.bias": (d,), "norm1.weight": (d,), "norm1.bias": (d,),
            "norm2.weight": (d,), "norm2.bias": (d,),
        }.items():
            put(p + name, shape)
    model = DecisionHead(weights, hidden_size=d, nhead=heads)
    x = mx.zeros((1, 7, d))
    logits, act = model(x, mx.array([[1, 1, 1, 1, 1, 0, 0]]),
                        mx.array([[2, 4]]), mx.array([[1, 1]]), mx.array([0]))
    mx.eval(logits, act)
    if logits.shape != (1, 2) or act.shape != (1, acts):
        raise AssertionError(f"unexpected synthetic output shapes: {logits.shape}, {act.shape}")
    return logits.shape, act.shape


if __name__ == "__main__":  # pragma: no cover - manual local smoke test
    print("synthetic forward OK:", synthetic_forward_smoke())
