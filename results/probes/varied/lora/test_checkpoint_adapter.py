"""Focused unit/smoke tests for the Qwen3 checkpoint adapter."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, ModelArgs

from checkpoint_adapter import (
    Qwen3Embedding,
    inject_upper_lora,
    map_decoder_weights,
)


def tiny_model() -> Model:
    return Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=8,
            num_hidden_layers=2,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=32,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
    )


def test_decoder_mapping_checks_count_shape_dtype_and_namespaces() -> None:
    model = tiny_model()
    expected = tree_flatten(model.parameters())
    unprefixed = {name.removeprefix("model."): mx.ones_like(value) for name, value in expected}
    mapped = map_decoder_weights(unprefixed, model)
    assert [name for name, _ in mapped] == sorted(dict(expected))
    # Existing nested weights are accepted without being double-prefixed.
    nested = {name: mx.ones_like(value) for name, value in expected}
    assert [name for name, _ in map_decoder_weights(nested, model)] == sorted(nested)

    bad_shape = dict(unprefixed)
    bad_shape["norm.weight"] = mx.ones((9,), dtype=mx.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        map_decoder_weights(bad_shape, model)

    bad_dtype = dict(unprefixed)
    bad_dtype["norm.weight"] = mx.ones((8,), dtype=mx.int32)
    with pytest.raises(ValueError, match="unsupported dtype"):
        map_decoder_weights(bad_dtype, model)

    extra = dict(unprefixed)
    extra["lm_head.weight"] = mx.ones((32, 8), dtype=mx.float32)
    with pytest.raises(ValueError, match="count mismatch"):
        map_decoder_weights(extra, model)


def test_upper_lora_weights_receive_autograd() -> None:
    model = tiny_model()
    loaded = Qwen3Embedding(
        model=model,
        tokenizer=None,  # not used by this token-ID gradient smoke
        max_length=32,
        checkpoint_dtype=mx.float32,
    )
    assert inject_upper_lora(loaded, layers=1, projections=("q_proj",), rank=2)
    ids = mx.array([[1, 2, 3]], dtype=mx.int32)

    def objective(module: Model) -> mx.array:
        hidden = module.model(ids).astype(mx.float32)
        return mx.mean(mx.square(hidden))

    value, gradients = nn.value_and_grad(model, objective)(model)
    mx.eval(value, gradients)
    lora_grads = {
        name: grad for name, grad in tree_flatten(gradients) if ".lora_" in name
    }
    assert lora_grads
    assert any(float(mx.sum(mx.abs(grad)).item()) > 0 for grad in lora_grads.values())


@pytest.mark.hardware
def test_installed_checkpoints_validate_and_embed() -> None:
    from checkpoint_adapter import DEFAULT_MODEL_STORE, load_checkpoint

    for folder in (
        "Qwen--Qwen3-Embedding-0.6B",
        "microsoft--harrier-oss-v1-0.6b",
    ):
        model_dir = DEFAULT_MODEL_STORE / "models" / folder
        if not model_dir.exists():
            pytest.skip(f"installed checkpoint is unavailable: {model_dir}")
        loaded = load_checkpoint(model_dir)
        assert len(tree_flatten(loaded.model.parameters())) == 310
        vector = loaded.embed("train-only adapter smoke", pooling_dtype="float32")
        mx.eval(vector)
        assert vector.shape == (1, 1024)
        assert float(mx.abs(mx.sqrt(mx.sum(mx.square(vector))) - 1).item()) < 1e-5
