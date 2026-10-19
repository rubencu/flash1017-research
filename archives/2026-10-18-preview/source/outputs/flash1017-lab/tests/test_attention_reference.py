"""Regression at the actual main/draft attention output projection boundary.

The independent NumPy oracle enumerates E4M3 values; it does not call the
runtime's quantizer. The reference checkpoint rounds groups of 32 before wo_a.
"""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import Attention
from omlx.patches.deepseek_v41.dspark import DSparkAttention


def fp8_reference(values):
    codes = np.arange(127, dtype=np.uint8)
    exponent = ((codes >> 3) & 15).astype(np.int32)
    fraction = (codes & 7).astype(np.float32)
    levels = np.where(exponent == 0, fraction * 2.0**-9,
                      np.ldexp(1.0 + fraction / 8, exponent - 7))
    # IEEE ties select the even code. Sort even codes before odd for argmin.
    order = np.r_[np.arange(0, 127, 2), np.arange(1, 127, 2)]
    rows = values.reshape(*values.shape[:-1], -1, 32)
    scale = np.exp2(np.ceil(np.log2(np.maximum(
        np.abs(rows).max(-1, keepdims=True) / 448.0, 2.0**-126))))
    distances = np.abs(np.abs(rows / scale)[..., None] - levels[order])
    rounded = np.copysign(levels[order[distances.argmin(-1)]], rows)
    return (rounded * scale).reshape(values.shape)


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_attention_output_matches_1017_reference(monkeypatch, draft, enabled):
    import omlx.patches.deepseek_v41.language as language
    import omlx.patches.deepseek_v41.dspark as dspark

    config = ModelConfig(
        dim=32, n_heads=2, head_dim=32, rope_head_dim=8, o_groups=2,
        o_lora_rank=16, q_lora_rank=32, n_layers=1, n_mtp_layers=0,
        compress_ratios=(0,), kv_source_layers=(), index_source_layers=(),
        window_size=4, index_head_dim=32, index_n_heads=2,
    )
    config.attention_output_quantization = enabled
    attention = (DSparkAttention if draft else Attention)(config, 0)
    rng = np.random.default_rng(1017)
    output = rng.normal(size=(1, 3, 2, 32)).astype(np.float32)
    weight = rng.normal(size=(2, 16, 32)).astype(np.float32)
    attention.wo_a.weight = mx.array(weight.reshape(32, 32))
    attention.wo_b = nn.Identity()
    module = dspark if draft else language
    monkeypatch.setattr(module, "rope", lambda x, *a, **kw: x)
    operands = []
    einsum = mx.einsum

    def capture(pattern, x, w):
        operands.append(np.asarray(x))
        return einsum(pattern, x, w)

    monkeypatch.setattr(mx, "einsum", capture)
    if draft:
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention",
                            lambda *a, **kw: mx.array(output.transpose(0, 2, 1, 3)))
        cache = SimpleNamespace(offset=0, keys=mx.zeros((1, 1, 0, 32)))
        actual = attention(mx.zeros((1, 3, 32)), cache)
    else:
        monkeypatch.setattr(language, "packed_sparse_attention",
                            lambda *a, **kw: mx.array(output))
        cache = [None, None]
        actual = attention(mx.zeros((1, 3, 32)), cache, {}, 0)
    grouped = fp8_reference(output) if enabled else output
    # Inspect the actual projection input, separating activation arithmetic
    # from Metal's shape-dependent matmul reduction/accumulation precision.
    assert len(operands) == 1
    np.testing.assert_array_equal(operands[0], grouped)
    assert np.isfinite(np.asarray(actual)).all()
    assert np.max(np.abs(fp8_reference(output) - output)) > 0.01
