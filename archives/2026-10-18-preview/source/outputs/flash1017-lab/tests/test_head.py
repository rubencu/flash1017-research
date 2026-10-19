import mlx.core as mx
import numpy as np
import pytest

from flash1017.head import fused_head
from omlx.patches.deepseek_v41.head import project_logits


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("width", [256, 5120])
def test_projection_is_bit_exact_to_existing_kernel(rows, width):
    mx.random.seed(1017)
    weight = mx.random.normal((4097, width), dtype=mx.bfloat16)
    x = mx.random.normal((1, rows, width), dtype=mx.bfloat16)
    a, b = project_logits(x, weight), fused_head(x, weight)
    np.testing.assert_array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))


def test_six_rows_matches_float64_reference_with_bounded_error():
    mx.random.seed(1017)
    weight = mx.random.normal((4097, 256), dtype=mx.bfloat16)
    x = mx.random.normal((6, 256), dtype=mx.bfloat16)
    reference = np.asarray(x.astype(mx.float32)).astype(np.float64) @ np.asarray(weight.astype(mx.float32)).astype(np.float64).T
    actual = np.asarray(fused_head(x, weight))
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-6)
