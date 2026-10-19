"""Reuse BF16 vocabulary weights across DSpark verification rows.

The reduction order matches oMLX's existing v41_bf16_head_fp32_output kernel.
This changes how work is grouped, not dot-product arithmetic, for rows 2..5.
Six-row verification is an optional numerical change from the stock fallback.
"""
from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    float sums[M];
    for (uint q = 0; q < M; ++q) sums[q] = 0.0f;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(w[offset], w[offset + 1],
                               w[offset + 2], w[offset + 3]);
        for (uint q = 0; q < M; ++q) {
            const float4 b = float4(x[q * K + k], x[q * K + k + 1],
                                   x[q * K + k + 2], x[q * K + k + 3]);
            sums[q] += dot(a, b);
        }
    }
    for (uint offset = KL / 2; offset > 0; offset /= 2) {
        for (uint q = 0; q < M; ++q)
            sums[q] += simd_shuffle_down(sums[q], offset);
    }
    if (lane == 0) {
        for (uint q = 0; q < M; ++q) y[q * N + row] = sums[q];
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="flash1017_weight_reuse_head", input_names=["x", "w"],
        output_names=["y"], source=_SOURCE,
    )


def fused_head(x, weight):
    n, k = weight.shape
    m = x.size // k
    if weight.dtype != mx.bfloat16 or x.shape[-1] != k or k % 4 or not 1 <= m <= 8:
        raise ValueError("Expected BF16 weight matrix and 1..8 compatible input rows")
    return _kernel()(
        inputs=[x.astype(mx.float32), weight],
        template=[("N", n), ("K", k), ("KL", 16), ("M", m)],
        grid=(((n * 16 + 63) // 64) * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], n)], output_dtypes=[mx.float32],
    )[0]


def install(max_rows=5):
    """Install only in this experiment process; return the original function."""
    from omlx.patches.deepseek_v41 import head, language, dspark
    original = getattr(head, "_flash1017_original", head.project_logits)

    def project(x, weight):
        n, k = weight.shape
        if (weight.dtype == mx.bfloat16 and x.shape[-1] == k and k % 4 == 0
                and 2 <= x.size // k <= max_rows and n >= 4096 and k >= 256
                and mx.default_device() != mx.cpu):
            return fused_head(x, weight)
        return original(x, weight)

    head._flash1017_original = original
    head.project_logits = language.project_logits = dspark.project_logits = project
    return original


def uninstall():
    from omlx.patches.deepseek_v41 import head, language, dspark
    original = getattr(head, "_flash1017_original", head.project_logits)
    head.project_logits = language.project_logits = dspark.project_logits = original
