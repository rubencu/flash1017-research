"""Compare every vocabulary logit on real 1017 activations, outside timing runs.

Forward remaining CLI arguments to flash1017.bench. Use --report for the
numerical evidence. These instrumented runs are NEVER speed benchmarks.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

from . import bench, head


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--single-row-reference', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    original_fused = head.fused_head
    counts = defaultdict(lambda: {'calls': 0, 'logits_checked': 0, 'different_bits': 0,
                                  'argmax_mismatches': 0, 'max_abs_error': 0.0})
    report = {'kind': 'instrumented full-checkpoint numerical validation; not a speed benchmark',
              'checkpoint': 'DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp', 'rows': counts,
              'reference': 'stock_single_row' if args.single_row_reference else 'stock_same_shape'}

    def save():
        args.report.write_text(json.dumps(report, indent=2) + '\n')

    def checked(x, weight):
        from omlx.patches.deepseek_v41 import head as upstream
        actual = original_fused(x, weight)
        m = x.size // weight.shape[-1]
        if args.single_row_reference:
            flat = x.reshape(m, weight.shape[-1])
            expected = mx.concatenate([
                upstream._flash1017_original(flat[i:i+1], weight)
                for i in range(m)
            ], axis=0).reshape(actual.shape)
        else:
            expected = upstream._flash1017_original(x, weight)
        mx.eval(actual, expected)
        a, b = np.asarray(actual), np.asarray(expected)
        row = counts[str(m)]
        row['calls'] += 1
        row['logits_checked'] += a.size
        row['different_bits'] += int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
        row['argmax_mismatches'] += int(np.count_nonzero(a.argmax(-1) != b.argmax(-1)))
        row['max_abs_error'] = max(row['max_abs_error'], float(np.max(np.abs(a - b))))
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ArithmeticError('Non-finite vocabulary logits')
        save()
        if (m <= 5 or args.single_row_reference) and row['different_bits']:
            raise ArithmeticError('The arithmetic-preserving kernel changed a logit bit')
        return actual

    head.fused_head = checked
    sys.argv = [sys.argv[0]] + remaining
    try:
        bench.main()
        if not {'2', '3', '4', '5', '6'}.issubset(counts):
            raise RuntimeError('Numerical validation did not exercise every requested row count')
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        head.fused_head = original_fused
        save()


if __name__ == '__main__':
    main()
