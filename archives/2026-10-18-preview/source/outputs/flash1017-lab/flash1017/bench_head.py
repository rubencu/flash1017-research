"""Paired microbenchmark at the exact 1017 vocabulary projection dimensions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import median
import time

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41.head import project_logits
from .head import fused_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mx.random.seed(1017)
    # Synthetic weights, exact checkpoint geometry. This is kernel evidence,
    # never substituted for full-model speed or task-quality measurements.
    weight = mx.random.normal((129280, 5120), dtype=mx.bfloat16) * .02
    mx.eval(weight)
    results = []
    rng = random.Random(1017)
    for m in (1, 2, 3, 4, 5, 6):
        x = mx.random.normal((1, m, 5120), dtype=mx.bfloat16)
        mx.eval(x)
        baseline, candidate = project_logits(x, weight), fused_head(x, weight)
        mx.eval(baseline, candidate)
        a, b = np.asarray(baseline), np.asarray(candidate)
        row = {"rows": m, "bit_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
               "max_absolute_error": float(np.max(np.abs(a - b))),
               "argmax_equal": bool(np.array_equal(a.argmax(-1), b.argmax(-1))),
               "baseline_ms": [], "candidate_ms": []}
        for _ in range(args.repeats):
            order = [("baseline_ms", project_logits), ("candidate_ms", fused_head)]
            rng.shuffle(order)
            for name, function in order:
                mx.synchronize()
                begin = time.perf_counter()
                output = function(x, weight)
                mx.eval(output)
                row[name].append((time.perf_counter() - begin) * 1000)
        row["median_speedup"] = median(row["baseline_ms"]) / median(row["candidate_ms"])
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}), flush=True)
    args.output.write_text(json.dumps({"device": mx.device_info(),
                                      "weight_shape": list(weight.shape),
                                      "synthetic_weights": True,
                                      "other_work": "Model download and existing oMLX service may be active",
                                      "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
