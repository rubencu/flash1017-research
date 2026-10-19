"""Fit a frozen DSpark policy from calibration runs, never test-set outcomes."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def log_loss(logits, labels, temperature):
    total = 0.0
    for logit, label in zip(logits, labels):
        x = logit / temperature
        total += max(x, 0) - x * label + math.log1p(math.exp(-abs(x)))
    return total / len(logits)


def fit(records, width=5):
    costs = defaultdict(list)
    observed = [[] for _ in range(width)]
    for run in records:
        if run.get("type") != "run" or run.get("warmup"):
            continue
        profile = run["profile"]
        if profile.get("draft_width") != width or "forced_depth" not in profile:
            raise ValueError("Calibration requires fixed-width, forced-depth runs")
        # Startup, terminal clamps and context copies do not describe a steady
        # DSpark cycle. Final-cycle stop tokens can censor acceptance labels.
        for event in run["events"][3:-1]:
            if event["copied"] or event["remaining"] <= width + 1:
                continue
            k, m = event["verified"], event["accepted"]
            costs[k].append(event["cycle_ms"])
            logits = event["confidence_logits"]
            # A later token after the first rejection has no acceptance label.
            for index in range(min(k, m + 1)):
                observed[index].append((logits[index], int(index < m)))
    if any(len(costs[k]) < 10 for k in range(width + 1)):
        raise ValueError("Need at least 10 uncensored cycles at every verify depth")
    temperatures, diagnostics = [], []
    # Search temperature on a fixed log grid, centered on the identity. This
    # is fitted on calibration prompts only; no task accuracy enters this fit.
    grid = [math.exp(x / 16) for x in range(-32, 49)]
    for values in observed:
        if len(values) < 20:
            temperatures.append(1.0)
            diagnostics.append({"n": len(values), "fit": "insufficient labels"})
            continue
        logits, labels = zip(*values)
        losses = [(log_loss(logits, labels, temp), temp) for temp in grid]
        loss, best = min(losses)
        temperatures.append(best)
        diagnostics.append({"n": len(values), "empirical_acceptance": sum(labels) / len(labels),
                            "loss_before": log_loss(logits, labels, 1.0), "loss_after": loss})
    return {"cycle_ms": [median(costs[k]) for k in range(width + 1)],
            "temperatures": temperatures,
            "diagnostics": {"cost_samples": [len(costs[k]) for k in range(width + 1)],
                            "confidence": diagnostics}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = args.input.read_bytes()
    result = fit([json.loads(line) for line in data.splitlines()])
    result["calibration_sha256"] = hashlib.sha256(data).hexdigest()
    result["calibration_file"] = str(args.input.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
