"""Summarize raw experiments, retaining exclusions and paired token checks."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import median


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def paired_summary(records, baseline, candidate):
    by_key = defaultdict(dict)
    for row in records:
        if row["profile"]["name"] in (baseline, candidate):
            key = row["prompt_id"], row["repeat"], row["seed"]
            by_key[key][row["profile"]["name"]] = row
    ratios, matches, pairs = [], [], []
    for key, rows in sorted(by_key.items()):
        if baseline not in rows or candidate not in rows:
            continue
        a, b = rows[baseline], rows[candidate]
        if a["input_tokens_sha256"] != b["input_tokens_sha256"]:
            raise ValueError("Paired runs used different tokenized prompts")
        ratio = b["decode_tokens_per_second"] / a["decode_tokens_per_second"]
        ratios.append(ratio)
        matches.append(a["token_ids"] == b["token_ids"])
        pairs.append({"prompt": key[0], "repeat": key[1], "ratio": ratio,
                      "tokens_equal": matches[-1], "baseline_seconds": a["total_seconds"],
                      "candidate_seconds": b["total_seconds"]})
    if not ratios:
        return {"baseline": baseline, "candidate": candidate, "pairs": []}
    rng = random.Random(1017)
    boot = sorted(geometric_mean(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return {"baseline": baseline, "candidate": candidate, "pairs": pairs,
            "paired_geometric_speedup": geometric_mean(ratios),
            "bootstrap_95_percent_interval": [boot[49], boot[1949]],
            "identical_token_streams": sum(matches), "total_pairs": len(matches),
            "interval_note": "Simple paired bootstrap; few runs or shared host conditions limit inference."}


def analyze(rows):
    records = [row for row in rows if row.get("type") == "run" and not row.get("warmup")]
    excluded = [{"profile": row["profile"]["name"], "prompt": row["prompt_id"],
                 "repeat": row["repeat"], "reason": "EOS before end of forced-length generation"}
                for row in records if row.get("fixed_length_valid") is False]
    records = [row for row in records if row.get("fixed_length_valid") is not False]
    grouped = defaultdict(list)
    for row in records:
        grouped[row["profile"]["name"], row["prompt_id"]].append(row)
    summaries = []
    for (profile, prompt), values in sorted(grouped.items()):
        summaries.append({"profile": profile, "prompt": prompt, "n": len(values),
                          "median_decode_tps": median(v["decode_tokens_per_second"] for v in values),
                          "median_ttft_seconds": median(v["ttft_seconds"] for v in values),
                          "median_total_seconds": median(v["total_seconds"] for v in values),
                          "peak_process_gib": max(v["resources"]["process_footprint_peak_bytes"] for v in values) / 2**30,
                          "max_swap_increase_bytes": max(v["resources"]["swap_peak_bytes"] - v["resources"]["swap_start_bytes"] for v in values)})
    profiles = sorted({row["profile"]["name"] for row in records})
    pairs = [paired_summary(records, name, name + "-head")
             for name in profiles if name + "-head" in profiles]
    return {"summaries": summaries, "kernel_pairs": pairs, "exclusions": excluded,
            "errors": [row for row in rows if row.get("type") == "error"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    args.output.write_text(json.dumps(analyze(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
