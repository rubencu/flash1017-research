"""CPU-only arithmetic/record audit of the immediate overall comparison.

Input labels declare arm and process block; no runtime is imported. This audits
records, not whether a runner really loaded its claimed runtime/native ABI.
Usage: python agent_analyze.py --plan agent-proposed-plan-02.json \
  --input native2048:0:/path/run.jsonl ... --output /new/report.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gm(values):
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def timing_ok(row):
    keys = ("total_seconds", "prefill_seconds", "ttft_seconds", "decode_tokens_per_second")
    return all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and row[k] > 0 for k in keys)


def analyze(plan, inputs):
    issues, provenance, records, warmups = [], [], {}, []
    fixtures = {x["id"]: x for x in plan["fixtures"]}
    arms = tuple(plan["arms"])
    blocks = {x["block"]: x for x in plan["process_blocks"]}
    supplied = set()
    for arm, block, path in inputs:
        key = (arm, block)
        if key in supplied:
            issues.append(f"Duplicate process input {key}")
            continue
        supplied.add(key)
        if arm not in arms or block not in blocks:
            issues.append(f"Unexpected process input {key}")
            continue
        raw = Path(path).read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        provenance.append({"arm": arm, "block": block, "path": str(Path(path).resolve()),
                           "sha256": hashlib.sha256(raw).hexdigest(), "records": len(rows)})
        failures = [r for r in rows if r.get("type") in ("error", "failure", "abort")]
        if failures:
            issues.append(f"Recorded failures in {key}: {len(failures)}")
        runs = [r for r in rows if r.get("type") == "run"]
        warms = [r for r in runs if r.get("warmup")]
        measured = [r for r in runs if not r.get("warmup")]
        if len(warms) != 1:
            issues.append(f"Expected one warmup in {key}, got {len(warms)}")
        for row in warms:
            warmups.append({"arm": arm, "block": block, "prompt_id": row.get("prompt_id"),
                            "generated_tokens": row.get("generated_tokens"),
                            "total_seconds": row.get("total_seconds")})
            if row.get("generated_tokens") != plan["warmup"]["max_tokens"]:
                issues.append(f"Wrong warmup output count in {key}")
        actual_order = [r.get("prompt_id") for r in measured]
        if actual_order != blocks[block]["case_order"]:
            issues.append(f"Measured order/membership mismatch in {key}")
        for row in measured:
            name = row.get("prompt_id")
            rk = (arm, block, name)
            if rk in records:
                issues.append(f"Duplicate measured record {rk}")
                continue
            records[rk] = row
            if name not in fixtures:
                issues.append(f"Unexpected fixture {rk}")
                continue
            expected = fixtures[name]
            checks = {
                "input count": row.get("input_tokens") == expected["input_tokens"],
                "input hash": row.get("input_tokens_sha256") == expected["input_tokens_sha256"],
                "seed": row.get("seed") == blocks[block]["seed_by_case"][name],
                "output count": row.get("generated_tokens") == expected["max_tokens"],
                "output IDs": isinstance(row.get("token_ids"), list) and len(row["token_ids"]) == expected["max_tokens"],
                "fixed length": row.get("fixed_length_valid") is True,
                "timings": timing_ok(row),
                "native MTP activity": bool(row.get("mtp_stats")),
            }
            for check, ok in checks.items():
                if not ok:
                    issues.append(f"Invalid {check}: {rk}")
    expected_processes = {(a, b) for a in arms for b in blocks}
    if supplied != expected_processes:
        issues.append(f"Missing/extra process labels: missing={sorted(expected_processes-supplied)} extra={sorted(supplied-expected_processes)}")
    expected_count = len(arms) * len(blocks) * len(fixtures)
    if len(records) != expected_count:
        issues.append(f"Expected {expected_count} measured records, got {len(records)}")

    pairs = []
    for baseline in ("native2048", "native8192"):
        for block in blocks:
            for name, fixture in fixtures.items():
                a, b = records.get((baseline, block, name)), records.get(("complete", block, name))
                if a is None or b is None or not timing_ok(a) or not timing_ok(b):
                    continue
                at, bt = a.get("token_ids", []), b.get("token_ids", [])
                first = next((i for i, (x, y) in enumerate(zip(at, bt)) if x != y), None)
                if first is None and len(at) != len(bt):
                    first = min(len(at), len(bt))
                pairs.append({"baseline": baseline, "block": block, "fixture": name,
                              "cohort": fixture["cohort"], "input_tokens": fixture["input_tokens"],
                              "baseline_total_seconds": a["total_seconds"], "candidate_total_seconds": b["total_seconds"],
                              "total_latency_ratio_candidate_over_baseline": b["total_seconds"] / a["total_seconds"],
                              "prefill_latency_ratio_candidate_over_baseline": b["prefill_seconds"] / a["prefill_seconds"],
                              "ttft_latency_ratio_candidate_over_baseline": b["ttft_seconds"] / a["ttft_seconds"],
                              "decode_tps_ratio_candidate_over_baseline": b["decode_tokens_per_second"] / a["decode_tokens_per_second"],
                              "fixed_length_valid": a.get("fixed_length_valid") is True and b.get("fixed_length_valid") is True,
                              "output_ids_identical": at == bt, "first_differing_output_position_zero_based": first})
    summaries = []
    for baseline in ("native2048", "native8192"):
        for cohort in ("all-eight-cases", "natural-source", "synthetic-generation"):
            selected = [x for x in pairs if x["baseline"] == baseline and (cohort == "all-eight-cases" or x["cohort"] == cohort)]
            item = {"baseline": baseline, "cohort": cohort, "pairs": len(selected),
                    "scope": "provisional incomplete subset; not an accepted speed claim" if issues else "complete fixed suite; two blocks on one host"}
            for metric in ("total", "prefill", "ttft"):
                val = gm([p[f"{metric}_latency_ratio_candidate_over_baseline"] for p in selected])
                item[f"{metric}_latency_reduction_percent"] = None if val is None else 100 * (1 - val)
            val = gm([p["decode_tps_ratio_candidate_over_baseline"] for p in selected])
            item["decode_tps_gain_percent"] = None if val is None else 100 * (val - 1)
            item["candidate_faster_whole_requests"] = sum(p["total_latency_ratio_candidate_over_baseline"] < 1 for p in selected)
            item["identical_output_streams"] = sum(p["output_ids_identical"] for p in selected)
            summaries.append(item)
    return {"status": "complete_record_and_arithmetic_checks_passed" if not issues else "incomplete_or_invalid_preserved",
            "issues": issues, "measured_records": len(records), "warmups": warmups,
            "inputs": provenance, "summaries": summaries, "pairs": pairs,
            "limits": ["This analyzer checks records and arithmetic, not runtime/native-ABI identity or actual activation; those need the runner provenance/telemetry audit.",
                       "Direct model-prefill-through-generation timing, excluding loading and HTTP transport.",
                       "Different output streams are expected; no quality equivalence or improved intelligence is established.",
                       "No OS-cache purge, independent cold-disk claim, confidence interval, or tail-latency claim.",
                       "Installed-stock denominator is not current-main unless independently established. No multiplication of historical component gains.",
                       "All provisional pair arithmetic remains visible on invalid/incomplete runs; no invalid run is silently promoted into an accepted aggregate."]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--input", action="append", required=True, help="ARM:BLOCK:JSONL_PATH")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = []
    for item in args.input:
        arm, block, path = item.split(":", 2)
        inputs.append((arm, int(block), Path(path)))
    report = analyze(json.loads(args.plan.read_text()), inputs)
    report.update(plan_sha256=digest(args.plan), analyzer_sha256=digest(__file__))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "issues": len(report["issues"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
