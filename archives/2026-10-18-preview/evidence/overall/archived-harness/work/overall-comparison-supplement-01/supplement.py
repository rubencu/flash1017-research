"""CPU-only supplementary analysis of completed, independently audited records.

Never changes the original fixed512 qualification. Prefill/TTFT may qualify
separately. Forced512 observations remain descriptive. Optional128 metrics are
post-hoc delivery prefixes of the original512 runs, never simulated128 runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def gm(values):
    if not values or not all(finite_positive(v) for v in values):
        raise ValueError("Geometric mean requires a nonempty set of finite positive observations")
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


GROUPS = ("all", "natural", "synthetic", "1K", "16K", "64K", "block0", "block1")


def in_group(row, name):
    if name == "all":
        return True
    if name in ("natural", "synthetic"):
        return row["cohort"] == {"natural": "natural-source", "synthetic": "synthetic-generation"}[name]
    if name in ("1K", "16K", "64K"):
        return row["input_tokens"] == {"1K": 1024, "16K": 16384, "64K": 65536}[name]
    return row["block"] == int(name[-1])


def metric_value(row, metric):
    if metric == "prefix128_delivery_seconds":
        return row["arrival_seconds"][127]
    if metric == "prefix128_post_first_tokens_per_second":
        return 127 / (row["arrival_seconds"][127] - row["arrival_seconds"][0])
    return row[metric]


def metrics_for_pairs(pairs, metrics):
    output = {}
    for metric in metrics:
        baseline = gm([metric_value(a, metric) for a, _ in pairs])
        candidate = gm([metric_value(b, metric) for _, b in pairs])
        ratio = gm([metric_value(b, metric) / metric_value(a, metric) for a, b in pairs])
        is_rate = "tokens_per_second" in metric
        output[metric] = {"baseline_geometric_mean": baseline, "candidate_geometric_mean": candidate,
            "ratio_candidate_over_baseline": ratio, "change_percent": 100 * (ratio - 1),
            "improvement_percent": 100 * ((ratio - 1) if is_rate else (1 - ratio)),
            "direction_of_improvement": "higher" if is_rate else "lower"}
    return output


def analyze(audit_path, plan_path, include_prefix128=False):
    audit_path, plan_path = Path(audit_path), Path(plan_path)
    original = json.loads(audit_path.read_text())
    plan = json.loads(plan_path.read_text())
    integrity_issues, audit_issues, numeric_issues = [], [], []
    original_excerpt = {k: original.get(k) for k in ("status", "performance_claim_qualified", "restoration_confirmed",
        "measured_requests", "structural_issues", "fixed_length_issues", "restoration_issues", "aggregate_scope")}
    if original.get("source_digests", {}).get("plan") != sha(plan_path):
        integrity_issues.append("Independent audit is bound to a different executable plan")
    if original.get("structural_issues") != []:
        audit_issues.append("Independent structural audit has issues or is missing")
    if original.get("restoration_issues") != [] or original.get("restoration_confirmed") is not True:
        audit_issues.append("Restoration is not independently confirmed")
    if original.get("measured_requests") != 48:
        audit_issues.append("Independent audit does not contain all48 measured requests")
    observed = original.get("processes", [])
    expected = plan.get("processes", [])
    complete_processes = len(observed) == len(expected) == 6
    if not complete_processes:
        integrity_issues.append("Exactly six audited processes are required")
    requests, raw_bindings = [], []
    for index, process in enumerate(observed):
        if index >= len(expected):
            integrity_issues.append("Unexpected process in independent audit")
            continue
        wanted = expected[index]
        arm, block = wanted["arm"], wanted["block"]
        label = f"process{index}/{arm}/block{block}"
        if process.get("index") != index or process.get("arm") != arm or process.get("block") != block:
            integrity_issues.append("Audited process identity differs from plan: " + label)
        if process.get("exit_code") != 0:
            complete_processes = False
            integrity_issues.append("Process exit was not confirmed successful: " + label)
        path = Path(process["output"])
        if not path.is_file() or sha(path) != process.get("output_sha256"):
            integrity_issues.append("Raw file missing or changed since independent audit: " + str(path))
            complete_processes = False
            continue
        try:
            rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        except ValueError:
            complete_processes = False
            integrity_issues.append("Incomplete/malformed raw file: " + str(path))
            continue
        raw_bindings.append({"arm": arm, "block": block, "path": str(path), "sha256": sha(path)})
        trace = process.get("load_trace", {})
        trace_path = Path(trace.get("path", "/nonexistent-load-trace"))
        if not trace_path.is_file() or sha(trace_path) != trace.get("sha256"):
            integrity_issues.append("Same-process load trace missing or changed: " + label)
        headers = [r for r in rows if r.get("type") == "manifest"]
        completions = [r for r in rows if r.get("type") == "complete"]
        cleanups = [r for r in rows if r.get("type") == "cleanup"]
        if len(completions) != 1 or completions[0].get("jobs") != 9 or len(cleanups) != 1:
            complete_processes = False
            integrity_issues.append("Missing complete/cleanup records: " + label)
        if len(headers) != 1 or headers[0].get("plan_sha256") != sha(plan_path) or headers[0].get("manifest_sha256") != original.get("source_digests", {}).get("manifest"):
            integrity_issues.append("Raw manifest binding differs from independent audit: " + label)
        runs = [r for r in rows if r.get("type") == "run"]
        if len(runs) != len(wanted["jobs"]) or len(runs) != 9:
            complete_processes = False
            integrity_issues.append("Expected nine ordered requests including warmup: " + label)
        for job_index, row in enumerate(runs):
            if job_index >= len(wanted["jobs"]):
                continue
            job = wanted["jobs"][job_index]
            prompt = plan["prompts"][job["prompt"]]
            rid = label + "/" + job["prompt"]
            if (row.get("prompt_id") != job["prompt"] or row.get("warmup") is not job["warmup"] or
                row.get("seed") != job["seed"] or row.get("job_index") != job_index or
                row.get("arm") != arm or row.get("block") != block or row.get("process_index") != index):
                integrity_issues.append("Request order/identity/seed differs: " + rid)
            if job["warmup"]:
                continue
            expected_hash = hashlib.sha256(json.dumps(prompt["tokens"]).encode()).hexdigest()
            if row.get("input_tokens_sha256") != expected_hash or row.get("input_tokens") != len(prompt["tokens"]):
                integrity_issues.append("Measured input differs: " + rid)
            if row.get("generated_tokens") != 512 or len(row.get("token_ids", [])) != 512:
                integrity_issues.append("Measured output is not512 tokens: " + rid)
            if row.get("cohort") != prompt["cohort"]:
                integrity_issues.append("Measured cohort differs: " + rid)
            for metric in ("prefill_seconds", "ttft_seconds", "total_seconds", "decode_tokens_per_second"):
                if not finite_positive(row.get(metric)):
                    numeric_issues.append("Nonfinite/nonpositive " + metric + ": " + rid)
            arrivals = row.get("arrival_seconds", [])
            good_arrivals = len(arrivals) == 512 and all(finite_positive(v) for v in arrivals)
            good_arrivals = good_arrivals and all(x <= y for x, y in zip(arrivals, arrivals[1:]))
            good_arrivals = good_arrivals and arrivals[127] > arrivals[0] and arrivals[-1] > arrivals[0]
            if not good_arrivals:
                numeric_issues.append("Invalid512-token arrival sequence: " + rid)
            elif all(finite_positive(row.get(k)) for k in ("prefill_seconds", "ttft_seconds", "total_seconds", "decode_tokens_per_second")):
                if not (row["prefill_seconds"] <= row["ttft_seconds"] == arrivals[0] <= arrivals[-1] <= row["total_seconds"]):
                    numeric_issues.append("Timing boundaries inconsistent: " + rid)
                expected_rate = 511 / (arrivals[-1] - arrivals[0])
                if not math.isclose(row["decode_tokens_per_second"], expected_rate, rel_tol=1e-12):
                    numeric_issues.append("Stored512 decode-rate arithmetic differs: " + rid)
            eos = row.get("first_eos_position")
            if eos is not None and (type(eos) is not int or not 0 <= eos < 512):
                integrity_issues.append("Invalid first EOS position: " + rid)
            if row.get("fixed_length_valid") is not (eos is None or eos == 511):
                integrity_issues.append("EOS/fixed-length fields inconsistent: " + rid)
            requests.append(row)
    by_key = {(r.get("arm"), r.get("block"), r.get("prompt_id")): r for r in requests}
    if len(requests) != 48 or len(by_key) != 48:
        integrity_issues.append("All48 unique measured requests are required; no subset permitted")
    complete_processes = complete_processes and len(requests) == 48
    source_complete = complete_processes and not integrity_issues
    qualified = source_complete and not audit_issues and not numeric_issues
    eos_blockers = [{"arm": r["arm"], "block": r["block"], "fixture": r["prompt_id"], "first_eos_position": r["first_eos_position"]}
        for r in requests if type(r.get("first_eos_position")) is int and r["first_eos_position"] < 128]
    prefix_available = bool(include_prefix128 and qualified and not eos_blockers)
    base_metrics = ["prefill_seconds", "ttft_seconds", "total_seconds", "decode_tokens_per_second"]
    metrics = base_metrics + (["prefix128_delivery_seconds", "prefix128_post_first_tokens_per_second"] if prefix_available else [])
    pair_summaries, case_summaries, arm_summaries = [], [], []
    # All calculations require the complete record set. Fixed512 validity is
    # deliberately NOT a filtering criterion, and never becomes a new pass.
    if source_complete and not numeric_issues:
        for baseline in ("native2048", "native8192"):
            pairs = []
            for candidate in requests:
                if candidate["arm"] == "complete":
                    key = (baseline, candidate["block"], candidate["prompt_id"])
                    if key not in by_key:
                        raise ValueError("Missing paired request; cannot calculate a subset")
                    pairs.append((by_key[key], candidate))
            if len(pairs) != 16:
                raise ValueError("Expected16 complete pairs per native denominator")
            for group in GROUPS:
                chosen = [(a, b) for a, b in pairs if in_group(b, group)]
                pair_summaries.append({"baseline": baseline, "group": group, "pairs": len(chosen), "metrics": metrics_for_pairs(chosen, metrics)})
            for fixture in sorted({b["prompt_id"] for _, b in pairs}):
                chosen = [(a, b) for a, b in pairs if b["prompt_id"] == fixture]
                if len(chosen) != 2:
                    raise ValueError("Per-case summary requires both blocks")
                case_summaries.append({"baseline": baseline, "fixture": fixture, "pairs": 2,
                    "cohort": chosen[0][1]["cohort"], "input_tokens": chosen[0][1]["input_tokens"],
                    "metrics": metrics_for_pairs(chosen, metrics)})
        for arm in ("native2048", "native8192", "complete"):
            for group in GROUPS:
                chosen = [r for r in requests if r["arm"] == arm and in_group(r, group)]
                arm_summaries.append({"arm": arm, "group": group, "requests": len(chosen),
                    "geometric_means": {metric: gm([metric_value(r, metric) for r in chosen]) for metric in metrics}})
    return {"schema": "overall-comparison-supplement/1", "status": "completed_supplement" if qualified else "refused_qualification",
        "input_run_completed": complete_processes, "original_qualification_unchanged": original_excerpt,
        "audit_path": str(audit_path.resolve()), "audit_sha256": sha(audit_path), "plan_sha256": sha(plan_path),
        "raw_bindings": raw_bindings, "integrity_issues": integrity_issues, "audit_issues": audit_issues,
        "numeric_issues": numeric_issues, "measured_requests": len(requests),
        "prefill_ttft": {"qualified_separately": qualified,
            "scope": "Prefill/first-token endpoints do not include post-EOS continuation; original fixed512 qualification remains unchanged."},
        "forced512": {"status": "descriptive_observations_only" if source_complete and not numeric_issues else "unavailable_incomplete_or_invalid_records",
            "qualified_fixed512_response_claim": False,
            "early_eos_count": sum(type(r.get("first_eos_position")) is int and r["first_eos_position"] < 511 for r in requests),
            "scope": "All forced512 continuation timings, including post-EOS work. Not validated useful-answer or ordinary completion latency."},
        "prefix128": {"requested": bool(include_prefix128), "available_for_all48": prefix_available,
            "horizon_tokens": 128, "early_eos_blockers": eos_blockers, "subset_or_shorter_horizon_used": False,
            "scope": "Post-hoc observed delivery of the first128 pre-EOS tokens within the original512-token runs. Not a new128-token request, terminal synchronization/cleanup latency, or isolated GPU compute."},
        "pair_group_summaries": pair_summaries, "per_case_two_block_summaries": case_summaries, "per_arm_group_summaries": arm_summaries,
        "limits": ["No data or acceptance rule in the original frozen experiment or independent audit is changed.",
            "No selective sample deletion; all48 records and both native denominators are required. EOS127 blocks the entire prefix supplement; EOS128 allows first128 positions0..127.",
            "MTP buffers accepted tokens and may predraft further tokens. Arrival times measure observed delivery and can include work for tokens beyond128.",
            "A512-token request uses different remaining-budget/end-clamp behavior from a128-token request. No counterfactual shortened-run performance is inferred.",
            "Generated streams differ across arms. This does not establish quality equivalence or a speedup for equivalent completed tasks.",
            "Installed-stock0.7.1rc1 with native adaptive MTP, not current main; model load and HTTP transport excluded.",
            "Two process blocks on one machine, persistent OS caches, declared row-cache history; no confidence interval, cold-disk, tail-latency or broad generalization claim.",
            "Times/rates and ratios are geometric means over the same matched cases; neither historical component gains nor different endpoints are multiplied."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prefix128", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior supplementary evidence: " + str(args.output))
    report = analyze(args.audit, args.plan, args.prefix128)
    if not report["input_run_completed"]:
        raise RuntimeError("Run is incomplete; no supplementary output written")
    report["analyzer_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "prefill_ttft_qualified": report["prefill_ttft"]["qualified_separately"],
        "prefix128_available": report["prefix128"]["available_for_all48"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
