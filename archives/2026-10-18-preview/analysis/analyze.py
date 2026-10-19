#!/usr/bin/env python3
"""Recalculate this release's saved observations. Python >=3.9, standard library only.

This reads JSON; it never imports a model/runtime, follows recorded source paths,
contacts a service, or reruns inference. See README.txt for evidence boundaries.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

PLAN_SHA256 = "f668ef98840aa9c13a736e4b8329b396ad1f69032c46836d1640fdf9346972db"
SOURCE_MANIFEST_SHA256 = "767355abf8eedac3427b0537612bceda3b646b01142d93bc0dee0152faedc570"
ARMS = ("native2048", "native8192", "complete")
ORDER = (("native2048", 0), ("native8192", 0), ("complete", 0),
         ("complete", 1), ("native8192", 1), ("native2048", 1))
FILES = tuple("%02d-%s-block%d.jsonl" % (i, arm, block)
              for i, (arm, block) in enumerate(ORDER))
CASES = {
    "held-code-1k": (1024, "synthetic-generation"),
    "held-prose-1k": (1024, "synthetic-generation"),
    "performance-cpython-16384": (16384, "natural-source"),
    "performance-curl-16384": (16384, "natural-source"),
    "performance-moby-dick-16384": (16384, "natural-source"),
    "performance-pride-prejudice-16384": (16384, "natural-source"),
    "performance-cpython-65536": (65536, "natural-source"),
    "performance-pride-prejudice-65536": (65536, "natural-source"),
}
WARMUP = "authored-warmup-16384"
# Exact checkpoint tokenizer contract, not a heuristic inferred from positions.
EOS_TOKEN_ID = 1
VOCAB_SIZE = 129280
GROUPS = ("all", "natural", "synthetic", "1K", "16K", "64K", "block0", "block1")
BASE_METRICS = ("prefill_seconds", "ttft_seconds", "total_seconds", "decode_tokens_per_second")
PREFIX_METRICS = ("prefix128_delivery_seconds", "prefix128_post_first_tokens_per_second")


class ValidationError(ValueError):
    """A complete valid record set is required; never calculate a subset."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError("Non-finite JSON constant: " + value)


def decode(text):
    return json.loads(text, object_pairs_hook=strict_pairs, parse_constant=reject_constant)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def token_hash(tokens):
    # Matches the frozen producer's encoding, including its ordinary JSON spaces.
    return sha_bytes(json.dumps(tokens).encode("utf-8"))


def number(value, positive=True):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def gm(values):
    values = list(values)
    require(bool(values) and all(number(v) for v in values), "Invalid geometric-mean inputs")
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def tokens_valid(tokens, count):
    return (isinstance(tokens, list) and len(tokens) == count
            and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in tokens))


def validate_plan(plan):
    require(plan.get("status") == "frozen_before_any_model_run", "Plan was not frozen")
    require(plan.get("counts") == {"measured": 48, "excluded_warmups": 6, "processes": 6},
            "Wrong study counts")
    require(set(plan.get("profiles", {})) == set(ARMS), "Wrong study arms")
    prompts = plan.get("prompts", {})
    require(set(prompts) == set(CASES) | {WARMUP}, "Missing or extra plan case")
    for name, prompt in prompts.items():
        n, cohort = (16384, "excluded-warmup") if name == WARMUP else CASES[name]
        require(tokens_valid(prompt.get("tokens"), n), "Invalid plan tokens: " + name)
        require(prompt.get("prompt_tokens") == n and prompt.get("cohort") == cohort,
                "Wrong plan length/cohort: " + name)
        require(prompt.get("input_tokens_sha256") == token_hash(prompt["tokens"]),
                "Plan token hash mismatch: " + name)
        require(prompt.get("max_tokens") == (64 if name == WARMUP else 512)
                and prompt.get("force_length") is True and prompt.get("thinking") is False
                and prompt.get("temperature") == 1 and prompt.get("top_p") == 1,
                "Wrong frozen generation settings: " + name)
    processes = plan.get("processes", [])
    require(len(processes) == 6, "Expected six planned processes")
    seeds = {}
    for index, (process, identity) in enumerate(zip(processes, ORDER)):
        require((process.get("arm"), process.get("block")) == identity, "Wrong planned process order")
        jobs = process.get("jobs", [])
        require(len(jobs) == 9 and jobs[0].get("prompt") == WARMUP
                and jobs[0].get("warmup") is True, "Missing or misplaced warmup")
        require({j.get("prompt") for j in jobs[1:]} == set(CASES)
                and all(j.get("warmup") is False for j in jobs[1:]), "Missing/duplicate planned case")
        for job in jobs:
            require(type(job.get("seed")) is int, "Seed must be an integer")
            key = (identity[1], job["prompt"])
            require(key not in seeds or seeds[key] == job["seed"], "Unmatched planned seeds")
            seeds[key] = job["seed"]


def validate_resources(resource, label):
    fields = ("process_footprint_peak_bytes", "system_available_min_bytes", "swap_start_bytes",
              "swap_peak_bytes", "swap_end_bytes", "disk_read_bytes_delta")
    require(isinstance(resource, dict), "Missing resources: " + label)
    for key in fields:
        require(type(resource.get(key)) is int and resource[key] >= 0, "Invalid resource " + key + ": " + label)
    require(resource["process_footprint_peak_bytes"] > 0, "Zero footprint: " + label)
    require(resource["swap_peak_bytes"] >= max(resource["swap_start_bytes"], resource["swap_end_bytes"]),
            "Swap peak lower than endpoint: " + label)
    require(resource["system_available_min_bytes"] >= 8 * 2**30, "Memory headroom violation: " + label)
    require(resource["swap_peak_bytes"] - resource["swap_start_bytes"] <= 512 * 2**20,
            "Swap-growth violation: " + label)


def validate_run(row, job, prompt, profile, process_index, job_index, arm, block):
    label = "%s/block%d/%s" % (arm, block, job["prompt"])
    expected = {"prompt_id": job["prompt"], "seed": job["seed"], "arm": arm, "block": block,
                "process_index": process_index, "job_index": job_index, "repeat": block,
                "cohort": prompt["cohort"], "profile": profile}
    require(all(row.get(k) == v for k, v in expected.items()), "Request identity/order/profile mismatch: " + label)
    require(row.get("warmup") is job["warmup"], "Wrong warmup flag: " + label)
    require(row.get("input_tokens") == len(prompt["tokens"])
            and row.get("input_tokens_sha256") == prompt["input_tokens_sha256"], "Input mismatch: " + label)
    count = prompt["max_tokens"]
    tokens = row.get("token_ids")
    require(type(row.get("generated_tokens")) is int and row["generated_tokens"] == count
            and tokens_valid(tokens, count), "Wrong output length/token type: " + label)
    require(row.get("finish_reason") == "length", "Unexpected finish reason: " + label)
    eos = tokens.index(EOS_TOKEN_ID) if EOS_TOKEN_ID in tokens else None
    recorded = row.get("first_eos_position")
    require((recorded is None or type(recorded) is int) and recorded == eos, "EOS disagrees with tokens: " + label)
    require(row.get("fixed_length_valid") is (eos is None or eos == count - 1), "Inconsistent fixed-length flag: " + label)
    for metric in BASE_METRICS:
        require(number(row.get(metric)), "Invalid timing/rate " + metric + ": " + label)
    arrivals = row.get("arrival_seconds")
    require(isinstance(arrivals, list) and len(arrivals) == count
            and all(number(v) for v in arrivals), "Invalid arrival array: " + label)
    require(all(a <= b for a, b in zip(arrivals, arrivals[1:])), "Arrival timestamps decrease: " + label)
    require(arrivals[-1] > arrivals[0], "Zero decode duration: " + label)
    require(row["prefill_seconds"] <= row["ttft_seconds"] == arrivals[0]
            <= arrivals[-1] <= row["total_seconds"], "Inconsistent timing endpoints: " + label)
    require(math.isclose(row["decode_tokens_per_second"], (count - 1) / (arrivals[-1] - arrivals[0]), rel_tol=1e-12),
            "Stored decode rate disagrees with timestamps: " + label)
    if not job["warmup"]:
        require(arrivals[127] > arrivals[0], "Zero first-128 decode duration: " + label)
    validate_resources(row.get("resources"), label)
    mtp = row.get("mtp_stats")
    require(isinstance(mtp, list) and mtp and all(type(s.get("cycles")) is int and s["cycles"] > 0 for s in mtp),
            "Missing MTP execution: " + label)
    candidate = row.get("candidate", {})
    require(candidate.get("passed") is True, "Failed candidate execution checks: " + label)
    if arm == "complete":
        checks = candidate.get("checks")
        require(isinstance(checks, dict) and checks and all(v is True for v in checks.values()),
                "Missing/failed candidate checks: " + label)
        after = candidate.get("after", {})
        eligible = sum(min(8192, len(prompt["tokens"]) - 1 - start) >= 2668
                       for start in range(0, len(prompt["tokens"]) - 1, 8192))
        cone = after.get("cone", {})
        require(candidate.get("expected_eligible_chunks") == eligible
                and cone.get("cone_counts", {}).get("cone_calls", 0) == eligible
                and cone.get("prime_counts", {}).get("padded_appends", 0) == eligible
                and cone.get("prime_counts", {}).get("wkv_pads", 0) == 3 * eligible
                and cone.get("counts", {}).get("call_failures", 0) == 0,
                "Replay/priming telemetry mismatch: " + label)
        retained = after.get("rows", {}).get("retained_bytes")
        require(type(retained) is int and 0 <= retained <= 512 * 2**20, "Invalid row-cache bound: " + label)
        require(any(e.get("confidence_logits") is not None for e in row.get("events", [])),
                "Missing confidence execution: " + label)
    return row


def validate_records(plan, process_rows):
    """Validate saved fields; this does not authenticate the historical execution."""
    validate_plan(plan)
    require(len(process_rows) == 6, "Expected all six raw processes")
    measured, warmups, headers = [], [], []
    previous_end = 0
    for index, (rows, process) in enumerate(zip(process_rows, plan["processes"])):
        arm, block = ORDER[index]
        label = FILES[index]
        require(all(isinstance(r, dict) for r in rows), "Non-object JSONL record: " + label)
        types = [r.get("type") for r in rows]
        wanted_types = ["manifest", "load", "source_activation", "run"]
        if arm == "complete":
            wanted_types.append("row_cache_reset_after_warmup")
        wanted_types += ["run"] * 8 + ["source_activation", "complete", "cleanup"]
        require(types == wanted_types, "Missing, duplicate or misordered records: " + label)
        header, load, before, complete, cleanup = rows[0], rows[1], rows[2], rows[-2], rows[-1]
        profile = dict(plan["profiles"][arm], name=arm)
        require(header.get("arm") == arm and header.get("block") == block
                and header.get("process_index") == index and header.get("profile") == profile,
                "Wrong process identity/profile: " + label)
        require(header.get("plan_sha256") == PLAN_SHA256
                and header.get("manifest_sha256") == SOURCE_MANIFEST_SHA256,
                "Recorded original source/plan binding differs: " + label)
        start, end = header.get("started_at_unix"), complete.get("completed_at_unix")
        require(number(start) and number(end) and plan["created_at_unix"] <= start
                and previous_end <= start < end, "Invalid/overlapping process times: " + label)
        previous_end = end
        require(number(load.get("seconds")) and complete.get("jobs") == 9, "Invalid load/completion record: " + label)
        headers.append(header)
        after = rows[-3]
        for trace, phase in ((before, "before_runs"), (after, "after_runs")):
            require(trace.get("phase") == phase and trace.get("passed") is True
                    and trace.get("checks") and all(v is True for v in trace["checks"].values())
                    and not trace.get("source_escapes"), "Recorded source activation failed: " + label)
            require(trace.get("runtime_root") == header.get("runtime_root"), "Activation root differs: " + label)
        require(before.get("functions") == after.get("functions") and before.get("state") == after.get("state"),
                "Activation function/state identity changed: " + label)
        require(cleanup.get("native_close") is True, "Native cleanup missing: " + label)
        require(cleanup.get("candidate_close") is (True if arm == "complete" else None), "Candidate cleanup mismatch: " + label)
        if arm == "complete":
            closed = cleanup.get("rows_after_native_close", {})
            require(closed.get("retained_bytes") == 0 and closed.get("live_tables") == [], "Rows retained after cleanup")
            require(rows[4].get("snapshot", {}).get("retained_bytes") == 0, "Warmup cache was not reset")
        runs = [r for r in rows if r["type"] == "run"]
        for job_index, (row, job) in enumerate(zip(runs, process["jobs"])):
            validate_run(row, job, plan["prompts"][job["prompt"]], profile, index, job_index, arm, block)
            (warmups if job["warmup"] else measured).append(row)
        require(load["seconds"] + sum(r["total_seconds"] for r in runs) <= end - start + 0.001,
                "Process wall time cannot contain recorded work: " + label)
    require(len(measured) == 48 and len(warmups) == 6, "Wrong measured/warmup totals")
    require(previous_end - headers[0]["started_at_unix"] <= plan["model_execution_wall_limit_seconds"],
            "Recorded execution exceeds frozen wall limit")
    identities = {(r["arm"], r["block"], r["prompt_id"]) for r in measured}
    require(identities == {(a, b, c) for a in ARMS for b in (0, 1) for c in CASES}, "Missing/duplicate measured identity")
    for key in ("device", "packages"):
        require(headers[0].get(key) and all(h.get(key) == headers[0][key] for h in headers), "Arms differ in " + key)
    policies = [{k: v for k, v in h.get("memory_policy", {}).items() if k != "previous_wired_limit_bytes"} for h in headers]
    require(policies[0] and all(p == policies[0] for p in policies), "Different common memory policies")
    require(policies[0].get("cache_limit_bytes") == plan["cache_limit_bytes"], "Cache limit differs from plan")
    return measured, warmups, headers


def in_group(row, group):
    if group == "all":
        return True
    if group in ("natural", "synthetic"):
        return row["cohort"] == {"natural": "natural-source", "synthetic": "synthetic-generation"}[group]
    if group in ("1K", "16K", "64K"):
        return row["input_tokens"] == {"1K": 1024, "16K": 16384, "64K": 65536}[group]
    return row["block"] == int(group[-1])


def value(row, metric):
    if metric == "prefix128_delivery_seconds":
        return row["arrival_seconds"][127]
    if metric == "prefix128_post_first_tokens_per_second":
        return 127 / (row["arrival_seconds"][127] - row["arrival_seconds"][0])
    return row[metric]


def pair_metrics(pairs, metrics):
    result = {}
    for metric in metrics:
        ratio = gm(value(b, metric) / value(a, metric) for a, b in pairs)
        rate = "tokens_per_second" in metric
        result[metric] = {"baseline_geometric_mean": gm(value(a, metric) for a, _ in pairs),
                          "candidate_geometric_mean": gm(value(b, metric) for _, b in pairs),
                          "ratio_candidate_over_baseline": ratio, "change_percent": 100 * (ratio - 1),
                          "improvement_percent": 100 * (ratio - 1 if rate else 1 - ratio),
                          "direction_of_improvement": "higher" if rate else "lower"}
    return result


def summarize(plan, process_rows):
    measured, warmups, headers = validate_records(plan, process_rows)
    early = [{"arm": r["arm"], "block": r["block"], "fixture": r["prompt_id"],
              "first_eos_position": r["first_eos_position"], "generated_tokens": 512,
              "token_ids_sha256": token_hash(r["token_ids"])} for r in measured
             if r["first_eos_position"] is not None and r["first_eos_position"] < 511]
    blockers = [r for r in early if r["first_eos_position"] < 128]
    metrics = BASE_METRICS + (() if blockers else PREFIX_METRICS)
    by_key = {(r["arm"], r["block"], r["prompt_id"]): r for r in measured}
    pair_summaries, case_summaries, arm_summaries, resources = [], [], [], []
    for baseline in ARMS[:2]:
        pairs = [(by_key[(baseline, b, c)], by_key[("complete", b, c)]) for b in (0, 1) for c in sorted(CASES)]
        for group in GROUPS:
            selected = [(a, b) for a, b in pairs if in_group(b, group)]
            pair_summaries.append({"baseline": baseline, "group": group, "pairs": len(selected), "metrics": pair_metrics(selected, metrics)})
        for case in sorted(CASES):
            selected = [(a, b) for a, b in pairs if b["prompt_id"] == case]
            case_summaries.append({"baseline": baseline, "fixture": case, "pairs": 2,
                                   "cohort": CASES[case][1], "input_tokens": CASES[case][0], "metrics": pair_metrics(selected, metrics)})
    for arm in ARMS:
        selected = [r for r in measured if r["arm"] == arm]
        for group in GROUPS:
            grouped = [r for r in selected if in_group(r, group)]
            arm_summaries.append({"arm": arm, "group": group, "requests": len(grouped),
                                  "geometric_means": {m: gm(value(r, m) for r in grouped) for m in metrics}})
        resources.append({"arm": arm, "requests": len(selected),
                          "geometric_mean_total_seconds": gm(r["total_seconds"] for r in selected),
                          "geometric_mean_decode_tokens_per_second": gm(r["decode_tokens_per_second"] for r in selected),
                          "max_process_footprint_gib": max(r["resources"]["process_footprint_peak_bytes"] for r in selected) / 2**30,
                          "minimum_system_available_gib": min(r["resources"]["system_available_min_bytes"] for r in selected) / 2**30,
                          "max_request_swap_growth_gib": max(r["resources"]["swap_peak_bytes"] - r["resources"]["swap_start_bytes"] for r in selected) / 2**30,
                          "process_disk_read_gib_sum": sum(r["resources"]["disk_read_bytes_delta"] for r in selected) / 2**30})
    report = {"schema": "flash1017-portable-observation-analysis/1", "status": "validated_saved_observations",
              "baseline": "Untouched installed oMLX 0.7.1rc1, native adaptive MTP; NOT current main",
              "counts": {"processes": 6, "measured_requests": len(measured), "excluded_warmups": len(warmups), "unique_cases": 8},
              "fixed512": {"qualified": not early, "early_eos_count": len(early), "status": "failed_early_eos" if early else "no_early_eos",
                           "scope": "Original frozen fixed-length rule; full-512 metrics remain descriptive forced continuation, including post-EOS work."},
              "prefill_ttft": {"qualified_separately": True, "scope": "Endpoints contain no post-EOS continuation; no answer-quality equivalence claim."},
              "prefix128": {"post_hoc": True, "available_for_all48": not blockers, "horizon_tokens": 128,
                            "early_eos_blockers": blockers, "subset_or_shorter_horizon_used": False,
                            "scope": "Saved token-128 delivery within original 512-token executions. NOT new 128-token requests, terminal completion or isolated GPU time."},
              "early_eos_records": early, "pair_group_summaries": pair_summaries,
              "per_case_two_block_summaries": case_summaries, "per_arm_group_summaries": arm_summaries,
              "resource_summary": resources, "device": headers[0]["device"], "packages": headers[0]["packages"],
              "common_memory_policy": headers[0]["memory_policy"],
              "limits": ["All 48 records retained; two process blocks on one machine, equal-case geometric means, no confidence interval.",
                         "Warm runtime, retained OS caches, declared row-cache history; model loading and HTTP transport excluded.",
                         "Outputs can differ. No equivalent-answer speed, task-quality, generic-model, hardware or current-main claim.",
                         "Sampled footprint may miss transients; process disk counters are not exclusively Engram or physical SSD traffic.",
                         "This recalculates packaged observations and checks recorded activation fields. It does not rerun or reauthenticate the original runtime/source audit."]}
    report["headline"] = next(p["metrics"]["ttft_seconds"] for p in pair_summaries if p["baseline"] == "native2048" and p["group"] == "all")
    report["memory_tradeoff_gib_vs_native2048"] = resources[2]["max_process_footprint_gib"] - resources[0]["max_process_footprint_gib"]
    report["ttft_regressions_vs_native2048"] = [c for c in case_summaries if c["baseline"] == "native2048" and c["metrics"]["ttft_seconds"]["change_percent"] > 0]
    return report


def compare_values(actual, expected, label):
    """Recursive exact structure comparison; allow only floating-point roundoff."""
    if type(expected) is float:
        require(number(actual, positive=False) if expected >= 0 else type(actual) in (int, float) and math.isfinite(actual), "Invalid comparison value: " + label)
        require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), "Saved review disagrees: " + label)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Review keys differ: " + label)
        for key in expected:
            compare_values(actual[key], expected[key], label + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "Review list length differs: " + label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_values(a, b, label + "/" + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, "Saved review differs: " + label)


def compare_review(report, review_dir):
    audit = decode((review_dir / "independent-audit.json").read_text(encoding="utf-8"))
    supplement = decode((review_dir / "supplement.json").read_text(encoding="utf-8"))
    require(audit.get("structural_issues") == [] and audit.get("restoration_issues") == []
            and audit.get("restoration_confirmed") is True and audit.get("measured_requests") == 48,
            "Historical audit did not accept structure/restoration")
    require(audit.get("source_digests", {}).get("plan") == PLAN_SHA256, "Historical audit plan differs")
    require(audit.get("performance_claim_qualified") is report["fixed512"]["qualified"], "Original fixed512 result changed")
    require(len(audit.get("fixed_length_issues", [])) == report["fixed512"]["early_eos_count"], "Original fixed512 failures changed")
    compare_values(report["early_eos_records"], audit["early_eos_records"], "early_eos_records")
    compare_values(report["resource_summary"], audit["resource_summary"], "resource_summary")
    require(supplement.get("integrity_issues") == [] and supplement.get("audit_issues") == []
            and supplement.get("numeric_issues") == [] and supplement.get("measured_requests") == 48,
            "Historical supplement has unresolved issues")
    require(supplement.get("plan_sha256") == PLAN_SHA256, "Historical supplement plan differs")
    for key, sort_keys in (("pair_group_summaries", ("baseline", "group")),
                           ("per_case_two_block_summaries", ("baseline", "fixture")),
                           ("per_arm_group_summaries", ("arm", "group"))):
        order = lambda row: tuple(row[k] for k in sort_keys)
        compare_values(sorted(report[key], key=order), sorted(supplement[key], key=order), key)
    require(supplement.get("prefix128", {}).get("available_for_all48") is report["prefix128"]["available_for_all48"],
            "Historical prefix128 gate differs")
    report["saved_review_comparison"] = {"matched": True, "scope": "All aggregate, case and arm metrics, resource summaries, EOS records and original failure status."}


def verify_inventory(plan_path, raw_dir, review_dir, inventory_path):
    inventory_bytes = inventory_path.read_bytes()
    inventory = decode(inventory_bytes.decode("utf-8"))
    require(inventory.get("schema") == "flash1017-evidence-inventory/1", "Unknown evidence inventory schema")
    entries = inventory.get("files", [])
    by_path = {entry.get("path"): entry for entry in entries}
    require(len(by_path) == len(entries), "Duplicate evidence inventory paths")
    selected = [("evidence/overall/plan-01.json", plan_path)]
    selected += [("evidence/overall/raw/" + name, raw_dir / name) for name in FILES]
    selected += [("evidence/overall/review/" + name, review_dir / name)
                 for name in ("independent-audit.json", "supplement.json")]
    for logical, actual in selected:
        require(logical in by_path, "Evidence missing from inventory: " + logical)
        entry, data = by_path[logical], actual.read_bytes()
        require(entry.get("sha256") == sha_bytes(data) and entry.get("bytes") == len(data),
                "Packaged bytes disagree with inventory: " + logical)
    audit = decode((review_dir / "independent-audit.json").read_text(encoding="utf-8"))
    require(len(audit.get("processes", [])) == 6, "Historical audit lacks six process records")
    for index, name in enumerate(FILES):
        require(by_path["evidence/overall/raw/" + name].get("original_sha256")
                == audit["processes"][index].get("output_sha256"), "Original/public raw hash mapping differs: " + name)
    require(by_path["evidence/overall/plan-01.json"].get("original_sha256") == PLAN_SHA256,
            "Original/public plan identity differs")
    return sha_bytes(inventory_bytes)


def analyze(plan_path, raw_dir, review_dir, inventory_path=None):
    plan_bytes = plan_path.read_bytes()
    require(sha_bytes(plan_bytes) == PLAN_SHA256, "Frozen plan bytes changed")
    plan = decode(plan_bytes.decode("utf-8"))
    record_files = {p.name for p in raw_dir.iterdir() if re.fullmatch(r"\d+-[^/]+-block\d+\.jsonl", p.name)}
    require(record_files == set(FILES), "Missing or extra overall raw file")
    if inventory_path is None:
        inventory_path = Path(__file__).resolve().parent.parent / "EVIDENCE-INVENTORY.json"
    inventory_sha256 = verify_inventory(plan_path, raw_dir, review_dir, inventory_path)
    all_rows, bindings = [], []
    for filename in FILES:
        data = (raw_dir / filename).read_bytes()
        lines = data.decode("utf-8").splitlines()
        require(lines and all(line.strip() for line in lines), "Empty raw file/record: " + filename)
        all_rows.append([decode(line) for line in lines])
        bindings.append({"file": filename, "packaged_sha256": sha_bytes(data), "bytes": len(data)})
    report = summarize(plan, all_rows)
    compare_review(report, review_dir)
    report["inputs"] = {"plan_sha256": sha_bytes(plan_bytes), "original_source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                        "evidence_inventory_sha256": inventory_sha256,
                        "raw": bindings, "normalization_note": "Packaged hashes identify path-normalized bytes; original audit hashes identify original bytes. No recorded absolute path is accessed."}
    report["analyzer_sha256"] = sha_bytes(Path(__file__).read_bytes())
    return report


def render(report):
    h = report["headline"]
    lines = ["DeepSeek-V4.1-Flash-1017: saved-observation recalculation",
             report["baseline"],
             "TTFT: %.2f%% less time (%.3fs -> %.3fs, equal-case geometric means)." %
             (h["improvement_percent"], h["baseline_geometric_mean"], h["candidate_geometric_mean"]),
             "48 measured requests; 8 cases; 3 arms; 2 process blocks per arm; 6 excluded warmups.",
             "Original fixed-512 qualification: %s; %d early-EOS records retained." %
             ("PASS" if report["fixed512"]["qualified"] else "FAIL", report["fixed512"]["early_eos_count"]),
             "Prefill and TTFT qualify separately; no answer-quality equivalence claim."]
    if report["prefix128"]["available_for_all48"]:
        p = next(x["metrics"]["prefix128_delivery_seconds"] for x in report["pair_group_summaries"]
                 if x["baseline"] == "native2048" and x["group"] == "all")
        lines += ["POST HOC time to token 128: %.2f%% less (%.3fs -> %.3fs), all 48 original runs." %
                  (p["improvement_percent"], p["baseline_geometric_mean"], p["candidate_geometric_mean"]), report["prefix128"]["scope"]]
    else:
        lines.append("POST HOC prefix128 unavailable: at least one request ended before that horizon; no subset used.")
    lines += ["", "Per-case TTFT change vs native2048 (positive means slower):"]
    for c in report["per_case_two_block_summaries"]:
        if c["baseline"] == "native2048":
            lines.append("  %-38s %+7.2f%%" % (c["fixture"], c["metrics"]["ttft_seconds"]["change_percent"]))
    lines += ["", "Sampled process footprint peaks:"]
    for r in report["resource_summary"]:
        lines.append("  %-12s %.2f GiB" % (r["arm"], r["max_process_footprint_gib"]))
    lines += ["Complete package adds %.2f GiB vs native2048; includes its changed chunk configuration." % report["memory_tradeoff_gib_vs_native2048"],
              "Full-512 elapsed/rate fields in JSON are DESCRIPTIVE FORCED CONTINUATION, not useful-answer completion.",
              "Saved independent audit/supplement: numerical comparison matched.", ""]
    lines += report["limits"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    default = Path(__file__).resolve().parent.parent / "evidence" / "overall"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=default / "plan-01.json")
    parser.add_argument("--raw-dir", type=Path, default=default / "raw")
    parser.add_argument("--review-dir", type=Path, default=default / "review")
    parser.add_argument("--inventory", type=Path, default=default.parent.parent / "EVIDENCE-INVENTORY.json")
    parser.add_argument("--json", type=Path, help="Write machine-readable output; refuses overwrite")
    parser.add_argument("--text", type=Path, help="Write human-readable output; otherwise print it")
    args = parser.parse_args(argv)
    try:
        for destination in (args.json, args.text):
            require(destination is None or not destination.exists(), "Refusing to overwrite output: " + str(destination))
        require(args.json is None or args.text is None or args.json.resolve() != args.text.resolve(), "JSON/text destinations must differ")
        report = analyze(args.plan, args.raw_dir, args.review_dir, args.inventory)
        if args.json:
            with args.json.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        text = render(report)
        if args.text:
            with args.text.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Analysis refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
