"""Independent CPU-only audit/report for the frozen immediate comparison.

Reads completed saved records; never imports the benchmark/runtime or contacts a
server. Raw hashes bind timing, load, activation and cleanup from each process.
Incomplete/early-EOS evidence is retained with explicitly provisional arithmetic.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values)) if values else None


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def equal_hash(path, expected):
    return Path(path).is_file() and sha(path) == expected


def under(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def audit(plan_path, manifest_path, proposal_path, supervisor_path, runtime_roots=None):
    plan_path, manifest_path, proposal_path, supervisor_path = map(Path, (plan_path, manifest_path, proposal_path, supervisor_path))
    plan, frozen, proposal, supervisor = map(read_json, (plan_path, manifest_path, proposal_path, supervisor_path))
    digests = {"plan": sha(plan_path), "manifest": sha(manifest_path), "proposal": sha(proposal_path), "supervisor": sha(supervisor_path)}
    workspace = plan_path.resolve().parents[2]
    runtime_roots = runtime_roots or {"native2048": "/Applications/oMLX.app/Contents/Resources/omlx",
        "native8192": "/Applications/oMLX.app/Contents/Resources/omlx",
        "complete": str(workspace / "outputs/flash1017-lab/_runtime/omlx")}
    structural, fixed_length, restoration = [], [], []

    def check(condition, label, bucket=structural):
        if not condition:
            bucket.append(label)

    check(plan.get("proposal_sha256") == digests["proposal"], "Plan/proposal binding")
    check(supervisor.get("plan_sha256") == digests["plan"], "Supervisor/plan binding")
    check(supervisor.get("manifest_sha256") == digests["manifest"], "Supervisor/manifest binding")
    check(frozen.get("stock_is_current_main") is False, "Installed-stock denominator label")
    check(frozen.get("checkpoint_revision") == "78456115d301eecf9fadcdcc81c0cf6f8b2c8eab", "Exact checkpoint revision")
    pins = frozen["sources_sha256"]
    for path, expected in pins.items():
        check(equal_hash(path, expected), "Frozen source changed/missing: " + path)
    check(pins.get(str(plan_path.resolve())) == digests["plan"], "Plan present in source closure")
    check(pins.get(str(proposal_path.resolve())) == digests["proposal"], "Proposal present in source closure")
    check(supervisor.get("benchmark_status") == "all_processes_completed", "Supervisor did not complete all processes")
    check(not supervisor.get("forced_shutdown"), "A benchmark process required forced shutdown")
    check(supervisor.get("restoration_status") == "restored", "Original service restoration not confirmed", restoration)
    before = supervisor.get("prior_state", {})
    after = supervisor.get("restored_state", {})
    check(before.get("loaded_models") == after.get("loaded_models"), "Restored model identity differs", restoration)
    check(before.get("memory") == after.get("memory"), "Restored memory settings differ", restoration)
    expected_processes = plan["processes"]
    observed_processes = supervisor.get("processes", [])
    check(len(expected_processes) == len(observed_processes) == 6, "Expected six process records")
    check(len({p.get("pid") for p in observed_processes}) == len(observed_processes), "Process IDs not distinct")
    execution_seconds = None
    if observed_processes and positive(observed_processes[-1].get("completed_at_unix")) and positive(supervisor.get("model_execution_started_at")):
        execution_seconds = observed_processes[-1]["completed_at_unix"] - supervisor["model_execution_started_at"]
        check(0 < execution_seconds <= plan["model_execution_wall_limit_seconds"] + 1,
              "Frozen model-execution wall limit exceeded (one-second polling tolerance)")
    else:
        check(False, "Model-execution interval not fully recorded")
    fixture_meta = {p["id"]: p for p in proposal["fixtures"]}
    process_audits, requests, memory_policies, device_ids, packages_seen = [], [], [], [], []

    for index, expected_process in enumerate(expected_processes):
        if index >= len(observed_processes):
            break
        process = observed_processes[index]
        arm, block = expected_process["arm"], expected_process["block"]
        label = f"process{index}/{arm}/block{block}"
        start_issues = len(structural)
        check(process.get("index") == index and process.get("arm") == arm and process.get("block") == block,
              "Supervisor process order/identity: " + label)
        check(process.get("exit_code") == 0 and positive(process.get("completed_at_unix")), "Child exit not confirmed successful: " + label)
        if index:
            prior = observed_processes[index - 1]
            check(process.get("started_at_unix", 0) >= prior.get("completed_at_unix", math.inf), "Process lifetimes overlap/unconfirmed: " + label)
        output = Path(process["output"])
        if not output.is_file():
            check(False, "Missing raw process file: " + label)
            continue
        try:
            rows = read_lines(output)
        except (ValueError, OSError) as exc:
            check(False, "Unreadable/incomplete process records: " + label + "/" + repr(exc))
            process_audits.append({"index": index, "arm": arm, "block": block,
                                   "output": str(output), "output_sha256": sha(output),
                                   "structural_issues": structural[start_issues:]})
            continue
        by_type = collections.defaultdict(list)
        for position, row in enumerate(rows):
            by_type[row.get("type")].append((position, row))
        check(not by_type["error"], "Worker error record: " + label)
        for kind in ("manifest", "load", "complete", "cleanup"):
            check(len(by_type[kind]) == 1, f"Expected one {kind}: {label}")
        process_summary = {"index": index, "arm": arm, "block": block, "output": str(output),
                           "output_sha256": sha(output), "record_counts": {k: len(v) for k, v in by_type.items()},
                           "exit_code": process.get("exit_code")}
        if any(len(by_type[k]) != 1 for k in ("manifest", "load", "complete", "cleanup")):
            process_summary["structural_issues"] = structural[start_issues:]
            process_audits.append(process_summary)
            continue
        manifest_index, header = by_type["manifest"][0]
        load_index, load = by_type["load"][0]
        complete_index, complete = by_type["complete"][0]
        cleanup_index, cleanup = by_type["cleanup"][0]
        wanted_profile = plan["profiles"][arm] | {"name": arm}
        check(header.get("plan_sha256") == digests["plan"] and header.get("manifest_sha256") == digests["manifest"], "Worker source-manifest binding: " + label)
        check(header.get("process_index") == index and header.get("block") == block and header.get("arm") == arm, "Worker identity: " + label)
        check(header.get("profile") == wanted_profile, "Worker profile: " + label)
        check(header.get("runtime_root") == str(Path(runtime_roots[arm]).resolve()), "Wrong actual runtime root: " + label)
        check(header.get("packages") == frozen["packages"], "Dependency version mismatch: " + label)
        check(header.get("memory_policy", {}).get("cache_limit_bytes") == plan["cache_limit_bytes"], "Allocator-cache limit mismatch: " + label)
        memory_policies.append({k: v for k, v in header.get("memory_policy", {}).items() if k != "previous_wired_limit_bytes"})
        device_ids.append(header.get("device"))
        packages_seen.append(header.get("packages"))
        check(positive(load.get("seconds")), "Invalid separate load duration: " + label)
        process_summary["load"] = load

        traces = by_type["source_activation"]
        check([t.get("phase") for _, t in traces] == ["before_runs", "after_runs"], "Missing/incorrect activation phases: " + label)
        for position, trace in traces:
            phase = trace.get("phase")
            check(trace.get("passed") is True and trace.get("checks") and all(v is True for v in trace["checks"].values()), "Source activation failed: " + label + "/" + str(phase))
            check(not trace.get("source_escapes"), "Source root escape: " + label)
            check(trace.get("runtime_root") == str(Path(runtime_roots[arm]).resolve()), "Activation runtime identity: " + label)
            required_functions = {"head", "target_head", "draft_head", "target_call", "target_prefill", "draft_loop", "verify_loop", "loader"}
            check(required_functions.issubset(trace.get("functions", {})), "Missing function activation identities: " + label)
            functions = trace.get("functions", {})
            check(functions.get("head") == functions.get("target_head") == functions.get("draft_head"), "Projection alias source identities differ: " + label)
            for module_name, source in trace.get("sources", {}).items():
                check(under(source["path"], runtime_roots[arm]), "Imported module outside root: " + module_name)
                check(pins.get(source["path"]) == source["sha256"], "Imported source not pinned/mismatch: " + source["path"])
            for function_name, source in trace.get("functions", {}).items():
                check(bool(source.get("path")) and pins.get(source.get("path")) == source.get("sha256"), "Executed function source not pinned/mismatch: " + label + "/" + function_name)
            state = trace.get("state", {})
            shared_state = {"mtp_enabled": True, "dspark_enabled": True, "mtp_depth": wanted_profile["depth"],
                            "mtp_fixed": wanted_profile["fixed"], "ced_prefill": False,
                            "prefetch_bytes": wanted_profile["engram_prefetch_bytes"],
                            "prefill_step_requested": wanted_profile["prefill_step"]}
            if arm == "complete":
                shared_state.update(attention_rounding=True, attention_rounding_field_present=True,
                                    confidence_policy=True, draft_width=5, cone_attached=True,
                                    row_cache_attached=True, head_max_rows=6)
            else:
                shared_state.update(attention_rounding=False, attention_rounding_field_present=False,
                                    confidence_policy=False, cone_attached=False, row_cache_attached=False,
                                    native_target_class=True)
            for key, value in shared_state.items():
                check(state.get(key) == value, "Activation state mismatch: " + label + "/" + key)
            if phase == "before_runs":
                check(manifest_index < load_index < position, "Load/activation ordering: " + label)
            elif phase == "after_runs":
                check(position < complete_index < cleanup_index, "Completion/cleanup ordering: " + label)
        check(cleanup.get("native_close") is True, "Native model close not confirmed: " + label)
        if arm == "complete":
            check(cleanup.get("candidate_close") is True, "Candidate close not confirmed: " + label)
            closed_rows = cleanup.get("rows_after_native_close", {})
            check(closed_rows.get("retained_bytes") == 0 and closed_rows.get("live_tables") == [], "Native close retained candidate rows: " + label)
        else:
            check(cleanup.get("candidate_close") is None, "Candidate cleanup unexpectedly present in baseline: " + label)
        check(complete.get("jobs") == len(expected_process["jobs"]), "Completion job count: " + label)
        run_entries = by_type["run"]
        check(len(run_entries) == len(expected_process["jobs"]) == 9, "Request count: " + label)
        resets = by_type["row_cache_reset_after_warmup"]
        if arm == "complete":
            check(len(resets) == 1, "Missing/surplus warmup cache reset: " + label)
            if len(resets) == 1:
                reset_position, reset = resets[0]
                check(reset.get("snapshot", {}).get("retained_bytes") == 0, "Warmup row cache not cleared: " + label)
                if len(run_entries) >= 2:
                    check(run_entries[0][0] < reset_position < run_entries[1][0], "Cache-reset ordering: " + label)
        else:
            check(not resets, "Candidate cache reset in baseline: " + label)
        for job_index, (position, row) in enumerate(run_entries):
            if job_index >= len(expected_process["jobs"]):
                break
            job = expected_process["jobs"][job_index]
            prompt = plan["prompts"][job["prompt"]]
            rid = label + "/" + job["prompt"]
            check(row.get("job_index") == job_index and row.get("process_index") == index and row.get("arm") == arm and row.get("block") == block, "Request identity/order: " + rid)
            check(row.get("prompt_id") == job["prompt"] and row.get("warmup") is job["warmup"], "Request fixture/order: " + rid)
            check(row.get("seed") == job["seed"] and row.get("profile") == wanted_profile, "Request seed/profile: " + rid)
            token_hash = hashlib.sha256(json.dumps(prompt["tokens"]).encode()).hexdigest()
            check(row.get("input_tokens_sha256") == token_hash == prompt["input_tokens_sha256"], "Input identity: " + rid)
            check(row.get("input_tokens") == len(prompt["tokens"]), "Input length: " + rid)
            check(row.get("generated_tokens") == prompt["max_tokens"] and len(row.get("token_ids", [])) == prompt["max_tokens"], "Output length: " + rid)
            if not job["warmup"]:
                check(row.get("fixed_length_valid") is True and row.get("first_eos_position") in (None, prompt["max_tokens"] - 1), "Early EOS invalidates fixed-length benchmark: " + rid, fixed_length)
            check(bool(row.get("mtp_stats")), "No observed MTP activity: " + rid)
            check(all(positive(row.get(k)) for k in ("total_seconds", "prefill_seconds", "ttft_seconds", "decode_tokens_per_second")), "Invalid request timing: " + rid)
            arrivals = row.get("arrival_seconds", [])
            check(len(arrivals) == prompt["max_tokens"] and all(x <= y for x, y in zip(arrivals, arrivals[1:])), "Arrival sequence: " + rid)
            if len(arrivals) > 1 and arrivals[-1] > arrivals[0]:
                expected_tps = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                check(math.isclose(row["decode_tokens_per_second"], expected_tps, rel_tol=1e-12), "Decode rate arithmetic: " + rid)
                check(row["ttft_seconds"] == arrivals[0] and row["prefill_seconds"] <= arrivals[0] <= arrivals[-1] <= row["total_seconds"], "Timing boundary ordering: " + rid)
            resources = row.get("resources", {})
            check(resources.get("system_available_min_bytes", 0) >= 8 * 2**30, "Memory headroom threshold: " + rid)
            check(resources.get("swap_peak_bytes", math.inf) - resources.get("swap_start_bytes", 0) <= 512 * 2**20, "Swap-growth threshold: " + rid)
            if arm == "complete":
                candidate = row.get("candidate", {})
                check(candidate.get("passed") is True and candidate.get("checks") and all(v is True for v in candidate["checks"].values()), "Candidate execution checks: " + rid)
                after_state = candidate.get("after", {})
                eligible = sum(min(wanted_profile["prefill_step"], len(prompt["tokens"]) - 1 - start) >= 2668 for start in range(0, len(prompt["tokens"]) - 1, wanted_profile["prefill_step"]))
                cone = after_state.get("cone", {})
                check(candidate.get("expected_eligible_chunks") == eligible, "Eligibility arithmetic: " + rid)
                check(cone.get("cone_counts", {}).get("cone_calls", 0) == eligible, "Observed replay count: " + rid)
                check(cone.get("prime_counts", {}).get("padded_appends", 0) == eligible and cone.get("prime_counts", {}).get("wkv_pads", 0) == 3 * eligible, "Observed draft-priming counts: " + rid)
                check(cone.get("counts", {}).get("call_failures", 0) == 0, "Replay call failure: " + rid)
                rows_after = after_state.get("rows", {})
                rows_before = candidate.get("before", {}).get("rows", {})
                delta = sum(t["unique_queries"] for t in rows_after.get("live_tables", [])) - sum(t["unique_queries"] for t in rows_before.get("live_tables", []))
                check(delta > 0, "No observed row-cache queries: " + rid)
                check(0 <= rows_after.get("retained_bytes", -1) <= 512 * 2**20, "Retained row bound: " + rid)
                check(any(e.get("confidence_logits") is not None for e in row.get("events", [])), "No confidence-policy execution: " + rid)
            if not job["warmup"]:
                requests.append(row)
        load_trace = output.with_suffix(".load.jsonl")
        check(load_trace.is_file(), "Missing same-process load trace: " + label)
        if load_trace.is_file():
            try:
                entries = read_lines(load_trace)
            except (ValueError, OSError) as exc:
                entries = []
                check(False, "Unreadable/incomplete load trace: " + label + "/" + repr(exc))
            check(bool(entries) and len(entries) % 2 == 0, "Incomplete shard-load trace: " + label)
            for a, b in zip(entries[::2], entries[1::2]):
                check(a.get("stage") == "before" and b.get("stage") == "after" and a.get("shard") == b.get("shard"), "Unpaired shard-load trace: " + label)
                check(a.get("available_bytes", 0) >= 12 * 2**30 and b.get("available_bytes", 0) >= 12 * 2**30, "Load reserve violation: " + label)
            process_summary["load_trace"] = {"path": str(load_trace), "sha256": sha(load_trace), "observations": len(entries), "shard_calls": len(entries) // 2}
        process_summary["structural_issues"] = structural[start_issues:]
        process_audits.append(process_summary)
    for field, values in (("memory policy", memory_policies), ("device", device_ids), ("packages", packages_seen)):
        check(bool(values) and all(x == values[0] for x in values), "Arms did not share " + field)
    check(len(requests) == 48, "Expected all48 measured requests")
    by_identity = {(r["arm"], r["block"], r["prompt_id"]): r for r in requests}
    check(len(by_identity) == len(requests), "Duplicate measured identity")
    pair_rows = []
    for baseline in ("native2048", "native8192"):
        for block in (0, 1):
            for fixture in fixture_meta:
                a = by_identity.get((baseline, block, fixture))
                b = by_identity.get(("complete", block, fixture))
                if a is None or b is None:
                    continue
                if not all(positive(r.get(k)) for r in (a, b) for k in ("total_seconds", "prefill_seconds", "ttft_seconds", "decode_tokens_per_second")):
                    continue
                pair = {"baseline": baseline, "block": block, "fixture": fixture,
                    "cohort": fixture_meta[fixture]["cohort"], "input_tokens": a["input_tokens"],
                    "baseline_total_seconds": a["total_seconds"], "candidate_total_seconds": b["total_seconds"],
                    "identical_output_ids": a["token_ids"] == b["token_ids"],
                    "fixed_length_valid": a.get("fixed_length_valid") is True and b.get("fixed_length_valid") is True}
                for metric in ("total_seconds", "prefill_seconds", "ttft_seconds", "decode_tokens_per_second"):
                    pair[metric + "_candidate_over_baseline"] = b[metric] / a[metric]
                pair_rows.append(pair)
    aggregates = []
    predicates = [("all", lambda r: True), ("natural", lambda r: r["cohort"] == "natural-source"),
                  ("synthetic", lambda r: r["cohort"] == "synthetic-generation")]
    predicates += [(f"tokens{n}", lambda r, n=n: r["input_tokens"] == n) for n in (1024, 16384, 65536)]
    predicates += [(f"block{n}", lambda r, n=n: r["block"] == n) for n in (0, 1)]
    for baseline in ("native2048", "native8192"):
        for group, predicate in predicates:
            selected = [p for p in pair_rows if p["baseline"] == baseline and predicate(p)]
            item = {"baseline": baseline, "group": group, "pairs": len(selected),
                "faster_whole_requests": sum(p["total_seconds_candidate_over_baseline"] < 1 for p in selected),
                "identical_output_streams": sum(p["identical_output_ids"] for p in selected)}
            for metric in ("total_seconds", "prefill_seconds", "ttft_seconds", "decode_tokens_per_second"):
                value = geometric_mean([p[metric + "_candidate_over_baseline"] for p in selected])
                item[metric + "_ratio"] = value
                item[metric + "_change_percent"] = None if value is None else 100 * (value - 1)
            aggregates.append(item)
    resource_report = []
    for arm in plan["profiles"]:
        selected = [r for r in requests if r["arm"] == arm]
        if not selected:
            continue
        resource_report.append({"arm": arm, "requests": len(selected),
            "geometric_mean_total_seconds": geometric_mean([r["total_seconds"] for r in selected]),
            "geometric_mean_decode_tokens_per_second": geometric_mean([r["decode_tokens_per_second"] for r in selected]),
            "max_process_footprint_gib": max(r["resources"]["process_footprint_peak_bytes"] for r in selected) / 2**30,
            "minimum_system_available_gib": min(r["resources"]["system_available_min_bytes"] for r in selected) / 2**30,
            "max_request_swap_growth_gib": max(r["resources"]["swap_peak_bytes"] - r["resources"]["swap_start_bytes"] for r in selected) / 2**30,
            "process_disk_read_gib_sum": sum(r["resources"].get("disk_read_bytes_delta", 0) for r in selected) / 2**30})
    accepted = not structural and not fixed_length and not restoration
    return {"status": "accepted_complete_fixed_length_record_audit" if accepted else "not_qualified_all_observations_preserved",
        "performance_claim_qualified": not structural and not fixed_length,
        "restoration_confirmed": not restoration, "structural_issues": structural,
        "fixed_length_issues": fixed_length, "restoration_issues": restoration,
        "source_digests": digests, "frozen_source_files_checked": len(pins), "processes": process_audits,
        "model_execution_seconds": execution_seconds, "model_execution_wall_limit_seconds": plan["model_execution_wall_limit_seconds"],
        "measured_requests": len(requests), "resource_summary": resource_report,
        "aggregate_scope": "accepted fixed suite" if not structural and not fixed_length else "provisional arithmetic on retained complete records; no accepted speed claim",
        "aggregates": aggregates, "pairs": pair_rows,
        "early_eos_records": [{"arm": r["arm"], "block": r["block"], "fixture": r["prompt_id"],
            "first_eos_position": r.get("first_eos_position"), "generated_tokens": r["generated_tokens"],
            "token_ids_sha256": hashlib.sha256(json.dumps(r["token_ids"]).encode()).hexdigest()}
            for r in requests if r.get("fixed_length_valid") is not True],
        "limits": ["Installed-stock0.7.1rc1, not current upstream main; native adaptive MTP enabled in both stock baselines.",
            "Whole response means loaded-model prefill through synchronized512-token generation; excludes loading and HTTP transport.",
            "Natural source and synthetic short-context cohorts are separately reported; fixed512 can include post-EOS output and invalidates the benchmark when EOS occurs early.",
            "Different generated streams are permitted in this package comparison and do not establish quality equivalence.",
            "Two fresh processes per arm; OS/file caches naturally persist. Candidate HotRows are reset after warmup then retained through each measured sequence.",
            "Sampled process I/O is whole-request traffic, not Engram-only physical disk bytes; memory samples can miss peaks.",
            "No multiplication of historical gains, confidence interval, population generalization, cold-disk or tail-latency claim.",
            "This is an independent saved-record/hash/activation audit, not external peer review, clean-machine replication, or a new inference run."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.plan, args.manifest, args.proposal, args.supervisor)
    report["analyzer_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "structural_issues": len(report["structural_issues"]),
                     "fixed_length_issues": len(report["fixed_length_issues"]), "restoration_issues": len(report["restoration_issues"]),
                     "output": str(args.output)}))


if __name__ == "__main__":
    main()
