"""Strict CPU-only interpretation of the frozen persistent-serving performance suite.

No inference modules are imported. Provenance is read from the run's archived
.sources copies, never from a subsequently changed live runtime/profile. A
partial, divergent or internally inconsistent run is rejected without a speed
summary. Raw observations remain untouched.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ARMS = ("native64", "fastpath64")
TABLE_BUDGET = 256 * 2**20
TOTAL_BUDGET = 2 * TABLE_BUDGET
ROW_BYTES = 160
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def loads(text):
    return json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def read_json(path):
    return loads(Path(path).read_text())

def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer: " + name)
    return value


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            "Invalid finite number: " + name)
    return value


def counter(value, name):
    require(isinstance(value, dict), "Missing counter: " + name)
    for key, val in value.items():
        integer(val, name + "." + key)
    return Counter(value)


def chunks(prompt):
    prefix = prompt["prompt_tokens"] - 1
    return [min(8192, prefix - start) for start in range(0, prefix, 8192)]


def validate_plan(plan, profile):
    require(plan["design"]["suite"] == "performance", "Boundary timing cannot be analyzed as performance")
    design = plan["design"]
    require((design["measured_pairs"], design["measured_jobs"], design["warmup_jobs"], design["generation_tokens"])
            == (12, 24, 2, 128), "Expected12 measured pairs/26 jobs/128 outputs")
    require(set(plan["profiles"]) == set(ARMS), "Unexpected profile arms")
    runtime = profile["runtime"]
    require(profile["qualified"] is True and runtime["ced_prefill"] is False and
            runtime["prefill_step"] == 8192 and runtime["engram_prefetch_bytes"] == 64 * 2**20,
            "Expected qualified native64 runtime")
    require(all(p == runtime for p in plan["profiles"].values()), "Both arms must use the same native64 configuration")
    jobs = plan["jobs"]
    require(len(jobs) == 26, "Expected exactly26 ordered jobs")
    require([bool(j.get("warmup", False)) for j in jobs] == [True, True] + [False] * 24,
            "Warmups must be the first separate pair")
    seen, orders, lengths, sources = set(), Counter(), Counter(), Counter()
    for i in range(0, 26, 2):
        a, b = jobs[i:i+2]
        require({a["profile"], b["profile"]} == set(ARMS), "Missing arm in adjacent pair")
        require({k:v for k,v in a.items() if k != "profile"} == {k:v for k,v in b.items() if k != "profile"},
                "Pair differs beyond arm")
        p = plan["prompts"][a["prompt"]]
        require(p["id"] == a["prompt"] and p["max_tokens"] == 128 and p["force_length"] is True and
                p["temperature"] == 1.0 and p["top_p"] == 1.0, "Unexpected frozen prompt settings")
        ids = p["tokens"]
        require(len(ids) == p["prompt_tokens"] and all(type(t) is int and t >= 0 for t in ids), "Invalid input token vector")
        require(digest(ids) == p["input_tokens_sha256"], "Input token digest differs")
        if i:
            require(a["prompt"] not in seen, "Repeated measured pair")
            seen.add(a["prompt"])
            orders["AB" if a["profile"] == ARMS[0] else "BA"] += 1
            lengths[p["prompt_tokens"]] += 1
            sources[p["source"]] += 1
    require(orders == {"AB":6,"BA":6}, "Measured pair orders must be balanced6AB/6BA")
    require(lengths == {16384:4,32768:4,65536:3,131072:1}, "Measured length strata changed")
    require(sources == {"cpython":3,"curl":3,"moby-dick":3,"pride-prejudice":3}, "Measured sources changed")


def validate_rows(snapshot, *, final=False):
    require(snapshot["per_table_budget_bytes"] == TABLE_BUDGET and snapshot["max_retained_bytes"] == TOTAL_BUDGET,
            "Cache budget changed")
    require(type(snapshot["enabled"]) is bool, "Invalid cache enable state")
    tables = snapshot["retired_tables"] if final else snapshot["live_tables"]
    require(len(tables) == 2 and {t["cache_id"] for t in tables} == {0,1}, "Exactly two cache identities required")
    require(len({t["table"] for t in tables}) == 2, "Duplicate cache table identity")
    require(not snapshot["live_tables"] if final else not snapshot["retired_tables"], "Unexpected cache retirement")
    require(snapshot["status"] == ("closed" if final else "active"), "Incorrect cache lifecycle status")
    retained = 0
    for table in tables:
        require(table["budget"] == TABLE_BUDGET, "Per-table budget changed")
        retained += integer(table["nbytes"], "retained table bytes")
        require(table["nbytes"] <= TABLE_BUDGET, "Per-table memory bound exceeded")
        require(table["cache_disabled_after_allocation_failure"] is False, "Cache disabled after allocation failure")
        require(table["closed"] is final and table["resident"] is False, "Unexpected table storage lifecycle")
        counts = counter(table["counts"], "table counts")
        for key in ("read_failures", "cache_allocation_failures"):
            require(counts[key] == 0, "Cache/native read failure")
        for key in ("unique_hits", "unique_queries", "admitted"):
            integer(table[key], key)
        require(table["unique_hits"] <= table["unique_queries"], "Cache hits exceed queries")
        require(table["unique_queries"] == counts["unique_rows"], "Query counter disagrees with unique rows")
        require(counts["miss_rows"] == table["unique_queries"] - table["unique_hits"], "Hits/misses do not reconcile")
        require(counts["requested_rows"] == counts["unique_rows"] + counts["duplicate_rows_elided"], "Duplicate count inconsistent")
        require(counts["calls"] == counts["cache_calls"] + counts["bypass_calls"], "Read-call accounting inconsistent")
        require(counts["native_read_calls"] == counts["calls"] - counts["all_hit_calls"], "Native read-call accounting inconsistent")
        require(counts["native_selected_bytes"] == counts["native_requested_rows"] * ROW_BYTES,
                "Native selected bytes do not match pinned160-byte rows")
    require(snapshot["retained_bytes"] == retained <= TOTAL_BUDGET, "Total retained memory inconsistent")
    if final:
        require(retained == 0 and snapshot["enabled"] is False, "Final cache was not released")
    return {t["cache_id"]: t for t in tables}


def row_delta(before, after, enabled):
    a, b = validate_rows(before), validate_rows(after)
    require(after["enabled"] is enabled, "Cache mode does not match arm")
    total = Counter()
    for i in (0,1):
        require(a[i]["table"] == b[i]["table"], "Cache table identity changed")
        left, right = Counter(a[i]["counts"]), Counter(b[i]["counts"])
        delta = {k:right[k]-left[k] for k in left.keys()|right.keys()}
        require(all(v >= 0 for v in delta.values()), "Cache counters regressed")
        for key in ("unique_hits", "unique_queries", "admitted"):
            delta[key] = b[i][key] - a[i][key]
            require(delta[key] >= 0, "Cache lifetime history reset")
        require(b[i]["nbytes"] >= a[i]["nbytes"], "Cache allocation disappeared between requests")
        require(delta.get("calls",0) > 0, "Missing native row activity")
        if enabled:
            require(delta["unique_queries"] > 0 and delta.get("cache_calls",0) > 0 and delta.get("bypass_calls",0) == 0,
                    "Candidate silently bypassed its cache")
            require(delta.get("native_requested_rows",0) == delta.get("miss_rows",0), "Candidate native I/O differs from misses")
        else:
            require(delta["unique_queries"] == delta["unique_hits"] == delta["admitted"] == 0 and
                    delta.get("cache_calls",0) == 0 and delta.get("bypass_calls",0) == delta["calls"], "Baseline changed candidate cache history")
            require(a[i]["nbytes"] == b[i]["nbytes"], "Baseline changed candidate retained allocation")
        total.update(delta)
    return dict(total) | {"retained_bytes_before":before["retained_bytes"],
                          "retained_bytes_after":after["retained_bytes"],
                          "unique_row_hit_fraction":total["unique_hits"]/total["unique_queries"] if total["unique_queries"] else None}


def validate_cone(after, prompt, enabled):
    require(after["enabled"] is enabled and after["closed"] is False and after["owned_instances"] > 0,
            "Cone controller lifecycle/mode mismatch")
    counts = counter(after["counts"], "cone controller")
    cone = counter(after["cone_counts"], "cone counts")
    shape = counter(after["shape_counts"], "shape counts")
    prime = counter(after["prime_counts"], "prime counts")
    require(counts["model_calls"] > 0 and counts["call_failures"] == counts["prefill_trace_evictions"] == 0,
            "Failed/incomplete cone invocation traces")
    lengths = chunks(prompt)
    eligible = [n for n in lengths if n >= 2668] if enabled else []
    require(cone["cone_calls"] == prime["padded_appends"] == prime["eligible_prefills"] == len(eligible) and
            prime["wkv_pads"] == 3 * len(eligible), "Actual cone/prime activation count differs")
    require(prime["append_failures"] == prime["prefill_failures"] == 0, "DSpark priming failure")
    require(len(after["prefill_trace"]) == len(lengths), "Missing or extra cache-only chunk traces")
    start = 0
    for rows, event in zip(lengths, after["prefill_trace"]):
        if not enabled:
            require(event == {"cone":[],"shape":[],"prime":[]}, "Disabled arm ran candidate arithmetic")
            start += rows
            continue
        require(len(event["cone"]) == len(event["prime"]) == 1, "Chunk lacks cone/prime frame")
        c, p = event["cone"][0], event["prime"][0]
        require(c["input_tokens"] == rows and p["native_rows"] == rows and p["complete"] is True,
                "Chunk geometry or completion differs")
        if rows >= 2668:
            end = start + rows
            require(c["mode"] == "cone" and c["reason"] is None and c["start"] == start and c["end"] == end and
                    c["chunk_tokens"] == rows and c["next_layer"] == 40, "Invalid cone absolute frame")
            require(len(c["layers"]) == 20, "Incomplete decoder cone traversal")
            input_rows = rows
            for layer, block in zip(range(20,40), c["layers"]):
                tail = 128 + (40-layer)*127
                require(block == {"layer":layer,"input_rows":input_rows,"query_rows":tail,
                                  "input_start":end-input_rows,"query_start":end-tail,"endpoint":end},
                        "Cone layer geometry differs")
                input_rows = tail
            require(p["reason"] is None and p["start"] == start and p["end"] == end and
                    p["append_start"] == end-128 and p["append_count"] == 1 and p["append_complete"] is True and
                    not p.get("append_error") and not p.get("error"), "DSpark frame or endpoint differs")
            require(p["stages"] == [{"stage":i,"input_rows":128,"native_rows":rows,"output_rows":128} for i in range(3)],
                    "Exactly three ordered original-shape DSpark projections required")
            kinds = Counter(s["kind"] for s in event["shape"])
            require(kinds["index-pad"] == 5 and kinds["gate-pad"] == 20 and kinds["final-moe-256"] == 1,
                    "Backbone shape stabilization path incomplete")
            require(all(s["native_rows"] == rows for s in event["shape"]), "Shape stabilization lost original chunk rows")
        else:
            require(c["mode"] == "full" and c["reason"] == "short-chunk" and p["reason"] == "short-chunk" and
                    not p["stages"] and p["append_count"] == 0, "Short chunk did not take full-native fallback")
        start += rows
    if not enabled:
        require(not cone and not shape and not prime and counts["disabled_calls"] == counts["model_calls"],
                "Disabled arm altered cone helper state")
    return len(eligible)


def functional_events(row):
    return [{k:v for k,v in item.items() if k != "cycle_ms"} for item in row["events"]]


def functional_stats(row):
    return [{k:v for k,v in item.items() if not k.endswith("_ms")} for item in row["mtp_stats"]]


def check_pair(a,b):
    return {
        "same_input":a["input_tokens_sha256"] == b["input_tokens_sha256"],
        "same_seed":a["seed"] == b["seed"],
        "all_output_ids_identical":a["token_ids"] == b["token_ids"],
        "decoded_text_identical":a["text"] == b["text"],
        "finish_reason_identical":a["finish_reason"] == b["finish_reason"],
        "functional_mtp_cycles_identical":functional_events(a) == functional_events(b),
        "functional_mtp_totals_identical":functional_stats(a) == functional_stats(b),
        "full_requested_length":a["generated_tokens"] == b["generated_tokens"] == 128,
        "no_early_eos":all(r["fixed_length_valid"] is True and r["first_eos_position"] in (None,127) for r in (a,b)),
        "intended_paths_exercised":a["qualification"]["activation"]["passed"] is True and b["qualification"]["activation"]["passed"] is True,
    }


def validate_records(records, plan, profile, qualification):
    validate_plan(plan, profile)
    require([r.get("type") for r in records] == ["manifest","load"] + ["run"]*26,
            "Incomplete, extra or unordered raw records")
    meta, load, *runs = records
    require(meta["plan"] == plan and meta["packages"] == profile["packages"], "Raw plan/package pins differ")
    expected_runtime = {k.removeprefix("_runtime/omlx/"):v for k,v in profile["inference_files_sha256"].items() if k.startswith("_runtime/omlx/")}
    expected_experiment = {k.removeprefix("flash1017/"):v for k,v in profile["inference_files_sha256"].items() if k.startswith("flash1017/") and k.endswith(".py")}
    require(meta["runtime_files_sha256"] == expected_runtime, "Raw runtime source pins differ")
    recorded_experiment = meta["experiment_files_sha256"]
    require(all(recorded_experiment.get(k) == v for k,v in expected_experiment.items()),
            "A qualified bench/inference source pin differs")
    require(all(isinstance(k,str) and Path(k).name == k and k.endswith(".py") and
                isinstance(v,str) and len(v)==64 and all(c in "0123456789abcdef" for c in v)
                for k,v in recorded_experiment.items()), "Malformed recorded Python source metadata")
    q = qualification
    require(q["status"] == "complete_validated" and q["mode"] == "performance" and q["model_cleanup_completed"] is True and
            q["instance_hooks_restored"] is True and q["source_hashes_reverified"] is True and not q["mismatch_pending"],
            "Qualification is incomplete, mismatched or unclean")
    require(len(q["pairs"]) == 13, "Qualification has incomplete pair results")
    require(len(q["worker_threads"]) == 2 and len(set(q["worker_threads"] + [q["loading_thread"]])) == 3,
            "Worker migration was not exercised")
    require(q["completed_at_unix"] >= q["started_at_unix"], "Invalid run completion timestamps")
    previous = q["installation"]["rows"]
    validate_rows(previous)
    require(previous["retained_bytes"] == 0 and all(not t["counts"] for t in previous["live_tables"]),
            "Initial cache was not empty")
    deltas, pairs = [], []
    source_seen = set()
    for i,(row,job) in enumerate(zip(runs,plan["jobs"])):
        p = plan["prompts"][job["prompt"]]
        arm = job["profile"]
        require(row["profile"] == plan["profiles"][arm] | {"name":arm}, "Runtime profile mismatch")
        require((row["prompt_id"],row["repeat"],row["seed"],row["warmup"],row["input_tokens"],row["input_tokens_sha256"])
                == (job["prompt"],job.get("repeat",0),job["seed"],bool(job.get("warmup",False)),p["prompt_tokens"],p["input_tokens_sha256"]),
                "Raw request differs from ordered frozen job")
        require(row["generated_tokens"] == len(row["token_ids"]) == len(row["arrival_seconds"]) == 128 and
                all(type(t) is int and t >= 0 for t in row["token_ids"]), "Output count/IDs differ")
        require(row["finish_reason"] == "length" and row["fixed_length_valid"] is True and row["first_eos_position"] in (None,127),
                "Early EOS or truncated generation")
        require(row["events"] and row["mtp_stats"], "Missing functional speculation evidence")
        require(sum(s["cycles"] for s in row["mtp_stats"]) == len(row["events"]), "MTP cycle count inconsistent")
        require(sum(s.get(k,0) for s in row["mtp_stats"] for k in ("init_emits","draft_emits","bonus_emits","verify_emits")) == 128,
                "MTP emission accounting disagrees with128 outputs")
        for key in ("prefill_seconds","ttft_seconds","total_seconds","decode_tokens_per_second"):
            number(row[key],key,positive=True)
        require(row["prefill_seconds"] <= row["ttft_seconds"] <= row["total_seconds"], "Timing order inconsistent")
        arrivals = row["arrival_seconds"]
        require(all(number(t,"arrival") >= 0 for t in arrivals) and arrivals == sorted(arrivals) and
                arrivals[0] == row["ttft_seconds"] and arrivals[-1] <= row["total_seconds"], "Token arrival sequence inconsistent")
        require(math.isclose(row["decode_tokens_per_second"],127/(arrivals[-1]-arrivals[0]),rel_tol=1e-8), "Decode telemetry inconsistent")
        for key in ("process_footprint_peak_bytes","system_available_min_bytes","swap_start_bytes","swap_peak_bytes",
                    "swap_end_bytes","disk_read_bytes_delta","disk_write_bytes_delta","mlx_peak_bytes"):
            number(row["resources"][key],key)
        r = row["resources"]
        require(r["swap_peak_bytes"] >= max(r["swap_start_bytes"],r["swap_end_bytes"]), "Swap peaks inconsistent")
        require(row["cache_state"].startswith("fresh KV"), "Request KV was not fresh")
        info = row["qualification"]
        expected_route = "VLMModelAdapter" if (i//2)%2 == 0 else "LanguageModel._omlx_prefill"
        require((info["job_index"],info["pair_index"],info["worker_thread"],info["prefill_route"],info["requested_output_tokens"])
                == (i,i//2,q["worker_threads"][(i//2)%2],expected_route,128), "Worker/job/adapter-route mismatch")
        before, after = row["persistent_adapter"]["before"], row["persistent_adapter"]["after"]
        require(before["rows"] == previous, "Persistent row-cache history was reset or lost between jobs")
        delta = row_delta(before["rows"],after["rows"],arm == "fastpath64")
        expected = validate_cone(after["cone"],p,arm == "fastpath64")
        activation = {"expected_cone_chunks":expected,"actual_cone_chunks":expected,
                      "actual_padded_appends":expected,"actual_stage_pads":expected*3,
                      "row_cache_unique_queries":delta["unique_queries"],"retained_cache_bytes":after["rows"]["retained_bytes"],
                      "prefill_trace_count":len(chunks(p)),"prefill_trace_evictions":0,"complete_chunk_traces":True,"passed":True}
        require(info["activation"] == activation, "Reported activation disagrees with raw traces/counters")
        deltas.append(delta)
        previous = after["rows"]
        if i%2:
            left,right = runs[i-1:i+1]
            a,b = (left,right) if left["profile"]["name"] == "native64" else (right,left)
            checks = check_pair(a,b)
            require(all(checks.values()), "Pair output or functional speculation diverged: " + job["prompt"])
            recorded = q["pairs"][i//2]
            require(recorded == {"checks":checks,"passed":True,"compared_output_tokens":128,
                                 "compared_mtp_cycles":len(a["events"]),"prompt_id":job["prompt"],
                                 "warmup":bool(job.get("warmup",False)),"pair_index":i//2},
                    "Qualification pair summary disagrees with raw records")
            candidate_index = i-1 if left["profile"]["name"] == "fastpath64" else i
            source = p.get("source","authored-warmup")
            pair = {"pair_index":i//2,"prompt_id":p["id"],"source":source,"input_tokens":p["prompt_tokens"],
                    "order":"AB" if left["profile"]["name"] == "native64" else "BA",
                    "warmup":bool(job.get("warmup",False)),"candidate_history_phase":"empty-cache warmup" if i==1 else
                    ("warmup-primed first measured request" if i==3 else "persistent reuse after previous measured requests"),
                    "first_measured_request_for_source":source not in source_seen,
                    "native64":a,"fastpath64":b,"cache_delta":deltas[candidate_index]}
            if i>1:
                source_seen.add(source)
            pairs.append(pair)
    final = validate_rows(q["row_cache_after_model_close"],final=True)
    last = {t["cache_id"]:t for t in previous["live_tables"]}
    for i in (0,1):
        require({k:v for k,v in final[i].items() if k not in ("closed","nbytes")} ==
                {k:v for k,v in last[i].items() if k not in ("closed","nbytes")}, "Final closure lost/reset cache history")
    return pairs


def gm(values):
    return math.exp(sum(math.log(x) for x in values)/len(values))


def summarize(pairs):
    result = {"pairs":len(pairs),"output_tokens_compared":128*len(pairs),
              "mtp_cycles_compared":sum(len(p["native64"]["events"]) for p in pairs)}
    for metric in ("prefill_seconds","ttft_seconds","total_seconds"):
        a,b = ([p[arm][metric] for p in pairs] for arm in ARMS)
        ratio = gm([x/y for x,y in zip(a,b)])
        result[metric] = {"native64_geomean":gm(a),"fastpath64_geomean":gm(b),"paired_speed_ratio_geomean":ratio,
                          "latency_reduction_percent":100*(1-1/ratio),"candidate_faster_pairs":sum(y<x for x,y in zip(a,b))}
    result["measured_decode_tokens_per_second"] = {arm:gm([p[arm]["decode_tokens_per_second"] for p in pairs]) for arm in ARMS}
    result["decode_scope"] = "Observed arrival-based throughput, not an inferred decode benefit from prefill optimization."
    result["resources"] = {arm:{"disk_read_bytes_sum":sum(p[arm]["resources"]["disk_read_bytes_delta"] for p in pairs),
        "disk_write_bytes_sum":sum(p[arm]["resources"]["disk_write_bytes_delta"] for p in pairs),
        "process_footprint_peak_bytes":max(p[arm]["resources"]["process_footprint_peak_bytes"] for p in pairs),
        "system_available_min_bytes":min(p[arm]["resources"]["system_available_min_bytes"] for p in pairs),
        "swap_peak_bytes":max(p[arm]["resources"]["swap_peak_bytes"] for p in pairs),
        "max_per_request_swap_growth_bytes":max(p[arm]["resources"]["swap_end_bytes"]-p[arm]["resources"]["swap_start_bytes"] for p in pairs)} for arm in ARMS}
    totals = Counter()
    for p in pairs:
        totals.update({k:v for k,v in p["cache_delta"].items() if type(v) is int and not k.startswith("retained_")})
    result["candidate_row_cache"] = dict(totals) | {"weighted_unique_hit_fraction":totals["unique_hits"]/totals["unique_queries"]}
    return result


def build_report(pairs):
    warmup, measured = pairs[:1],pairs[1:]
    out = {"status":"accepted_complete_token_and_functional_trace_identical",
           "measured":summarize(measured),"warmup_separate":summarize(warmup),
           "all_measured_orders":dict(Counter(p["order"] for p in measured)),"ordered_pairs":[],
           "by_input_tokens":{},"by_source":{},"by_pair_order":{},"history_strata":{},
           "limits":["One adjacent pair per source/length; no confidence interval or independent cold-replicate claim.",
                     "Lengths of one source share its prefix and are correlated; each source appears in three requests.",
                     "Candidate cache survives baseline bypass. Both arms share warmed OS/file caches; no OS cache purge.",
                     "The first measured candidate is already warmup-primed. Later reuse combines new-source and repeated-source effects.",
                     "512MiB bounds retained row arrays only; prefetch, gather temporaries, model/GPU memory are additional.",
                     "Baseline after first allocation carries the candidate's retained cache memory; peak RSS is not an independent memory-cost estimate.",
                     "The one131K pair is exploratory capacity evidence, not a long-context benchmark or a1M result.",
                     "Combined gains are measured directly; separate prefetch/cone/cache gains must not be multiplied."]}
    for key,target in (("input_tokens","by_input_tokens"),("source","by_source"),("order","by_pair_order")):
        for value in sorted({p[key] for p in measured}):
            out[target][str(value)] = summarize([p for p in measured if p[key]==value])
    out["history_strata"]["warmup_primed_first_measured"] = summarize(measured[:1])
    out["history_strata"]["later_persistent_reuse"] = summarize(measured[1:])
    for p in pairs:
        item = {k:v for k,v in p.items() if k not in ARMS}
        item["timings"] = {arm:{key:p[arm][key] for key in ("prefill_seconds","ttft_seconds","total_seconds","decode_tokens_per_second")} for arm in ARMS}
        item["paired_prefill_speed_ratio"] = p["native64"]["prefill_seconds"]/p["fastpath64"]["prefill_seconds"]
        out["ordered_pairs"].append(item)
    return out


def archived_evidence(raw_path, plan_path, manifest_path):
    raw_path, plan_path, manifest_path = map(lambda p:Path(p).resolve(),(raw_path,plan_path,manifest_path))
    folder = raw_path.with_suffix(".sources")
    index_path = folder/"manifest.json"
    index = read_json(index_path)
    archived = {}
    for original,item in index.items():
        name = item["snapshot"]
        require(Path(name).name == name, "Snapshot traversal path refused")
        path = folder/name
        require(sha(path) == item["sha256"], "Archived source hash mismatch: " + original)
        archived[original] = path
    require(str(plan_path) in archived and str(manifest_path) in archived, "Plan/execution manifest missing from snapshot")
    plan,manifest = read_json(archived[str(plan_path)]),read_json(archived[str(manifest_path)])
    require(set(index) == set(manifest["sources_sha256"]) | {str(manifest_path)}, "Snapshot source set differs from frozen manifest")
    for original,expected in manifest["sources_sha256"].items():
        require(index[original]["sha256"] == expected, "Snapshot and execution source pins disagree")
    profile_paths = [p for k,p in archived.items() if k.endswith("/qualified-profile.json")]
    checkpoint_paths = [p for k,p in archived.items() if k.endswith("/results/checkpoint-hashes.json")]
    require(len(profile_paths) == len(checkpoint_paths) == 1, "Need one archived qualified profile and checkpoint manifest")
    profile,checkpoint = read_json(profile_paths[0]),read_json(checkpoint_paths[0])
    baseline = manifest["baseline"]
    require(sha(profile_paths[0]) == baseline["profile_sha256"] and sha(checkpoint_paths[0]) == baseline["checkpoint_manifest_sha256"],
            "Archived baseline/checkpoint hashes differ")
    require(profile["packages"] == baseline["packages"] and len(profile["inference_files_sha256"]) == baseline["inference_files_verified"],
            "Baseline package/source count mismatch")
    require(checkpoint.get("repo") == "mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp" and
            checkpoint.get("revision") == "78456115d301eecf9fadcdcc81c0cf6f8b2c8eab", "Different checkpoint revision")
    require(checkpoint["status"] == "complete_hash_verified" and not checkpoint["missing"] and
            len(checkpoint["files"]) == baseline["checkpoint_fingerprints_verified"] and
            all(v["verified"] is True and len(v["fingerprint"]) == 3 for v in checkpoint["files"].values()),
            "Checkpoint baseline incomplete")
    fixture = str(Path(plan["fixture_manifest"]).resolve())
    require(fixture in archived and sha(archived[fixture]) == plan["fixture_manifest_sha256"], "Fixture manifest pin mismatch")
    fixtures = read_json(archived[fixture])
    require(fixtures["baseline_runtime"] == profile["runtime"] and fixtures["source_manifest_sha256"] == plan["source_manifest_sha256"],
            "Frozen fixture provenance disagrees")
    require(plan["source_manifest_sha256"] in [v["sha256"] for v in index.values()], "Source corpus manifest is not archived")
    sidecar = read_json(raw_path.with_suffix(".qualification.json"))
    require(sidecar["plan_sha256"] == sha(archived[str(plan_path)]) and
            sidecar["manifest_sha256"] == sha(archived[str(manifest_path)]) and sidecar["raw_output_sha256"] == sha(raw_path),
            "Raw/sidecar/plan/source hashes disagree")
    records = [loads(line) for line in raw_path.read_text().splitlines()]
    require(records[0]["model"] == checkpoint["directory"], "Run used a different checkpoint directory")
    completed = sidecar["completed_request_files_sha256"]
    expected_names = {f"job{i:02d}.json" for i in range(26)}
    require({Path(p).name for p in completed} == expected_names and len(completed) == 26, "Completed request backups missing/extra")
    for i,row in enumerate(records[2:]):
        matches = [p for p in completed if Path(p).name == f"job{i:02d}.json"]
        require(len(matches)==1, "Missing completed request backup")
        path = Path(matches[0])
        require(path.parent.resolve() == raw_path.with_suffix(".completed") and sha(path) == completed[str(path)], "Completed request backup pin differs")
        original = read_json(path)
        expected = {k:v for k,v in row.items() if k not in ("persistent_adapter","warmup")}
        expected["qualification"] = {k:v for k,v in expected["qualification"].items() if k != "activation"}
        require(original == expected, "Completed request backup differs from enriched raw output")
    return records,plan,profile,sidecar,{"raw_sha256":sha(raw_path),"plan_sha256":sidecar["plan_sha256"],
        "execution_manifest_sha256":sidecar["manifest_sha256"],"run_source_snapshot_manifest_sha256":sha(index_path),
        "source_files_verified":len(index),
        "qualified_experiment_python_pins_verified":sum(k.startswith("flash1017/") and k.endswith(".py") for k in profile["inference_files_sha256"]),
        "additional_recorded_python_files":sorted(set(records[0]["experiment_files_sha256"]) -
            {k.removeprefix("flash1017/") for k in profile["inference_files_sha256"] if k.startswith("flash1017/") and k.endswith(".py")}),
        "additional_python_scope":"Additional bench-reported file hashes are metadata, not qualified inference pins.",
        "checkpoint_validation":"Archived full-hash verification manifest; no current weight file reads.",
        "analyzer":str(Path(__file__).resolve()),"analyzer_sha256":sha(__file__)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True,help="Original frozen plan identity; contents read from run snapshot")
    parser.add_argument("--manifest",type=Path,required=True,help="Original execution-manifest identity; contents read from run snapshot")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),"Refuse to replace analysis evidence")
    try:
        records,plan,profile,qualification,pins=archived_evidence(args.raw,args.plan,args.manifest)
        pairs=validate_records(records,plan,profile,qualification)
        report=build_report(pairs)|{"provenance":pins,"analyzed_at_unix":time.time(),"raw":str(args.raw.resolve())}
    except Exception as exc:
        report={"status":"rejected_no_speed_claim","error":str(exc),"exception_type":type(exc).__name__,
                "raw":str(args.raw.resolve()),"analyzer":str(Path(__file__).resolve()),"analyzer_sha256":sha(__file__)}
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"measured":report["measured"]["prefill_seconds"]},indent=2))


if __name__=="__main__":
    main()
