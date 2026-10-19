"""Independent CPU/NumPy audit of the frozen serving-v3 boundary experiment.

Imports no inference code, and reads no model weights. Recomputes archive and
decompressed array hashes, compares all saved bytes directly, and reconciles
ordered requests against the frozen plan. A partial/failed run never passes.
Boundary timings contain instrumentation and are deliberately not summarized.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "work/serving-qualification-v3/plans/boundary-01.json"
EXECUTION = ROOT / "work/serving-integration-v3/execution-manifest-01.json"
PLAN_SHA = "d4095452c17564db3d57c106e041ef042a16355ecaa1243687201999c8047dcf"
EXECUTION_SHA = "2c8ebc4ca96ae514e9d237aac7cb1c5a6555b46f5d4d6ecb6c9234ba7045bcad"
ARMS = {"native64", "fastpath64"}
PREFIXES = {2667,2668,2669,8190,8191,8192,8193,10859,10860,10861,16383,16384}
SLOTS = ("offset", "local_kv", "global_kv", "index_k", "partial_kv", "partial_gates", "engram_history")
COMPRESSORS = {2:2,8:2,14:2,20:1}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def byte_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def loads(text):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(text, parse_constant=invalid)


def read(path):
    return loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def validate_lab_pins(recorded, profile):
    expected={k.removeprefix("flash1017/"):v for k,v in profile["inference_files_sha256"].items() if k.startswith("flash1017/") and k.endswith(".py")}
    # Bench records every lab Python file, including unrelated reporting tools;
    # the qualified manifest intentionally pins only the inference subset.
    require(all(recorded.get(k) == v for k,v in expected.items()), "Raw lab inference source pins differ")
    require(all(isinstance(k,str) and isinstance(v,str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for k,v in recorded.items()), "Invalid recorded lab source digest")


def source_evidence(raw, q):
    folder = raw.with_suffix(".sources")
    index = read(folder / "manifest.json")
    archived = {}
    for original, item in index.items():
        require(Path(item["snapshot"]).name == item["snapshot"], "Unsafe source snapshot name")
        path = folder / item["snapshot"]
        require(sha(path) == item["sha256"], "Snapshot byte hash differs: " + original)
        archived[original] = path
    require(index[str(PLAN)]["sha256"] == q["plan_sha256"] == PLAN_SHA, "Different frozen boundary plan")
    require(index[str(EXECUTION)]["sha256"] == q["manifest_sha256"] == EXECUTION_SHA, "Different frozen execution")
    execution, plan = read(archived[str(EXECUTION)]), read(archived[str(PLAN)])
    require(set(index) == set(execution["sources_sha256"]) | {str(EXECUTION)}, "Archived source set differs")
    for original, digest in execution["sources_sha256"].items():
        require(index[original]["sha256"] == digest, "Execution source pin differs")
    profile_path = archived[str(ROOT / "outputs/flash1017-lab/qualified-profile.json")]
    checkpoint_path = archived[str(ROOT / "outputs/flash1017-lab/results/checkpoint-hashes.json")]
    profile, checkpoint = read(profile_path), read(checkpoint_path)
    baseline = execution["baseline"]
    require(sha(profile_path) == baseline["profile_sha256"] and sha(checkpoint_path) == baseline["checkpoint_manifest_sha256"], "Baseline source pins differ")
    require(checkpoint["repo"] == "mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp" and checkpoint["revision"] == "78456115d301eecf9fadcdcc81c0cf6f8b2c8eab", "Checkpoint identity differs")
    require(checkpoint["status"] == "complete_hash_verified" and not checkpoint["missing"] and all(x["verified"] is True for x in checkpoint["files"].values()), "Checkpoint provenance incomplete")
    require(len(checkpoint["files"]) == baseline["checkpoint_fingerprints_verified"], "Checkpoint fingerprint count differs")
    fixture = read(archived[plan["fixture_manifest"]])
    require(sha(archived[plan["fixture_manifest"]]) == plan["fixture_manifest_sha256"], "Fixture manifest differs")
    require(fixture["source_manifest_sha256"] == plan["source_manifest_sha256"] and plan["source_manifest_sha256"] in {x["sha256"] for x in index.values()}, "Corpus provenance differs")
    require(profile["qualified"] is True and profile["runtime"]["engram_prefetch_bytes"] == 64*2**20 and profile["runtime"]["ced_prefill"] is False, "Expected qualified native64")
    require(plan["design"]["suite"] == "boundary" and plan["profiles"] == {arm:profile["runtime"] for arm in ARMS}, "Wrong suite/profile")
    require(len(plan["jobs"]) == 26 and plan["design"]["generation_tokens"] == 32, "Wrong boundary suite size")
    return plan, profile, checkpoint, {"source_snapshot_index_sha256":sha(folder/"manifest.json"), "source_files_verified":len(index), "execution_manifest_sha256":EXECUTION_SHA, "plan_sha256":PLAN_SHA}


def expected_keys():
    return {f"layer{i:02d}_{slot}" for i in range(40) for slot in ([f"slot{j}" for j in range(7)] + ["left_padding", "lengths"])} | {f"prime{i}_physical_keys" for i in range(3)}


def validate_archive(boundary, prefix, expected_path):
    archive = boundary["archive"]
    path = Path(archive["path"])
    require(path.resolve() == expected_path.resolve(), "Unexpected array archive path")
    require(path.stat().st_size == archive["bytes"] and sha(path) == archive["sha256"], "Array archive hash/size differs")
    require(boundary["layers"] == [{"layer":i,"offset":prefix,"meta_state":["deepseek_v41","3",str(COMPRESSORS.get(i,0))]} for i in range(40)], "Target cache layer/offset metadata differs")
    require(boundary["prime"] == {"present":True,"rings":[{"stage":i,"offset":prefix,"max_size":128} for i in range(3)],"expected_target_offset":prefix}, "DSpark ring metadata differs")
    metadata = boundary["arrays"]
    require(set(metadata) == expected_keys(), "Missing/extra target or draft state slot")
    values = {}
    with np.load(path, allow_pickle=False) as saved:
        keys = {k for k,v in metadata.items() if v["none"] is False} | {"probe_logits"}
        require(set(saved.files) == keys and len(saved.files) == len(keys), "NPZ entries differ from declared slots")
        for key, meta in metadata.items():
            if meta["none"] is True:
                require(meta == {"none":True}, "Unexpected None-slot metadata")
                values[key] = None
                continue
            require(meta["none"] is False, "Invalid None-slot flag")
            array = saved[key]
            require(array.dtype == np.uint8 and array.ndim == 1, "Saved state must be a raw uint8 vector")
            require(all(type(d) is int and d >= 0 for d in meta["shape"]), "Invalid state dimensions")
            width = 2 if meta["dtype"] == "bfloat16" else np.dtype(meta["dtype"]).itemsize
            require(math.prod(meta["shape"]) == meta["elements"] and meta["elements"]*width == meta["bytes"] == array.size, "State dimensions/byte counts differ")
            require(byte_sha(array) == meta["sha256"], "Decompressed raw state hash differs: " + key)
            if key.endswith("_slot0"):
                require(meta["shape"] == [1] and meta["dtype"] == "int32" and int(np.frombuffer(array.tobytes(),dtype=np.int32)[0]) == prefix, "Raw target offset differs")
            if key.startswith("prime"):
                require(meta["dtype"] == "bfloat16" and meta["shape"] == [1,1,128,512], "Physical draft ring layout differs")
            values[key] = array.copy()
        logits = saved["probe_logits"]
        require(logits.dtype == np.float32 and logits.shape == (1,1,129280) and np.isfinite(logits).all(), "Invalid full-vocabulary probe")
        values["probe_logits"] = logits.copy()
    return values


def compare_values(a, b, metadata_a, metadata_b):
    differences = []
    compared_bytes = 0
    for key in sorted(expected_keys()):
        x,y = a[key],b[key]
        if metadata_a[key] != metadata_b[key]:
            differences.append({"slot":key,"kind":"metadata"})
        if x is None or y is None:
            if x is not None or y is not None:
                differences.append({"slot":key,"kind":"presence"})
        elif x.shape != y.shape:
            differences.append({"slot":key,"kind":"byte_shape","baseline":list(x.shape),"candidate":list(y.shape)})
        else:
            compared_bytes += x.nbytes
            n = int(np.count_nonzero(x != y))
            if n:
                differences.append({"slot":key,"kind":"bytes","different_bytes":n})
    x,y = a["probe_logits"], b["probe_logits"]
    logit_equal = x.tobytes() == y.tobytes()
    return {"state_differences":differences,"state_bytes_compared":compared_bytes,
            "target_slots_compared":360,"physical_draft_rings_compared":3,
            "logits_compared":x.size,"logits_byte_identical":logit_equal,
            "logit_max_absolute_difference":float(np.max(np.abs(x-y))),
            "logit_different_elements":int(np.count_nonzero(x != y)),
            "logit_baseline_sha256":byte_sha(x),"logit_candidate_sha256":byte_sha(y),
            "passed":not differences and logit_equal}


def validate_trace(row, prompt):
    enabled = row["profile"]["name"] == "fastpath64"
    cone = row["persistent_adapter"]["after"]["cone"]
    require(cone["enabled"] is enabled and cone["closed"] is False, "Wrong adapter lifecycle/mode")
    counts, pc, cc = map(Counter, (cone["counts"],cone["prime_counts"],cone["cone_counts"]))
    require(counts["call_failures"] == counts["prefill_trace_evictions"] == pc["append_failures"] == pc["prefill_failures"] == 0, "Failed or incomplete adapter traces")
    prefix = len(prompt["tokens"]) - 1
    chunks = [min(8192,prefix-start) for start in range(0,prefix,8192)]
    require(len(cone["prefill_trace"]) == len(chunks), "Missing chunk trace")
    eligible = sum(n >= 2668 for n in chunks) if enabled else 0
    require(cc["cone_calls"] == pc["padded_appends"] == pc["eligible_prefills"] == eligible and pc["wkv_pads"] == 3*eligible, "Missing actual cone or priming activation")
    start = 0
    for rows, trace in zip(chunks,cone["prefill_trace"]):
        end = start + rows
        if not enabled:
            require(trace == {"cone":[],"shape":[],"prime":[]}, "Baseline ran candidate path")
        else:
            require(len(trace["cone"]) == len(trace["prime"]) == 1, "Incomplete prefill frame")
            c,p = trace["cone"][0],trace["prime"][0]
            require(c["input_tokens"] == p["native_rows"] == rows and p["complete"] is True, "Wrong chunk size or unfinished prime")
            if rows < 2668:
                require(c["mode"] == "full" and c["reason"] == p["reason"] == "short-chunk" and p["append_count"] == 0 and p["stages"] == [], "Unsupported short chunk did not fall back")
                require(not trace["shape"], "Short chunk ran shape shim")
            else:
                require(c["mode"] == "cone" and c["reason"] is None and c["start"] == start and c["end"] == end and c["next_layer"] == 40, "Wrong dependency-cone frame")
                blocks=[]
                incoming=rows
                for layer in range(20,40):
                    outgoing=128+(40-layer)*127
                    blocks.append({"layer":layer,"input_rows":incoming,"query_rows":outgoing,"input_start":end-incoming,"query_start":end-outgoing,"endpoint":end})
                    incoming=outgoing
                require(c["layers"] == blocks, "Dependency-cone layer bounds differ")
                require(p["reason"] is None and p["start"] == start and p["end"] == end and p["append_start"] == end-128 and p["append_count"] == 1 and p["append_complete"] is True, "Wrong DSpark prime interval")
                require(p["stages"] == [{"stage":i,"input_rows":128,"native_rows":rows,"output_rows":128} for i in range(3)], "Missing ordered DSpark projections")
                shape=Counter(x["kind"] for x in trace["shape"])
                require(shape["index-pad"] == 5-int(rows == 2668) and shape["gate-pad"] == 20-int(rows == 2668) and shape["final-moe-256"] == 1, "Shape stabilization activation differs")
                require(all(x["native_rows"] == rows for x in trace["shape"]), "Shape shim lost original chunk size")
        start=end
    activation=row["qualification"]["activation"]
    require(activation["complete_chunk_traces"] is True and activation["passed"] is True and activation["prefill_trace_count"] == len(chunks) and activation["prefill_trace_evictions"] == 0, "Reported trace completeness differs")
    require(activation["expected_cone_chunks"] == activation["actual_cone_chunks"] == activation["actual_padded_appends"] == eligible and activation["actual_stage_pads"] == 3*eligible, "Reported activation differs")
    return eligible, len(chunks)-eligible if enabled else 0


def functional(row):
    return ([{k:v for k,v in e.items() if k != "cycle_ms"} for e in row["events"]],
            [{k:v for k,v in s.items() if not k.endswith("_ms")} for s in row["mtp_stats"]])


def validate_rows(before,after,enabled):
    require(after["enabled"] is enabled and after["status"] == "active" and after["retained_bytes"] <= 512*2**20, "Row-cache lifecycle/memory differs")
    require(after["max_retained_bytes"] == 512*2**20 and after["per_table_budget_bytes"] == 256*2**20, "Row-cache budget differs")
    require({t["cache_id"] for t in before["live_tables"]} == {t["cache_id"] for t in after["live_tables"]} == {0,1}, "Row-cache identities differ")
    queries=0
    for left in before["live_tables"]:
        right=next(x for x in after["live_tables"] if x["cache_id"] == left["cache_id"])
        require(left["table"] == right["table"] and right["nbytes"] <= 256*2**20 and not right["closed"] and not right["cache_disabled_after_allocation_failure"], "Invalid row-cache table")
        a,b=Counter(left["counts"]),Counter(right["counts"])
        delta={k:b[k]-a[k] for k in a.keys()|b.keys()}
        require(all(v >= 0 for v in delta.values()), "Row-cache history regressed")
        require(b["native_selected_bytes"] == 160*b["native_requested_rows"] and b["read_failures"] == b["cache_allocation_failures"] == 0, "Native storage byte/failure accounting differs")
        query=right["unique_queries"]-left["unique_queries"]
        hits=right["unique_hits"]-left["unique_hits"]
        require(query >= hits >= 0 and query == delta.get("unique_rows",0), "Row-cache query accounting differs")
        if enabled:
            require(query > 0 and delta.get("native_requested_rows",0) == query-hits == delta.get("miss_rows",0) and delta.get("bypass_calls",0) == 0, "Candidate bypassed miss-only storage")
        else:
            require(query == hits == 0 and left["nbytes"] == right["nbytes"] and delta.get("cache_calls",0) == 0, "Baseline changed cache history")
        queries += query
    require(after["retained_bytes"] == sum(t["nbytes"] for t in after["live_tables"]), "Retained table byte accounting differs")
    return queries


def analyze(raw):
    raw=Path(raw).resolve()
    q=read(raw.with_suffix(".qualification.json"))
    require(q["status"] == "complete_validated", "Run is incomplete/failed: " + str(q.get("status")))
    require(q["mode"] == "boundary" and all(q.get(k) is True for k in ("model_cleanup_completed","instance_hooks_restored","source_hashes_reverified")) and q["mismatch_pending"] is False, "Run cleanup or qualification failed")
    require(q["raw_output_sha256"] == sha(raw), "Final raw byte hash differs")
    plan,profile,checkpoint,pins=source_evidence(raw,q)
    records=[loads(line) for line in raw.read_text().splitlines()]
    require([r.get("type") for r in records] == ["manifest","load"]+["run"]*26, "Incomplete/extra/error raw records")
    meta=records[0]
    require(meta["plan"] == plan and meta["model"] == checkpoint["directory"] and meta["packages"] == profile["packages"], "Raw model/plan/packages differ")
    require(meta["runtime_files_sha256"] == {k.removeprefix("_runtime/omlx/"):v for k,v in profile["inference_files_sha256"].items() if k.startswith("_runtime/omlx/")}, "Raw runtime source pins differ")
    validate_lab_pins(meta["experiment_files_sha256"],profile)
    require(len(q["pairs"]) == 13 and len(q["worker_threads"]) == 2 and len(set(q["worker_threads"]+[q["loading_thread"]])) == 3, "Incomplete pairs or worker migration")
    backups={str((ROOT/Path(p)).resolve()):digest for p,digest in q["completed_request_files_sha256"].items()}
    require(len(backups) == len(q["completed_request_files_sha256"]), "Duplicate normalized backup paths")
    expected_backups={str(raw.with_suffix(".completed")/f"job{i:02d}.json") for i in range(26)}
    require(set(backups) == expected_backups and {str(p) for p in raw.with_suffix(".completed").glob("*.json")} == expected_backups, "Completed worker backups incomplete/extra")
    previous=q["installation"]["rows"]
    require(previous["retained_bytes"] == 0, "Cache not initially empty")
    runs=records[2:]
    pairs=[]
    orders=Counter()
    all_eligible=all_fallback=0
    for i,(row,job) in enumerate(zip(runs,plan["jobs"])):
        p=plan["prompts"][job["prompt"]]
        warmup=i<2
        require(bool(job.get("warmup",False)) is warmup and row["warmup"] is warmup, "Warmup order differs")
        require(identity(p["tokens"]) == p["input_tokens_sha256"] and len(p["tokens"]) == p["prompt_tokens"], "Frozen input hash differs")
        require(row["profile"] == profile["runtime"] | {"name":job["profile"]}, "Runtime profile differs")
        require((row["prompt_id"],row["repeat"],row["seed"],row["input_tokens"],row["input_tokens_sha256"]) == (job["prompt"],job.get("repeat",0),job["seed"],p["prompt_tokens"],p["input_tokens_sha256"]), "Raw ordered request differs")
        require(row["generated_tokens"] == len(row["token_ids"]) == p["max_tokens"] == 32 and all(type(t) is int and t >= 0 for t in row["token_ids"]), "Invalid generated output")
        require(row["finish_reason"] == "length" and row["fixed_length_valid"] is True and row["first_eos_position"] in (None,31), "Early EOS/truncated generation")
        require(row["events"] and row["mtp_stats"] and sum(s["cycles"] for s in row["mtp_stats"]) == len(row["events"]), "Incomplete MTP cycle evidence")
        require(sum(s.get(k,0) for s in row["mtp_stats"] for k in ("init_emits","draft_emits","bonus_emits","verify_emits")) == 32, "MTP emission accounting differs")
        info=row["qualification"]
        require((info["job_index"],info["pair_index"],info["worker_thread"],info["prefill_route"],info["requested_output_tokens"]) == (i,i//2,q["worker_threads"][(i//2)%2],"VLMModelAdapter" if (i//2)%2 == 0 else "LanguageModel._omlx_prefill",32), "Job/worker/route differs")
        backup=raw.with_suffix(".completed")/f"job{i:02d}.json"
        require(sha(backup) == backups[str(backup)], "Worker backup byte hash differs")
        core={k:v for k,v in row.items() if k not in ("persistent_adapter","boundary","warmup")}
        core["qualification"]={k:v for k,v in info.items() if k != "activation"}
        require(read(backup) == core, "Worker completed inference differs from enriched raw record")
        before,after=row["persistent_adapter"]["before"],row["persistent_adapter"]["after"]
        require(before["rows"] == previous, "Cache history lost between requests")
        queries=validate_rows(before["rows"],after["rows"],job["profile"] == "fastpath64")
        require(info["activation"]["row_cache_unique_queries"] == queries and info["activation"]["retained_cache_bytes"] == after["rows"]["retained_bytes"], "Reported cache activation differs")
        previous=after["rows"]
        eligible,fallback=validate_trace(row,p)
        if not warmup:
            all_eligible+=eligible
            all_fallback+=fallback
        if i%2:
            left,right=runs[i-1:i+1]
            require({left["profile"]["name"],right["profile"]["name"]} == ARMS, "Pair missing arm")
            a,b=(left,right) if left["profile"]["name"] == "native64" else (right,left)
            require(a["input_tokens_sha256"] == b["input_tokens_sha256"] and a["seed"] == b["seed"] and a["token_ids"] == b["token_ids"] and a["text"] == b["text"] and functional(a) == functional(b), "Pair output/input/functional MTP diverged")
            recorded=q["pairs"][i//2]
            require(recorded["passed"] is True and all(recorded["checks"].values()) and recorded["compared_output_tokens"] == 32 and recorded["compared_mtp_cycles"] == len(a["events"]) and recorded["pair_index"] == i//2 and recorded["warmup"] is warmup, "Reported pair summary differs")
            if not warmup:
                prefix=len(p["tokens"])-1
                av=validate_archive(a["boundary"],prefix,raw.with_suffix(".arrays")/f"job{a['qualification']['job_index']:02d}-native64.npz")
                bv=validate_archive(b["boundary"],prefix,raw.with_suffix(".arrays")/f"job{b['qualification']['job_index']:02d}-fastpath64.npz")
                cmp=compare_values(av,bv,a["boundary"]["arrays"],b["boundary"]["arrays"])
                cmp.update(pair_index=i//2,prompt_id=p["id"],prefix_tokens=prefix,source=p["source"],order="AB" if left is a else "BA",mtp_cycles_compared=len(a["events"]))
                require(cmp["passed"], "Raw state/logit mismatch: " + json.dumps(cmp))
                state=recorded["state"]
                require(state["issues"] == [] and all(state[k] is True for k in ("encoder_cache_all_bit_identical","global_kv_and_index_cache_all_bit_identical","all_cache_slots_bit_identical","all_prime_rings_bit_identical","structural_and_unchanged_state_checks_passed","all_target_metadata_identical","all_prime_metadata_identical")), "Recorded state summary differs")
                for item in state["cache_slots"]:
                    suffix=f"slot{item['slot']}" if "slot" in item else item["name"]
                    key=f"layer{item['layer']:02d}_{suffix}"
                    require(item["baseline"] == a["boundary"]["arrays"][key] and item["candidate"] == b["boundary"]["arrays"][key] and item["byte_identical"] is True and item["different_bytes"] == 0, "Recorded slot comparison differs")
                require(len(state["cache_slots"]) == 360, "Recorded cache-slot count differs")
                probe=recorded["next_token_full_vocabulary"]
                require(probe["baseline_sha256"] == cmp["logit_baseline_sha256"] and probe["candidate_sha256"] == cmp["logit_candidate_sha256"] and probe["byte_identical"] is True and probe["different_elements"] == 0, "Recorded probe comparison differs")
                pairs.append(cmp)
                orders[cmp["order"]]+=1
    require({p["prefix_tokens"] for p in pairs} == PREFIXES and orders == {"AB":6,"BA":6}, "Boundary coverage/order differs")
    require(len(list(raw.with_suffix(".arrays").glob("*.npz"))) == 24, "Unexpected archive count")
    final=q["row_cache_after_model_close"]
    require(final["status"] == "closed" and final["retained_bytes"] == 0 and final["live_tables"] == [] and final["enabled"] is False and len(final["retired_tables"]) == 2 and all(t["closed"] and t["nbytes"] == 0 for t in final["retired_tables"]), "Retained cache not fully released")
    return {"status":"accepted_independent_raw_state_audit","raw":str(raw),"raw_sha256":sha(raw),"qualification_sha256":sha(raw.with_suffix(".qualification.json")),"provenance":pins,
            "measured_pairs":len(pairs),"measured_output_ids_compared":32*len(pairs),"warmup_output_ids_compared_separately":32,
            "measured_target_slot_comparisons":360*len(pairs),"measured_physical_draft_ring_comparisons":3*len(pairs),
            "measured_full_vocabulary_logit_values_compared":129280*len(pairs),"measured_state_bytes_compared":sum(p["state_bytes_compared"] for p in pairs),
            "measured_mtp_cycles_compared":sum(p["mtp_cycles_compared"] for p in pairs),"candidate_eligible_chunks":all_eligible,"candidate_short_chunk_fallbacks":all_fallback,
            "orders":dict(orders),"pairs":pairs,"limits":["All saved target/DSpark state and full next-token probe bytes match for these12 boundaries; this is finite validation, not a universal arithmetic proof.","Code/prose sources were held out from local tuning, not from model training; no independent benchmark-quality score is implied.","Both prefill entry routes and sequential worker migration are exercised; this is not a concurrent HTTP/cancellation/vision test.","Fresh request KV and retained candidate row-cache history; OS caches were not purged.","Boundary timings include state copies and probe observers and are not performance evidence.","Model weights were not read by this CPU audit; the archived full-hash manifest and execution fingerprint checks provide checkpoint provenance."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(), "Refuse to overwrite independent evidence")
    error=None
    try:
        report=analyze(args.raw)
    except Exception as exc:
        error=exc
        report={"status":"not_accepted","raw":str(args.raw.resolve()),"error":str(exc),"exception_type":type(exc).__name__,"partial_evidence_preserved":True}
    report.update(analyzer=str(Path(__file__).resolve()),analyzer_sha256=sha(__file__),analyzed_at_unix=time.time(),numpy_version=np.__version__,inference_frameworks_imported=False)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("status","error","measured_pairs","measured_state_bytes_compared") if k in report}))
    if error:
        raise error


if __name__ == "__main__":
    main()
