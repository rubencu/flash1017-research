"""Frozen paired qualification of persistent per-model serving adapters.

The existing bench retains model loading, generation, sampling, telemetry and
cleanup. Only the selected candidate owns additional per-instance hooks. Two
single-worker executors alternate complete matched pairs, exercising migration
away from the loading thread without concurrent GPU/model execution.

Boundary mode additionally archives every target-cache/DSpark byte before the
last unprefilled token and all logits for that token. Its timings include these
diagnostics and are not performance measurements. Performance mode has no array
instrumentation. Any output/state/functional-speculation mismatch ends the run
after retaining both arms; speed regressions never stop a performance run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
import sys
import threading
import time
import traceback

from evidence import (HERE, LAB, ROOT, exclusive_model, import_path,
                      require_fresh_output, sha, snapshot, validate_frozen, write_json)


def functional_events(result):
    return [{k: v for k, v in event.items() if k != "cycle_ms"}
            for event in result["events"]]


def functional_stats(result):
    return [{k: v for k, v in stats.items() if not k.endswith("_ms")}
            for stats in result["mtp_stats"]]


def result_comparison(a, b):
    checks = {
        "same_input": a["input_tokens_sha256"] == b["input_tokens_sha256"],
        "same_seed": a["seed"] == b["seed"],
        "all_output_ids_identical": a["token_ids"] == b["token_ids"],
        "decoded_text_identical": a["text"] == b["text"],
        "finish_reason_identical": a["finish_reason"] == b["finish_reason"],
        "functional_mtp_cycles_identical": functional_events(a) == functional_events(b),
        "functional_mtp_totals_identical": functional_stats(a) == functional_stats(b),
        "full_requested_length": a["generated_tokens"] == b["generated_tokens"] == a["qualification"]["requested_output_tokens"],
        "no_early_eos": all(row["fixed_length_valid"] is True and row["first_eos_position"] in
                            (None, row["generated_tokens"]-1) for row in (a,b)),
        "intended_paths_exercised": a["qualification"]["activation"]["passed"]
            and b["qualification"]["activation"]["passed"],
    }
    return {"checks": checks, "passed": all(checks.values()),
            "compared_output_tokens": len(a["token_ids"]),
            "compared_mtp_cycles": len(a["events"])}


def await_drained_future(future):
    """An interrupted waiting thread must never close a running worker's model."""
    try:
        return future.result()
    except BaseException:
        if not future.done():
            print("INTERRUPTED: waiting for the active model worker to drain before cleanup", flush=True)
            while not future.done():
                try:
                    future.result(timeout=0.5)
                except (FutureTimeout, KeyboardInterrupt):
                    continue
                except BaseException:
                    # Worker exceptions mark the future done; preserve the
                    # original interruption as the cause of this shutdown.
                    if not future.done():
                        continue
        raise


def validate_plan(plan, profile):
    if set(plan["profiles"]) != {"native64", "fastpath64"}:
        raise ValueError("Unexpected qualification arms")
    if any(p != profile["runtime"] for p in plan["profiles"].values()):
        raise ValueError("Both arms must retain the qualified runtime configuration")
    jobs = plan["jobs"]
    if len(jobs) % 2:
        raise ValueError("Expected adjacent complete matched pairs")
    for index in range(0, len(jobs), 2):
        a, b = jobs[index:index+2]
        if {a["profile"], b["profile"]} != {"native64", "fastpath64"}:
            raise ValueError("Each pair needs both arms")
        if {k:v for k,v in a.items() if k != "profile"} != {k:v for k,v in b.items() if k != "profile"}:
            raise ValueError("Paired jobs differ beyond arm")
    for name, prompt in plan["prompts"].items():
        ids = prompt.get("tokens")
        if not isinstance(ids, list) or len(ids) < 2 or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("Invalid frozen token IDs: " + name)
        import hashlib
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != prompt["input_tokens_sha256"]:
            raise ValueError("Input token hash mismatch: " + name)


@contextmanager
def boundary_observers(target, adapter, *, via_adapter, prompt, audit, mx, np, capture):
    """Instance-only observers; no arithmetic replacements or global patches."""
    namespace = object.__getattribute__(target, "__dict__")
    saved = {name: (name in namespace, namespace.get(name)) for name in ("_omlx_prefill", "_forward")}
    prefill = adapter._omlx_prefill if via_adapter else target._omlx_prefill
    forward = target._forward
    prefix = len(prompt["tokens"]) - 1

    def observed_prefill(ids, cache=None, **kwargs):
        result = prefill(ids, cache=cache, **kwargs)
        if int(cache[0].size()) == prefix:
            if "state" in capture:
                raise RuntimeError("Duplicate final prefill boundary")
            capture["state"] = audit.snapshot_boundary(cache, target, mx, np)
        return result

    def observed_forward(ids, cache=None, **kwargs):
        probe = ("state" in capture and "logits" not in capture and
                 ids.shape == (1, 1) and int(cache[0].size()) == prefix)
        result = forward(ids, cache=cache, **kwargs)
        if probe:
            logits = result[0] if isinstance(result, tuple) else result
            if logits.shape[:2] != (1, 1):
                raise RuntimeError("Expected one full-vocabulary probe row")
            mx.eval(logits)
            capture["logits"] = np.asarray(logits.astype(mx.float32)).copy()
        return result

    object.__setattr__(target, "_omlx_prefill", observed_prefill)
    object.__setattr__(target, "_forward", observed_forward)
    try:
        yield
    finally:
        for name, (present, value) in saved.items():
            if present:
                object.__setattr__(target, name, value)
            else:
                object.__delattr__(target, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("boundary", "performance"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    require_fresh_output(args.output, (".qualification.json", ".sources", ".arrays", ".completed"))
    frozen, profile, model_path = validate_frozen(args.manifest)
    if str(args.plan.resolve()) not in frozen["sources_sha256"]:
        raise ValueError("Plan is not pinned in the execution manifest")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, profile)
    if plan["design"]["suite"] != args.mode:
        raise ValueError("Execution mode must match the frozen plan suite")
    if args.validate_only:
        write_json(args.output, {"status":"cpu_validation_passed", "model_loaded":False,
                                "gpu_executed":False, "mode":args.mode, "jobs":len(plan["jobs"]),
                                "manifest_sha256":sha(args.manifest), "plan_sha256":sha(args.plan)})
        return

    with exclusive_model():
        validate_frozen(args.manifest)
        snapshot(args.manifest, args.output.with_suffix(".sources"))
        from flash1017 import bench
        import mlx.core as mx
        import numpy as np
        from omlx.models.vlm import VLMModelAdapter
        from runtime import Candidate, drained, serving_prefill_entry
        audit = import_path("_serving_v3_array_helpers", ROOT / "work/cone-experiment/verify_real.py")
        if Path(bench.__file__).resolve() != LAB / "flash1017/bench.py":
            raise RuntimeError("Invoke with the isolated lab launcher")
        arrays = args.output.with_suffix(".arrays")
        completed = args.output.with_suffix(".completed")
        completed.mkdir()
        if args.mode == "boundary":
            arrays.mkdir()
        report = {"status":"running", "mode":args.mode, "plan_sha256":sha(args.plan),
                  "manifest_sha256":sha(args.manifest), "started_at_unix":time.time(),
                  "loading_thread":threading.get_ident(), "pairs":[],
                  "scope":"Fresh KV, native64 bypass versus persistent model-owned cone+HotRows. OS cache is not purged. Row-cache memory remains allocated during subsequent baseline arms.",
                  "timing_interpretation":"Diagnostic only: state copies and full-vocabulary probe capture included." if args.mode == "boundary" else "Completed paired requests; excludes state/probe observers."}
        sidecar = args.output.with_suffix(".qualification.json")
        save = lambda: write_json(sidecar, report)
        save()
        original_configure, original_run, original_argv = bench.configure, bench.run_one, sys.argv
        owner = adapter = None
        index = 0
        pending = []
        captures = []
        workers = [ThreadPoolExecutor(max_workers=1, thread_name_prefix="qualification-a"),
                   ThreadPoolExecutor(max_workers=1, thread_name_prefix="qualification-b")]
        report["worker_threads"] = [worker.submit(threading.get_ident).result() for worker in workers]
        if len(set(report["worker_threads"] + [report["loading_thread"]])) != 3:
            raise RuntimeError("Expected two workers distinct from the loading thread")

        def configure(model, actual_profile):
            result = original_configure(model, actual_profile)
            owner.set_enabled(actual_profile["name"] == "fastpath64")
            return result

        def run_one(model, processor, prompt, actual_profile, repeat, seed):
            nonlocal owner, adapter, index, pending, captures
            job = plan["jobs"][index]
            if (actual_profile["name"], prompt["id"], repeat, seed) != (job["profile"], job["prompt"], job.get("repeat", 0), job["seed"]):
                raise RuntimeError("Request order differs from the frozen plan")
            if owner is None:
                original_configure(model, actual_profile)
                owner = Candidate(model, record_events=True)
                adapter = VLMModelAdapter(model)
                report["installation"] = owner.snapshot()
                save()
            owner.verify()
            # Both arms of each pair use the same worker and adapter route.
            pair_index = index // 2
            via_adapter = pair_index % 2 == 0
            captured = {}
            before = owner.snapshot(clear_cone=True)

            def execute():
                if args.mode == "boundary" and not job.get("warmup", False):
                    context = boundary_observers(model.language_model, adapter, via_adapter=via_adapter,
                        prompt=prompt, audit=audit, mx=mx, np=np, capture=captured)
                elif via_adapter:
                    context = serving_prefill_entry(model.language_model, adapter)
                else:
                    from contextlib import nullcontext
                    context = nullcontext()
                try:
                    with context:
                        value = original_run(model, processor, prompt, actual_profile, repeat, seed)
                finally:
                    # Complete queued GPU work before a worker exception can
                    # propagate into bench.main's native model.close finally.
                    mx.synchronize()
                value["qualification"] = {"job_index":index, "pair_index":pair_index,
                    "worker_thread":threading.get_ident(), "prefill_route":"VLMModelAdapter" if via_adapter else "LanguageModel._omlx_prefill",
                    "requested_output_tokens":prompt["max_tokens"]}
                # Retain each completed inference before any optional state
                # export/comparison can fail. Bench writes the enriched record.
                write_json(completed / f"job{index:02d}.json", value)
                return value

            result = await_drained_future(workers[pair_index % 2].submit(execute))
            drained(model.language_model)
            owner.verify()
            result["persistent_adapter"] = {"before":before, "after":owner.snapshot()}
            after = result["persistent_adapter"]["after"]
            eligible = sum(min(8192, len(prompt["tokens"])-1-start) >= 2668
                           for start in range(0, len(prompt["tokens"])-1, 8192))
            expected = eligible if actual_profile["name"] == "fastpath64" else 0
            query_before = sum(table["unique_queries"] for table in before["rows"]["live_tables"])
            query_after = sum(table["unique_queries"] for table in after["rows"]["live_tables"])
            activation = {
                "expected_cone_chunks":expected,
                "actual_cone_chunks":after["cone"]["cone_counts"].get("cone_calls",0),
                "actual_padded_appends":after["cone"]["prime_counts"].get("padded_appends",0),
                "actual_stage_pads":after["cone"]["prime_counts"].get("wkv_pads",0),
                "row_cache_unique_queries":query_after-query_before,
                "retained_cache_bytes":after["rows"]["retained_bytes"],
                "prefill_trace_count":len(after["cone"]["prefill_trace"]),
                "prefill_trace_evictions":after["cone"]["counts"].get("prefill_trace_evictions",0),
            }
            chunks = [min(8192,len(prompt["tokens"])-1-start)
                      for start in range(0,len(prompt["tokens"])-1,8192)]
            trace_valid = activation["prefill_trace_count"] == len(chunks) and activation["prefill_trace_evictions"] == 0
            if actual_profile["name"] == "fastpath64" and trace_valid:
                for rows, trace in zip(chunks, after["cone"]["prefill_trace"]):
                    if len(trace["prime"]) != 1 or len(trace["cone"]) != 1:
                        trace_valid = False
                        break
                    prime_event, cone_event = trace["prime"][0], trace["cone"][0]
                    wanted = rows >= 2668
                    trace_valid &= (prime_event.get("complete") is True
                        and prime_event.get("native_rows") == rows
                        and prime_event.get("append_count") == int(wanted)
                        and cone_event.get("mode") == ("cone" if wanted else "full")
                        and (prime_event.get("append_complete") is True if wanted else prime_event.get("reason") == "short-chunk"))
            activation["complete_chunk_traces"] = bool(trace_valid)
            activation["passed"] = (activation["actual_cone_chunks"] == expected
                and activation["actual_padded_appends"] == expected
                and activation["actual_stage_pads"] == 3*expected
                and (activation["row_cache_unique_queries"] > 0 if actual_profile["name"] == "fastpath64"
                     else activation["row_cache_unique_queries"] == 0)
                and activation["retained_cache_bytes"] <= 512*2**20
                and activation["complete_chunk_traces"]
                and not after["cone"]["counts"].get("call_failures",0))
            result["qualification"]["activation"] = activation
            if args.mode == "boundary" and not job.get("warmup", False):
                if set(captured) != {"state", "logits"}:
                    raise RuntimeError("Missing full state or one-token probe capture")
                state = captured["state"]
                result["boundary"] = {"layers":state["layers"], "prime":state["prime"],
                    "arrays":{k:v["metadata"] for k,v in state["slots"].items()},
                    "archive":audit.export_arrays(arrays / f"job{index:02d}-{job['profile']}.npz", state, captured["logits"], np)}
            pending.append(result)
            captures.append(captured)
            index += 1
            if index % 2 == 0:
                a, b = pending
                if a["profile"]["name"] != "native64":
                    a, b = b, a
                    captures.reverse()
                comparison = result_comparison(a, b)
                comparison.update(prompt_id=prompt["id"], warmup=bool(job.get("warmup", False)), pair_index=pair_index)
                if args.mode == "boundary" and not job.get("warmup", False):
                    x, y = captures
                    state_cmp = audit.compare_boundaries(x["state"], y["state"], len(prompt["tokens"])-1, np)
                    logits_cmp = audit.compare_logits(x["logits"], y["logits"], np)
                    state_cmp["all_target_metadata_identical"] = x["state"]["layers"] == y["state"]["layers"]
                    state_cmp["all_prime_metadata_identical"] = x["state"]["prime"] == y["state"]["prime"]
                    comparison.update(state=state_cmp, next_token_full_vocabulary=logits_cmp)
                    comparison["passed"] &= (state_cmp["structural_and_unchanged_state_checks_passed"]
                        and state_cmp["all_cache_slots_bit_identical"] and state_cmp["all_prime_rings_bit_identical"]
                        and state_cmp["all_target_metadata_identical"] and state_cmp["all_prime_metadata_identical"]
                        and logits_cmp["byte_identical"] and logits_cmp["both_finite"])
                report["pairs"].append(comparison)
                save()
                # The second raw arm must be written by bench.main before an
                # assertion can stop execution; defer mismatch to next call or
                # final completion rather than losing the negative result.
                report["mismatch_pending"] = not comparison["passed"]
                pending, captures = [], []
                gc.collect()
            return result

        def guarded_run(*args_, **kwargs_):
            if report.get("mismatch_pending"):
                raise RuntimeError("Qualification mismatch retained in preceding paired evidence")
            return run_one(*args_, **kwargs_)

        try:
            bench.configure, bench.run_one = configure, guarded_run
            sys.argv = [sys.argv[0], "--model", str(model_path), "--plan", str(args.plan), "--output", str(args.output)]
            bench.main()
            report["model_cleanup_completed"] = True
            if index != len(plan["jobs"]) or report.get("mismatch_pending"):
                raise RuntimeError("Incomplete or failed qualification")
            report["status"] = "complete_validated"
        except BaseException as exc:
            report.update(status="failed", exception=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            for worker in workers:
                worker.shutdown(wait=True, cancel_futures=True)
            bench.configure, bench.run_one, sys.argv = original_configure, original_run, original_argv
            try:
                if owner is not None:
                    report["row_cache_after_model_close"] = owner.rows.snapshot()
                    if (report["row_cache_after_model_close"]["retained_bytes"] != 0
                            or report["row_cache_after_model_close"]["live_tables"]):
                        raise RuntimeError("Model close did not release all row-cache tables")
                    owner.close()
                    report["instance_hooks_restored"] = owner.closed
                    adapter = None
                    gc.collect()
                validate_frozen(args.manifest)
                report["source_hashes_reverified"] = True
            except BaseException as exc:
                report.update(status="failed", cleanup_exception=repr(exc))
                raise
            finally:
                report["completed_at_unix"] = time.time()
                if args.output.exists():
                    report["raw_output_sha256"] = sha(args.output)
                report["completed_request_files_sha256"] = {str(p):sha(p) for p in completed.glob("*.json")}
                save()


if __name__ == "__main__":
    main()
