"""One fresh process for one arm of the frozen overall comparison.

The existing bench supplies identical timing, sampling and generation plumbing.
Stock imports only installed oMLX and uses its own inference implementation.
Candidate uses the unchanged qualified source and model-owned replay/cache.
This measures loaded-model prompt+generation, not HTTP or installation time.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "outputs/flash1017-lab"
APP = Path("/Applications/oMLX.app/Contents/Resources")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def validate(plan, manifest):
    for path, digest in manifest["sources_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Frozen source changed: " + path)
    checkpoint = json.loads((LAB / "results/checkpoint-hashes.json").read_text())
    if checkpoint["status"] != "complete_hash_verified" or checkpoint.get("missing"):
        raise ValueError("Missing checkpoint verification")
    model_path = Path(checkpoint["directory"])
    for name, record in checkpoint["files"].items():
        stat = (model_path / name).stat()
        if not record["verified"] or [stat.st_ino, stat.st_size, stat.st_mtime_ns] != record["fingerprint"]:
            raise ValueError("Checkpoint fingerprint changed: " + name)
    for name, prompt in plan["prompts"].items():
        if hashlib.sha256(json.dumps(prompt["tokens"]).encode()).hexdigest() != prompt["input_tokens_sha256"]:
            raise ValueError("Prompt token identity changed: " + name)
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    manifest = json.loads(args.manifest.read_text())
    model_path = validate(plan, manifest)
    process = plan["processes"][args.process_index]
    arm = process["arm"]
    profile = plan["profiles"][arm] | {"name": arm}
    is_candidate = arm == "complete"
    expected_root = (LAB / "_runtime/omlx") if is_candidate else APP / "omlx"
    if args.output.exists():
        raise FileExistsError("Preserve prior results: " + str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        args.output.write_text(json.dumps({"status": "cpu_validated", "model_loaded": False,
            "arm": arm, "jobs": len(process["jobs"]), "plan_sha256": sha(args.plan)}, indent=2) + "\n")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with (LAB / "server-state/server.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import importlib.metadata
        import mlx.core as mx
        import psutil
        import omlx
        from flash1017 import bench
        from flash1017.memory import prepare_memory
        from omlx.patches.deepseek_v41 import apply_patch
        from omlx.patches.deepseek_v41.loading import load
        from omlx.patches.deepseek_v41 import storage
        from omlx.patches.mlx_lm_mtp import batch_generator, set_mtp_active
        from omlx.model_settings import ModelSettings
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        if Path(omlx.__file__).resolve().parent != expected_root.resolve():
            raise RuntimeError("Wrong oMLX import root: " + omlx.__file__)
        if Path(bench.__file__).resolve() != LAB / "flash1017/bench.py":
            raise RuntimeError("Wrong common benchmark")
        packages = {name: importlib.metadata.version(name) for name in manifest["packages"]}
        if packages != manifest["packages"]:
            raise RuntimeError("Changed dependency versions")
        if psutil.virtual_memory().available < 210 * 2**30:
            raise MemoryError("Less than 210 GiB available; no model load attempted")
        maybe_apply_pre_load_patches(str(model_path), ModelSettings(
            mtp_enabled=True, mtp_adaptive_max_depth=profile["depth"],
            deepseek_v41_engram_ssd_offload=True), for_vlm=True)
        apply_patch()
        set_mtp_active(True)
        batch_generator.apply()
        memory_policy = prepare_memory(args.output.with_suffix(".load.jsonl"), plan["cache_limit_bytes"])
        audit = load_module("overall_source_activation", HERE / "runtime-audit/source_activation.py")
        model = processor = owner = None
        row = lambda data: write(args.output, data)
        row({"type": "manifest", "started_at_unix": time.time(), "process_index": args.process_index,
             "block": process["block"], "arm": arm, "profile": profile,
             "plan_sha256": sha(args.plan), "manifest_sha256": sha(args.manifest),
             "runtime_root": str(expected_root), "packages": packages,
             "device": mx.device_info(), "memory_policy": memory_policy,
             "scope": plan["scope"]})
        meter = bench.Meter().start()
        begun = time.perf_counter()
        try:
            kwargs = {"engram_ssd_offload": True}
            if is_candidate:
                kwargs["attention_output_quantization"] = False
            model, processor = load(model_path, **kwargs)
            row({"type": "load", "seconds": time.perf_counter()-begun, "resources": meter.stop()})
            print(json.dumps({"event": "loaded", "arm": arm, "block": process["block"]}), flush=True)
            candidate_configure = bench.configure
            if is_candidate:
                sys.path.insert(0, str(ROOT / "work/serving-integration-v3b"))
                runtime = load_module("overall_candidate_runtime", ROOT / "work/serving-integration-v3b/runtime.py")
                candidate_configure(model, profile)
                owner = runtime.Candidate(model, record_events=True)
                owner.set_enabled(True)

                def configure(actual_model, actual_profile):
                    experiment = candidate_configure(actual_model, actual_profile)
                    owner.set_enabled(True)
                    return experiment
            else:
                def configure(actual_model, actual_profile):
                    if storage.PREFETCH_BYTES != 16 * 2**20:
                        raise RuntimeError("Stock prefetch budget changed")
                    target = actual_model.language_model
                    if hasattr(target._config, "attention_output_quantization"):
                        raise RuntimeError("Local reference patch present in stock")
                    target._config.ced_prefill = False
                    target.configure_mtp(True, actual_profile["depth"])
                    target._omlx_mtp_depth_fixed = False
                    return None
            bench.configure = configure
            configure(model, profile)

            def source_check(phase):
                captured = audit.capture(runtime_root=expected_root, model=model,
                    arm="candidate" if is_candidate else "upstream", profile=profile, owner=owner)
                row({"type": "source_activation", "phase": phase, **captured})
                if not captured["passed"]:
                    raise RuntimeError("Source/feature activation check failed: " + repr(captured["checks"]))
            source_check("before_runs")
            for index, job in enumerate(process["jobs"]):
                prompt = plan["prompts"][job["prompt"]] | {"id": job["prompt"]}
                if owner:
                    before = owner.snapshot(clear_cone=True)
                print(json.dumps({"event": "start", "arm": arm, "block": process["block"],
                    "job": index, "prompt": job["prompt"], "warmup": job["warmup"]}), flush=True)
                result = bench.run_one(model, processor, prompt, profile, process["block"], job["seed"])
                mx.synchronize()
                result.update(arm=arm, block=process["block"], process_index=args.process_index,
                    job_index=index, warmup=job["warmup"], cohort=prompt["cohort"])
                if owner:
                    runtime.drained(model.language_model)
                    owner.verify()
                    after = owner.snapshot()
                    eligible = sum(min(profile["prefill_step"], len(prompt["tokens"])-1-start) >= 2668
                        for start in range(0, len(prompt["tokens"])-1, profile["prefill_step"]))
                    queries_before = sum(t["unique_queries"] for t in before["rows"]["live_tables"])
                    queries_after = sum(t["unique_queries"] for t in after["rows"]["live_tables"])
                    checks = {
                        "expected_cone_calls": after["cone"]["cone_counts"].get("cone_calls", 0) == eligible,
                        "expected_padded_appends": after["cone"]["prime_counts"].get("padded_appends", 0) == eligible,
                        "expected_wkv_pads": after["cone"]["prime_counts"].get("wkv_pads", 0) == 3*eligible,
                        "no_cone_call_failure": after["cone"]["counts"].get("call_failures", 0) == 0,
                        "row_cache_queried": queries_after > queries_before,
                        "retained_rows_bounded": after["rows"]["retained_bytes"] <= 512*2**20,
                        "confidence_executed": any(e.get("confidence_logits") is not None for e in result["events"]),
                    }
                    result["candidate"] = {"before": before, "after": after,
                        "expected_eligible_chunks": eligible, "checks": checks, "passed": all(checks.values())}
                else:
                    result["candidate"] = {"passed": True, "scope": "No candidate installed; see source activation record"}
                row(result)
                print(json.dumps({"event": "done", "arm": arm, "block": process["block"],
                    "job": index, "prompt": job["prompt"], "warmup": job["warmup"],
                    "prefill_seconds": result["prefill_seconds"], "total_seconds": result["total_seconds"],
                    "decode_tokens_per_second": result["decode_tokens_per_second"],
                    "generated_tokens": result["generated_tokens"], "first_eos_position": result["first_eos_position"],
                    "activation_passed": result["candidate"]["passed"]}), flush=True)
                if not result["candidate"]["passed"]:
                    raise RuntimeError("Expected complete candidate execution not observed")
                if result["generated_tokens"] != prompt["max_tokens"]:
                    raise RuntimeError("Requested benchmark length not completed")
                if result["resources"]["system_available_min_bytes"] < 8*2**30:
                    raise MemoryError("Less than 8 GiB system headroom observed")
                if result["resources"]["swap_peak_bytes"]-result["resources"]["swap_start_bytes"] > 512*2**20:
                    raise MemoryError("More than 512 MiB swap growth observed")
                if job["warmup"] and owner:
                    owner.rows.set_enabled(True, clear=True)
                    row({"type": "row_cache_reset_after_warmup", "snapshot": owner.rows.snapshot()})
            source_check("after_runs")
            validate(plan, manifest)
            row({"type": "complete", "completed_at_unix": time.time(), "jobs": len(process["jobs"])})
        except BaseException as exc:
            row({"type": "error", "exception": repr(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            if not meter.done.is_set():
                meter.stop()
            cleanup = {"type": "cleanup", "native_close": False, "candidate_close": None}
            try:
                mx.synchronize()
                if model is not None:
                    model.close()
                    cleanup["native_close"] = True
            finally:
                try:
                    if owner is not None:
                        try:
                            cleanup["rows_after_native_close"] = owner.rows.snapshot()
                        finally:
                            owner.close()
                            cleanup["candidate_close"] = owner.closed
                        if cleanup["rows_after_native_close"]["retained_bytes"] or cleanup["rows_after_native_close"]["live_tables"]:
                            raise RuntimeError("Native close retained candidate row tables")
                finally:
                    row(cleanup)


if __name__ == "__main__":
    main()
