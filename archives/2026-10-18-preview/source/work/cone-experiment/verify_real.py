"""Opt-in full-checkpoint numerical audit of the shrinking decoder cone.

Run with outputs/flash1017-lab/run.sh and an unused --output JSON path.
This script loads one model, holds the lab process lock until model.close(),
and requires 210 GiB available before loading. It does not change qualified
files, server configuration, or checkpoint files. Its author did not run it.

Both arms use the qualified profile and native selector. Only the candidate
installs cone.py. Prefixes are fresh-cache runs of the same frozen token stream.
Cache and physical DSpark ring bytes are copied before the one-token probe.
The probe evaluates one vocabulary row; discarded wide prefill logits remain
lazy. Dependency exactness does not imply identical floating-point arithmetic.
All differences are retained; timing is diagnostic, never a benchmark claim.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "outputs/flash1017-lab"
SLOT_NAMES = ("offset", "local_kv", "global_kv", "index_k", "partial_kv", "partial_gates", "engram_history")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_report(path, report):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_qualified():
    path = LAB / "qualified-profile.json"
    qualified = json.loads(path.read_text())
    if not qualified.get("qualified"):
        raise RuntimeError("Profile is not qualified")
    for relative, digest in qualified["inference_files_sha256"].items():
        if sha((LAB / relative).read_bytes()) != digest:
            raise RuntimeError("Qualified source changed: " + relative)
    for package, version in qualified["packages"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("Qualified dependency changed: " + package)
    hashes_path = LAB / "results/checkpoint-hashes.json"
    hashes = json.loads(hashes_path.read_text())
    model_path = Path(hashes["directory"])
    for name, record in hashes["files"].items():
        stat = (model_path / name).stat()
        if [stat.st_ino, stat.st_size, stat.st_mtime_ns] != record["fingerprint"]:
            raise RuntimeError("Verified checkpoint changed: " + name)
    return qualified, model_path, {
        "qualified_profile_sha256": sha(path.read_bytes()),
        "inference_files_checked": len(qualified["inference_files_sha256"]),
        "checkpoint_hash_manifest_sha256": sha(hashes_path.read_bytes()),
        "checkpoint_files_fingerprint_checked": len(hashes["files"]),
        "checkpoint_verification": "Compare inode, size and mtime_ns to previously hashed checkpoint; no new whole-weight hash pass.",
        "packages": qualified["packages"],
    }


def frozen_tokens(plan_path, prompt_id, needed):
    plan = json.loads(plan_path.read_text())
    prompt = plan["prompts"][prompt_id]
    tokens = prompt.get("tokens")
    if not isinstance(tokens, list) or len(tokens) < needed:
        raise ValueError("Selected prompt needs at least prefix+1 frozen token IDs")
    if any(type(x) is not int or x < 0 for x in tokens):
        raise ValueError("Token IDs must be nonnegative integers")
    full_hash = sha(json.dumps(tokens).encode())
    declared = prompt.get("input_tokens_sha256")
    if declared and declared != full_hash:
        raise ValueError("Frozen token IDs disagree with the input manifest")
    return tokens[:needed], {
        "plan": str(plan_path.resolve()), "plan_sha256": sha(plan_path.read_bytes()),
        "prompt_id": prompt_id, "source_token_count": len(tokens),
        "source_input_tokens_sha256": full_hash,
        "used_token_count": needed,
        "used_tokens_sha256": sha(json.dumps(tokens[:needed]).encode()),
        "natural_source": prompt.get("natural_source"),
        "interpretation": "Prefixes of a frozen natural source-code prompt, possibly ending inside source text; this is a state audit, not a completed user-question evaluation.",
    }


def raw_array(value, mx, np):
    if value is None:
        return {"metadata": {"none": True}, "raw": None}
    # MLX view guarantees the binary representation, unlike BF16 widening.
    # Explicitly flatten logical elements first, so strides do not enter the
    # byte hash. A zero-length cache slot is legal.
    raw = np.asarray(value.reshape((value.size,)).view(mx.uint8)).copy()
    metadata = {"none": False, "shape": list(value.shape),
                "dtype": str(value.dtype).split(".")[-1],
                "elements": int(value.size), "bytes": int(raw.size),
                "sha256": sha(raw.tobytes())}
    return {"metadata": metadata, "raw": raw}


def numeric_array(snapshot, np):
    if snapshot["raw"] is None:
        return None
    meta, raw = snapshot["metadata"], snapshot["raw"]
    dtype = meta["dtype"]
    if dtype == "bfloat16":
        # Lossless representation of all finite BF16 values for error metrics;
        # identity is always computed on original bytes, including NaN payloads.
        value = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        value = raw.view(np.dtype(dtype))
    return value.reshape(meta["shape"])


def array_comparison(baseline, candidate, np, numerical=False):
    a, b = baseline["metadata"], candidate["metadata"]
    same_none = a["none"] and b["none"]
    same_layout = same_none or (not a["none"] and not b["none"]
                               and a["shape"] == b["shape"] and a["dtype"] == b["dtype"])
    row = {"baseline": a, "candidate": b, "same_layout": same_layout,
           "byte_identical": same_none or (same_layout and a["sha256"] == b["sha256"]),
           "different_bytes": None, "different_elements": None}
    if same_none:
        row.update(different_bytes=0, different_elements=0)
    elif same_layout:
        different = baseline["raw"] != candidate["raw"]
        row["different_bytes"] = int(np.count_nonzero(different))
        itemsize = a["bytes"] // a["elements"] if a["elements"] else 1
        row["different_elements"] = int(np.count_nonzero(different.reshape(-1, itemsize).any(1)))
        if numerical:
            x, y = numeric_array(baseline, np), numeric_array(candidate, np)
            finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
            row["both_finite"] = finite
            row["max_absolute_difference"] = (float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                                                if finite and x.size else (0.0 if finite else None))
    return row


def unpack_local(snapshot, np):
    """CPU copy of native v41_fp8 + per-32-channel exponent decoding."""
    value = numeric_array(snapshot, np)
    if value is None or value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 528:
        return None
    code = value[..., :512]
    magnitude = code & 127
    exponent, mantissa = magnitude >> 3, magnitude & 7
    decoded = np.where(exponent == 0, np.ldexp(mantissa.astype(np.float64), -9),
                       np.ldexp(1.0 + mantissa.astype(np.float64) / 8, exponent.astype(np.int32) - 7))
    decoded = np.where(code & 128, -decoded, decoded)
    # Match the native decoder even for code 0x7f; it does not special-case NaN.
    scales = np.repeat(value[..., 512:].astype(np.int32) - 127, 32, axis=-1)
    return np.ldexp(decoded, scales)


def materialize(cache, language, mx):
    arrays = [value for item in cache for value in item.cache if value is not None]
    context = getattr(language, "_omlx_mtp_prime_ctx", None)
    if context is not None:
        arrays.extend(item.keys for item in context.caches if item.keys is not None)
    mx.eval(arrays)
    mx.synchronize()


def snapshot_boundary(cache, language, mx, np):
    materialize(cache, language, mx)
    slots, metadata = {}, []
    for layer, item in enumerate(cache):
        metadata.append({"layer": layer, "offset": item.size(), "meta_state": list(item.meta_state)})
        for slot, value in enumerate(item.cache):
            slots[f"layer{layer:02d}_slot{slot}"] = raw_array(value, mx, np)
        for attr in ("left_padding", "lengths"):
            slots[f"layer{layer:02d}_{attr}"] = raw_array(getattr(item, attr, None), mx, np)
    context = getattr(language, "_omlx_mtp_prime_ctx", None)
    prime = {"present": context is not None, "rings": []}
    if context is not None:
        prime["expected_target_offset"] = int(context.expected_target_offset)
        for stage, item in enumerate(context.caches):
            prime["rings"].append({"stage": stage, "offset": int(item.offset), "max_size": int(item.max_size)})
            slots[f"prime{stage}_physical_keys"] = raw_array(item.keys, mx, np)
    return {"slots": slots, "layers": metadata, "prime": prime}


def export_arrays(path, snapshot, logits, np):
    arrays = {key: value["raw"] for key, value in snapshot["slots"].items() if value["raw"] is not None}
    arrays["probe_logits"] = logits
    # Raw ring bytes retain physical absolute-position % window ordering.
    # Their shape/dtype and all None slots are described in the JSON manifest.
    np.savez_compressed(path, **arrays)
    return {"path": str(path.resolve()), "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def compare_boundaries(baseline, candidate, prefix, np):
    rows, rings, issues = [], [], []
    for layer in range(len(baseline["layers"])):
        for slot, name in enumerate(SLOT_NAMES):
            key = f"layer{layer:02d}_slot{slot}"
            row = {"layer": layer, "slot": slot, "name": name,
                   **array_comparison(baseline["slots"][key], candidate["slots"][key], np)}
            if slot == 1 and row["same_layout"]:
                x, y = unpack_local(baseline["slots"][key], np), unpack_local(candidate["slots"][key], np)
                if x is not None and y is not None:
                    row["unpacked_local_kv_max_absolute_difference"] = float(np.max(np.abs(x - y))) if x.size else 0.0
                    row["unpacked_local_kv_different_elements"] = int(np.count_nonzero(x != y))
            rows.append(row)
            if not row["same_layout"]:
                issues.append(f"Cache layout differs at layer {layer}, slot {slot}")
        for attr in ("left_padding", "lengths"):
            key = f"layer{layer:02d}_{attr}"
            row = {"layer": layer, "name": attr,
                   **array_comparison(baseline["slots"][key], candidate["slots"][key], np)}
            rows.append(row)
            if not row["byte_identical"]:
                issues.append(f"Cache {attr} differs at layer {layer}")
    for arm, snapshot in (("baseline", baseline), ("cone", candidate)):
        if any(item["offset"] != prefix for item in snapshot["layers"]):
            issues.append(arm + " target cache offset mismatch")
        prime = snapshot["prime"]
        if not prime["present"] or len(prime["rings"]) != 3:
            issues.append(arm + " missing expected three-stage DSpark prime context")
        elif prime["expected_target_offset"] != prefix or any(item["offset"] != prefix for item in prime["rings"]):
            issues.append(arm + " DSpark prime timeline mismatch")
    if baseline["prime"]["present"] and candidate["prime"]["present"]:
        for a, b in zip(baseline["prime"]["rings"], candidate["prime"]["rings"]):
            key = f"prime{a['stage']}_physical_keys"
            row = {"stage": a["stage"], "baseline_offset": a["offset"], "candidate_offset": b["offset"],
                   "same_ring_metadata": a == b,
                   **array_comparison(baseline["slots"][key], candidate["slots"][key], np, numerical=True)}
            rings.append(row)
            if not row["same_layout"] or not row.get("both_finite", False) or not row["same_ring_metadata"]:
                issues.append(f"Invalid DSpark ring {a['stage']}")
    encoder_equal = all(row["byte_identical"] for row in rows if row["layer"] < 20)
    global_equal = all(row["byte_identical"] for row in rows if row.get("slot") in (2, 3))
    if not encoder_equal:
        issues.append("Unchanged encoder cache bytes differ")
    if not global_equal:
        issues.append("Unchanged global KV/index cache bytes differ")
    return {"cache_slots": rows, "dspark_physical_rings_before_probe": rings,
            "encoder_cache_all_bit_identical": encoder_equal,
            "global_kv_and_index_cache_all_bit_identical": global_equal,
            "all_cache_slots_bit_identical": all(row["byte_identical"] for row in rows),
            "all_prime_rings_bit_identical": len(rings) == 3 and all(row["byte_identical"] for row in rings),
            "structural_and_unchanged_state_checks_passed": not issues, "issues": issues}


def compare_logits(baseline, candidate, np):
    a, b = baseline.astype(np.float64).reshape(-1), candidate.astype(np.float64).reshape(-1)
    same_shape = baseline.shape == candidate.shape
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    row = {"baseline_shape": list(baseline.shape), "candidate_shape": list(candidate.shape),
           "same_shape": same_shape, "both_finite": finite,
           "baseline_sha256": sha(baseline.tobytes()), "candidate_sha256": sha(candidate.tobytes()),
           "byte_identical": same_shape and baseline.dtype == candidate.dtype and baseline.tobytes() == candidate.tobytes()}
    if not same_shape or not finite:
        return row
    la = a - (np.log(np.exp(a - a.max()).sum()) + a.max())
    lb = b - (np.log(np.exp(b - b.max()).sum()) + b.max())
    pa, pb = np.exp(la), np.exp(lb)
    # Lexicographic token-ID tie breaking makes overlap diagnostics repeatable.
    ids = np.arange(a.size)
    order_a, order_b = np.lexsort((ids, -a)), np.lexsort((ids, -b))
    row.update({"max_absolute_difference": float(np.max(np.abs(a - b))),
                "rms_difference": float(np.sqrt(np.mean((a - b) ** 2))),
                "different_elements": int(np.count_nonzero(a != b)),
                "kl_baseline_to_cone": float(np.sum(pa * (la - lb))),
                "kl_cone_to_baseline": float(np.sum(pb * (lb - la))),
                "total_variation": float(0.5 * np.abs(pa - pb).sum()),
                "baseline_top1": int(order_a[0]), "cone_top1": int(order_b[0]),
                "top1_equal": bool(order_a[0] == order_b[0]),
                "baseline_top10": order_a[:10].tolist(), "cone_top10": order_b[:10].tolist(),
                "topk_overlap": {str(k): int(len(set(order_a[:k]) & set(order_b[:k]))) for k in (5, 10, 50, 100)}})
    return row


def run_arm(model, tokens, prefix, arm, profile, folder, report, save, mx, np, bench, cone, drop_primed):
    language = model.language_model
    bench.configure(model, profile)
    drop_primed(language)
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    cache = language.make_cache()
    record = {"prefix_tokens": prefix, "arm": arm, "status": "running", "started_at": now(),
              "chunks": [], "profile": profile, "probe_token": tokens[prefix]}
    report["arms"].append(record)
    save()
    context = cone.installed(language, record_events=True) if arm == "cone" else nullcontext(None)
    try:
        with context as experiment:
            for start in range(0, prefix, 8192):
                end = min(start + 8192, prefix)
                before = time.perf_counter()
                # skip_lm_head is the serving prefill contract. The native
                # implementation can still construct lazy wide logits; do
                # not keep or evaluate the returned value.
                language._omlx_prefill(mx.array([tokens[start:end]]), cache=cache, skip_lm_head=True)
                materialize(cache, language, mx)
                elapsed = time.perf_counter() - before
                trace = dict(experiment.events[-1]) if experiment is not None else {
                    "mode": "full", "reason": "qualified-ced-disabled", "input_tokens": end - start}
                record["chunks"].append({"start": start, "end": end, "seconds": elapsed,
                                          "target_offsets": [item.size() for item in cache], "trace": trace})
                record["completed_prefill_seconds"] = sum(item["seconds"] for item in record["chunks"])
                save()
                print(json.dumps({"audit": arm, "prefix": prefix, "chunk_end": end,
                                  "seconds": elapsed, "mode": trace["mode"]}), flush=True)
            boundary = snapshot_boundary(cache, language, mx, np)
            record["boundary_before_probe"] = {"layers": boundary["layers"], "prime": boundary["prime"],
                                               "arrays": {k: v["metadata"] for k, v in boundary["slots"].items()}}
            save()
            before = time.perf_counter()
            result = language(mx.array([[tokens[prefix]]]), cache=cache)
            if isinstance(result, tuple):
                raise TypeError("Ordinary one-token forward unexpectedly returned a tuple")
            if tuple(result.shape[:2]) != (1, 1):
                raise ValueError("One-token probe must produce exactly one vocabulary row")
            mx.eval(result)
            materialize(cache, language, mx)
            record["probe_seconds"] = time.perf_counter() - before
            logits = np.asarray(result.astype(mx.float32)).copy()
            del result
            record["probe_offsets"] = [item.size() for item in cache]
            if any(offset != prefix + 1 for offset in record["probe_offsets"]):
                raise RuntimeError("Probe did not advance each target cache by one")
            if experiment is not None:
                record["cone_events"] = experiment.events
                record["cone_counts"] = dict(experiment.counts)
                record["required_rows"] = experiment.required
                expected_cone_calls = sum(min(8192, prefix - start) >= experiment.required
                                          for start in range(0, prefix, 8192))
                if experiment.counts["cone_calls"] != expected_cone_calls:
                    raise RuntimeError("Expected large chunks did not exercise the cone")
                if experiment.events[-1]["mode"] != "full":
                    raise RuntimeError("Normal one-token forward did not use the full path")
            record["arrays"] = export_arrays(folder / f"prefix{prefix}-{arm}.npz", boundary, logits, np)
            record.update(status="complete", finished_at=now())
            save()
            return boundary, logits, record
    finally:
        drop_primed(language)
        del cache
        mx.synchronize()
        gc.collect()
        mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "work/prefill-fullmodel/natural.json")
    parser.add_argument("--prompt", default="natural-code-32768")
    parser.add_argument("--prefixes", type=int, nargs="+", default=[8192, 16400])
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".arrays").exists() or args.output.with_suffix(".sources").exists():
        raise FileExistsError("Use a new output stem to preserve existing evidence")
    if not args.prefixes or len(set(args.prefixes)) != len(args.prefixes) or any(n < 8192 for n in args.prefixes):
        raise ValueError("Use distinct prefix lengths of at least 8192 tokens")
    qualified, model_path, verification = verify_qualified()
    profile = json.loads(json.dumps(qualified["runtime"]))
    if profile.get("ced_prefill") is not False or profile.get("prefill_step") != 8192:
        raise RuntimeError("Expected the qualified full-prefill 8192-token profile")
    tokens, token_manifest = frozen_tokens(args.plan, args.prompt, max(args.prefixes) + 1)
    import psutil
    available = psutil.virtual_memory().available
    if available < 210 * 2**30:
        raise MemoryError("Less than 210 GiB available; unload other models before this audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folder = args.output.with_suffix(".arrays")
    sources = args.output.with_suffix(".sources")
    folder.mkdir(exist_ok=False)
    sources.mkdir(exist_ok=False)
    source_paths = [Path(__file__), Path(__file__).with_name("cone.py"), args.plan,
                    LAB / "qualified-profile.json", LAB / "provenance.json", LAB / "results/checkpoint-hashes.json"]
    source_manifest = {}
    for path in source_paths:
        data = path.read_bytes()
        (sources / path.name).write_bytes(data)
        source_manifest[str(path.resolve())] = sha(data)
    (sources / "frozen-used-tokens.json").write_text(json.dumps(tokens) + "\n")
    report = {"status": "created", "created_at": now(), "pid": os.getpid(), "platform": platform.platform(),
              "byte_order": sys.byteorder, "model": str(model_path), "verification": verification,
              "available_bytes_before_load": available, "source_sha256": source_manifest,
              "tokens": token_manifest, "prefixes": args.prefixes, "profile": profile,
              "scope": "Full exact-1017 checkpoint numerical audit, dependency-exact cone prototype. Decoder bit identity is measured, not assumed. No quality or performance qualification.",
              "timing_scope": "One fresh-cache paired run per prefix; fixed baseline/cone order, no warmup or variance estimate. Completed prefill and one-token times are diagnostics only.",
              "arms": [], "comparisons": []}
    (sources / "manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    save = lambda: write_report(args.output, report)
    save()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model = None
    try:
        with (LAB / "server-state/server.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Recheck after acquiring the exclusive lock, before any load.
            if psutil.virtual_memory().available < 210 * 2**30:
                raise MemoryError("Available memory dropped below the 210 GiB preload floor")
            import mlx.core as mx
            import numpy as np
            import cone
            from flash1017 import bench
            from flash1017.memory import prepare_memory
            from omlx.model_settings import ModelSettings
            from omlx.patches.deepseek_v41 import apply_patch
            from omlx.patches.deepseek_v41.loading import load
            from omlx.patches.mlx_lm_mtp import batch_generator, set_mtp_active
            from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import drop_primed
            from omlx.utils.model_loading import maybe_apply_pre_load_patches
            try:
                loaded_modules = {name: Path(sys.modules[name].__file__).resolve() for name in (
                    "cone", "flash1017.bench", "omlx.patches.deepseek_v41.language",
                    "omlx.patches.deepseek_v41.loading")}
                expected_paths = {"cone": Path(__file__).with_name("cone.py").resolve(),
                                  "flash1017.bench": (LAB / "flash1017/bench.py").resolve(),
                                  "omlx.patches.deepseek_v41.language": (LAB / "_runtime/omlx/patches/deepseek_v41/language.py").resolve(),
                                  "omlx.patches.deepseek_v41.loading": (LAB / "_runtime/omlx/patches/deepseek_v41/loading.py").resolve()}
                if loaded_modules != expected_paths:
                    raise RuntimeError("Loaded modules are outside the isolated lab; invoke this script through outputs/flash1017-lab/run.sh")
                report["loaded_module_paths"] = {name: str(path) for name, path in loaded_modules.items()}
                maybe_apply_pre_load_patches(str(model_path), ModelSettings(
                    mtp_enabled=True, mtp_adaptive_max_depth=3,
                    deepseek_v41_engram_ssd_offload=True), for_vlm=True)
                apply_patch()
                set_mtp_active(True)
                batch_generator.apply()
                report["memory_policy"] = prepare_memory(args.output.with_suffix(".load.jsonl"), 8 * 2**30)
                report.update(status="loading", device=mx.device_info())
                save()
                started = time.perf_counter()
                model, processor = load(model_path, engram_ssd_offload=True, attention_output_quantization=False)
                report.update(status="running", load_seconds=time.perf_counter() - started)
                save()
                print("MODEL_LOADED_FOR_CONE_AUDIT", flush=True)
                for prefix in args.prefixes:
                    baseline, baseline_logits, baseline_record = run_arm(model, tokens, prefix, "baseline", profile, folder, report, save, mx, np, bench, cone, drop_primed)
                    candidate, candidate_logits, candidate_record = run_arm(model, tokens, prefix, "cone", profile, folder, report, save, mx, np, bench, cone, drop_primed)
                    comparison = compare_boundaries(baseline, candidate, prefix, np)
                    comparison.update({"prefix_tokens": prefix, "probe_token": tokens[prefix],
                                       "probe_logits": compare_logits(baseline_logits, candidate_logits, np),
                                       "completed_paired_prefill_seconds": {"baseline": baseline_record["completed_prefill_seconds"], "cone": candidate_record["completed_prefill_seconds"]}})
                    report["comparisons"].append(comparison)
                    save()
                    print(json.dumps({"prefix": prefix, "unchanged_state_checks": comparison["structural_and_unchanged_state_checks_passed"],
                                      "logits": comparison["probe_logits"]}), flush=True)
                    del baseline, candidate, baseline_logits, candidate_logits
                    gc.collect()
                report["source_files_unchanged_during_audit"] = all(sha(Path(path).read_bytes()) == digest for path, digest in source_manifest.items())
                report.update(status="complete", finished_at=now())
                report["structural_and_unchanged_state_checks_passed"] = all(
                    row["structural_and_unchanged_state_checks_passed"] and row["probe_logits"]["same_shape"] and row["probe_logits"]["both_finite"]
                    for row in report["comparisons"])
                save()
            finally:
                # Keep the process lock held through native mmap cleanup.
                if model is not None:
                    try:
                        drop_primed(model.language_model)
                        mx.synchronize()
                    finally:
                        model.close()
                        report["model_closed"] = True
                        save()
    except BaseException as exc:
        report.update(status="failed", finished_at=now(), exception=repr(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
