"""Single-request, fixed-checkpoint experiments; raw evidence is JSONL.

Example: python -m flash1017.bench --model /path/to/model --plan plan.json
Every plan specifies its profiles and prompts. No hidden benchmark tuning,
automatic model downloads, global app settings, or network requests occur here.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import threading
import time

import mlx.core as mx
import psutil

from .policy import Experiment, PrefixPolicy


def process_io():
    # Reuse oMLX's macOS ABI definition, including Metal-backed footprint.
    from omlx.utils.proc_memory import _RusageInfoV4, _proc_pid_rusage, _RUSAGE_INFO_V4
    if _proc_pid_rusage is None:
        return {}
    info = _RusageInfoV4()
    if _proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)):
        return {}
    return {"disk_read_bytes": int(info.ri_diskio_bytesread),
            "disk_write_bytes": int(info.ri_diskio_byteswritten)}


class Meter:
    """Sample the process ledger and system state; report the sampling interval."""
    def __init__(self):
        self.done = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _snapshot(self):
        from omlx.utils.proc_memory import get_phys_footprint
        vm = psutil.virtual_memory()
        self.samples.append({"time": time.perf_counter(),
                             "footprint_bytes": get_phys_footprint(),
                             "system_used_bytes": vm.used,
                             "available_bytes": vm.available,
                             "swap_used_bytes": psutil.swap_memory().used,
                             "mlx_cache_bytes": mx.get_cache_memory()})

    def _sample(self):
        while not self.done.wait(.2):
            self._snapshot()

    def start(self):
        self._snapshot()
        self.io_before = process_io()
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        self.thread.join()
        self._snapshot()
        io = process_io()
        return {
            "sampling_interval_seconds": .2,
            "process_footprint_peak_bytes": max(s["footprint_bytes"] for s in self.samples),
            "system_used_peak_bytes": max(s["system_used_bytes"] for s in self.samples),
            "system_available_min_bytes": min(s["available_bytes"] for s in self.samples),
            "swap_start_bytes": self.samples[0]["swap_used_bytes"],
            "swap_peak_bytes": max(s["swap_used_bytes"] for s in self.samples),
            "swap_end_bytes": self.samples[-1]["swap_used_bytes"],
            "mlx_cache_peak_bytes": max(s["mlx_cache_bytes"] for s in self.samples),
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
            **{f"{key}_delta": value - self.io_before.get(key, value)
               for key, value in io.items()},
        }


def configure(model, profile):
    # Configure only between drained requests or immediately after loading.
    # This process owns one model; the native prefetch budget is module-global.
    budget = profile.get("engram_prefetch_bytes", 16 * 2**20)
    if type(budget) is not int or not 1 <= budget <= 64 * 2**20:
        raise ValueError("engram_prefetch_bytes must be an integer from 1 to 67108864")
    prefetch = getattr(model.language_model, "_engram_prefetch", None)
    if prefetch is not None:
        if prefetch._closed:
            raise RuntimeError("Cannot configure a closed Engram prefetch executor")
        if prefetch._pending is not None:
            raise RuntimeError("Drain Engram prefetch before configuring the model")
    from omlx.patches.deepseek_v41 import storage
    storage.PREFETCH_BYTES = budget
    from . import head
    if profile.get("fused_head", False):
        head.install(max_rows=profile.get("head_max_rows", 5))
    else:
        head.uninstall()
    language = model.language_model
    for _, module in model.named_modules():
        config = getattr(module, "_config", None)
        if config is not None and hasattr(config, "attention_output_quantization"):
            config.attention_output_quantization = profile.get("attention_rounding", False)
    language._config.ced_prefill = profile.get("ced_prefill", False)
    if language._config.ced_prefill and not language._config.ced_layout_supported():
        raise ValueError("CED is unsupported by this checkpoint layout")
    language.configure_mtp(profile.get("mtp", True), profile.get("depth", 3))
    language._omlx_mtp_depth_fixed = profile.get("fixed", False)
    experiment = None
    if profile.get("instrument", False):
        spec = profile.get("policy")
        policy = PrefixPolicy(tuple(spec["cycle_ms"]), tuple(spec.get("temperatures", ()))) if spec else None
        experiment = Experiment(draft_width=profile.get("draft_width"),
                                forced_depth=profile.get("forced_depth"), policy=policy)
    language._flash1017_experiment = experiment
    return experiment


def prepare(processor, prompt):
    if "tokens" in prompt:
        return list(prompt["tokens"])
    messages = prompt.get("messages", [{"role": "user", "content": prompt.get("text", "")}])
    tokens = processor.apply_chat_template(messages, tokenize=True,
                                           enable_thinking=prompt.get("thinking", False),
                                           tools=prompt.get("tools"))
    wanted = prompt.get("prompt_tokens")
    if wanted:
        # Throughput corpus is explicitly synthetic. Insert source tokens
        # inside the user message, preserving the full chat framing and query.
        body = prompt.get("context", "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\n\n")
        filler = processor.tokenizer.encode(body, add_special_tokens=False)
        if wanted < len(tokens) or not filler:
            raise ValueError("Requested prompt length cannot contain the full query")
        count = wanted - len(tokens)
        marker = processor.tokenizer.encode("<｜User｜>", add_special_tokens=False)
        positions = [i for i in range(len(tokens)) if tokens[i:i + len(marker)] == marker]
        if len(positions) != 1 or len(messages) != 1:
            raise ValueError("Synthetic throughput padding needs one user message")
        at = positions[0] + len(marker)
        tokens = tokens[:at] + (filler * (count // len(filler) + 1))[:count] + tokens[at:]
    return tokens


def run_one(model, processor, prompt, profile, repeat, seed):
    from mlx_lm.generate import BatchGenerator
    from omlx.utils.sampling import make_sampler
    from omlx.patches.mlx_lm_mtp import batch_generator

    experiment = configure(model, profile)
    tokens = prepare(processor, prompt)
    language = model.language_model
    count = prompt.get("max_tokens", 256)
    sampler = make_sampler(temp=prompt.get("temperature", 0),
                           top_p=prompt.get("top_p", 1), top_k=0, min_p=0)
    stops = None if prompt.get("force_length", False) else [[processor.tokenizer.eos_token_id]]
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(seed)
    meter = Meter().start()
    started = time.perf_counter()
    arrivals, generated = [], []
    finish_reason = None
    mtp_stats = []
    original_log = batch_generator._log_mtp_stats

    def capture_stats(uid, stats, reason):
        mtp_stats.append(asdict(stats) | {"finish_reason": reason})
        original_log(uid, stats, reason)

    batch_generator._log_mtp_stats = capture_stats
    cache = language.make_cache()
    prefill_step = profile.get("prefill_step", 2048)
    generator = BatchGenerator(language, max_tokens=count, stop_tokens=stops,
                               sampler=sampler, prefill_step_size=prefill_step,
                               completion_batch_size=1, prefill_batch_size=1)
    try:
        # Mirror oMLX's external cache-only prefill. This is necessary to
        # exercise CED; mlx-lm's bare BatchGenerator does not call that hook.
        # Match this path for EVERY profile, including stock and MTP-off.
        with mx.stream(generator.stream):
            for start in range(0, len(tokens) - 1, prefill_step):
                chunk = tokens[start:min(start + prefill_step, len(tokens) - 1)]
                language._omlx_prefill(mx.array([chunk]), cache=cache, skip_lm_head=True)
                mx.eval([item.state for item in cache])
            prefilled = time.perf_counter()
        generator.insert([tokens[-1:]], caches=[cache], all_tokens=[tokens[:-1]])
        while len(generated) < count:
            _, responses = generator.next()
            now = time.perf_counter()
            for response in responses:
                generated.append(int(response.token))
                arrivals.append(now - started)
                if response.finish_reason is not None:
                    finish_reason = response.finish_reason
            if finish_reason is not None:
                break
            if now - started > prompt.get("timeout_seconds", 1800):
                raise TimeoutError("Generation exceeded the explicit per-prompt limit")
            if not responses and not len(generator._generation_batch) and not generator._unprocessed_sequences:
                raise RuntimeError("Generator exhausted before a finishing response")
        mx.synchronize(generator.stream)
        ended = time.perf_counter()
        resources = meter.stop()
    finally:
        batch_generator._log_mtp_stats = original_log
        generator.close()
        if not meter.done.is_set():
            meter.stop()
    first = arrivals[0] if arrivals else None
    duration = ended - started
    if profile.get("mtp", True) and len(generated) > 8 and not mtp_stats:
        raise RuntimeError("MTP was requested but no completed MTP activity was recorded")
    eos = processor.tokenizer.eos_token_id
    first_eos = generated.index(eos) if eos in generated else None
    result = {
        "type": "run", "prompt_id": prompt["id"], "profile": profile,
        "repeat": repeat, "seed": seed, "input_tokens": len(tokens),
        "input_tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "generated_tokens": len(generated), "token_ids": generated,
        "text": processor.tokenizer.decode(generated, skip_special_tokens=False),
        "finish_reason": finish_reason, "total_seconds": duration,
        "first_eos_position": first_eos,
        "fixed_length_valid": (first_eos is None or first_eos == len(generated) - 1)
                              if prompt.get("force_length", False) else None,
        "prefill_seconds": prefilled - started, "ttft_seconds": first,
        "decode_tokens_per_second": ((len(generated) - 1) / (arrivals[-1] - first)
                                      if len(generated) > 1 and arrivals[-1] > first else None),
        "arrival_seconds": arrivals, "resources": resources,
        "events": experiment.events if experiment else [],
        "mtp_stats": mtp_stats,
        "cache_state": "fresh KV; OS file cache and compiled kernels not purged",
    }
    del generator, cache
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        raise ValueError("Output already exists; use a new name to preserve evidence")
    if not args.allow_low_memory and psutil.virtual_memory().available < 210 * 2**30:
        raise MemoryError("Less than 210 GiB available; unload other models first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from omlx.patches.deepseek_v41 import apply_patch
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.mlx_lm_mtp import batch_generator, set_mtp_active
    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import maybe_apply_pre_load_patches
    maybe_apply_pre_load_patches(
        str(args.model),
        ModelSettings(mtp_enabled=True, mtp_adaptive_max_depth=3,
                      deepseek_v41_engram_ssd_offload=True),
        for_vlm=True,
    )
    apply_patch()
    set_mtp_active(True)
    batch_generator.apply()
    from .memory import prepare_memory
    memory_policy = prepare_memory(args.output.with_suffix(".load.jsonl"), plan.get("cache_limit_bytes", 8 * 2**30))

    def write(record):
        with args.output.open("a") as file:
            file.write(json.dumps(record) + "\n")
        if record["type"] == "run":
            print(json.dumps({key: record[key] for key in ("prompt_id", "profile", "generated_tokens", "ttft_seconds", "decode_tokens_per_second", "total_seconds")}), flush=True)

    provenance = json.loads((Path(__file__).parents[1] / "provenance.json").read_text())
    import omlx
    runtime_root = Path(omlx.__file__).parent
    runtime_hashes = {
        str(path.relative_to(runtime_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(runtime_root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".so", ".dylib", ".metallib"}
    }
    experiment_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(Path(__file__).parent.glob("*.py"))}
    write({"type": "manifest", "created_at": time.time(), "model": str(args.model.resolve()),
           "plan": plan, "provenance": provenance, "platform": platform.platform(),
           "device": mx.device_info(), "pid": os.getpid(), "memory_policy": memory_policy,
           "runtime_files_sha256": runtime_hashes, "experiment_files_sha256": experiment_hashes,
           "packages": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "mlx-vlm")}})
    started = time.perf_counter()
    meter = Meter().start()
    model = None
    try:
        model, processor = load(args.model, engram_ssd_offload=True, attention_output_quantization=False)
        write({"type": "load", "seconds": time.perf_counter() - started, "resources": meter.stop()})
        print("MODEL_LOADED", flush=True)
        for job in plan["jobs"]:
            profile = plan["profiles"][job["profile"]] | {"name": job["profile"]}
            prompt = plan["prompts"][job["prompt"]] | {"id": job["prompt"]}
            result = run_one(model, processor, prompt, profile,
                             job.get("repeat", 0), job.get("seed", 1017))
            result["warmup"] = job.get("warmup", False)
            write(result)
    except BaseException as exc:
        write({"type": "error", "exception": repr(exc)})
        raise
    finally:
        if not meter.done.is_set():
            meter.stop()
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()
