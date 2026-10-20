# SPDX-License-Identifier: Apache-2.0
"""Run one frozen installed-wheel performance process via public streaming HTTP.

Run under the selected wheel's isolated Python with --process-id, --arm,
--model, --wheel, --plan, --base-path, --output and an unused --port. This tool
does not modify inference functions. Importing it is CPU-only and does not
start a model or server. The separate quality harness supplies configure and
the request allowlist, and both source files must be pinned by the final plan.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path

CHECKPOINT_METADATA = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def load_quality_harness():
    path = Path(__file__).with_name("run_installed_quality.py")
    spec = importlib.util.spec_from_file_location("performance_quality_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def native_chat_keepalive(event):
    """Accept only the pinned server's exact empty, non-witness frame."""
    if event.get("model") != "keepalive":
        return False
    valid = (
        set(event) == {"id", "object", "created", "model", "choices"}
        and type(event["id"]) is str
        and re.fullmatch(r"chatcmpl-[0-9a-f]{8}", event["id"]) is not None
        and event["object"] == "chat.completion.chunk"
        and type(event["created"]) is int
        and event["created"] == 0
        and type(event["choices"]) is list
        and len(event["choices"]) == 1
    )
    if valid:
        choice = event["choices"][0]
        valid = (
            type(choice) is dict
            and set(choice) == {"index", "delta", "finish_reason"}
            and type(choice["index"]) is int
            and choice["index"] == 0
            and choice["finish_reason"] is None
            and type(choice["delta"]) is dict
            and choice["delta"] == {"role": "assistant", "content": ""}
        )
    if not valid:
        raise ValueError("Malformed native keepalive frame")
    return True


def consume_sse(
    lines,
    row,
    started_ns,
    clock=time.perf_counter_ns,
    expected_model="DeepSeek-V4.1-Flash-1017",
):
    """Consume complete SSE frames, retaining partial results if a frame fails.

    ``lines`` yields decoded lines without terminators. The caller supplies the
    request-start timestamp and persists ``row`` even if this function raises.
    A role-only event never starts the visible-content clock. We require a
    terminal DONE frame, a finish reason and a final usage object, and reject
    unfinished frames or data after DONE instead of accepting partial answers.
    """
    row.update(text="", reasoning_content="", finish_reason=None, usage=None)
    row.update(
        sse_events=[], stream_done=False, first_content_ns=None, last_content_ns=None
    )
    data_lines = []
    done = False
    for line in lines:
        if not isinstance(line, str):
            raise TypeError("SSE iterator must yield decoded text lines")
        if line.startswith(":"):
            continue
        if line:
            if done:
                raise ValueError("SSE data after terminal DONE")
            field, separator, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data" and separator:
                data_lines.append(value)
            elif field == "event" and value == "error":
                row["invalid_sse_event_name"] = value
                raise ValueError("Explicit SSE error event")
            elif field not in {"id", "event", "retry"}:
                raise ValueError(f"Unexpected SSE field {field!r}")
            continue
        if not data_lines:
            continue
        received_ns = clock()
        data = "\n".join(data_lines)
        data_lines = []
        if data == "[DONE]":
            done = True
            row["stream_done"] = True
            row["done_ns"] = received_ns
            row["wall_seconds"] = (received_ns - started_ns) / 1e9
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            row["malformed_sse_data"] = data
            raise ValueError("Malformed JSON in SSE event") from exc
        if not isinstance(event, dict) or "error" in event:
            row["invalid_sse_event"] = event
            raise ValueError("Invalid or error SSE event")
        row["sse_events"].append({"received_ns": received_ns, "data": event})
        if native_chat_keepalive(event):
            # Preserve observation only: never set content, usage, finish or
            # timing witnesses, and never restart the request's clock.
            continue
        if event.get("model") != expected_model:
            raise ValueError("SSE response model differs from requested model")
        if event.get("usage") is not None:
            if not isinstance(event["usage"], dict):
                raise ValueError("SSE usage must be an object")
            row["usage"] = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected one choice or an empty usage-only choice list")
        if not choices:
            if event.get("usage") is None:
                raise ValueError("Empty SSE choices without usage")
            continue
        choice = choices[0]
        if (
            not isinstance(choice, dict)
            or type(choice.get("index")) is not int
            or choice["index"] != 0
        ):
            raise ValueError("Unexpected SSE choice index")
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise TypeError("SSE delta must be an object")
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        content = "" if content is None else content
        reasoning = "" if reasoning is None else reasoning
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise TypeError("SSE content/reasoning must be text")
        if row["finish_reason"] is not None and (content or reasoning):
            raise ValueError("SSE content after finish reason")
        if delta.get("tool_calls") or delta.get("function_call"):
            raise ValueError("Unexpected tool output in a text-only workload")
        row["reasoning_content"] += reasoning
        if content:
            row["text"] += content
            row["last_content_ns"] = received_ns
            if row["first_content_ns"] is None:
                row["first_content_ns"] = received_ns
                row["client_ttft_seconds"] = (received_ns - started_ns) / 1e9
        finish = choice.get("finish_reason")
        if finish is not None:
            if row["finish_reason"] is not None:
                raise ValueError("Multiple SSE finish reasons")
            row["finish_reason"] = finish
    if data_lines:
        raise ValueError("Truncated SSE frame at EOF")
    if not done:
        raise ValueError("SSE stream ended without DONE")
    if row["finish_reason"] is None:
        raise ValueError("SSE stream ended without finish reason")
    if row["usage"] is None:
        raise ValueError("SSE stream ended without usage")
    if row["first_content_ns"] is None:
        raise ValueError("SSE stream contained no visible assistant content")
    return row


def validate_response(row, job):
    if row["finish_reason"] not in {"stop", "length"}:
        raise ValueError("Unexpected finish reason")
    if row["reasoning_content"]:
        raise ValueError("Unexpected reasoning output with thinking disabled")
    usage = row["usage"]
    if (
        type(usage.get("prompt_tokens")) is not int
        or usage["prompt_tokens"] != job["expected_prompt_tokens"]
    ):
        raise ValueError("Actual prompt token count differs from frozen fixture")
    completion_tokens = usage.get("completion_tokens")
    if (
        type(completion_tokens) is not int
        or not 0 < completion_tokens <= job["max_tokens"]
    ):
        raise ValueError("Invalid completion token count")
    details = usage.get("prompt_tokens_details")
    if (
        not isinstance(details, dict)
        or type(details.get("cached_tokens")) is not int
        or details["cached_tokens"] != 0
    ):
        raise ValueError("Expected explicit zero prefix-cache hits")
    for name in (
        "prompt_eval_duration",
        "generation_duration",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    ):
        value = usage.get(name)
        if type(value) not in (int, float) or not 0 < value < float("inf"):
            raise ValueError(f"Missing or invalid native timing: {name}")
    row["output_sha256"] = digest(row["text"].encode())
    row["completed_answer_latency_eligible"] = row["finish_reason"] == "stop"


def verify_wheel(package, wheel_path, expected_sha):
    """Bind all installed Python/native package files to the supplied wheel."""
    if digest(wheel_path.read_bytes()) != expected_sha:
        raise ValueError("Wheel archive differs from frozen plan")
    actual = {
        str(path.relative_to(package)): digest(path.read_bytes())
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".so", ".metallib", ".pyd"}
    }
    with zipfile.ZipFile(wheel_path) as wheel:
        expected = {
            name.removeprefix("omlx/"): digest(wheel.read(name))
            for name in wheel.namelist()
            if name.startswith("omlx/")
            and Path(name).suffix in {".py", ".so", ".metallib", ".pyd"}
        }
    if not expected or actual != expected:
        different = sorted(
            k
            for k in actual.keys() | expected.keys()
            if actual.get(k) != expected.get(k)
        )
        raise ValueError(f"Installed package differs from wheel: {different}")
    return {
        "python_sources": {k: v for k, v in actual.items() if k.endswith(".py")},
        "native_assets": {k: v for k, v in actual.items() if not k.endswith(".py")},
        "installed_manifest_sha256": digest(canonical(actual)),
    }


def normalized_packages(records):
    """Normalize distribution names, reject collisions, exclude only oMLX."""
    result, seen = {}, set()
    for name, version in records:
        if (
            type(name) is not str
            or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name) is None
            or type(version) is not str
            or not version
        ):
            raise ValueError("Invalid installed distribution name/version")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in seen:
            raise ValueError(f"Duplicate normalized distribution: {normalized}")
        seen.add(normalized)
        if normalized != "omlx":
            result[normalized] = version
    return dict(sorted(result.items()))


def validate_shared_environment_schema(plan):
    expected = plan.get("shared_environment")
    if type(expected) is not dict or set(expected) != {
        "packages",
        "python",
        "model_metadata_sha256",
        "native_assets",
    }:
        raise ValueError("Frozen plan must declare the complete shared_environment")
    packages = expected["packages"]
    if (
        type(packages) is not dict
        or not packages
        or normalized_packages(packages.items()) != packages
    ):
        raise ValueError("Expected package map must be normalized and exclude omlx")
    if type(expected["python"]) is not str or not expected["python"]:
        raise ValueError("Expected full Python version must be declared")
    metadata = expected["model_metadata_sha256"]
    if type(metadata) is not dict or set(metadata) != set(CHECKPOINT_METADATA):
        raise ValueError("Expected checkpoint metadata inventory is incomplete")
    for name, value in metadata.items():
        if value is None and name == "generation_config.json":
            continue
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"Invalid expected checkpoint metadata hash: {name}")
    assets = expected["native_assets"]
    if type(assets) is not dict or not assets:
        raise ValueError("Expected native asset map must be declared")
    for name, value in assets.items():
        if (
            type(name) is not str
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or str(Path(name)) != name
            or Path(name).suffix not in {".so", ".metallib", ".pyd"}
            or type(value) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise ValueError(f"Invalid expected native asset: {name!r}")
    return expected


def verify_shared_environment(
    plan, model_path, *, packages, python_version, native_assets
):
    """Check the complete declared environment before loading model weights.

    Package versions are identity metadata, not dependency-binary hashes. The
    native map is the wheel-verified oMLX asset map; checkpoint weight shards
    are deliberately not read by this check.
    """
    expected = validate_shared_environment_schema(plan)
    actual = {
        "packages": normalized_packages(packages),
        "python": python_version,
        "model_metadata_sha256": {
            name: (
                digest((model_path / name).read_bytes())
                if (model_path / name).is_file()
                else None
            )
            for name in CHECKPOINT_METADATA
        },
        "native_assets": dict(native_assets),
    }
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"Shared environment differs from frozen plan: {name}")
    return actual


def validate_timeouts(plan):
    """HTTPX limits I/O inactivity; a separate supervisor owns the wall bound."""
    wall = plan.get("process_wall_timeout_seconds")
    http = plan.get("http", {})
    inactivity = http.get("io_inactivity_timeout_seconds")
    if type(wall) is not int or wall <= 0:
        raise ValueError("A positive integer process_wall_timeout_seconds is required")
    if (
        type(inactivity) is not int
        or inactivity <= 0
        or http.get("timeout_semantics")
        != "per-I/O inactivity, not a cumulative request deadline"
        or "request_timeout_seconds" in http
    ):
        raise ValueError("Declare the positive per-I/O inactivity timeout explicitly")
    return wall, inactivity


def allocator_observation(mx):
    """Read counters only; never synchronize, evaluate or reset the peak."""
    return {
        "mlx_active_memory_bytes": mx.get_active_memory(),
        "mlx_cache_memory_bytes": mx.get_cache_memory(),
        "mlx_peak_memory_bytes": mx.get_peak_memory(),
        "mlx_peak_memory_scope": (
            "Cumulative process MLX allocator peak, including load/warmup; "
            "not a measured-request-only peak or full system resident memory"
        ),
    }


def verify_quality_report(plan, workspace_root):
    """Require the exact approved 618-pair report for these same two wheels."""
    path_value = plan.get("quality_report_path")
    expected_sha = plan.get("quality_report_sha256")
    if not path_value or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("Final quality report path and hash must be frozen first")
    path = workspace_root / path_value
    data = path.read_bytes()
    if digest(data) != expected_sha:
        raise ValueError("Quality report differs from frozen plan")
    report = json.loads(data)
    if (
        report.get("planned_acceptance", {}).get("status") != "passed"
        or report.get("integrity", {}).get("status") != "verified"
        or report.get("graded_pairs") != 618
        or report.get("paired_responses") != 618
        or report.get("observed_exact_response_parity_established") is not True
        or report.get("observed_task_score_preservation_established") is not True
    ):
        raise ValueError("Final 618-pair quality/parity gate has not passed")
    for arm in ("baseline", "candidate"):
        build = report.get("builds", {}).get(arm, {})
        for key in ("version", "wheel_sha256"):
            if not plan[arm].get(key) or build.get(key) != plan[arm][key]:
                raise ValueError(
                    f"Quality report was not approved for this {arm} {key}"
                )
    return path


def load_frozen_inputs(args, support, support_path):
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if (
        plan.get("schema") != "installed-http-performance-v1"
        or plan.get("status") != "frozen_before_any_model_run"
    ):
        raise ValueError("This runner requires a reviewed, frozen performance plan")
    if digest(Path(__file__).read_bytes()) != plan.get("performance_harness_sha256"):
        raise ValueError("Performance harness differs from frozen plan")
    if digest(support_path.read_bytes()) != plan.get("quality_harness_sha256"):
        raise ValueError("Imported quality harness differs from frozen plan")
    validate_shared_environment_schema(plan)
    validate_timeouts(plan)
    quality_report_path = verify_quality_report(plan, args.workspace_root)
    expected_cache = {
        "cache.enabled": False,
        "cache.hot_cache_max_size": "0",
        "cache.hot_cache_only": False,
    }
    if plan["cache_policy"]["global_settings_overrides"] != expected_cache:
        raise ValueError(
            "This qualification requires prefix reuse disabled in both arms"
        )
    factory = plan["request_factory"]
    tree = ast.parse(support_path.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "request_for"
    )
    if (
        digest(ast.dump(fn, include_attributes=False).encode())
        != factory["request_for_ast_sha256"]
    ):
        raise ValueError("Request allowlist changed")
    if factory["constants"] != {"MODEL_ID": support.MODEL_ID, "SEED": support.SEED}:
        raise ValueError("Request model or seed changed")
    paths = {}
    for name, record in plan["artifacts"].items():
        path = args.workspace_root / record["path"]
        if digest(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Frozen artifact changed: {name}")
        paths[name] = path
    paths["quality_report"] = quality_report_path
    pick = lambda prefix: next(
        path for name, path in paths.items() if name.startswith(prefix)
    )
    provenance = json.loads(pick("fixture-provenance-").read_bytes())
    tokenizer_record = provenance["tokenizer"]
    if args.model.resolve() != Path(tokenizer_record["path"]).parent.resolve():
        raise ValueError("Model directory differs from the pinned fixture checkpoint")
    if (
        digest((args.model / "tokenizer.json").read_bytes())
        != tokenizer_record["sha256"]
    ):
        raise ValueError("Checkpoint tokenizer differs from frozen fixture")
    jobs_list = read_jsonl(pick("requests-"))
    payloads_list = read_jsonl(pick("payloads-"))
    if len(jobs_list) != 9 or len(payloads_list) != 9:
        raise ValueError("Expected eight measured fixtures and one excluded warmup")
    jobs = {row["task_id"]: row for row in jobs_list}
    payloads = {row["task_id"]: row for row in payloads_list}
    if len(jobs) != 9 or jobs.keys() != payloads.keys():
        raise ValueError("Duplicate or mismatched fixture IDs")
    process = next(p for p in plan["processes"] if p["process_id"] == args.process_id)
    if (
        process["arm"] != args.arm
        or not process["fresh_process"]
        or not process["fresh_state_directory"]
    ):
        raise ValueError("Selected arm/process does not match frozen plan")
    sequence = [process["warmup_task_id"], *process["measured_task_ids"]]
    if len(sequence) != 9 or set(sequence) != jobs.keys():
        raise ValueError("Invalid frozen process sequence")
    for index, task_id in enumerate(sequence):
        job, record = jobs[task_id], payloads[task_id]
        if job["warmup"] != (index == 0) or job["max_tokens"] != (
            64 if index == 0 else 512
        ):
            raise ValueError("Warmup or output budget changed")
        payload = support.request_for(job)
        payload.update(stream=True, stream_options={"include_usage": True})
        if (
            payload != record["request"]
            or digest(canonical(payload)) != record["request_sha256"]
        ):
            raise ValueError("Actual request differs from frozen payload")
    return plan_bytes, plan, process, jobs, payloads, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--process-id", required=True)
    for name in ("model", "wheel", "plan", "base-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    support, support_path = load_quality_harness()
    plan_bytes, plan, process, jobs, payloads, paths = load_frozen_inputs(
        args, support, support_path
    )
    import httpx
    import omlx
    import psutil
    from omlx import cli
    from omlx.settings import GlobalSettings

    expected_version = plan[args.arm]["version"]
    if importlib.metadata.version("omlx") != expected_version:
        raise ValueError("Unexpected installed runtime version")
    package = Path(omlx.__file__).resolve().parent
    if "site-packages/omlx" not in str(package):
        raise ValueError("Use an installed wheel outside the source checkout")
    source_identity = verify_wheel(package, args.wheel, plan[args.arm]["wheel_sha256"])
    package_versions = [
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    ]
    shared_environment = verify_shared_environment(
        plan,
        args.model,
        packages=package_versions,
        python_version=sys.version,
        native_assets=source_identity["native_assets"],
    )
    process_wall_timeout, io_inactivity_timeout = validate_timeouts(plan)
    import mlx.core as mx

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite or silently resume a result run")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    if psutil.virtual_memory().available < 210 * 2**30:
        raise MemoryError("At least 210 GiB available memory is needed")
    device = mx.device_info()
    for key in ("device_name", "memory_size", "architecture"):
        if device[key] != plan["observed_reference_device"][key]:
            raise ValueError(f"Device differs from frozen qualification: {key}")
    limit = min(
        device["max_recommended_working_set_size"], device["memory_size"] - 32 * 2**30
    )
    if limit != plan["memory_limit_bytes"]:
        raise ValueError("Memory limit differs from frozen qualification")
    mx.set_wired_limit(limit)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(8 * 2**30)
    base = support.configure(args)
    settings = GlobalSettings.load(base_path=base)
    settings.cache.enabled = False
    settings.cache.hot_cache_max_size = "0"
    settings.cache.hot_cache_only = False
    settings.save()
    args.output.mkdir(parents=True, exist_ok=True)
    settings_files = ("settings.json", "model_settings.json")
    receipt = {
        "schema": "flash1017-performance-generation/1",
        "arm": args.arm,
        "process_id": args.process_id,
        "block": process["block"],
        "status": "starting",
        "plan_sha256": digest(plan_bytes),
        "artifact_hashes": {
            name: digest(path.read_bytes()) for name, path in paths.items()
        },
        "total_measured_tasks": 8,
        "measured_completed": 0,
        "attempts_completed": 0,
        "errors": 0,
        "truncated": 0,
        "runtime_version": expected_version,
        "runtime_source": "installed wheel",
        "wheel_sha256": digest(args.wheel.read_bytes()),
        "quality_report_sha256": plan["quality_report_sha256"],
        "model_directory": str(args.model.resolve()),
        "model_metadata_sha256": shared_environment["model_metadata_sha256"],
        "shared_environment": shared_environment,
        "shared_environment_sha256": digest(canonical(shared_environment)),
        **source_identity,
        "packages": dict(package_versions),
        "python": sys.version,
        "device": device,
        "memory_limit_bytes": limit,
        "process_wall_timeout_seconds": process_wall_timeout,
        "http_io_inactivity_timeout_seconds": io_inactivity_timeout,
        "timeout_scope": "HTTPX per-I/O inactivity; external supervisor must enforce process wall timeout",
        "settings_sha256": {
            name: digest((base / name).read_bytes()) for name in settings_files
        },
        "started_unix": time.time(),
        "responses": "responses.jsonl",
        "harness_sha256": digest(Path(__file__).read_bytes()),
        "quality_harness_sha256": digest(support_path.read_bytes()),
    }
    receipt_path = args.output / "receipt.json"
    controllers = []

    def save():
        support.atomic_json(receipt_path, receipt)

    def runtime_snapshot(require_loaded=True):
        from omlx import server

        memory = allocator_observation(mx)
        entry = server._server_state.engine_pool.get_entry(support.MODEL_ID)
        engine = None if entry is None else entry.engine
        if engine is None or getattr(engine, "_vlm_model", None) is None:
            if require_loaded:
                raise RuntimeError("Expected a loaded benchmark model")
            return {
                "loaded": False,
                "memory_available_gib": psutil.virtual_memory().available / 2**30,
                **memory,
            }
        model = engine._vlm_model
        scheduler = engine._engine.engine.scheduler
        language = model.language_model
        model_config = language._config
        controller = getattr(model, "_flash1017_runtime_controller", None)
        if controller is not None and controller not in controllers:
            controllers.append(controller)
        confidence = getattr(language, "_flash1017_experiment", None)
        result = {
            "loaded": True,
            **memory,
            "prefix_cache_enabled": bool(engine.prefix_cache_enabled),
            "prefill_step_size": scheduler.config.prefill_step_size,
            "max_num_seqs": scheduler.config.max_num_seqs,
            "attention_output_quantization": getattr(
                model_config, "attention_output_quantization", False
            ),
            "flash1017_enabled": getattr(model_config, "flash1017_enabled", False),
            "mtp_depth": getattr(language, "_omlx_mtp_depth", None),
            "mtp_depth_fixed": getattr(language, "_omlx_mtp_depth_fixed", None),
            "confidence": (
                None
                if confidence is None
                else {
                    "cycles": confidence.cycles,
                    "selected_depths": list(confidence.selected_depths),
                }
            ),
            "runtime": None if controller is None else controller.snapshot(),
            "engram_prefetch_bytes": getattr(
                getattr(language, "_engram_prefetch", None), "budget_bytes", None
            ),
            "memory_available_gib": psutil.virtual_memory().available / 2**30,
        }
        if result["prefix_cache_enabled"]:
            raise RuntimeError("Prefix cache unexpectedly active")
        if args.arm == "candidate" and controller is None:
            raise RuntimeError("Candidate profile did not activate")
        if args.arm == "baseline" and (
            controller is not None or confidence is not None
        ):
            raise RuntimeError("Baseline has candidate modifications active")
        expected = plan[args.arm].get("expected_runtime")
        if not expected:
            raise RuntimeError(
                "Frozen plan must declare expected runtime flags for each arm"
            )
        for name, value in expected.items():
            actual = result.get(name)
            if name == "confidence_enabled":
                actual = confidence is not None
            elif name == "replay_installed":
                actual = (
                    controller is not None and result["runtime"].get("cone") is not None
                )
            if actual != value:
                raise RuntimeError(
                    f"Runtime configuration mismatch for {name}: {actual!r} != {value!r}"
                )
        return result

    def safe_settings():
        effective = GlobalSettings.load(base_path=base)
        return {
            "cache_enabled": effective.cache.enabled,
            "hot_cache_max_size": effective.cache.hot_cache_max_size,
            "hot_cache_only": effective.cache.hot_cache_only,
            "max_concurrent_requests": effective.scheduler.max_concurrent_requests,
            "max_context_window": effective.sampling.max_context_window,
        }

    def client_work():
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{args.port}",
                timeout=io_inactivity_timeout,
                trust_env=False,
            ) as client:
                for _ in range(240):
                    try:
                        if client.get("/health").is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Server did not become healthy")
                receipt["runtime_initial"] = runtime_snapshot(require_loaded=False)
                receipt["poststart_settings_sha256"] = {
                    name: digest((base / name).read_bytes()) for name in settings_files
                }
                receipt["safe_effective_settings"] = safe_settings()
                if (
                    receipt["safe_effective_settings"]["cache_enabled"]
                    or receipt["safe_effective_settings"]["hot_cache_max_size"] != "0"
                ):
                    raise RuntimeError("Expected prefix-cache settings disabled")
                receipt["status"] = "running"
                save()
                sequence = [process["warmup_task_id"], *process["measured_task_ids"]]
                with (args.output / "responses.jsonl").open("x") as stream:
                    for index, task_id in enumerate(sequence):
                        if digest(args.plan.read_bytes()) != receipt["plan_sha256"]:
                            raise RuntimeError("Frozen plan changed during execution")
                        if any(
                            digest(path.read_bytes())
                            != receipt["artifact_hashes"][name]
                            for name, path in paths.items()
                        ):
                            raise RuntimeError(
                                "Frozen fixture changed during execution"
                            )
                        if psutil.virtual_memory().available < 8 * 2**30:
                            raise MemoryError(
                                "Available memory below 8 GiB between requests"
                            )
                        job = jobs[task_id]
                        row = {
                            "task_id": task_id,
                            "suite": job["suite"],
                            "arm": args.arm,
                            "block": process["block"],
                            "process_id": args.process_id,
                            "index": index,
                            "measured_index": None if index == 0 else index - 1,
                            "warmup": index == 0,
                            "preceding_task_ids": sequence[:index],
                            "request_sha256": payloads[task_id]["request_sha256"],
                            "max_tokens": job["max_tokens"],
                            "text": "",
                            "finish_reason": None,
                        }
                        # Observers, serialization, artifact hashing and disk flushes
                        # are outside the timed HTTP interval.
                        try:
                            row["runtime_before"] = runtime_snapshot(
                                require_loaded=index > 0
                            )
                            body = canonical(payloads[task_id]["request"])
                            row["request_start_ns"] = time.perf_counter_ns()
                            try:
                                with client.stream(
                                    "POST",
                                    "/v1/chat/completions",
                                    content=body,
                                    headers={"Content-Type": "application/json"},
                                ) as response:
                                    row["http_status"] = response.status_code
                                    if not response.is_success:
                                        row["http_error_body"] = response.read().decode(
                                            errors="replace"
                                        )[:8192]
                                    response.raise_for_status()
                                    if not response.headers.get(
                                        "content-type", ""
                                    ).startswith("text/event-stream"):
                                        raise ValueError(
                                            "Expected text/event-stream response"
                                        )
                                    consume_sse(
                                        response.iter_lines(),
                                        row,
                                        row["request_start_ns"],
                                        expected_model=payloads[task_id]["request"][
                                            "model"
                                        ],
                                    )
                            finally:
                                row["http_interval_end_ns"] = time.perf_counter_ns()
                                row["http_interval_seconds"] = (
                                    row["http_interval_end_ns"]
                                    - row["request_start_ns"]
                                ) / 1e9
                            validate_response(row, job)
                        except Exception as exc:  # noqa: BLE001
                            # Retain the failed observation before aborting.
                            row["error"] = repr(exc)
                        try:
                            row["runtime_after"] = runtime_snapshot()
                        except Exception as exc:  # noqa: BLE001
                            # Observer errors also invalidate, and remain in, the run.
                            row["observer_error"] = repr(exc)
                            row.setdefault(
                                "error", "Runtime observer failed after request"
                            )
                        row["completed_unix"] = time.time()
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        receipt["attempts_completed"] += 1
                        receipt["measured_completed"] += int(index > 0)
                        receipt["errors"] += int("error" in row)
                        receipt["truncated"] += int(
                            index > 0 and row["finish_reason"] == "length"
                        )
                        receipt["last_task_id"] = task_id
                        receipt["runtime_latest"] = row.get("runtime_after")
                        if index == 0:
                            receipt["warmup"] = {
                                "task_id": task_id,
                                "excluded": True,
                                "usage": row.get("usage"),
                                "finish_reason": row["finish_reason"],
                                "runtime_after": row.get("runtime_after"),
                            }
                        save()
                        print(
                            json.dumps(
                                {
                                    "process_id": args.process_id,
                                    "task": task_id,
                                    "warmup": index == 0,
                                    "finish": row["finish_reason"],
                                    "client_ttft_seconds": row.get(
                                        "client_ttft_seconds"
                                    ),
                                    "wall_seconds": row.get("wall_seconds"),
                                    "error": row.get("error"),
                                }
                            ),
                            flush=True,
                        )
                        if row.get("error"):
                            raise RuntimeError(
                                "Failed observation retained; stopping without retry"
                            )
                receipt["status"] = "generated-awaiting-shutdown"
        except BaseException:  # noqa: BLE001
            # Persist interruptions before requesting normal server shutdown.
            receipt["status"] = "failed"
            receipt["error"] = traceback.format_exc()
        finally:
            save()
            os.kill(os.getpid(), signal.SIGINT)

    save()
    worker = threading.Thread(
        target=client_work, name="performance-http-client", daemon=True
    )
    worker.start()
    try:
        sys.argv = [
            "omlx",
            "serve",
            "--base-path",
            str(base),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--max-concurrent-requests",
            "1",
        ]
        cli.main()
    except KeyboardInterrupt:
        pass
    finally:
        worker.join(timeout=10)
        receipt["controllers_closed"] = all(
            controller.closed for controller in controllers
        )
        receipt["client_worker_stopped"] = not worker.is_alive()
        receipt["finished_unix"] = time.time()
        try:
            final_identity = verify_wheel(
                package, args.wheel, plan[args.arm]["wheel_sha256"]
            )
            receipt["final_installed_manifest_sha256"] = final_identity[
                "installed_manifest_sha256"
            ]
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            receipt["status"] = "failed"
            receipt["error"] = f"Installed package identity changed: {exc!r}"
        if (
            receipt["status"] == "generated-awaiting-shutdown"
            and receipt["controllers_closed"]
            and receipt["client_worker_stopped"]
        ):
            receipt["status"] = "generated"
        elif receipt["status"] != "failed":
            receipt["status"] = "failed"
            receipt["error"] = (
                "Server shutdown or client-worker completion was incomplete"
            )
        save()
    if receipt["status"] != "generated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
