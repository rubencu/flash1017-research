# SPDX-License-Identifier: Apache-2.0
"""Audit all four saved HTTP performance processes; never run inference/code.

All 16 pairs remain visible on failure, but aggregate speed summaries require
the entire frozen comparison to pass. Wheel archives independently bind the
recorded installed Python/native manifests. Secret-bearing settings files are
not loaded: safe settings and runtime flags are checked as saved observations.
"""

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

METRICS = {
    "client_ttft_seconds": "latency",
    "prompt_tokens_per_second": "rate",
    "generation_tokens_per_second": "rate",
}
METADATA_NAMES = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def geomean(values):
    values = list(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


class Audit:
    def __init__(self):
        self.errors = []
        self.checks = 0

    def check(self, condition, code, detail):
        self.checks += 1
        if not condition:
            self.errors.append({"code": code, "detail": detail})
        return bool(condition)

    def read(self, path, *, lines=False):
        try:
            data = Path(path).read_bytes()
            parsed = (
                [json.loads(line) for line in data.splitlines() if line.strip()]
                if lines
                else json.loads(data)
            )
            if not self.check(
                isinstance(parsed, list if lines else dict), "json_shape", str(path)
            ):
                return [] if lines else {}
            if lines and not self.check(
                all(isinstance(row, dict) for row in parsed),
                "jsonl_row_shape",
                str(path),
            ):
                return []
            return parsed
        except (OSError, ValueError) as exc:
            self.check(False, "unreadable_evidence", f"{path}: {exc}")
            return [] if lines else {}

    def file_hash(self, path, expected=None):
        try:
            actual = digest(Path(path).read_bytes())
        except OSError as exc:
            self.check(False, "missing_file", f"{path}: {exc}")
            return None
        if expected is not None:
            self.check(actual == expected, "file_hash", str(path))
        return actual


def normalized_packages(packages, audit, label):
    result = {}
    if not audit.check(
        isinstance(packages, dict) and bool(packages), "packages_missing", label
    ):
        return result
    for name, version in packages.items():
        key = re.sub(r"[-_.]+", "-", name).lower()
        audit.check(key not in result, "package_name_collision", label + ":" + key)
        result[key] = version
    result.pop("omlx", None)
    return result


def wheel_manifest(path, expected_hash, audit, arm):
    if path is None:
        audit.check(False, "wheel_missing", arm)
        return {}, {}
    audit.file_hash(path, expected_hash)
    try:
        with zipfile.ZipFile(path) as wheel:
            records = {
                name.removeprefix("omlx/"): digest(wheel.read(name))
                for name in wheel.namelist()
                if name.startswith("omlx/")
                and Path(name).suffix in {".py", ".so", ".metallib", ".pyd"}
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        audit.check(False, "wheel_unreadable", f"{arm}: {exc}")
        return {}, {}
    python = {name: sha for name, sha in records.items() if name.endswith(".py")}
    native = {name: sha for name, sha in records.items() if not name.endswith(".py")}
    audit.check(bool(python) and bool(native), "wheel_manifest_empty", arm)
    return python, native


def snapshot_valid(snapshot, expected, audit, label, require_loaded=True):
    if not audit.check(isinstance(snapshot, dict), "runtime_snapshot_missing", label):
        return
    audit.check(type(snapshot.get("loaded")) is bool, "runtime_loaded_flag", label)
    if not snapshot.get("loaded") and not require_loaded:
        return
    audit.check(snapshot.get("loaded") is True, "runtime_not_loaded", label)
    audit.check(
        snapshot.get("prefix_cache_enabled") is False, "runtime_prefix_cache", label
    )
    for name, value in expected.items():
        actual = snapshot.get(name)
        if name == "confidence_enabled":
            audit.check(
                "confidence" in snapshot, "runtime_flag_missing", label + ":confidence"
            )
            actual = snapshot.get("confidence") is not None
        elif name == "replay_installed":
            audit.check(
                "runtime" in snapshot, "runtime_flag_missing", label + ":runtime"
            )
            actual = (
                isinstance(snapshot.get("runtime"), dict)
                and snapshot["runtime"].get("cone") is not None
            )
        audit.check(actual == value, "runtime_flag", f"{label}:{name}")


def validate_row(row, job, payload, process, index, expected_runtime, audit):
    start_errors = len(audit.errors)
    label = process["process_id"] + ":" + str(job["task_id"])
    sequence = [process["warmup_task_id"], *process["measured_task_ids"]]
    for key, expected in {
        "task_id": job["task_id"],
        "suite": job["suite"],
        "arm": process["arm"],
        "block": process["block"],
        "process_id": process["process_id"],
        "index": index,
        "measured_index": None if index == 0 else index - 1,
        "warmup": index == 0,
        "preceding_task_ids": sequence[:index],
        "request_sha256": payload["request_sha256"],
        "max_tokens": job["max_tokens"],
        "http_status": 200,
        "stream_done": True,
    }.items():
        audit.check(row.get(key) == expected, "response_field", f"{label}:{key}")
    audit.check(
        not row.get("error") and not row.get("observer_error"), "response_error", label
    )
    text = row.get("text")
    audit.check(isinstance(text, str) and bool(text), "empty_response", label)
    audit.check(row.get("reasoning_content") == "", "unexpected_reasoning", label)
    audit.check(row.get("finish_reason") in {"stop", "length"}, "finish_reason", label)
    audit.check(
        row.get("completed_answer_latency_eligible")
        == (row.get("finish_reason") == "stop"),
        "completion_latency_eligibility",
        label,
    )
    if isinstance(text, str):
        audit.check(
            row.get("output_sha256") == digest(text.encode()), "output_hash", label
        )
    usage = row.get("usage") or {}
    audit.check(
        usage.get("prompt_tokens") == job["expected_prompt_tokens"],
        "prompt_tokens",
        label,
    )
    count = usage.get("completion_tokens")
    audit.check(
        type(count) is int and 0 < count <= job["max_tokens"],
        "completion_tokens",
        label,
    )
    audit.check(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens") == 0,
        "prefix_cache_hit",
        label,
    )
    for metric in (
        "prompt_eval_duration",
        "generation_duration",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    ):
        audit.check(positive(usage.get(metric)), "native_timing", label + ":" + metric)
    timestamps = [
        row.get(k)
        for k in (
            "request_start_ns",
            "first_content_ns",
            "last_content_ns",
            "done_ns",
            "http_interval_end_ns",
        )
    ]
    if audit.check(all(type(v) is int for v in timestamps), "client_timestamps", label):
        started, first, last, done, interval_end = timestamps
        audit.check(
            started < first <= last <= done <= interval_end,
            "client_timestamp_order",
            label,
        )
        for field, value in {
            "client_ttft_seconds": (first - started) / 1e9,
            "wall_seconds": (done - started) / 1e9,
            "http_interval_seconds": (interval_end - started) / 1e9,
        }.items():
            audit.check(
                positive(row.get(field))
                and math.isclose(row[field], value, rel_tol=1e-10, abs_tol=1e-9),
                "client_timing_arithmetic",
                label + ":" + field,
            )
    events = row.get("sse_events")
    if audit.check(
        isinstance(events, list) and bool(events), "sse_events_missing", label
    ):
        content, finishes, event_times, content_times = [], [], [], []
        last_usage = None
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("data"), dict):
                audit.check(False, "sse_event_shape", label)
                continue
            event_times.append(event.get("received_ns"))
            data = event["data"]
            if data.get("usage") is not None:
                last_usage = data["usage"]
            choices = data.get("choices")
            if not audit.check(isinstance(choices, list), "sse_choice_shape", label):
                continue
            for choice in choices:
                if not audit.check(isinstance(choice, dict), "sse_choice_shape", label):
                    continue
                delta = choice.get("delta") or {}
                if not audit.check(isinstance(delta, dict), "sse_delta_shape", label):
                    continue
                piece = delta.get("content")
                if piece:
                    if isinstance(piece, str):
                        content.append(piece)
                        content_times.append(event.get("received_ns"))
                    else:
                        audit.check(False, "sse_content_type", label)
                audit.check(
                    not delta.get("reasoning_content") and not delta.get("tool_calls"),
                    "unexpected_sse_output",
                    label,
                )
                if choice.get("finish_reason") is not None:
                    finishes.append(choice["finish_reason"])
        audit.check("".join(content) == text, "sse_text_reconstruction", label)
        audit.check(
            finishes == [row.get("finish_reason")], "sse_finish_reconstruction", label
        )
        audit.check(last_usage == usage, "sse_usage_reconstruction", label)
        audit.check(
            bool(content_times)
            and content_times[0] == row.get("first_content_ns")
            and content_times[-1] == row.get("last_content_ns"),
            "sse_first_content",
            label,
        )
        if all(type(t) is int for t in event_times):
            audit.check(event_times == sorted(event_times), "sse_event_order", label)
            if (
                type(row.get("request_start_ns")) is int
                and type(row.get("done_ns")) is int
            ):
                audit.check(
                    all(
                        row["request_start_ns"] <= t <= row["done_ns"]
                        for t in event_times
                    ),
                    "sse_event_bounds",
                    label,
                )
        else:
            audit.check(False, "sse_event_timestamps", label)
    snapshot_valid(
        row.get("runtime_before"),
        expected_runtime,
        audit,
        label + ":before",
        require_loaded=index > 0,
    )
    snapshot_valid(row.get("runtime_after"), expected_runtime, audit, label + ":after")
    return len(audit.errors) == start_errors


def summarize(plan_path, run_paths, wheel_paths, workspace_root):
    audit = Audit()
    plan_path, workspace_root = Path(plan_path), Path(workspace_root)
    plan = audit.read(plan_path)
    plan_sha = audit.file_hash(plan_path)
    audit.check(
        plan.get("schema") == "installed-http-performance-v1"
        and plan.get("status") == "frozen_before_any_model_run",
        "plan_not_frozen",
        str(plan_path),
    )
    audit.file_hash(Path(__file__), plan.get("analysis_harness_sha256", "MISSING"))
    wall_limit = plan.get("process_wall_timeout_seconds")
    io_limit = plan.get("http", {}).get("io_inactivity_timeout_seconds")
    audit.check(
        type(wall_limit) is int
        and wall_limit > 0
        and type(io_limit) is int
        and io_limit > 0
        and plan.get("http", {}).get("timeout_semantics")
        == "per-I/O inactivity, not a cumulative request deadline",
        "timeout_contract",
        "plan",
    )
    for kind in ("performance", "quality"):
        path = plan.get(kind + "_harness_path")
        if audit.check(bool(path), "harness_source_missing", kind):
            audit.file_hash(
                workspace_root / path, plan.get(kind + "_harness_sha256", "MISSING")
            )
    quality_path = workspace_root / plan.get("quality_report_path", "MISSING")
    quality = audit.read(quality_path)
    audit.file_hash(quality_path, plan.get("quality_report_sha256", "MISSING"))
    audit.check(
        quality.get("planned_acceptance", {}).get("status") == "passed"
        and quality.get("integrity", {}).get("status") == "verified"
        and quality.get("graded_pairs") == quality.get("paired_responses") == 618
        and quality.get("observed_exact_response_parity_established") is True
        and quality.get("observed_task_score_preservation_established") is True,
        "quality_prerequisite",
        str(quality_path),
    )
    shared = plan.get("shared_environment", {})
    audit.check(
        set(shared) == {"packages", "python", "model_metadata_sha256", "native_assets"},
        "shared_environment_missing",
        "plan",
    )
    audit.check(
        bool(shared.get("packages"))
        and bool(shared.get("python"))
        and bool(shared.get("native_assets")),
        "shared_environment_empty",
        "plan",
    )
    metadata = shared.get("model_metadata_sha256", {})
    audit.check(
        set(metadata) == METADATA_NAMES
        and all(
            isinstance(metadata.get(k), str) and len(metadata[k]) == 64
            for k in METADATA_NAMES - {"generation_config.json"}
        ),
        "checkpoint_metadata_missing",
        "plan",
    )
    manifests = {}
    for arm in ("baseline", "candidate"):
        build = plan.get(arm, {})
        for key in ("version", "wheel_sha256"):
            audit.check(
                bool(build.get(key))
                and build.get(key) == quality.get("builds", {}).get(arm, {}).get(key),
                "quality_build_identity",
                arm + ":" + key,
            )
        audit.check(
            bool(build.get("expected_runtime")), "runtime_expectation_missing", arm
        )
        manifests[arm] = wheel_manifest(
            wheel_paths.get(arm), build.get("wheel_sha256", "MISSING"), audit, arm
        )
        audit.check(
            manifests[arm][1] == shared.get("native_assets"), "native_asset_pin", arm
        )
    artifacts = {}
    artifact_hashes = {}
    for name, record in plan.get("artifacts", {}).items():
        path = workspace_root / record["path"]
        artifact_hashes[name] = audit.file_hash(path, record.get("sha256", "MISSING"))
        artifacts[name] = path
    artifact_hashes["quality_report"] = audit.file_hash(quality_path)

    def artifact(prefix, *, lines=False):
        matches = [p for n, p in artifacts.items() if n.startswith(prefix)]
        if not audit.check(len(matches) == 1, "artifact_inventory", prefix):
            return [] if lines else {}
        return audit.read(matches[0], lines=lines)

    jobs_list = artifact("requests-", lines=True)
    payload_list = artifact("payloads-", lines=True)
    rendered_list = artifact("rendered-inputs-", lines=True)
    provenance = artifact("fixture-provenance-")
    jobs = {j.get("task_id"): j for j in jobs_list}
    payloads = {p.get("task_id"): p for p in payload_list}
    rendered = {r.get("task_id"): r for r in rendered_list}
    audit.check(
        len(jobs_list)
        == len(jobs)
        == len(payload_list)
        == len(payloads)
        == len(rendered_list)
        == len(rendered)
        == 9
        and jobs.keys() == payloads.keys() == rendered.keys(),
        "fixture_inventory",
        "nine unique matching fixtures",
    )
    for task_id, job in jobs.items():
        payload = payloads.get(task_id, {})
        request = payload.get("request", {})
        audit.check(
            digest(canonical(request)) == payload.get("request_sha256"),
            "payload_hash",
            str(task_id),
        )
        constants = plan.get("request_factory", {}).get("constants", {})
        expected_request = {
            "model": constants.get("MODEL_ID"),
            "messages": job.get("messages"),
            "max_tokens": 64 if job.get("warmup") else 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": constants.get("SEED"),
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        audit.check(
            request == expected_request
            and job.get("max_tokens") == expected_request["max_tokens"]
            and bool(constants.get("MODEL_ID"))
            and type(constants.get("SEED")) is int,
            "payload_contract",
            str(task_id),
        )
        tokens = rendered.get(task_id, {}).get("tokens", [])
        token_hash = digest(json.dumps(tokens).encode())
        audit.check(
            len(tokens) == job.get("expected_prompt_tokens")
            and token_hash
            == job.get("expected_prompt_tokens_sha256")
            == rendered.get(task_id, {}).get("tokens_sha256"),
            "rendered_input",
            str(task_id),
        )
    audit.check(
        provenance.get("tokenizer", {}).get("sha256") == metadata.get("tokenizer.json")
        and bool(metadata.get("tokenizer.json")),
        "tokenizer_pin",
        "fixtures/checkpoint",
    )
    fixtures = {f["task_id"]: f for f in provenance.get("fixtures", [])}
    measured_ids = [task for task, job in jobs.items() if job.get("warmup") is False]
    groups = {"synthetic_1k": [], "natural_16k": [], "natural_64k": []}
    for task in measured_ids:
        fixture = fixtures.get(task, {})
        if (
            fixture.get("synthetic") is True
            and fixture.get("historical_prompt_tokens") == 1024
        ):
            groups["synthetic_1k"].append(task)
        elif fixture.get("synthetic") is False and fixture.get(
            "historical_prompt_tokens"
        ) in (16384, 65536):
            groups[
                (
                    "natural_16k"
                    if fixture["historical_prompt_tokens"] == 16384
                    else "natural_64k"
                )
            ].append(task)
        else:
            audit.check(False, "unknown_workload_group", task)
    audit.check(
        {k: len(v) for k, v in groups.items()}
        == {"synthetic_1k": 2, "natural_16k": 4, "natural_64k": 2},
        "workload_group_counts",
        "plan",
    )
    groups["natural_all"] = groups["natural_16k"] + groups["natural_64k"]
    groups["all_eight_mixed"] = measured_ids
    processes = plan.get("processes", [])
    audit.check(
        [(p.get("block"), p.get("arm")) for p in processes]
        == [(0, "baseline"), (0, "candidate"), (1, "candidate"), (1, "baseline")],
        "process_counterbalance",
        "plan",
    )
    process_ids = [p.get("process_id") for p in processes]
    audit.check(len(set(process_ids)) == 4, "process_inventory", "plan")
    audit.check(
        not set(run_paths) - set(process_ids),
        "unexpected_run",
        sorted(set(run_paths) - set(process_ids)),
    )
    seen_rows, row_valid, process_results, run_identities = {}, {}, [], {}
    previous_finished = None
    for proc in processes:
        pid, arm, block = proc["process_id"], proc["arm"], proc["block"]
        audit.check(
            proc.get("fresh_process") is True
            and proc.get("fresh_state_directory") is True,
            "fresh_process_contract",
            pid,
        )
        build = plan[arm]
        sequence = [proc["warmup_task_id"], *proc["measured_task_ids"]]
        audit.check(
            len(sequence) == len(set(sequence)) == 9 and set(sequence) == set(jobs),
            "process_sequence",
            pid,
        )
        if block == 1 and processes:
            audit.check(
                proc["measured_task_ids"]
                == list(reversed(processes[0]["measured_task_ids"])),
                "case_counterbalance",
                pid,
            )
        elif processes:
            audit.check(
                proc["measured_task_ids"] == processes[0]["measured_task_ids"],
                "case_order",
                pid,
            )
        directory = run_paths.get(pid)
        if not audit.check(directory is not None, "missing_run", pid):
            process_results.append({"process_id": pid, "status": "missing", "rows": 0})
            continue
        directory = Path(directory)
        receipt = audit.read(directory / "receipt.json")
        rows = audit.read(directory / "responses.jsonl", lines=True)
        run_identities[pid] = {
            "receipt_sha256": audit.file_hash(directory / "receipt.json"),
            "responses_sha256": audit.file_hash(directory / "responses.jsonl"),
        }
        expected_fields = {
            "schema": "flash1017-performance-generation/1",
            "arm": arm,
            "block": block,
            "process_id": pid,
            "status": "generated",
            "plan_sha256": plan_sha,
            "artifact_hashes": artifact_hashes,
            "total_measured_tasks": 8,
            "measured_completed": 8,
            "attempts_completed": 9,
            "errors": 0,
            "runtime_version": build.get("version"),
            "runtime_source": "installed wheel",
            "wheel_sha256": build.get("wheel_sha256"),
            "quality_report_sha256": plan.get("quality_report_sha256"),
            "harness_sha256": plan.get("performance_harness_sha256"),
            "quality_harness_sha256": plan.get("quality_harness_sha256"),
            "controllers_closed": True,
            "client_worker_stopped": True,
            "memory_limit_bytes": plan.get("memory_limit_bytes"),
            "shared_environment": shared,
            "shared_environment_sha256": digest(canonical(shared)),
            "process_wall_timeout_seconds": plan.get("process_wall_timeout_seconds"),
            "http_io_inactivity_timeout_seconds": plan.get("http", {}).get(
                "io_inactivity_timeout_seconds"
            ),
            "timeout_scope": "HTTPX per-I/O inactivity; external supervisor must enforce process wall timeout",
        }
        for key, value in expected_fields.items():
            audit.check(
                key in receipt and receipt[key] == value,
                "receipt_field",
                pid + ":" + key,
            )
        python_sources, native_assets = manifests[arm]
        for key, value in {
            "python_sources": python_sources,
            "native_assets": native_assets,
            "python": shared.get("python"),
            "model_metadata_sha256": metadata,
        }.items():
            audit.check(
                bool(value) and receipt.get(key) == value,
                "receipt_identity",
                pid + ":" + key,
            )
        installed_hash = digest(canonical({**python_sources, **native_assets}))
        audit.check(
            receipt.get("installed_manifest_sha256")
            == receipt.get("final_installed_manifest_sha256")
            == installed_hash,
            "installed_source_identity",
            pid,
        )
        audit.check(
            normalized_packages(receipt.get("packages"), audit, pid)
            == shared.get("packages"),
            "dependency_identity",
            pid,
        )
        for key in ("device_name", "memory_size", "architecture"):
            audit.check(
                receipt.get("device", {}).get(key)
                == plan.get("observed_reference_device", {}).get(key)
                and key in receipt.get("device", {}),
                "hardware_identity",
                pid + ":" + key,
            )
        for key, value in {
            "cache_enabled": False,
            "hot_cache_max_size": "0",
            "hot_cache_only": False,
            "max_concurrent_requests": 1,
            "max_context_window": 131072,
        }.items():
            audit.check(
                receipt.get("safe_effective_settings", {}).get(key) == value,
                "effective_setting",
                pid + ":" + key,
            )
        for field in ("settings_sha256", "poststart_settings_sha256"):
            audit.check(
                all(
                    isinstance(receipt.get(field, {}).get(name), str)
                    and len(receipt[field][name]) == 64
                    for name in ("settings.json", "model_settings.json")
                ),
                "settings_identity_missing",
                pid + ":" + field,
            )
        start, finished = receipt.get("started_unix"), receipt.get("finished_unix")
        if positive(start) and positive(finished):
            audit.check(
                finished > start
                and finished - start <= plan.get("process_wall_timeout_seconds", 0),
                "process_duration",
                pid,
            )
            if previous_finished is not None:
                audit.check(start >= previous_finished, "process_execution_order", pid)
            previous_finished = finished
        else:
            audit.check(False, "process_timestamps", pid)
        audit.check(
            [r.get("task_id") for r in rows] == sequence,
            "response_order_or_missing",
            pid,
        )
        audit.check(
            receipt.get("truncated")
            == sum(
                r.get("finish_reason") == "length" for r in rows if not r.get("warmup")
            ),
            "truncation_counter",
            pid,
        )
        peaks = []
        for index, row in enumerate(rows):
            task = row.get("task_id")
            if task not in jobs or task not in payloads or index >= len(sequence):
                audit.check(False, "unexpected_response", pid + ":" + str(task))
                continue
            key = (arm, block, task)
            audit.check(key not in seen_rows, "duplicate_response", str(key))
            seen_rows[key] = row
            row_valid[key] = validate_row(
                row,
                jobs[task],
                payloads[task],
                proc,
                index,
                build.get("expected_runtime", {}),
                audit,
            )
            for snap in (row.get("runtime_before"), row.get("runtime_after")):
                if (
                    isinstance(snap, dict)
                    and type(snap.get("mlx_peak_memory_bytes")) is int
                    and snap["mlx_peak_memory_bytes"] >= 0
                ):
                    peaks.append(snap["mlx_peak_memory_bytes"])
        snapshot_valid(
            receipt.get("runtime_initial"),
            build.get("expected_runtime", {}),
            audit,
            pid + ":initial",
            require_loaded=False,
        )
        snapshot_valid(
            receipt.get("runtime_latest"),
            build.get("expected_runtime", {}),
            audit,
            pid + ":latest",
        )
        audit.check(
            receipt.get("warmup", {}).get("task_id") == proc["warmup_task_id"]
            and receipt.get("warmup", {}).get("excluded") is True,
            "warmup_receipt",
            pid,
        )
        process_results.append(
            {
                "process_id": pid,
                "arm": arm,
                "block": block,
                "status": receipt.get("status"),
                "rows": len(rows),
                "measured_rows": sum(r.get("warmup") is False for r in rows),
                "excluded_warmups": sum(r.get("warmup") is True for r in rows),
                "finish_reasons": dict(
                    Counter(str(r.get("finish_reason")) for r in rows)
                ),
                "errors": [
                    r.get("task_id")
                    for r in rows
                    if r.get("error") or r.get("observer_error")
                ],
                "observed_cumulative_mlx_peak_bytes": max(peaks) if peaks else None,
            }
        )
    pairs, case_ratios, completion_ids = [], {task: {} for task in measured_ids}, []
    for task in measured_ids:
        for arm in ("baseline", "candidate"):
            left, right = seen_rows.get((arm, 0, task)), seen_rows.get((arm, 1, task))
            if left and right:
                audit.check(
                    (
                        left.get("text"),
                        left.get("finish_reason"),
                        (left.get("usage") or {}).get("completion_tokens"),
                    )
                    == (
                        right.get("text"),
                        right.get("finish_reason"),
                        (right.get("usage") or {}).get("completion_tokens"),
                    ),
                    "within_arm_block_variation",
                    arm + ":" + task,
                )
        for block in (0, 1):
            pair = {"task_id": task, "block": block, "valid": False, "ratios": {}}
            baseline = seen_rows.get(("baseline", block, task))
            candidate = seen_rows.get(("candidate", block, task))
            pair["present"] = {
                "baseline": baseline is not None,
                "candidate": candidate is not None,
            }
            pair_start = len(audit.errors)
            if audit.check(
                baseline is not None and candidate is not None,
                "missing_pair",
                task + ":" + str(block),
            ):
                for field in ("request_sha256", "text", "finish_reason"):
                    audit.check(
                        baseline.get(field) == candidate.get(field),
                        "paired_" + field,
                        task + ":" + str(block),
                    )
                for field in ("prompt_tokens", "completion_tokens"):
                    audit.check(
                        (baseline.get("usage") or {}).get(field)
                        == (candidate.get("usage") or {}).get(field),
                        "paired_" + field,
                        task + ":" + str(block),
                    )
                pair["outputs"] = {
                    arm: {
                        "text_sha256": (
                            digest(row["text"].encode())
                            if isinstance(row.get("text"), str)
                            else None
                        ),
                        "finish_reason": row.get("finish_reason"),
                        "prompt_tokens": (row.get("usage") or {}).get("prompt_tokens"),
                        "completion_tokens": (row.get("usage") or {}).get(
                            "completion_tokens"
                        ),
                    }
                    for arm, row in (("baseline", baseline), ("candidate", candidate))
                }
                pair["measurements"] = {
                    arm: {
                        "client_ttft_seconds": row.get("client_ttft_seconds"),
                        "wall_seconds": row.get("wall_seconds"),
                        **{
                            key: (row.get("usage") or {}).get(key)
                            for key in (
                                "prompt_tokens_per_second",
                                "generation_tokens_per_second",
                                "time_to_first_token",
                                "prompt_eval_duration",
                                "generation_duration",
                            )
                        },
                    }
                    for arm, row in (("baseline", baseline), ("candidate", candidate))
                }
                pair["valid"] = (
                    len(audit.errors) == pair_start
                    and row_valid.get(("baseline", block, task), False)
                    and row_valid.get(("candidate", block, task), False)
                )
                if pair["valid"]:
                    for metric, kind in METRICS.items():
                        a, b = (
                            pair["measurements"]["baseline"][metric],
                            pair["measurements"]["candidate"][metric],
                        )
                        pair["ratios"][metric] = a / b if kind == "latency" else b / a
                    if (
                        baseline["finish_reason"]
                        == candidate["finish_reason"]
                        == "stop"
                    ):
                        pair["ratios"]["completed_answer_latency_seconds"] = (
                            baseline["wall_seconds"] / candidate["wall_seconds"]
                        )
                        completion_ids.append({"task_id": task, "block": block})
                    case_ratios[task][block] = pair["ratios"]
            pairs.append(pair)
    audit.check(
        len(pairs) == 16 and all(p["valid"] for p in pairs),
        "complete_paired_comparison",
        "all 16 measured pairs required",
    )
    aggregates = None
    if not audit.errors:
        per_case = {}
        for task in measured_ids:
            common = set(case_ratios[task][0]) & set(case_ratios[task][1])
            per_case[task] = {
                metric: {
                    "block_ratios": [
                        case_ratios[task][block][metric] for block in (0, 1)
                    ],
                    "geomean_ratio": geomean(
                        case_ratios[task][block][metric] for block in (0, 1)
                    ),
                }
                for metric in sorted(common)
            }
        aggregates = {"per_case": per_case, "groups": {}}
        for group, members in groups.items():
            metrics = {}
            for metric in [*METRICS, "completed_answer_latency_seconds"]:
                eligible = [task for task in members if metric in per_case[task]]
                if not eligible:
                    metrics[metric] = {
                        "eligible_task_ids": [],
                        "paired_observations": 0,
                        "geomean_ratio": None,
                    }
                    continue
                values = [per_case[task][metric]["geomean_ratio"] for task in eligible]
                metrics[metric] = {
                    "eligible_task_ids": eligible,
                    "paired_observations": 2 * len(eligible),
                    "geomean_ratio": geomean(values),
                    "minimum_case_ratio": min(values),
                    "maximum_case_ratio": max(values),
                }
            aggregates["groups"][group] = {
                "workloads": len(members),
                "task_ids": members,
                "metrics": metrics,
            }
    memory_deltas = []
    for block in (0, 1):
        values = {
            p.get("arm"): p.get("observed_cumulative_mlx_peak_bytes")
            for p in process_results
            if p.get("block") == block
        }
        a, b = values.get("baseline"), values.get("candidate")
        memory_deltas.append(
            {
                "block": block,
                "baseline_bytes": a,
                "candidate_bytes": b,
                "candidate_minus_baseline_bytes": (
                    b - a if a is not None and b is not None else None
                ),
            }
        )
    return {
        "schema": "flash1017-installed-performance-summary/1",
        "status": "failed" if audit.errors else "passed",
        "integrity": {
            "status": "failed" if audit.errors else "verified",
            "checks": audit.checks,
            "errors": audit.errors,
        },
        "plan_sha256": plan_sha,
        "analysis_harness_sha256": digest(Path(__file__).read_bytes()),
        "run_identities": run_identities,
        "expected": {
            "processes": 4,
            "measured_requests": 32,
            "excluded_warmups": 4,
            "paired_observations": 16,
        },
        "observed": {
            "processes": len(run_identities),
            "measured_requests": sum(
                p.get("measured_rows", 0) for p in process_results
            ),
            "valid_pairs": sum(p["valid"] for p in pairs),
            "excluded_warmups": sum(
                p.get("excluded_warmups", 0) for p in process_results
            ),
        },
        "processes": process_results,
        "pairs": pairs,
        "aggregates": aggregates,
        "completion_latency": {
            "eligible_pairs": completion_ids,
            "eligible_pair_count": len(completion_ids),
            "excluded_length_stops_remain_in_primary_metrics": True,
        },
        "memory": {
            "block_deltas": memory_deltas,
            "scope": "Largest observed cumulative process MLX allocator peak, including loading and warmup. Not per-request peak, full process RSS or total system memory; not comparable to the historical 7.75 GiB sampled-footprint change. Missing counters are unavailable, not zero.",
        },
        "interpretation": {
            "ratios": "Above 1 favors candidate: baseline/candidate for TTFT and completed-answer latency; candidate/baseline for token rates.",
            "aggregation": "Geometric mean of 2 paired block ratios per case, then equal-case geometric means. The all-eight group deliberately mixes synthetic and natural workloads.",
            "failure": "No aggregate speed summary is produced if any gate fails. Present pair measurements remain inspectable and are not permission to exclude failed pairs.",
            "quality": "Timing establishes no general quality result. Pair output equality establishes comparability of these requests; the separately verified 618-task report supplies bounded quality evidence.",
            "uncertainty": "Two temporal blocks on one device: no confidence intervals, p95/p99, population significance or other-hardware claim.",
            "memory_policy": plan.get("memory_limit_interpretation"),
            "cache_policy": plan.get("cache_policy"),
            "identity_scope": "Frozen artifacts, wheel contents and saved receipt/runtime observations are checked. Raw secret-bearing settings and live effective wired-memory equality are not independently asserted.",
            "checkpoint_scope": "The frozen metadata and tokenizer identities are compared; this analysis does not rehash model weight shards.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--run", action="append", default=[], metavar="PROCESS_ID=RESULT_DIRECTORY"
    )
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        pid, separator, path = value.partition("=")
        if not separator or not pid or not path or pid in runs:
            parser.error("Each --run must uniquely specify PROCESS_ID=RESULT_DIRECTORY")
        runs[pid] = Path(path)
    report = summarize(
        args.plan,
        runs,
        {"baseline": args.baseline_wheel, "candidate": args.candidate_wheel},
        args.workspace_root,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "errors": len(report["integrity"]["errors"]),
                "valid_pairs": report["observed"]["valid_pairs"],
                "aggregates_available": report["aggregates"] is not None,
                "output": str(args.output),
            }
        )
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
