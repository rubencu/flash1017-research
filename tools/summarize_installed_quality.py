#!/usr/bin/env python3
"""Validate saved paired generation/grades and preserve the original release gate.

CPU-only: reads JSON/JSONL, never imports a runtime or executes generated code.
Missing coding grades produce a pending result, never an inferred pass. Settings
directories are optional for an interim report; without them byte verification
of settings remains pending. Authentication settings are never emitted.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MODEL_ID = "DeepSeek-V4.1-Flash-1017"
CODING_SUITES = {"humaneval", "mbpp"}
SUITE_KEYS = {
    "humaneval": "humaneval_plus",
    "mbpp": "mbpp_plus",
    "gsm8k-test-64-hash-subset": "gsm8k_hash_subset",
    "synthetic-long-context-controls": "synthetic_long_context",
}
MODEL_DIFFERENCES = {
    "deepseek_v41_flash1017_enabled",
    "mtp_fixed_depth",
    "mtp_adaptive_max_depth",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def identity(path):
    path = Path(path)
    return {
        "file": path.name,
        "sha256": digest(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def payload(job, plan):
    sampling = plan["sampling"]
    return {
        "model": plan.get("request_model_id", MODEL_ID),
        "messages": job.get("messages") or [{"role": "user", "content": job["prompt"]}],
        "max_tokens": int(job["max_tokens"]),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": sampling["top_k"],
        "seed": sampling["seed"],
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": sampling["enable_thinking"]},
    }


def normalized_packages(packages):
    return {re.sub(r"[-_.]+", "-", k).lower(): v for k, v in packages.items()}


class Audit:
    def __init__(self):
        self.errors = []
        self.pending = []
        self.checks = 0
        self.settings_reconstructions = []

    def check(self, condition, code, detail):
        self.checks += 1
        if not condition:
            self.errors.append({"code": code, "detail": detail})

    def ordered(self, rows, expected, label):
        ids = [row.get("task_id") for row in rows]
        self.check(len(ids) == len(set(ids)), "duplicate_task", label)
        self.check(ids == expected, "task_order_or_membership", label)
        return {row.get("task_id"): row for row in rows}


def generation_summary(rows):
    return {
        "responses": len(rows),
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in rows)),
        "errors": [
            r["task_id"]
            for r in rows
            if r.get("error") or not isinstance(r.get("text"), str) or not r.get("text")
        ],
        "truncated": [r["task_id"] for r in rows if r.get("finish_reason") == "length"],
        "unfinished": [
            r["task_id"]
            for r in rows
            if r.get("finish_reason") not in {"stop", "eos", "length"}
        ],
    }


def validate_mtp(audit, selected, receipt, arm, specification):
    expected = specification.get("expected_runtime", {})
    if "mtp_depth_fixed" in expected:
        # V2 explicitly specifies native adaptive MTP. Its draft_width is the
        # ceiling, not a request for fixed depth as in the original protocol.
        fixed = expected["mtp_depth_fixed"]
        depth = expected.get("mtp_depth")
        audit.check(type(fixed) is bool, "mtp_mode", arm)
        audit.check(type(depth) is int and depth > 0, "mtp_depth", arm)
        audit.check(selected.get("mtp_enabled") is True, "mtp_enabled", arm)
        for field in ("draft_width", "adaptive_max_depth"):
            if field in specification:
                audit.check(specification[field] == depth, "mtp_plan_depth", arm)
        if fixed is False:
            audit.check(selected.get("mtp_fixed_depth") is None, "mtp_setting", arm)
            audit.check(
                selected.get("mtp_adaptive_max_depth") == depth,
                "mtp_setting",
                arm,
            )
        elif fixed is True:
            audit.check(selected.get("mtp_fixed_depth") == depth, "mtp_setting", arm)
        for name in ("runtime_initial", "runtime_latest"):
            runtime = receipt.get(name, {})
            audit.check(
                runtime.get("mtp_depth_fixed") is fixed,
                "mtp_runtime_mode",
                f"{arm}/{name}",
            )
            audit.check(
                type(runtime.get("mtp_depth")) is int and runtime["mtp_depth"] == depth,
                "mtp_runtime_depth",
                f"{arm}/{name}",
            )
        return
    # Preserve interpretation of historical protocols without explicit mode.
    if "adaptive_max_depth" in specification:
        audit.check(
            selected.get("mtp_adaptive_max_depth")
            == specification["adaptive_max_depth"],
            "mtp_setting",
            arm,
        )
    if "draft_width" in specification:
        audit.check(
            selected.get("mtp_fixed_depth") == specification["draft_width"],
            "mtp_setting",
            arm,
        )


def validate_settings(audit, directory, receipt, arm, plan):
    if directory is None:
        audit.pending.append(f"{arm}: settings bytes not supplied")
        return None
    directory = Path(directory)
    for name in ("settings.json", "model_settings.json"):
        actual = digest((directory / name).read_bytes())
        expected = receipt.get(
            "poststart_settings_sha256", receipt.get("settings_sha256", {})
        ).get(name)
        matches = actual == expected
        if (
            not matches
            and name == "settings.json"
            and not receipt.get("poststart_settings_sha256")
        ):
            # The original server auto-saves these two non-inference fields at
            # startup. Accept reconstruction only when the exact original byte
            # hash matches; any other change still fails. Never emit secrets.
            restored = read_json(directory / name)
            restored.setdefault("auth", {})["secret_key"] = None
            restored.setdefault("server", {})["server_aliases"] = []
            reconstructed = digest(
                json.dumps(restored, indent=2, ensure_ascii=False).encode()
            )
            matches = reconstructed == expected
            if matches:
                audit.settings_reconstructions.append(
                    {
                        "arm": arm,
                        "file": name,
                        "method": "Exact pre-start bytes reconstructed from final JSON by reverting only the server-generated auth.secret_key and server.server_aliases fields",
                        "startup_sha256": expected,
                        "current_sha256": actual,
                        "reconstructed_sha256": reconstructed,
                    }
                )
        audit.check(
            matches,
            "settings_hash",
            f"{arm}/{name}",
        )
    global_settings = read_json(directory / "settings.json")
    model_settings = read_json(directory / "model_settings.json")
    audit.check(
        global_settings.get("version") == "1.0",
        "settings_schema",
        f"{arm}/settings.json",
    )
    audit.check(
        model_settings.get("version") == 1,
        "settings_schema",
        f"{arm}/model_settings.json",
    )
    selected = model_settings.get("models", {}).get(
        plan.get("request_model_id", MODEL_ID), {}
    )
    audit.check(bool(selected), "model_settings_missing", arm)
    for name in ("temperature", "top_p"):
        audit.check(
            global_settings.get("sampling", {}).get(name) == plan["sampling"][name],
            "sampling_setting",
            f"{arm}/global/{name}",
        )
        audit.check(
            selected.get(name) == plan["sampling"][name],
            "sampling_setting",
            f"{arm}/model/{name}",
        )
    audit.check(selected.get("force_sampling") is False, "force_sampling", arm)
    audit.check(
        selected.get("enable_thinking") == plan["sampling"]["enable_thinking"],
        "thinking_setting",
        arm,
    )
    validate_mtp(audit, selected, receipt, arm, plan[arm])
    # Return only relevant, non-secret settings for internal pair comparison.
    cache = dict(global_settings.get("cache", {}))
    cache.pop("ssd_cache_dir", None)
    return {
        "sampling": global_settings.get("sampling"),
        "memory": global_settings.get("memory"),
        "scheduler": global_settings.get("scheduler"),
        "cache": cache,
        "model": {k: v for k, v in selected.items() if k not in MODEL_DIFFERENCES},
    }


def validate_arm(
    audit, arm, plan, plan_path, requests_path, requests, receipt, rows, state_dir
):
    ids = [job["task_id"] for job in requests]
    by_id = audit.ordered(rows, ids, f"{arm}/responses")
    summary = generation_summary(rows)
    audit.check(receipt.get("arm") == arm, "receipt_arm", arm)
    audit.check(receipt.get("status") == "generated", "receipt_status", arm)
    audit.check(receipt.get("controllers_closed") is True, "shutdown_incomplete", arm)
    reused = plan.get("reused_baseline") if arm == "baseline" else None
    audit.check(
        receipt.get("plan_sha256")
        == (reused["plan_sha256"] if reused else digest(Path(plan_path).read_bytes())),
        "plan_hash",
        arm,
    )
    audit.check(
        receipt.get("requests_sha256") == digest(Path(requests_path).read_bytes()),
        "requests_hash",
        arm,
    )
    # A reused baseline is bound to its entire old receipt and original plan
    # below. Its old harness identity must not be rewritten to the new one.
    if not reused:
        audit.check(
            receipt.get("harness_sha256") == plan["harness_sha256"], "harness_hash", arm
        )
    audit.check(
        receipt.get("runtime_version") == plan[arm]["version"], "runtime_version", arm
    )
    packages = normalized_packages(receipt.get("packages", {}))
    audit.check(packages.get("omlx") == plan[arm]["version"], "package_version", arm)
    for field, value in (
        ("total_tasks", len(requests)),
        ("completed", len(rows)),
        ("errors", len(summary["errors"])),
        ("truncated", len(summary["truncated"])),
    ):
        audit.check(receipt.get(field) == value, "receipt_count", f"{arm}/{field}")
    for name in ("runtime_initial", "runtime_latest"):
        runtime = receipt.get(name, {})
        audit.check(
            runtime.get("prefill_step_size") == plan[arm]["prefill_step"],
            "runtime_prefill",
            f"{arm}/{name}",
        )
        audit.check(
            runtime.get("max_num_seqs") == 1, "runtime_concurrency", f"{arm}/{name}"
        )
    for index, job in enumerate(requests):
        row = by_id.get(job["task_id"])
        if row is None:
            continue
        label = f"{arm}/{job['task_id']}"
        audit.check(row.get("index") == index, "response_index", label)
        audit.check(row.get("suite") == job["suite"], "response_suite", label)
        audit.check(
            row.get("max_tokens") == job["max_tokens"], "response_budget", label
        )
        audit.check(
            row.get("request_sha256") == digest(canonical(payload(job, plan))),
            "request_payload_hash",
            label,
        )
        audit.check(
            row.get("http_status") == 200 or bool(row.get("error")),
            "http_status",
            label,
        )
    settings = validate_settings(audit, state_dir, receipt, arm, plan)
    return by_id, summary, packages, settings


def load_supplement(audit, score_path, requests, responses, inputs):
    expected = [r["task_id"] for r in requests if r["suite"] not in CODING_SUITES]
    if not expected:
        return []
    if score_path is None:
        audit.pending.append("supplement grading not supplied")
        return []
    score_path = Path(score_path)
    score = read_json(score_path)
    inputs["supplement_scores"] = identity(score_path)
    pairs = score.get("pairs", [])
    audit.ordered(pairs, expected, "supplement grades")
    for arm in ("baseline", "candidate"):
        source = score_path.parent / score["inputs"][arm]["file"]
        inputs[f"{arm}_supplement_responses"] = identity(source)
        audit.check(
            digest(source.read_bytes()) == score["inputs"][arm]["sha256"],
            "supplement_input_hash",
            arm,
        )
        subset = read_jsonl(source)
        subset_by_id = audit.ordered(subset, expected, f"{arm}/supplement responses")
        for task_id in expected:
            audit.check(
                subset_by_id.get(task_id) == responses[arm].get(task_id),
                "supplement_response_binding",
                f"{arm}/{task_id}",
            )
    normalized = []
    for row in pairs:
        for arm in ("baseline", "candidate"):
            audit.check(
                type(row.get(arm, {}).get("correct")) is bool,
                "supplement_correct_type",
                f"{arm}/{row.get('task_id')}",
            )
        normalized.append(
            {
                "task_id": row["task_id"],
                "suite": row["suite"],
                "baseline_pass": row["baseline"]["correct"],
                "candidate_pass": row["candidate"]["correct"],
                "grade_kind": "supplement exact answer",
            }
        )
    return normalized


def load_coding(audit, paths, requests, responses, response_paths, inputs):
    expected = [r["task_id"] for r in requests if r["suite"] in CODING_SUITES]
    if not expected:
        return []
    if paths["baseline"] is None or paths["candidate"] is None:
        audit.pending.append("paired coding grades not supplied")
        return []
    grades, metadata = {}, {}
    for arm in ("baseline", "candidate"):
        path = Path(paths[arm])
        inputs[f"{arm}_coding_grades"] = identity(path)
        grades[arm] = audit.ordered(read_jsonl(path), expected, f"{arm}/coding grades")
        meta_path = path.with_suffix(path.suffix + ".metadata.json")
        inputs[f"{arm}_coding_metadata"] = identity(meta_path)
        meta = metadata[arm] = read_json(meta_path)
        audit.check(
            meta.get("samples_sha256")
            == digest(Path(response_paths[arm]).read_bytes()),
            "coding_samples_hash",
            arm,
        )
        audit.check(meta.get("tasks") == expected, "coding_metadata_tasks", arm)
        audit.check(
            meta.get("canonical_validation") is False, "canonical_not_generated", arm
        )
        for task_id, row in grades[arm].items():
            label = f"{arm}/{task_id}"
            audit.check(
                row.get("canonical_validation") is False,
                "canonical_not_generated",
                label,
            )
            audit.check(
                row.get("suite") == responses[arm].get(task_id, {}).get("suite"),
                "coding_suite",
                label,
            )
            for phase in ("base", "plus"):
                outcome = row.get(phase, {})
                status = outcome.get("status")
                audit.check(
                    status in {"pass", "fail", "timeout"},
                    "coding_phase_status",
                    f"{label}/{phase}",
                )
                if outcome.get("raw_sha256") is not None or status == "pass":
                    audit.check(
                        outcome.get("raw_sha256")
                        == digest(
                            responses[arm].get(task_id, {}).get("text", "").encode()
                        ),
                        "coding_raw_hash",
                        f"{label}/{phase}",
                    )
            base = row.get("base", {}).get("status") == "pass"
            plus = base and row.get("plus", {}).get("status") == "pass"
            audit.check(
                type(row.get("base_pass")) is bool and row["base_pass"] == base,
                "coding_base_consistency",
                label,
            )
            audit.check(
                type(row.get("plus_pass")) is bool and row["plus_pass"] == plus,
                "coding_plus_consistency",
                label,
            )
    for name in (
        "evalplus_commit",
        "datasets",
        "adapter_sha256",
        "export_helper_sha256",
        "python",
        "platform",
        "min_time_limit_seconds",
        "reference_time_factor",
        "phase_wall_budget",
        "memory",
        "cpu_hard_limit_seconds",
        "output_limit_bytes",
    ):
        audit.check(
            metadata["baseline"].get(name) == metadata["candidate"].get(name)
            and name in metadata["baseline"],
            "coding_evaluator_identity",
            name,
        )
    return [
        {
            "task_id": task_id,
            "suite": grades["baseline"][task_id]["suite"],
            "baseline_pass": grades["baseline"][task_id]["plus_pass"],
            "candidate_pass": grades["candidate"][task_id]["plus_pass"],
            "baseline_base_pass": grades["baseline"][task_id]["base_pass"],
            "candidate_base_pass": grades["candidate"][task_id]["base_pass"],
            "grade_kind": "EvalPlus base AND extended tests",
        }
        for task_id in expected
        if task_id in grades["baseline"] and task_id in grades["candidate"]
    ]


def summarize(args):
    audit = Audit()
    plan_path = Path(args.plan)
    plan = read_json(plan_path)
    requests_path = (
        Path(args.requests)
        if args.requests
        else plan_path.parent / plan["requests_path"]
    )
    requests = read_jsonl(requests_path)
    inputs = {
        "plan": identity(plan_path),
        "requests": identity(requests_path),
        "reporter": identity(__file__),
    }
    audit.check(
        digest(requests_path.read_bytes()) == plan["requests_sha256"],
        "frozen_requests_hash",
        "plan",
    )
    audit.check(len(requests) == plan["tasks"], "plan_task_count", "requests")
    audit.ordered(requests, [r["task_id"] for r in requests], "requests")
    counts = Counter(SUITE_KEYS.get(r["suite"], r["suite"]) for r in requests)
    audit.check(dict(counts) == plan["suites"], "plan_suite_counts", "requests")
    receipts, responses, generation, packages, settings = {}, {}, {}, {}, {}
    response_paths = {}
    original_plan = None
    reuse = plan.get("reused_baseline")
    if reuse:
        original_path = getattr(args, "baseline_original_plan", None)
        if original_path is None:
            audit.check(
                False,
                "reused_baseline_original_plan_missing",
                "supply --baseline-original-plan",
            )
        else:
            original_path = Path(original_path)
            original_plan = read_json(original_path)
            inputs["baseline_original_plan"] = identity(original_path)
            audit.check(
                digest(original_path.read_bytes()) == reuse.get("plan_sha256"),
                "reused_baseline_plan_hash",
                "baseline",
            )
            for field in ("requests_sha256", "model", "sampling", "baseline"):
                audit.check(
                    original_plan.get(field) == plan.get(field),
                    "reused_baseline_protocol_disagreement",
                    field,
                )
    for arm in ("baseline", "candidate"):
        receipt_path = Path(getattr(args, f"{arm}_receipt"))
        receipt = receipts[arm] = read_json(receipt_path)
        given = getattr(args, f"{arm}_responses")
        response_paths[arm] = (
            Path(given) if given else receipt_path.parent / receipt["responses"]
        )
        rows = read_jsonl(response_paths[arm])
        inputs[f"{arm}_receipt"] = identity(receipt_path)
        inputs[f"{arm}_responses"] = identity(response_paths[arm])
        if arm == "baseline" and reuse:
            audit.check(
                digest(receipt_path.read_bytes()) == reuse.get("receipt_sha256"),
                "reused_baseline_receipt_hash",
                arm,
            )
            audit.check(
                digest(response_paths[arm].read_bytes())
                == reuse.get("responses_sha256"),
                "reused_baseline_responses_hash",
                arm,
            )
            if original_plan is not None:
                audit.check(
                    receipt.get("harness_sha256")
                    == original_plan.get("harness_sha256"),
                    "reused_baseline_harness_hash",
                    arm,
                )
        responses[arm], generation[arm], packages[arm], settings[arm] = validate_arm(
            audit,
            arm,
            plan,
            plan_path,
            requests_path,
            requests,
            receipt,
            rows,
            getattr(args, f"{arm}_state_dir"),
        )
    non_runtime_packages = [
        {k: v for k, v in packages[a].items() if k != "omlx"}
        for a in ("baseline", "candidate")
    ]
    audit.check(
        non_runtime_packages[0] == non_runtime_packages[1]
        and bool(non_runtime_packages[0]),
        "dependency_disagreement",
        "all packages except omlx",
    )
    for name in ("native_assets", "python", "device", "memory_limit_bytes"):
        audit.check(
            bool(receipts["baseline"].get(name))
            and receipts["baseline"].get(name) == receipts["candidate"].get(name),
            "runtime_identity_disagreement",
            name,
        )
    for job in requests:
        task_id = job["task_id"]
        observed = [
            responses[arm].get(task_id, {}).get("usage", {}).get("prompt_tokens")
            for arm in ("baseline", "candidate")
        ]
        if any(value is not None for value in observed):
            audit.check(
                observed[0] == observed[1], "prompt_token_count_disagreement", task_id
            )
    if settings["baseline"] is not None and settings["candidate"] is not None:
        for name in settings["baseline"]:
            audit.check(
                settings["baseline"][name] == settings["candidate"][name],
                "common_settings_disagreement",
                name,
            )
    pairs = load_supplement(audit, args.supplement_scores, requests, responses, inputs)
    pairs += load_coding(
        audit,
        {a: getattr(args, f"{a}_coding_grades") for a in ("baseline", "candidate")},
        requests,
        responses,
        response_paths,
        inputs,
    )
    position = {r["task_id"]: i for i, r in enumerate(requests)}
    pairs.sort(key=lambda r: position.get(r["task_id"], len(position)))
    suites = {}
    for suite in sorted({r["suite"] for r in requests}):
        rows = [p for p in pairs if p["suite"] == suite]
        expected = sum(r["suite"] == suite for r in requests)
        suites[suite] = {
            "expected_tasks": expected,
            "graded_pairs": len(rows),
            "status": "complete" if len(rows) == expected else "pending",
            "baseline_pass": sum(r["baseline_pass"] for r in rows),
            "candidate_pass": sum(r["candidate_pass"] for r in rows),
            "regressions": [
                r["task_id"]
                for r in rows
                if r["baseline_pass"] and not r["candidate_pass"]
            ],
            "improvements": [
                r["task_id"]
                for r in rows
                if not r["baseline_pass"] and r["candidate_pass"]
            ],
            "both_fail": [
                r["task_id"]
                for r in rows
                if not r["baseline_pass"] and not r["candidate_pass"]
            ],
            "identical_text": sum(
                responses["baseline"].get(r["task_id"], {}).get("text")
                == responses["candidate"].get(r["task_id"], {}).get("text")
                for r in rows
            ),
        }
    regressions = [
        r["task_id"] for r in pairs if r["baseline_pass"] and not r["candidate_pass"]
    ]
    violations = []
    acceptance = plan["acceptance"]
    mode = acceptance.get("mode", "original-v1")
    audit.check(
        mode in {"original-v1", "amended-stock-arithmetic-v1"},
        "acceptance_mode",
        "unknown acceptance mode",
    )
    if mode == "amended-stock-arithmetic-v1":
        audit.check(
            bool(acceptance.get("amendment_reason")),
            "acceptance_amendment_reason",
            "acceptance",
        )
        audit.check(
            bool(reuse)
            and acceptance.get("original_protocol_sha256") == reuse.get("plan_sha256"),
            "acceptance_original_protocol",
            "acceptance",
        )
        audit.check(
            acceptance.get("allow_truncation") is True
            and acceptance.get("allow_candidate_only_truncation") is False
            and acceptance.get("require_exact_text_and_finish_reason") is True
            and acceptance.get("allow_missing_or_errors") is False
            and acceptance.get("maximum_baseline_pass_to_candidate_fail_tasks") == 0
            and acceptance.get("minimum_complete_pairs") == len(requests),
            "amended_acceptance_contract",
            "stock-arithmetic qualification",
        )
    candidate_only_truncated = sorted(
        set(generation["candidate"]["truncated"])
        - set(generation["baseline"]["truncated"])
    )
    text_or_finish_differences = [
        job["task_id"]
        for job in requests
        if any(
            responses["baseline"].get(job["task_id"], {}).get(key)
            != responses["candidate"].get(job["task_id"], {}).get(key)
            for key in ("text", "finish_reason")
        )
        or job["task_id"] not in responses["baseline"]
        or job["task_id"] not in responses["candidate"]
    ]
    paired_count = len(set(responses["baseline"]) & set(responses["candidate"]))
    if paired_count < acceptance["minimum_complete_pairs"]:
        violations.append("Insufficient paired responses")
    if not acceptance["allow_missing_or_errors"] and any(
        generation[a]["errors"] or generation[a]["unfinished"] for a in generation
    ):
        violations.append("Generation errors or unfinished responses")
    if not acceptance["allow_truncation"] and any(
        generation[a]["truncated"] for a in generation
    ):
        violations.append(
            "The frozen protocol disallows truncation; at least one arm truncated"
        )
    if len(regressions) > acceptance["maximum_baseline_pass_to_candidate_fail_tasks"]:
        violations.append(
            "Baseline-pass to candidate-fail tasks exceed the frozen limit"
        )
    if (
        acceptance.get("allow_candidate_only_truncation") is False
        and candidate_only_truncated
    ):
        violations.append("Candidate-only truncations violate the amended gate")
    if (
        acceptance.get("require_exact_text_and_finish_reason") is True
        and text_or_finish_differences
    ):
        violations.append(
            "Exact response text and finish-reason parity was required but not observed on every task"
        )
    audit.check(
        acceptance.get("quality_gains_offset_regressions") is False,
        "offsetting_not_supported",
        "acceptance",
    )
    if len(pairs) != len(requests):
        audit.pending.append(f"Only {len(pairs)} of {len(requests)} pairs graded")
    if audit.errors:
        violations.append("Evidence integrity checks failed")
    status = "failed" if violations else "pending" if audit.pending else "passed"
    gate = {
        "status": status,
        "mode": mode,
        "rules": acceptance,
        "violations": violations,
        "pending": audit.pending,
        "improvements_offset_regressions": False,
    }
    original_gate = gate
    if mode != "original-v1":
        original_truncation_failed = (
            bool(original_plan)
            and original_plan.get("acceptance", {}).get("allow_truncation") is False
            and any(generation[a]["truncated"] for a in generation)
        )
        original_gate = {
            "status": "failed" if original_truncation_failed else "not_recomputed",
            "protocol_sha256": acceptance.get("original_protocol_sha256"),
            "violations": (
                ["The original absolute zero-truncation prerequisite remains failed"]
                if original_truncation_failed
                else []
            ),
            "scope": "This records the original rule's continuing failure, not a replacement or regrading of the original candidate run. Preserve that separate report.",
        }
    return {
        "schema": "flash1017-quality-pair-summary/1",
        "inputs": inputs,
        "integrity": {
            "status": (
                "failed"
                if audit.errors
                else (
                    "pending"
                    if any("settings" in p for p in audit.pending)
                    else "verified"
                )
            ),
            "checks": audit.checks,
            "errors": audit.errors,
            "pending": audit.pending,
            "settings_reconstructions": audit.settings_reconstructions,
        },
        "builds": {a: plan[a] for a in ("baseline", "candidate")},
        "identity_scope": "Checks saved version, native assets, dependencies, request/settings hashes. Planned wheel hashes are declarations; receipts do not hash every installed Python source file.",
        "generation": generation,
        "paired_responses": paired_count,
        "graded_pairs": len(pairs),
        "suites": suites,
        "regressions": regressions,
        "candidate_only_truncated": candidate_only_truncated,
        "exact_text_and_finish_reason": {
            "matched": len(requests) - len(text_or_finish_differences),
            "expected": len(requests),
            "differences": text_or_finish_differences,
        },
        "planned_acceptance": gate,
        "original_acceptance": original_gate,
        "observed_exact_response_parity_established": (
            not audit.errors
            and paired_count == len(requests)
            and not text_or_finish_differences
        ),
        "observed_task_score_preservation_established": (
            not audit.errors and len(pairs) == len(requests) and not regressions
        ),
        "universal_quality_equivalence_proved": False,
        "pairs": pairs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--requests", type=Path)
    p.add_argument("--baseline-original-plan", type=Path)
    for arm in ("baseline", "candidate"):
        p.add_argument(f"--{arm}-receipt", type=Path, required=True)
        p.add_argument(f"--{arm}-responses", type=Path)
        p.add_argument(f"--{arm}-state-dir", type=Path)
        p.add_argument(f"--{arm}-coding-grades", type=Path)
    p.add_argument("--supplement-scores", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "integrity": result["integrity"]["status"],
                "graded_pairs": result["graded_pairs"],
                "original_acceptance": result["original_acceptance"]["status"],
                "planned_acceptance": result["planned_acceptance"]["status"],
                "regressions": result["regressions"],
                "output": str(args.output),
            }
        )
    )
    return 0 if result["planned_acceptance"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
