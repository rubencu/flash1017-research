# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU fixtures for the performance evidence contract; no inference."""

import copy
import importlib.util
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

SOURCE = Path(__file__).with_name("summarize_installed_performance.py")
SPEC = importlib.util.spec_from_file_location("performance_summary", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.plan_path = root / "plan.json"
        self.runs = {}
        self.wheels = {}
        self.rows = {}
        self.receipts = {}
        self.native = {"custom_kernels/test.metallib": MODULE.digest(b"native")}
        self.metadata = {name: "d" * 64 for name in MODULE.METADATA_NAMES}
        self.metadata["generation_config.json"] = None
        self.shared = {
            "packages": {"mlx": "test"},
            "python": "synthetic Python",
            "model_metadata_sha256": self.metadata,
            "native_assets": self.native,
        }
        self.plan = {
            "schema": "installed-http-performance-v1",
            "status": "frozen_before_any_model_run",
            "analysis_harness_sha256": MODULE.digest(SOURCE.read_bytes()),
            "shared_environment": self.shared,
            "request_factory": {
                "constants": {"MODEL_ID": "test-model", "SEED": 431017}
            },
            "process_wall_timeout_seconds": 5400,
            "http": {
                "io_inactivity_timeout_seconds": 1200,
                "timeout_semantics": "per-I/O inactivity, not a cumulative request deadline",
            },
            "memory_limit_bytes": 224 * 2**30,
            "observed_reference_device": {
                "device_name": "synthetic device",
                "memory_size": 256 * 2**30,
                "architecture": "test",
            },
            "memory_limit_interpretation": "Requested startup guideline; no claim of equal live wired limits.",
            "artifacts": {},
            "processes": [],
        }
        for kind in ("performance", "quality"):
            path = root / (kind + ".py")
            path.write_text("# synthetic source identity\n")
            self.plan[kind + "_harness_path"] = path.name
            self.plan[kind + "_harness_sha256"] = MODULE.digest(path.read_bytes())
        self.python_sources = {}
        for arm in ("baseline", "candidate"):
            wheel = root / (arm + ".whl")
            source = ("# " + arm + "\n").encode()
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("omlx/__init__.py", source)
                archive.writestr("omlx/custom_kernels/test.metallib", b"native")
            self.wheels[arm] = wheel
            self.python_sources[arm] = {"__init__.py": MODULE.digest(source)}
            self.plan[arm] = {
                "version": arm,
                "wheel_sha256": MODULE.digest(wheel.read_bytes()),
                "expected_runtime": {"prefill_step_size": 2048, "max_num_seqs": 1},
            }
        quality = {
            "planned_acceptance": {"status": "passed"},
            "integrity": {"status": "verified"},
            "graded_pairs": 618,
            "paired_responses": 618,
            "observed_exact_response_parity_established": True,
            "observed_task_score_preservation_established": True,
            "builds": {arm: self.plan[arm] for arm in ("baseline", "candidate")},
        }
        write_json(root / "quality.json", quality)
        self.plan["quality_report_path"] = "quality.json"
        self.plan["quality_report_sha256"] = MODULE.digest(
            (root / "quality.json").read_bytes()
        )
        self.jobs, self.payloads, rendered, fixtures = [], [], [], []
        for index in range(-1, 8):
            warmup = index == -1
            task = "warmup" if warmup else f"case-{index}"
            token_hash = MODULE.digest(json.dumps([1, 2]).encode())
            job = {
                "task_id": task,
                "suite": "warmup" if warmup else "performance",
                "messages": [{"role": "user", "content": task}],
                "max_tokens": 64 if warmup else 512,
                "warmup": warmup,
                "expected_prompt_tokens": 2,
                "expected_prompt_tokens_sha256": token_hash,
            }
            payload = {
                "model": "test-model",
                "messages": job["messages"],
                "max_tokens": job["max_tokens"],
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 431017,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            self.jobs.append(job)
            self.payloads.append(
                {
                    "task_id": task,
                    "request": payload,
                    "request_sha256": MODULE.digest(MODULE.canonical(payload)),
                }
            )
            rendered.append(
                {"task_id": task, "tokens": [1, 2], "tokens_sha256": token_hash}
            )
            fixtures.append(
                {
                    "task_id": task,
                    "synthetic": index < 2,
                    "historical_prompt_tokens": (
                        16384
                        if warmup
                        else 1024 if index < 2 else 16384 if index < 6 else 65536
                    ),
                }
            )
        provenance = {
            "fixtures": fixtures,
            "tokenizer": {"sha256": self.metadata["tokenizer.json"]},
        }
        for name, value in {
            "requests-test.jsonl": self.jobs,
            "payloads-test.jsonl": self.payloads,
            "rendered-inputs-test.jsonl": rendered,
            "fixture-provenance-test.json": provenance,
        }.items():
            (write_rows if name.endswith("jsonl") else write_json)(root / name, value)
            self.plan["artifacts"][name] = {
                "path": name,
                "sha256": MODULE.digest((root / name).read_bytes()),
            }
        order = [f"case-{i}" for i in range(8)]
        for block, arms in enumerate(
            (("baseline", "candidate"), ("candidate", "baseline"))
        ):
            for arm in arms:
                pid = f"block-{block}-{arm}"
                self.plan["processes"].append(
                    {
                        "process_id": pid,
                        "process_index": len(self.plan["processes"]),
                        "arm": arm,
                        "block": block,
                        "warmup_task_id": "warmup",
                        "measured_task_ids": (
                            order if block == 0 else list(reversed(order))
                        ),
                        "fresh_process": True,
                        "fresh_state_directory": True,
                    }
                )
        write_json(self.plan_path, self.plan)
        plan_hash = MODULE.digest(self.plan_path.read_bytes())
        artifact_hashes = {
            name: value["sha256"] for name, value in self.plan["artifacts"].items()
        }
        artifact_hashes["quality_report"] = self.plan["quality_report_sha256"]
        self.job_map = {j["task_id"]: j for j in self.jobs}
        self.payload_map = {j["task_id"]: j for j in self.payloads}
        for process in self.plan["processes"]:
            arm, block, pid = process["arm"], process["block"], process["process_id"]
            self.runs[pid] = root / pid
            sequence = ["warmup", *process["measured_task_ids"]]
            rows = [
                self.row(process, task, index) for index, task in enumerate(sequence)
            ]
            self.rows[pid] = rows
            source_hash = MODULE.digest(
                MODULE.canonical({**self.python_sources[arm], **self.native})
            )
            self.receipts[pid] = {
                "schema": "flash1017-performance-generation/1",
                "arm": arm,
                "block": block,
                "process_id": pid,
                "status": "generated",
                "plan_sha256": plan_hash,
                "artifact_hashes": artifact_hashes,
                "total_measured_tasks": 8,
                "measured_completed": 8,
                "attempts_completed": 9,
                "errors": 0,
                "truncated": 0,
                "runtime_version": arm,
                "runtime_source": "installed wheel",
                "wheel_sha256": self.plan[arm]["wheel_sha256"],
                "quality_report_sha256": self.plan["quality_report_sha256"],
                "harness_sha256": self.plan["performance_harness_sha256"],
                "quality_harness_sha256": self.plan["quality_harness_sha256"],
                "controllers_closed": True,
                "client_worker_stopped": True,
                "memory_limit_bytes": self.plan["memory_limit_bytes"],
                "shared_environment": copy.deepcopy(self.shared),
                "shared_environment_sha256": MODULE.digest(
                    MODULE.canonical(self.shared)
                ),
                "process_wall_timeout_seconds": 5400,
                "http_io_inactivity_timeout_seconds": 1200,
                "timeout_scope": "HTTPX per-I/O inactivity; external supervisor must enforce process wall timeout",
                "python_sources": self.python_sources[arm],
                "native_assets": self.native,
                "python": self.shared["python"],
                "model_metadata_sha256": self.metadata,
                "installed_manifest_sha256": source_hash,
                "final_installed_manifest_sha256": source_hash,
                "packages": {"mlx": "test", "omlx": arm},
                "device": self.plan["observed_reference_device"],
                "safe_effective_settings": {
                    "cache_enabled": False,
                    "hot_cache_max_size": "0",
                    "hot_cache_only": False,
                    "max_concurrent_requests": 1,
                    "max_context_window": 131072,
                },
                "settings_sha256": {
                    "settings.json": "a" * 64,
                    "model_settings.json": "b" * 64,
                },
                "poststart_settings_sha256": {
                    "settings.json": "c" * 64,
                    "model_settings.json": "b" * 64,
                },
                "started_unix": 100 + 100 * process["process_index"],
                "finished_unix": 150 + 100 * process["process_index"],
                "runtime_initial": {"loaded": False},
                "runtime_latest": rows[-1]["runtime_after"],
                "warmup": {"task_id": "warmup", "excluded": True},
            }
        self.save()

    def row(self, process, task, index):
        arm, block = process["arm"], process["block"]
        factor = 1 if task == "warmup" else int(task.split("-")[-1]) + 1
        ratio = factor * (2 if block == 0 else 8)
        rate_ratio = factor * (3 if block == 0 else 12)
        ttft = 10.0 if arm == "baseline" else 10.0 / ratio
        start = 1_000_000_000_000 + index * 100_000_000_000
        first = start + round(ttft * 1e9)
        last, finish, usage_time, done = [
            first + step * 100_000_000 for step in (1, 2, 3, 4)
        ]
        text = "answer-" + task
        usage = {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 0},
            "prompt_eval_duration": 1.0,
            "generation_duration": 1.0,
            "prompt_tokens_per_second": (
                100.0 if arm == "baseline" else 100.0 * rate_ratio
            ),
            "generation_tokens_per_second": (
                50.0 if arm == "baseline" else 50.0 * rate_ratio
            ),
            "time_to_first_token": ttft,
        }
        snapshot = {
            "loaded": True,
            "prefix_cache_enabled": False,
            "prefill_step_size": 2048,
            "max_num_seqs": 1,
            "mlx_peak_memory_bytes": (10 if arm == "baseline" else 12) * 2**30,
        }
        events = [
            {
                "received_ns": start + 1,
                "data": {
                    "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]
                },
            },
            {
                "received_ns": first,
                "data": {
                    "choices": [
                        {"delta": {"content": text[:-1]}, "finish_reason": None}
                    ]
                },
            },
            {
                "received_ns": last,
                "data": {
                    "choices": [
                        {"delta": {"content": text[-1:]}, "finish_reason": None}
                    ]
                },
            },
            {
                "received_ns": finish,
                "data": {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            },
            {
                "received_ns": usage_time,
                "data": {"choices": [], "usage": copy.deepcopy(usage)},
            },
        ]
        return {
            "task_id": task,
            "suite": self.job_map[task]["suite"],
            "arm": arm,
            "block": block,
            "process_id": process["process_id"],
            "index": index,
            "measured_index": None if index == 0 else index - 1,
            "warmup": index == 0,
            "preceding_task_ids": [
                process["warmup_task_id"],
                *process["measured_task_ids"],
            ][:index],
            "request_sha256": self.payload_map[task]["request_sha256"],
            "max_tokens": self.job_map[task]["max_tokens"],
            "http_status": 200,
            "stream_done": True,
            "text": text,
            "reasoning_content": "",
            "finish_reason": "stop",
            "output_sha256": MODULE.digest(text.encode()),
            "completed_answer_latency_eligible": True,
            "usage": usage,
            "request_start_ns": start,
            "first_content_ns": first,
            "last_content_ns": last,
            "done_ns": done,
            "http_interval_end_ns": done + 1,
            "client_ttft_seconds": (first - start) / 1e9,
            "wall_seconds": (done - start) / 1e9,
            "http_interval_seconds": (done + 1 - start) / 1e9,
            "sse_events": events,
            "runtime_before": copy.deepcopy(snapshot),
            "runtime_after": copy.deepcopy(snapshot),
        }

    def save(self):
        for pid in self.runs:
            write_json(self.runs[pid] / "receipt.json", self.receipts[pid])
            write_rows(self.runs[pid] / "responses.jsonl", self.rows[pid])

    def report(self):
        self.save()
        return MODULE.summarize(self.plan_path, self.runs, self.wheels, self.root)

    def change_text(self, pid, task, suffix):
        row = next(r for r in self.rows[pid] if r["task_id"] == task)
        row["text"] += suffix
        row["output_sha256"] = MODULE.digest(row["text"].encode())
        row["sse_events"][2]["data"]["choices"][0]["delta"]["content"] += suffix


class TestPerformanceSummary(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))

    def codes(self, report):
        return {row["code"] for row in report["integrity"]["errors"]}

    def test_complete_pairing_and_equal_case_geometric_means(self):
        result = self.fixture.report()
        self.assertEqual(result["status"], "passed", result["integrity"]["errors"])
        self.assertEqual(
            result["observed"],
            {
                "processes": 4,
                "measured_requests": 32,
                "valid_pairs": 16,
                "excluded_warmups": 4,
            },
        )
        case = result["aggregates"]["per_case"]["case-1"]["client_ttft_seconds"]
        self.assertEqual(case["block_ratios"], [4.0, 16.0])
        self.assertAlmostEqual(case["geomean_ratio"], 8.0)
        group = result["aggregates"]["groups"]["synthetic_1k"]
        self.assertAlmostEqual(
            group["metrics"]["client_ttft_seconds"]["geomean_ratio"], 4 * math.sqrt(2)
        )
        self.assertAlmostEqual(
            group["metrics"]["generation_tokens_per_second"]["geomean_ratio"],
            6 * math.sqrt(2),
        )
        self.assertEqual(result["completion_latency"]["eligible_pair_count"], 16)
        self.assertEqual(
            result["memory"]["block_deltas"][0]["candidate_minus_baseline_bytes"],
            2 * 2**30,
        )

    def test_paired_text_mismatch_blocks_every_aggregate(self):
        self.fixture.change_text("block-0-candidate", "case-0", " changed")
        result = self.fixture.report()
        self.assertEqual(result["status"], "failed")
        self.assertIn("paired_text", self.codes(result))
        self.assertIsNone(result["aggregates"])
        self.assertEqual(len(result["pairs"]), 16)

    def test_cross_block_variation_fails_even_when_each_pair_matches(self):
        for arm in ("baseline", "candidate"):
            self.fixture.change_text("block-1-" + arm, "case-0", " changed")
        result = self.fixture.report()
        self.assertEqual(result["observed"]["valid_pairs"], 16)
        self.assertIn("within_arm_block_variation", self.codes(result))
        self.assertIsNone(result["aggregates"])

    def test_matched_length_stops_keep_primary_metrics_not_completion_latency(self):
        for pid, rows in self.fixture.rows.items():
            row = next(r for r in rows if r["task_id"] == "case-0")
            row["finish_reason"] = "length"
            row["completed_answer_latency_eligible"] = False
            row["sse_events"][3]["data"]["choices"][0]["finish_reason"] = "length"
            self.fixture.receipts[pid]["truncated"] = 1
        result = self.fixture.report()
        self.assertEqual(result["status"], "passed", result["integrity"]["errors"])
        self.assertEqual(result["observed"]["valid_pairs"], 16)
        self.assertEqual(result["completion_latency"]["eligible_pair_count"], 14)
        group = result["aggregates"]["groups"]["synthetic_1k"]["metrics"]
        self.assertEqual(
            group["client_ttft_seconds"]["eligible_task_ids"], ["case-0", "case-1"]
        )
        self.assertEqual(
            group["completed_answer_latency_seconds"]["eligible_task_ids"], ["case-1"]
        )

    def test_missing_source_identity_fails(self):
        del self.fixture.receipts["block-0-candidate"]["python_sources"]
        result = self.fixture.report()
        self.assertIn("receipt_identity", self.codes(result))
        self.assertIsNone(result["aggregates"])

    def test_missing_process_remains_visible_in_incomplete_report(self):
        del self.fixture.runs["block-0-baseline"]
        result = self.fixture.report()
        self.assertIn("missing_run", self.codes(result))
        self.assertEqual(len(result["pairs"]), 16)
        self.assertEqual(result["observed"]["valid_pairs"], 8)
        self.assertIsNone(result["aggregates"])

    def test_prefix_cache_hit_fails_even_with_matching_text(self):
        row = self.fixture.rows["block-0-candidate"][1]
        row["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
        row["sse_events"][-1]["data"]["usage"] = copy.deepcopy(row["usage"])
        result = self.fixture.report()
        self.assertIn("prefix_cache_hit", self.codes(result))
        self.assertIsNone(result["aggregates"])

    def test_timing_claim_must_match_observed_first_content(self):
        self.fixture.rows["block-0-candidate"][1]["client_ttft_seconds"] = 0.0001
        result = self.fixture.report()
        self.assertIn("client_timing_arithmetic", self.codes(result))
        self.assertIsNone(result["aggregates"])

    def test_dependency_change_fails(self):
        self.fixture.receipts["block-0-candidate"]["packages"]["mlx"] = "different"
        result = self.fixture.report()
        self.assertIn("dependency_identity", self.codes(result))
        self.assertIsNone(result["aggregates"])

    def test_failed_partial_stream_is_reported_without_dropping_other_pairs(self):
        row = self.fixture.rows["block-0-candidate"][1]
        row.update(
            error="ReadTimeout", stream_done=False, usage=None, finish_reason=None
        )
        row["sse_events"] = row["sse_events"][:2]
        self.fixture.rows["block-0-candidate"] = self.fixture.rows["block-0-candidate"][
            :2
        ]
        self.fixture.receipts["block-0-candidate"].update(
            status="failed", errors=1, measured_completed=1, attempts_completed=2
        )
        result = self.fixture.report()
        self.assertEqual(result["status"], "failed")
        self.assertIn("response_error", self.codes(result))
        self.assertIn("missing_pair", self.codes(result))
        self.assertEqual(len(result["pairs"]), 16)
        self.assertIsNone(result["aggregates"])

    def test_paired_completion_count_mismatch_fails(self):
        row = self.fixture.rows["block-0-candidate"][1]
        row["usage"]["completion_tokens"] = 3
        row["sse_events"][-1]["data"]["usage"] = copy.deepcopy(row["usage"])
        result = self.fixture.report()
        self.assertIn("paired_completion_tokens", self.codes(result))
        self.assertIsNone(result["aggregates"])

    def test_changed_quality_report_is_not_silently_trusted(self):
        path = self.fixture.root / "quality.json"
        report = json.loads(path.read_text())
        report["planned_acceptance"]["status"] = "pending"
        write_json(path, report)
        result = self.fixture.report()
        self.assertIn("quality_prerequisite", self.codes(result))
        self.assertIn("file_hash", self.codes(result))
        self.assertIsNone(result["aggregates"])


if __name__ == "__main__":
    unittest.main()
