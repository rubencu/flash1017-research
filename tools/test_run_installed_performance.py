# SPDX-License-Identifier: Apache-2.0
"""CPU-only streaming and input-integrity tests; no server or model imports."""

import importlib.util
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "performance_runner", Path(__file__).with_name("run_installed_performance.py")
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
MODEL = "DeepSeek-V4.1-Flash-1017"

# Exact _chat_keepalive_chunk("chatcmpl-1234abcd") wire representation in
# pinned stock/candidate server.py. No server imports or inference required.
NATIVE_KEEPALIVE = (
    'data: {"id":"chatcmpl-1234abcd","object":"chat.completion.chunk",'
    '"created":0,"model":"keepalive",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    '"finish_reason":null}]}\n\n'
)


def choice(delta=None, finish=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def frame(value):
    if isinstance(value, dict):
        value = {"model": MODEL, **value}
    return ["data: " + (value if isinstance(value, str) else json.dumps(value)), ""]


USAGE = {
    "prompt_tokens": 1024,
    "completion_tokens": 2,
    "prompt_tokens_details": {"cached_tokens": 0},
    "prompt_eval_duration": 1.0,
    "generation_duration": 0.5,
    "prompt_tokens_per_second": 1024.0,
    "generation_tokens_per_second": 4.0,
}


class TestSSE(unittest.TestCase):
    def parse(self, lines, row=None):
        ticks = iter(range(1_000_000_000, 100_000_000_000, 1_000_000_000))
        return RUNNER.consume_sse(
            lines, {} if row is None else row, 0, lambda: next(ticks)
        )

    def standard(self):
        return (
            frame(choice({"role": "assistant"}))
            + frame(choice({"content": ""}))
            + frame(choice({"content": "Hello"}))
            + frame(choice({"content": " world"}))
            + frame(choice(finish="stop"))
            + frame({"choices": [], "usage": USAGE})
            + frame("[DONE]")
        )

    def test_visible_ttft_ignores_role_empty_and_usage_chunks(self):
        row = self.parse(self.standard())
        self.assertEqual(row["text"], "Hello world")
        self.assertEqual(row["client_ttft_seconds"], 3.0)
        self.assertEqual(row["first_content_ns"], 3_000_000_000)
        self.assertEqual(row["last_content_ns"], 4_000_000_000)
        self.assertEqual(row["wall_seconds"], 7.0)
        self.assertEqual(row["usage"], USAGE)
        self.assertEqual(row["finish_reason"], "stop")
        self.assertTrue(row["stream_done"])

    def test_comments_and_multiline_data(self):
        lines = [
            ": heartbeat",
            "",
            "event: message",
            'data: {"model":"DeepSeek-V4.1-Flash-1017","choices":',
            'data: [{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}',
            "",
        ]
        lines += frame({"choices": [], "usage": USAGE}) + frame("[DONE]")
        self.assertEqual(self.parse(lines)["text"], "Hi")

    def test_last_usage_is_authoritative(self):
        lines = frame({"choices": [], "usage": {"completion_tokens": 1}})
        row = self.parse(lines + self.standard())
        self.assertEqual(row["usage"], USAGE)

    def test_reasoning_is_preserved_but_does_not_start_content_clock(self):
        lines = frame(choice({"reasoning_content": "internal"})) + self.standard()
        row = self.parse(lines)
        self.assertEqual(row["client_ttft_seconds"], 4.0)
        self.assertEqual(row["reasoning_content"], "internal")

    def test_missing_done_is_failure_with_partial_answer_retained(self):
        row = {}
        with self.assertRaisesRegex(ValueError, "without DONE"):
            self.parse(self.standard()[:-2], row)
        self.assertEqual(row["text"], "Hello world")
        self.assertFalse(row["stream_done"])

    def test_truncated_frame_is_failure(self):
        with self.assertRaisesRegex(ValueError, "Truncated SSE frame"):
            self.parse(self.standard()[:-1])

    def test_malformed_json_is_preserved(self):
        row = {}
        with self.assertRaisesRegex(ValueError, "Malformed JSON"):
            self.parse(frame(choice({"content": "partial"})) + frame("{bad"), row)
        self.assertEqual(row["text"], "partial")
        self.assertEqual(row["malformed_sse_data"], "{bad")

    def test_data_after_done_fails(self):
        with self.assertRaisesRegex(ValueError, "after terminal DONE"):
            self.parse(self.standard() + frame(choice({"content": "late"})))

    def test_missing_usage_fails(self):
        lines = frame(choice({"content": "Hi"}, "stop")) + frame("[DONE]")
        with self.assertRaisesRegex(ValueError, "without usage"):
            self.parse(lines)

    def test_missing_finish_reason_fails(self):
        lines = (
            frame(choice({"content": "Hi"}))
            + frame({"choices": [], "usage": USAGE})
            + frame("[DONE]")
        )
        with self.assertRaisesRegex(ValueError, "without finish reason"):
            self.parse(lines)

    def test_role_only_has_no_visible_ttft(self):
        lines = (
            frame(choice({"role": "assistant"}, "stop"))
            + frame({"choices": [], "usage": USAGE})
            + frame("[DONE]")
        )
        row = {}
        with self.assertRaisesRegex(ValueError, "no visible"):
            self.parse(lines, row)
        self.assertIsNone(row["first_content_ns"])
        self.assertNotIn("client_ttft_seconds", row)

    def test_content_after_finish_fails(self):
        lines = frame(choice({"content": "Hi"}, "stop")) + frame(
            choice({"content": "late"})
        )
        with self.assertRaisesRegex(ValueError, "after finish reason"):
            self.parse(lines)

    def test_response_validation_preserves_length_stops(self):
        row = self.parse(self.standard())
        row["finish_reason"] = "length"
        RUNNER.validate_response(
            row, {"expected_prompt_tokens": 1024, "max_tokens": 512}
        )
        self.assertFalse(row["completed_answer_latency_eligible"])

    def test_prefix_hits_and_absent_timings_are_failures(self):
        for change in ("cache", "timing"):
            row = self.parse(self.standard())
            if change == "cache":
                row["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
            else:
                del row["usage"]["generation_tokens_per_second"]
            with self.assertRaises(ValueError):
                RUNNER.validate_response(
                    row, {"expected_prompt_tokens": 1024, "max_tokens": 512}
                )

    def test_explicit_empty_or_data_error_events_fail_before_later_success(self):
        for lines in (
            ["event: error", ""],
            ["event:error", 'data: {"message":"failed"}', ""],
            ['data: {"message":"failed"}', "event: error", ""],
        ):
            with self.subTest(lines=lines):
                row = {}
                with self.assertRaisesRegex(ValueError, "Explicit SSE error"):
                    self.parse(lines + self.standard(), row)
                self.assertEqual(row["invalid_sse_event_name"], "error")
                self.assertEqual(row["text"], "")
                self.assertFalse(row["stream_done"])

    def test_error_key_cannot_be_hidden_by_empty_value(self):
        for value in (None, "", {}, False):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "error SSE"
            ):
                self.parse(frame({"error": value}) + self.standard())

    def test_foreign_or_missing_model_is_not_an_answer(self):
        for model in ("another-model", "keepalive-with-content", None):
            event = {"choices": [{"index": 0, "delta": {"content": "wrong"}}]}
            if model is not None:
                event["model"] = model
            lines = ["data: " + json.dumps(event), ""]
            with self.subTest(model=model), self.assertRaisesRegex(
                ValueError, "response model"
            ):
                self.parse(lines + self.standard())

    def test_native_keepalive_is_saved_without_any_completion_witness(self):
        lines = NATIVE_KEEPALIVE.splitlines()
        row = self.parse(lines + self.standard())
        self.assertEqual(row["text"], "Hello world")
        self.assertEqual(row["sse_events"][0]["data"]["model"], "keepalive")
        self.assertEqual(row["first_content_ns"], 4_000_000_000)
        self.assertEqual(row["client_ttft_seconds"], 4.0)
        self.assertEqual(row["wall_seconds"], 8.0)
        incomplete = {}
        with self.assertRaisesRegex(ValueError, "without DONE"):
            self.parse(lines, incomplete)
        for key in ("first_content_ns", "last_content_ns", "usage", "finish_reason"):
            self.assertIsNone(incomplete[key])
        self.assertEqual(incomplete["text"], "")

    def test_malformed_keepalive_cannot_smuggle_content_or_usage(self):
        native = json.loads(NATIVE_KEEPALIVE.removeprefix("data: "))
        variants = []
        for key, value in (
            ("created", False),
            ("created", 0.0),
            ("created", 1),
            ("id", "chatcmpl-keepalive"),
            ("id", "chatcmpl-1234abcd-extra"),
            ("object", "chat.completion"),
            ("usage", USAGE),
            ("prompt_progress", {"processed": 1}),
            ("extra", None),
        ):
            event = copy.deepcopy(native)
            event[key] = value
            variants.append(event)
        for key, value in (
            ("index", False),
            ("index", 0.0),
            ("index", 1),
            ("finish_reason", "stop"),
            ("extra", None),
            ("delta", {"role": "assistant", "content": "visible"}),
            ("delta", {"role": "assistant", "content": "", "tool_calls": []}),
        ):
            event = copy.deepcopy(native)
            event["choices"][0][key] = value
            variants.append(event)
        event = copy.deepcopy(native)
        event["choices"] = []
        variants.append(event)
        for event in variants:
            with self.subTest(event=event), self.assertRaisesRegex(
                ValueError, "Malformed native keepalive"
            ):
                self.parse(frame(event) + self.standard())

    def test_keepalive_after_done_still_fails(self):
        with self.assertRaisesRegex(ValueError, "after terminal DONE"):
            self.parse(self.standard() + NATIVE_KEEPALIVE.splitlines())

    def test_native_index_requires_plain_zero_integer(self):
        for index in (False, 0.0, "0", None):
            event = choice({"content": "wrong"})
            if index is None:
                del event["choices"][0]["index"]
            else:
                event["choices"][0]["index"] = index
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "choice index"
            ):
                self.parse(frame(event) + self.standard())

    def test_token_counts_and_cache_hits_require_plain_integers(self):
        for field, value in (
            ("prompt_tokens", 1024.0),
            ("completion_tokens", True),
            ("completion_tokens", 2.0),
            ("cached_tokens", False),
            ("cached_tokens", 0.0),
        ):
            row = self.parse(self.standard())
            usage = row["usage"]
            target = (
                usage["prompt_tokens_details"] if field == "cached_tokens" else usage
            )
            target[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                RUNNER.validate_response(
                    row, {"expected_prompt_tokens": 1024, "max_tokens": 512}
                )

    def test_native_timing_rejects_boolean_nonfinite_or_nonpositive_values(self):
        for field in (
            "prompt_eval_duration",
            "generation_duration",
            "prompt_tokens_per_second",
            "generation_tokens_per_second",
        ):
            for value in (True, False, float("nan"), float("inf"), 0, -1, "1"):
                row = self.parse(self.standard())
                row["usage"][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    ValueError, "native timing"
                ):
                    RUNNER.validate_response(
                        row, {"expected_prompt_tokens": 1024, "max_tokens": 512}
                    )


class TestWheelIdentity(unittest.TestCase):
    def test_installed_package_must_match_wheel_not_only_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "omlx"
            package.mkdir()
            installed = package / "__init__.py"
            installed.write_bytes(b"VERSION='test'\n")
            wheel = root / "test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("omlx/__init__.py", installed.read_bytes())
            digest = RUNNER.digest(wheel.read_bytes())
            identity = RUNNER.verify_wheel(package, wheel, digest)
            self.assertIn("__init__.py", identity["python_sources"])
            installed.write_bytes(b"VERSION='same version, changed source'\n")
            with self.assertRaisesRegex(ValueError, "differs from wheel"):
                RUNNER.verify_wheel(package, wheel, digest)


class TestSharedEnvironment(unittest.TestCase):
    def fixture(self, root):
        model = root / "model"
        model.mkdir()
        for name in RUNNER.CHECKPOINT_METADATA:
            if name != "generation_config.json":
                (model / name).write_bytes((name + " pinned bytes").encode())
        packages = [("MLX", "0.32.4"), ("mlx_lm", "0.31.4"), ("oMLX", "baseline")]
        python = "3.12.15 (full build identity) [Clang 22.1.3]"
        assets = {"custom_kernels/head/_ext.so": "a" * 64, "head.metallib": "b" * 64}
        expected = {
            "packages": {"mlx": "0.32.4", "mlx-lm": "0.31.4"},
            "python": python,
            "native_assets": assets,
            "model_metadata_sha256": {
                name: (
                    RUNNER.digest((model / name).read_bytes())
                    if (model / name).exists()
                    else None
                )
                for name in RUNNER.CHECKPOINT_METADATA
            },
        }
        return (
            {"shared_environment": expected},
            model,
            {
                "packages": packages,
                "python_version": python,
                "native_assets": assets,
            },
        )

    def test_complete_shared_identity_accepts_only_omlx_version_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, model, actual = self.fixture(Path(tmp))
            observed = RUNNER.verify_shared_environment(plan, model, **actual)
            self.assertEqual(observed, plan["shared_environment"])
            actual["packages"] = [
                ("mlx", "0.32.4"),
                ("MLX-LM", "0.31.4"),
                ("omlx", "candidate"),
            ]
            self.assertEqual(
                RUNNER.verify_shared_environment(plan, model, **actual), observed
            )
            self.assertNotIn("omlx", observed["packages"])

    def test_changed_missing_or_extra_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, model, actual = self.fixture(Path(tmp))
            for packages in (
                [("mlx", "0.32.5"), ("mlx-lm", "0.31.4")],
                [("mlx", "0.32.4")],
                actual["packages"] + [("extra", "1.0")],
            ):
                with self.subTest(packages=packages), self.assertRaisesRegex(
                    ValueError, "environment.*packages"
                ):
                    RUNNER.verify_shared_environment(
                        plan, model, **{**actual, "packages": packages}
                    )

    def test_normalized_name_collisions_are_not_silently_overwritten(self):
        for records in (
            [("MLX_LM", "1"), ("mlx-lm", "1")],
            [("mlx.lm", "1"), ("MLX--LM", "2")],
            [("OMLX", "stock"), ("omlx", "candidate")],
        ):
            with self.subTest(records=records), self.assertRaisesRegex(
                ValueError, "Duplicate normalized"
            ):
                RUNNER.normalized_packages(records)

    def test_full_python_build_string_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, model, actual = self.fixture(Path(tmp))
            for version in ("3.12.15", actual["python_version"] + " changed build"):
                with self.subTest(version=version), self.assertRaisesRegex(
                    ValueError, "environment.*python"
                ):
                    RUNNER.verify_shared_environment(
                        plan, model, **{**actual, "python_version": version}
                    )

    def test_changed_or_missing_required_metadata_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, model, actual = self.fixture(Path(tmp))
            for name in RUNNER.CHECKPOINT_METADATA:
                if name == "generation_config.json":
                    continue
                path = model / name
                original = path.read_bytes()
                for action in ("change", "remove"):
                    if action == "change":
                        path.write_bytes(original + b" ")
                    else:
                        path.unlink()
                    with self.subTest(name=name, action=action), self.assertRaisesRegex(
                        ValueError, "environment.*metadata"
                    ):
                        RUNNER.verify_shared_environment(plan, model, **actual)
                    path.write_bytes(original)

    def test_optional_generation_metadata_absence_is_an_explicit_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, model, actual = self.fixture(Path(tmp))
            path = model / "generation_config.json"
            path.write_bytes(b'{"temperature": 1.0}')
            with self.assertRaisesRegex(ValueError, "environment.*metadata"):
                RUNNER.verify_shared_environment(plan, model, **actual)
            plan["shared_environment"]["model_metadata_sha256"][path.name] = (
                RUNNER.digest(path.read_bytes())
            )
            RUNNER.verify_shared_environment(plan, model, **actual)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "environment.*metadata"):
                RUNNER.verify_shared_environment(plan, model, **actual)

    def test_native_asset_bytes_and_inventory_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, model, actual = self.fixture(Path(tmp))
            for assets in (
                {**actual["native_assets"], "head.metallib": "c" * 64},
                {"head.metallib": "b" * 64},
                {**actual["native_assets"], "extra.so": "c" * 64},
            ):
                with self.subTest(assets=assets), self.assertRaisesRegex(
                    ValueError, "environment.*native_assets"
                ):
                    RUNNER.verify_shared_environment(
                        plan, model, **{**actual, "native_assets": assets}
                    )

    def test_incomplete_or_noncanonical_expected_maps_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, model, actual = self.fixture(Path(tmp))
            bad = []
            for name in plan["shared_environment"]:
                changed = copy.deepcopy(plan)
                del changed["shared_environment"][name]
                bad.append(changed)
            for key, value in (
                ("packages", {"MLX": "0.32.4"}),
                ("packages", {"mlx": "0.32.4", "omlx": "baseline"}),
                ("packages", {}),
                ("python", ""),
                ("model_metadata_sha256", {"config.json": "a" * 64}),
                ("native_assets", {}),
                ("native_assets", {"../kernel.so": "a" * 64}),
                ("native_assets", {"kernel.so": "invalid"}),
            ):
                changed = copy.deepcopy(plan)
                changed["shared_environment"][key] = value
                bad.append(changed)
            for changed in bad:
                with self.subTest(plan=changed), self.assertRaises(ValueError):
                    RUNNER.verify_shared_environment(changed, model, **actual)


class TestTimeoutDeclaration(unittest.TestCase):
    def fixture(self):
        return {
            "process_wall_timeout_seconds": 5400,
            "http": {
                "io_inactivity_timeout_seconds": 1200,
                "timeout_semantics": "per-I/O inactivity, not a cumulative request deadline",
            },
        }

    def test_wall_bound_and_io_inactivity_are_separate(self):
        self.assertEqual(RUNNER.validate_timeouts(self.fixture()), (5400, 1200))

    def test_missing_invalid_or_ambiguous_limits_are_rejected(self):
        for field in ("process_wall_timeout_seconds", "io_inactivity_timeout_seconds"):
            for value in (None, False, 0, -1, 1200.0, "1200"):
                plan = self.fixture()
                target = plan if field.startswith("process") else plan["http"]
                if value is None:
                    del target[field]
                else:
                    target[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(
                    ValueError
                ):
                    RUNNER.validate_timeouts(plan)
        plan = self.fixture()
        plan["http"]["request_timeout_seconds"] = 1200
        with self.assertRaises(ValueError):
            RUNNER.validate_timeouts(plan)
        plan = self.fixture()
        plan["http"]["timeout_semantics"] = "cumulative request deadline"
        with self.assertRaises(ValueError):
            RUNNER.validate_timeouts(plan)


class TestAllocatorObservation(unittest.TestCase):
    def test_only_read_only_counters_are_called_without_peak_reset(self):
        calls = []

        class FakeMLX:
            def get_active_memory(self):
                calls.append("active")
                return 100

            def get_cache_memory(self):
                calls.append("cache")
                return 20

            def get_peak_memory(self):
                calls.append("peak")
                return 180

            def __getattr__(self, name):
                raise AssertionError("Unexpected MLX operation: " + name)

        result = RUNNER.allocator_observation(FakeMLX())
        self.assertEqual(calls, ["active", "cache", "peak"])
        self.assertEqual(result["mlx_active_memory_bytes"], 100)
        self.assertEqual(result["mlx_cache_memory_bytes"], 20)
        self.assertEqual(result["mlx_peak_memory_bytes"], 180)
        self.assertIn("including load/warmup", result["mlx_peak_memory_scope"])
        self.assertIn("not a measured-request-only", result["mlx_peak_memory_scope"])


class TestQualityPrerequisite(unittest.TestCase):
    def fixture(self, root):
        builds = {
            "baseline": {"version": "stock", "wheel_sha256": "a" * 64},
            "candidate": {"version": "candidate", "wheel_sha256": "b" * 64},
        }
        report = {
            "planned_acceptance": {"status": "passed"},
            "integrity": {"status": "verified"},
            "graded_pairs": 618,
            "paired_responses": 618,
            "observed_exact_response_parity_established": True,
            "observed_task_score_preservation_established": True,
            "builds": builds,
        }
        path = root / "report.json"
        path.write_text(json.dumps(report))
        plan = {
            **builds,
            "quality_report_path": path.name,
            "quality_report_sha256": RUNNER.digest(path.read_bytes()),
        }
        return plan, report, path

    def test_passed_exact_report_for_same_wheels_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _, path = self.fixture(root)
            self.assertEqual(RUNNER.verify_quality_report(plan, root), path)

    def test_changed_report_bytes_fail_even_if_still_passed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _, path = self.fixture(root)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "differs from frozen plan"):
                RUNNER.verify_quality_report(plan, root)

    def test_pending_incomplete_or_nonidentical_quality_is_not_eligible(self):
        changes = [
            {"planned_acceptance": {"status": "pending"}},
            {"integrity": {"status": "failed"}},
            {"graded_pairs": 617},
            {"paired_responses": 617},
            {"observed_exact_response_parity_established": False},
            {"observed_task_score_preservation_established": False},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for change in changes:
                with self.subTest(change=change):
                    plan, report, path = self.fixture(root)
                    report.update(change)
                    path.write_text(json.dumps(report))
                    plan["quality_report_sha256"] = RUNNER.digest(path.read_bytes())
                    with self.assertRaisesRegex(ValueError, "has not passed"):
                        RUNNER.verify_quality_report(plan, root)

    def test_passed_report_for_another_wheel_is_not_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _, _ = self.fixture(root)
            plan["candidate"] = {"version": "candidate", "wheel_sha256": "c" * 64}
            with self.assertRaisesRegex(ValueError, "not approved for this candidate"):
                RUNNER.verify_quality_report(plan, root)


if __name__ == "__main__":
    unittest.main()
