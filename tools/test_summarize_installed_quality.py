#!/usr/bin/env python3
"""CPU-only fixture tests for paired evidence integrity and strict acceptance."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "summarize_installed_quality",
    Path(__file__).with_name("summarize_installed_quality.py"),
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class PairIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = [
            {
                "task_id": "HumanEval/0",
                "suite": "humaneval",
                "prompt": "Write a function",
                "max_tokens": 2048,
            },
            {
                "task_id": "math/0",
                "suite": "gsm8k-test-64-hash-subset",
                "prompt": "One plus one?",
                "max_tokens": 2048,
            },
        ]
        write_jsonl(self.root / "requests.jsonl", self.jobs)
        self.plan = {
            "requests_path": "requests.jsonl",
            "requests_sha256": reporter.digest(
                (self.root / "requests.jsonl").read_bytes()
            ),
            "tasks": 2,
            "suites": {"humaneval_plus": 1, "gsm8k_hash_subset": 1},
            "sampling": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "seed": 431017,
                "enable_thinking": False,
            },
            "baseline": {"version": "stock", "prefill_step": 2048},
            "candidate": {"version": "fork", "prefill_step": 8192},
            "harness_sha256": "frozen-harness",
            "acceptance": {
                "minimum_complete_pairs": 2,
                "allow_missing_or_errors": False,
                "allow_truncation": False,
                "maximum_baseline_pass_to_candidate_fail_tasks": 0,
                "quality_gains_offset_regressions": False,
            },
        }
        write_json(self.root / "plan.json", self.plan)
        self.responses = {}
        for arm in ("baseline", "candidate"):
            state = self.root / f"{arm}-state"
            settings = {
                "version": "1.0",
                "sampling": {"temperature": 0, "top_p": 1},
                "memory": {},
                "scheduler": {},
                "cache": {"ssd_cache_dir": str(state / "cache")},
                "auth": {"secret_key": "NEVER-PRINT-THIS-SECRET"},
            }
            model_settings = {
                "version": 1,
                "models": {
                    reporter.MODEL_ID: {
                        "temperature": 0,
                        "top_p": 1,
                        "force_sampling": False,
                        "enable_thinking": False,
                    }
                },
            }
            write_json(state / "settings.json", settings)
            write_json(state / "model_settings.json", model_settings)
            rows = [
                {
                    "task_id": j["task_id"],
                    "suite": j["suite"],
                    "index": i,
                    "max_tokens": j["max_tokens"],
                    "request_sha256": reporter.digest(
                        reporter.canonical(reporter.payload(j, self.plan))
                    ),
                    "text": "#### 2",
                    "finish_reason": "stop",
                    "http_status": 200,
                }
                for i, j in enumerate(self.jobs)
            ]
            self.responses[arm] = rows
            write_jsonl(self.root / arm / "responses.jsonl", rows)
            runtime = {
                "prefill_step_size": self.plan[arm]["prefill_step"],
                "max_num_seqs": 1,
            }
            receipt = {
                "arm": arm,
                "status": "generated",
                "controllers_closed": True,
                "plan_sha256": reporter.digest((self.root / "plan.json").read_bytes()),
                "requests_sha256": self.plan["requests_sha256"],
                "harness_sha256": self.plan["harness_sha256"],
                "runtime_version": self.plan[arm]["version"],
                "packages": {"omlx": self.plan[arm]["version"], "mlx": "0.32.4"},
                "total_tasks": 2,
                "completed": 2,
                "errors": 0,
                "truncated": 0,
                "runtime_initial": runtime,
                "runtime_latest": runtime,
                "native_assets": {"native.so": "samehash"},
                "python": "samepython",
                "device": {"name": "samegpu"},
                "memory_limit_bytes": 100,
                "settings_sha256": {
                    name: reporter.digest((state / name).read_bytes())
                    for name in ("settings.json", "model_settings.json")
                },
                "responses": "responses.jsonl",
            }
            write_json(self.root / arm / "receipt.json", receipt)
        self.make_supplement()
        self.args = SimpleNamespace(
            plan=self.root / "plan.json",
            requests=None,
            baseline_receipt=self.root / "baseline/receipt.json",
            candidate_receipt=self.root / "candidate/receipt.json",
            baseline_responses=None,
            candidate_responses=None,
            baseline_state_dir=self.root / "baseline-state",
            candidate_state_dir=self.root / "candidate-state",
            baseline_coding_grades=None,
            candidate_coding_grades=None,
            supplement_scores=self.root / "supplement.json",
        )

    def make_supplement(self, b=True, c=True):
        inputs = {}
        for arm in ("baseline", "candidate"):
            path = self.root / f"{arm}-supplement.jsonl"
            write_jsonl(path, [self.responses[arm][1]])
            inputs[arm] = {
                "file": path.name,
                "sha256": reporter.digest(path.read_bytes()),
            }
        write_json(
            self.root / "supplement.json",
            {
                "inputs": inputs,
                "pairs": [
                    {
                        "task_id": "math/0",
                        "suite": "gsm8k-test-64-hash-subset",
                        "baseline": {"correct": b},
                        "candidate": {"correct": c},
                    }
                ],
            },
        )

    def make_coding(self, b=True, c=True):
        for arm, correct in (("baseline", b), ("candidate", c)):
            outcome = {
                "status": "pass" if correct else "fail",
                "raw_sha256": reporter.digest(self.responses[arm][0]["text"].encode()),
            }
            grade = {
                "task_id": "HumanEval/0",
                "suite": "humaneval",
                "canonical_validation": False,
                "base": outcome,
                "plus": outcome,
                "base_pass": correct,
                "plus_pass": correct,
            }
            path = self.root / f"{arm}-grades.jsonl"
            write_jsonl(path, [grade])
            meta = {
                k: "same"
                for k in (
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
                )
            }
            meta.update(
                samples_sha256=reporter.digest(
                    (self.root / arm / "responses.jsonl").read_bytes()
                ),
                tasks=["HumanEval/0"],
                canonical_validation=False,
            )
            write_json(path.with_suffix(path.suffix + ".metadata.json"), meta)
            setattr(self.args, f"{arm}_coding_grades", path)

    def edit_json(self, relative, update):
        path = self.root / relative
        value = reporter.read_json(path)
        update(value)
        write_json(path, value)

    def codes(self, result):
        return {e["code"] for e in result["integrity"]["errors"]}

    def test_missing_coding_never_infers_pass(self):
        result = reporter.summarize(self.args)
        self.assertEqual(result["integrity"]["status"], "verified")
        self.assertEqual(result["original_acceptance"]["status"], "pending")
        self.assertEqual(result["graded_pairs"], 1)

    def test_complete_clean_pair_passes_only_observation_gate(self):
        self.make_coding()
        result = reporter.summarize(self.args)
        self.assertEqual(result["original_acceptance"]["status"], "passed")
        self.assertFalse(result["universal_quality_equivalence_proved"])
        self.assertTrue(result["observed_task_score_preservation_established"])
        self.assertNotIn("NEVER-PRINT", json.dumps(result))

    def test_gain_does_not_offset_regression(self):
        self.make_coding(b=False, c=True)
        self.make_supplement(b=True, c=False)
        result = reporter.summarize(self.args)
        self.assertEqual(result["original_acceptance"]["status"], "failed")
        self.assertEqual(result["regressions"], ["math/0"])

    def test_baseline_truncation_preserves_original_failure_even_when_candidate_passes(
        self,
    ):
        self.responses["baseline"][0]["finish_reason"] = "length"
        write_jsonl(self.root / "baseline/responses.jsonl", self.responses["baseline"])
        self.edit_json("baseline/receipt.json", lambda v: v.update(truncated=1))
        self.make_coding()
        result = reporter.summarize(self.args)
        self.assertEqual(result["integrity"]["status"], "verified")
        self.assertEqual(result["original_acceptance"]["status"], "failed")
        self.assertEqual(result["generation"]["baseline"]["truncated"], ["HumanEval/0"])

    def test_response_order_change_is_detected(self):
        write_jsonl(
            self.root / "candidate/responses.jsonl",
            list(reversed(self.responses["candidate"])),
        )
        self.assertIn(
            "task_order_or_membership", self.codes(reporter.summarize(self.args))
        )

    def test_request_payload_hash_change_is_detected(self):
        self.responses["candidate"][0]["request_sha256"] = "wrong"
        write_jsonl(
            self.root / "candidate/responses.jsonl", self.responses["candidate"]
        )
        self.assertIn("request_payload_hash", self.codes(reporter.summarize(self.args)))

    def test_native_or_dependency_change_is_detected(self):
        self.edit_json(
            "candidate/receipt.json",
            lambda v: v.update(
                native_assets={"native.so": "changed"},
                packages={"omlx": "fork", "mlx": "different"},
            ),
        )
        codes = self.codes(reporter.summarize(self.args))
        self.assertIn("runtime_identity_disagreement", codes)
        self.assertIn("dependency_disagreement", codes)

    def test_stale_settings_hash_is_detected_without_secret_leak(self):
        self.edit_json(
            "candidate-state/settings.json",
            lambda v: v["auth"].update(secret_key="OTHER-SECRET"),
        )
        result = reporter.summarize(self.args)
        self.assertIn("settings_hash", self.codes(result))
        self.assertNotIn("OTHER-SECRET", json.dumps(result))

    def test_supplement_output_binding_is_checked(self):
        self.responses["candidate"][1]["text"] = "#### 3"
        write_jsonl(
            self.root / "candidate/responses.jsonl", self.responses["candidate"]
        )
        self.assertIn(
            "supplement_response_binding", self.codes(reporter.summarize(self.args))
        )

    def test_coding_source_hash_is_checked(self):
        self.make_coding()
        path = self.root / "candidate-grades.jsonl"
        row = reporter.read_jsonl(path)[0]
        row["plus"]["raw_sha256"] = "wrong"
        write_jsonl(path, [row])
        self.assertIn("coding_raw_hash", self.codes(reporter.summarize(self.args)))

    def test_coding_plus_boolean_must_match_both_test_phases(self):
        self.make_coding()
        path = self.root / "candidate-grades.jsonl"
        row = reporter.read_jsonl(path)[0]
        row["plus"]["status"] = "fail"
        write_jsonl(path, [row])
        self.assertIn(
            "coding_plus_consistency", self.codes(reporter.summarize(self.args))
        )

    def test_no_settings_files_leaves_integrity_pending(self):
        self.args.candidate_state_dir = None
        self.make_coding()
        result = reporter.summarize(self.args)
        self.assertEqual(result["integrity"]["status"], "pending")
        self.assertEqual(result["original_acceptance"]["status"], "pending")

    def amend(self):
        original = self.root / "plan.json"
        new_plan = json.loads(json.dumps(self.plan))
        new_plan["harness_sha256"] = "new-observer-harness"
        old_hash = reporter.digest(original.read_bytes())
        new_plan["reused_baseline"] = {
            "plan_sha256": old_hash,
            "receipt_sha256": reporter.digest(
                (self.root / "baseline/receipt.json").read_bytes()
            ),
            "responses_sha256": reporter.digest(
                (self.root / "baseline/responses.jsonl").read_bytes()
            ),
        }
        new_plan["acceptance"].update(
            mode="amended-stock-arithmetic-v1",
            amendment_reason="Baseline itself reaches the finite output budget; require exact paired outputs instead of absolute no-truncation",
            original_protocol_sha256=old_hash,
            allow_truncation=True,
            allow_candidate_only_truncation=False,
            require_exact_text_and_finish_reason=True,
        )
        amended = self.root / "amended-plan.json"
        write_json(amended, new_plan)
        self.args.plan = amended
        self.args.baseline_original_plan = original
        self.edit_json(
            "candidate/receipt.json",
            lambda v: v.update(
                plan_sha256=reporter.digest(amended.read_bytes()),
                harness_sha256="new-observer-harness",
            ),
        )

    def test_exact_startup_field_reconstruction_accepts_only_original_byte_hash(self):
        path = self.root / "candidate-state/settings.json"
        original = reporter.read_json(path)
        original["auth"]["secret_key"] = None
        original["server"] = {"server_aliases": []}
        startup = json.dumps(original, indent=2, ensure_ascii=False).encode()
        self.edit_json(
            "candidate/receipt.json",
            lambda v: v["settings_sha256"].update(
                {"settings.json": reporter.digest(startup)}
            ),
        )
        original["auth"]["secret_key"] = "GENERATED-SECRET"
        original["server"]["server_aliases"] = ["localhost"]
        path.write_text(json.dumps(original, indent=2))
        result = reporter.summarize(self.args)
        self.assertNotIn("settings_hash", self.codes(result))
        self.assertEqual(len(result["integrity"]["settings_reconstructions"]), 1)
        self.assertNotIn("GENERATED-SECRET", json.dumps(result))

    def test_reused_baseline_all_hashes_bind_old_harness_to_new_plan(self):
        self.make_coding()
        self.amend()
        result = reporter.summarize(self.args)
        self.assertEqual(result["integrity"]["status"], "verified")
        self.assertEqual(result["planned_acceptance"]["status"], "passed")
        self.assertEqual(result["original_acceptance"]["status"], "not_recomputed")

    def test_reused_baseline_cannot_waive_receipt_hash(self):
        self.make_coding()
        self.amend()
        self.edit_json(
            "baseline/receipt.json", lambda v: v.update(harness_sha256="unbound")
        )
        result = reporter.summarize(self.args)
        self.assertIn("reused_baseline_receipt_hash", self.codes(result))
        self.assertIn("reused_baseline_harness_hash", self.codes(result))

    def test_amended_parity_gate_can_pass_while_original_truncation_rule_stays_failed(
        self,
    ):
        for arm in ("baseline", "candidate"):
            self.responses[arm][0]["finish_reason"] = "length"
            write_jsonl(self.root / arm / "responses.jsonl", self.responses[arm])
            self.edit_json(f"{arm}/receipt.json", lambda v: v.update(truncated=1))
        self.make_coding()
        self.amend()
        result = reporter.summarize(self.args)
        self.assertEqual(result["planned_acceptance"]["status"], "passed")
        self.assertEqual(result["original_acceptance"]["status"], "failed")

    def test_amended_gate_requires_exact_text_even_with_all_correct_scores(self):
        self.responses["candidate"][0]["text"] = "A different but passing solution"
        write_jsonl(
            self.root / "candidate/responses.jsonl", self.responses["candidate"]
        )
        self.make_coding()
        self.amend()
        result = reporter.summarize(self.args)
        self.assertEqual(result["planned_acceptance"]["status"], "failed")
        self.assertEqual(result["regressions"], [])
        self.assertEqual(
            result["exact_text_and_finish_reason"]["differences"], ["HumanEval/0"]
        )

    def test_amended_gate_rejects_candidate_only_length_stop(self):
        self.responses["candidate"][0]["finish_reason"] = "length"
        write_jsonl(
            self.root / "candidate/responses.jsonl", self.responses["candidate"]
        )
        self.edit_json("candidate/receipt.json", lambda v: v.update(truncated=1))
        self.make_coding()
        self.amend()
        result = reporter.summarize(self.args)
        self.assertEqual(result["planned_acceptance"]["status"], "failed")
        self.assertEqual(result["candidate_only_truncated"], ["HumanEval/0"])


if __name__ == "__main__":
    unittest.main()
