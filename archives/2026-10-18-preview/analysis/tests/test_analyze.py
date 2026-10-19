"""CPU-only contract tests against packaged observations; no inference imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portable_analysis", HERE / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
EVIDENCE = HERE.parent / "evidence" / "overall"


class SavedObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = analysis.decode((EVIDENCE / "plan-01.json").read_text())
        cls.rows = [[analysis.decode(line) for line in (EVIDENCE / "raw" / name).read_text().splitlines()]
                    for name in analysis.FILES]

    def setUp(self):
        self.data = copy.deepcopy(self.rows)

    def first_measured(self):
        return next(r for r in self.data[0] if r["type"] == "run" and not r["warmup"])

    def reject(self, expression):
        with self.assertRaisesRegex(analysis.ValidationError, expression):
            analysis.summarize(self.plan, self.data)

    def test_complete_data_matches_saved_reviews(self):
        report = analysis.analyze(EVIDENCE / "plan-01.json", EVIDENCE / "raw", EVIDENCE / "review")
        self.assertEqual(report["counts"], {"processes": 6, "measured_requests": 48, "excluded_warmups": 6, "unique_cases": 8})
        self.assertFalse(report["fixed512"]["qualified"])
        self.assertEqual(report["fixed512"]["early_eos_count"], 9)
        self.assertTrue(report["prefix128"]["available_for_all48"])
        self.assertAlmostEqual(report["headline"]["improvement_percent"], 20.06083918786373, places=10)
        self.assertAlmostEqual(report["memory_tradeoff_gib_vs_native2048"], 7.748337350785732, places=9)
        self.assertEqual([r["fixture"] for r in report["ttft_regressions_vs_native2048"]], ["performance-cpython-65536"])
        self.assertTrue(report["saved_review_comparison"]["matched"])

    def test_missing_process_rejected(self):
        self.data.pop()
        self.reject("six raw processes")

    def test_missing_request_rejected(self):
        self.data[0].pop(4)
        self.reject("Missing, duplicate or misordered")

    def test_duplicate_request_rejected(self):
        self.data[0].insert(4, copy.deepcopy(self.data[0][4]))
        self.reject("Missing, duplicate or misordered")

    def test_duplicate_case_with_same_record_count_rejected(self):
        self.data[0][5] = copy.deepcopy(self.data[0][4])
        self.reject("identity/order/profile")

    def test_missing_cleanup_rejected(self):
        self.data[0].pop()
        self.reject("Missing, duplicate or misordered")

    def test_request_order_rejected(self):
        self.data[0][4], self.data[0][5] = self.data[0][5], self.data[0][4]
        self.reject("identity/order/profile")

    def test_seed_mismatch_rejected(self):
        self.first_measured()["seed"] += 1
        self.reject("identity/order/profile")

    def test_prompt_hash_mismatch_rejected(self):
        self.first_measured()["input_tokens_sha256"] = "0" * 64
        self.reject("Input mismatch")

    def test_wrong_output_length_rejected(self):
        self.first_measured()["token_ids"].pop()
        self.reject("output length")

    def test_boolean_token_rejected(self):
        self.first_measured()["token_ids"][0] = True
        self.reject("output length/token type")

    def test_out_of_vocabulary_token_rejected(self):
        self.first_measured()["token_ids"][0] = analysis.VOCAB_SIZE
        self.reject("output length/token type")

    def test_eos_claim_checked_against_actual_token_ids(self):
        row = self.first_measured()
        self.assertIsNone(row["first_eos_position"])
        row["first_eos_position"] = 130
        row["fixed_length_valid"] = False
        self.reject("EOS disagrees with tokens")

    def test_unreported_eos_rejected(self):
        self.first_measured()["token_ids"][130] = analysis.EOS_TOKEN_ID
        self.reject("EOS disagrees with tokens")

    def set_eos(self, position):
        row = self.first_measured()
        self.assertNotIn(analysis.EOS_TOKEN_ID, row["token_ids"])
        row["token_ids"][position] = analysis.EOS_TOKEN_ID
        row["first_eos_position"] = position
        row["fixed_length_valid"] = position == 511

    def test_eos127_blocks_entire_prefix_analysis_without_subset(self):
        self.set_eos(127)
        report = analysis.summarize(self.plan, self.data)
        self.assertEqual(report["counts"]["measured_requests"], 48)
        self.assertFalse(report["prefix128"]["available_for_all48"])
        self.assertFalse(report["prefix128"]["subset_or_shorter_horizon_used"])
        self.assertNotIn("prefix128_delivery_seconds", report["pair_group_summaries"][0]["metrics"])
        self.assertEqual(report["fixed512"]["early_eos_count"], 10)

    def test_eos128_allows_positions_zero_through127(self):
        self.set_eos(128)
        report = analysis.summarize(self.plan, self.data)
        self.assertTrue(report["prefix128"]["available_for_all48"])
        self.assertFalse(report["fixed512"]["qualified"])
        self.assertEqual(report["fixed512"]["early_eos_count"], 10)

    def test_eos_at_last_token_is_not_early(self):
        self.set_eos(511)
        report = analysis.summarize(self.plan, self.data)
        self.assertEqual(report["fixed512"]["early_eos_count"], 9)

    def test_fixed_length_boolean_must_match_eos(self):
        self.first_measured()["fixed_length_valid"] = False
        self.reject("fixed-length flag")

    def test_non_monotonic_arrivals_rejected(self):
        a = self.first_measured()["arrival_seconds"]
        a[100] = a[99] - 0.01
        self.reject("timestamps decrease")

    def test_first_token_endpoint_mismatch_rejected(self):
        self.first_measured()["ttft_seconds"] += 0.01
        self.reject("timing endpoints")

    def test_nan_timing_rejected(self):
        self.first_measured()["total_seconds"] = float("nan")
        self.reject("Invalid timing")

    def test_zero_prefix_duration_rejected(self):
        a = self.first_measured()["arrival_seconds"]
        a[:128] = [a[0]] * 128
        self.reject("Zero first-128")

    def test_stored_rate_arithmetic_rejected(self):
        self.first_measured()["decode_tokens_per_second"] *= 1.01
        self.reject("rate disagrees")

    def test_resource_negative_rejected(self):
        self.first_measured()["resources"]["disk_read_bytes_delta"] = -1
        self.reject("Invalid resource")

    def test_memory_guard_rejected(self):
        self.first_measured()["resources"]["system_available_min_bytes"] = 1024
        self.reject("headroom violation")

    def test_source_activation_failure_rejected(self):
        self.data[0][2]["passed"] = False
        self.reject("source activation failed")

    def test_candidate_priming_mismatch_rejected(self):
        row = next(r for r in self.data[2] if r["type"] == "run")
        row["candidate"]["after"]["cone"]["prime_counts"]["wkv_pads"] += 1
        self.reject("Replay/priming telemetry")

    def test_overlapping_processes_rejected(self):
        self.data[1][0]["started_at_unix"] = self.data[0][0]["started_at_unix"]
        self.reject("overlapping process")

    def test_warmup_cannot_be_measured(self):
        self.data[0][3]["warmup"] = False
        self.reject("warmup flag")

    def test_changed_saved_review_rejected(self):
        report = analysis.summarize(self.plan, self.data)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            audit = analysis.decode((EVIDENCE / "review" / "independent-audit.json").read_text())
            supplement = analysis.decode((EVIDENCE / "review" / "supplement.json").read_text())
            supplement["pair_group_summaries"][0]["metrics"]["ttft_seconds"]["improvement_percent"] += 0.1
            (target / "independent-audit.json").write_text(json.dumps(audit))
            (target / "supplement.json").write_text(json.dumps(supplement))
            with self.assertRaisesRegex(analysis.ValidationError, "review disagrees"):
                analysis.compare_review(report, target)

    def test_frozen_plan_hash_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "plan.json"
            bad.write_text("{}")
            with self.assertRaisesRegex(analysis.ValidationError, "plan bytes changed"):
                analysis.analyze(bad, EVIDENCE / "raw", EVIDENCE / "review")

    def test_missing_raw_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(analysis.ValidationError, "Missing or extra overall raw file"):
                analysis.analyze(EVIDENCE / "plan-01.json", Path(temp), EVIDENCE / "review")

    def test_extra_raw_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            for name in analysis.FILES + ("06-complete-block2.jsonl",):
                (target / name).write_text("{}\n")
            with self.assertRaisesRegex(analysis.ValidationError, "Missing or extra overall raw file"):
                analysis.analyze(EVIDENCE / "plan-01.json", target, EVIDENCE / "review")

    def test_wrong_packaged_hash_rejected(self):
        inventory = analysis.decode((HERE.parent / "EVIDENCE-INVENTORY.json").read_text())
        entry = next(e for e in inventory["files"] if e["path"] == "evidence/overall/raw/" + analysis.FILES[0])
        entry["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(analysis.ValidationError, "Packaged bytes disagree"):
                analysis.analyze(EVIDENCE / "plan-01.json", EVIDENCE / "raw", EVIDENCE / "review", path)

    def test_original_to_public_hash_mapping_rejected(self):
        inventory = analysis.decode((HERE.parent / "EVIDENCE-INVENTORY.json").read_text())
        entry = next(e for e in inventory["files"] if e["path"] == "evidence/overall/raw/" + analysis.FILES[0])
        entry["original_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(analysis.ValidationError, "Original/public raw hash mapping differs"):
                analysis.analyze(EVIDENCE / "plan-01.json", EVIDENCE / "raw", EVIDENCE / "review", path)

    def test_strict_json_parser_rejects_duplicate_keys_and_nan(self):
        for value in ('{"arm": 1, "arm": 2}', '{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}', '{"unfinished":'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analysis.decode(value)


if __name__ == "__main__":
    unittest.main()
