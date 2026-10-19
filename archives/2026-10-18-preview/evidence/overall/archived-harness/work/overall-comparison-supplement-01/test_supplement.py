"""Synthetic saved-record tests; no model, server, or actual benchmark reads."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("overall_supplement", HERE / "supplement.py")
supplement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supplement)
fixture_spec = importlib.util.spec_from_file_location("independent_fixture_tests", HERE.parent / "overall-comparison-report-01/test_independent_report.py")
fixture_module = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)


class SupplementTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the prior CPU test's temporary synthetic record builder.
        # Its entire original independent audit is then run over these records.
        self.fixture = fixture_module.RecordAuditTests(methodName="test_complete_and_independent_aggregate")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan_path = self.fixture.plan_path
        plan = json.loads(self.plan_path.read_text())
        for name, prompt in plan["prompts"].items():
            prompt["cohort"] = "excluded-warmup" if name == "warmup" else ("synthetic-generation" if len(prompt["tokens"]) == 1024 else "natural-source")
        fixture_module.save(self.plan_path, plan)
        self.plan = plan
        manifest = json.loads(self.fixture.manifest_path.read_text())
        manifest["sources_sha256"][str(self.plan_path)] = supplement.sha(self.plan_path)
        fixture_module.save(self.fixture.manifest_path, manifest)
        for path in self.fixture.raw_paths:
            rows = fixture_module.reporter.read_lines(path)
            for row in rows:
                if row["type"] == "manifest":
                    row["plan_sha256"] = supplement.sha(self.plan_path)
                    row["manifest_sha256"] = supplement.sha(self.fixture.manifest_path)
                if row["type"] == "run":
                    row["cohort"] = plan["prompts"][row["prompt_id"]]["cohort"]
            fixture_module.save_lines(path, rows)
        self.fixture.supervisor.update(plan_sha256=supplement.sha(self.plan_path), manifest_sha256=supplement.sha(self.fixture.manifest_path))
        fixture_module.save(self.fixture.supervisor_path, self.fixture.supervisor)
        self.audit_path = self.fixture.supervisor_path.parent / "independent-audit.json"
        self.refresh_audit()

    def refresh_audit(self):
        self.audit = self.fixture.audit()
        fixture_module.save(self.audit_path, self.audit)

    def run_analysis(self, prefix=True):
        return supplement.analyze(self.audit_path, self.plan_path, prefix)

    def set_eos(self, value):
        self.fixture.mutate(0, lambda r: r["type"] == "run" and not r["warmup"],
            lambda r: r.update(first_eos_position=value, fixed_length_valid=value is None or value == 511))
        self.refresh_audit()

    def rebind_changed_raw(self, index):
        # Deliberately simulate an auditor that missed bad numbers; the new
        # supplementary numeric checks must independently reject them.
        self.audit["processes"][index]["output_sha256"] = supplement.sha(self.fixture.raw_paths[index])
        fixture_module.save(self.audit_path, self.audit)

    def test_complete_groups_actual_gms_and_prefix_math(self):
        report = self.run_analysis()
        self.assertTrue(report["prefill_ttft"]["qualified_separately"])
        self.assertTrue(report["prefix128"]["available_for_all48"])
        self.assertEqual(report["measured_requests"], 48)
        self.assertEqual(len(report["pair_group_summaries"]), 16)
        self.assertEqual(len(report["per_case_two_block_summaries"]), 16)
        self.assertEqual(len(report["per_arm_group_summaries"]), 24)
        groups = {r["group"]: r for r in report["pair_group_summaries"] if r["baseline"] == "native2048"}
        self.assertEqual({k: v["pairs"] for k, v in groups.items()}, {"all": 16, "natural": 12, "synthetic": 4, "1K": 4, "16K": 8, "64K": 4, "block0": 8, "block1": 8})
        values = groups["all"]["metrics"]["prefix128_delivery_seconds"]
        self.assertAlmostEqual(values["baseline_geometric_mean"], 50.5 + 127 / 40)
        self.assertAlmostEqual(values["candidate_geometric_mean"], 40.5 + 127 / 50)
        rate = groups["all"]["metrics"]["prefix128_post_first_tokens_per_second"]
        self.assertAlmostEqual(rate["ratio_candidate_over_baseline"], 1.25)
        self.assertTrue(all(x["pairs"] == 2 for x in report["per_case_two_block_summaries"]))

    def test_eos127_blocks_whole_prefix_without_subsetting(self):
        self.set_eos(127)
        report = self.run_analysis()
        self.assertFalse(report["original_qualification_unchanged"]["performance_claim_qualified"])
        self.assertTrue(report["prefill_ttft"]["qualified_separately"])
        self.assertFalse(report["prefix128"]["available_for_all48"])
        self.assertFalse(report["prefix128"]["subset_or_shorter_horizon_used"])
        self.assertEqual(report["measured_requests"], 48)
        self.assertEqual(len(report["prefix128"]["early_eos_blockers"]), 1)
        self.assertTrue(all("prefix128_delivery_seconds" not in x["metrics"] for x in report["pair_group_summaries"] + report["per_case_two_block_summaries"]))
        self.assertEqual(report["pair_group_summaries"][0]["pairs"], 16)

    def test_eos128_allows_prefix_but_not_fixed512_qualification(self):
        self.set_eos(128)
        report = self.run_analysis()
        self.assertTrue(report["prefix128"]["available_for_all48"])
        self.assertFalse(report["original_qualification_unchanged"]["performance_claim_qualified"])
        self.assertFalse(report["forced512"]["qualified_fixed512_response_claim"])
        self.assertEqual(report["forced512"]["early_eos_count"], 1)

    def test_later_eos_still_preserves_original_failure(self):
        self.set_eos(322)
        audit_before = self.audit_path.read_bytes()
        report = self.run_analysis()
        self.assertTrue(report["prefix128"]["available_for_all48"])
        self.assertEqual(self.audit_path.read_bytes(), audit_before)
        self.assertEqual(report["original_qualification_unchanged"]["fixed_length_issues"], self.audit["fixed_length_issues"])

    def test_incomplete_independent_audit_refuses_qualification(self):
        self.audit["structural_issues"] = ["missing final activation"]
        self.audit["performance_claim_qualified"] = False
        fixture_module.save(self.audit_path, self.audit)
        report = self.run_analysis()
        self.assertFalse(report["prefill_ttft"]["qualified_separately"])
        self.assertFalse(report["prefix128"]["available_for_all48"])
        self.assertEqual(report["status"], "refused_qualification")

    def test_restoration_failure_refuses_supplementary_qualification(self):
        self.audit["restoration_confirmed"] = False
        self.audit["restoration_issues"] = ["not restored"]
        fixture_module.save(self.audit_path, self.audit)
        report = self.run_analysis()
        self.assertFalse(report["prefill_ttft"]["qualified_separately"])
        self.assertFalse(report["prefix128"]["available_for_all48"])

    def test_zero_time_refuses_all_numeric_aggregates(self):
        self.fixture.mutate(0, lambda r: r["type"] == "run" and not r["warmup"], lambda r: r.update(ttft_seconds=0))
        self.rebind_changed_raw(0)
        report = self.run_analysis()
        self.assertFalse(report["prefill_ttft"]["qualified_separately"])
        self.assertTrue(report["numeric_issues"])
        self.assertEqual(report["pair_group_summaries"], [])

    def test_nonfinite_arrival_refuses_all_numeric_aggregates(self):
        self.fixture.mutate(0, lambda r: r["type"] == "run" and not r["warmup"], lambda r: r["arrival_seconds"].__setitem__(127, float("nan")))
        self.rebind_changed_raw(0)
        report = self.run_analysis()
        self.assertFalse(report["prefix128"]["available_for_all48"])
        self.assertTrue(report["numeric_issues"])
        self.assertEqual(report["per_case_two_block_summaries"], [])

    def test_changed_raw_hash_refuses_all_aggregates(self):
        self.fixture.mutate(0, lambda r: r["type"] == "run", lambda r: r.update(total_seconds=123))
        report = self.run_analysis()
        self.assertTrue(report["integrity_issues"])
        self.assertFalse(report["prefix128"]["available_for_all48"])
        self.assertEqual(report["pair_group_summaries"], [])

    def test_prefix_is_explicitly_optional(self):
        report = self.run_analysis(prefix=False)
        self.assertTrue(report["prefill_ttft"]["qualified_separately"])
        self.assertFalse(report["prefix128"]["requested"])
        self.assertFalse(report["prefix128"]["available_for_all48"])
        self.assertTrue(all("prefix128_delivery_seconds" not in x["metrics"] for x in report["pair_group_summaries"]))

    def test_cli_refuses_overwrite(self):
        output = self.audit_path.parent / "already-exists.json"
        output.write_text("keep this exact evidence")
        result = subprocess.run([sys.executable, str(HERE / "supplement.py"), "--audit", str(self.audit_path), "--plan", str(self.plan_path), "--prefix128", "--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "keep this exact evidence")

    def test_cli_writes_nothing_for_incomplete_run(self):
        self.audit["processes"].pop()
        self.audit["measured_requests"] = 40
        fixture_module.save(self.audit_path, self.audit)
        output = self.audit_path.parent / "must-not-exist.json"
        result = subprocess.run([sys.executable, str(HERE / "supplement.py"), "--audit", str(self.audit_path), "--plan", str(self.plan_path), "--prefix128", "--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
