#!/usr/bin/env python3
"""Unit checks for answer extraction and the no-observed-regression gate."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "score_quality_supplement", Path(__file__).with_name("score_quality_supplement.py")
)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class SupplementScoringTest(unittest.TestCase):
    def task(self, task_id="one", answer="12"):
        return {
            "task_id": task_id,
            "suite": "gsm8k-test-64-hash-subset",
            "expected_answer": answer,
            "scorer": "gsm8k-official-regex-string-v1",
        }

    def result(self, answer="12", finish_reason="stop"):
        return {"text": f"#### {answer}", "finish_reason": finish_reason}

    def test_official_numeric_extraction_uses_first_marker_and_only_comma_removal(self):
        extract = scorer.extract_answer
        name = "gsm8k-official-regex-string-v1"
        self.assertEqual(extract("work\n#### -1,234.5", name), "-1234.5")
        self.assertEqual(extract("#### 12\n#### 13", name), "12")
        self.assertEqual(extract("#### 12.0", name), "12.0")
        self.assertIsNone(extract("The answer is 12", name))

    def test_synthetic_extraction_requires_only_one_exact_line(self):
        extract = scorer.extract_answer
        name = "exact-marked-string-v1"
        self.assertEqual(extract("\n#### ABC123\n", name), "ABC123")
        self.assertIsNone(extract("#### abc123", name))
        self.assertIsNone(extract("Explanation\n#### ABC123", name))
        self.assertIsNone(extract("#### ABC123\n#### DEF456", name))

    def test_equal_complete_correct_results_pass_observation_gate(self):
        result = scorer.score(
            {"one": self.task()}, {"one": self.result()}, {"one": self.result()}
        )
        self.assertTrue(result["zero_observed_regression_gate_passed"])
        self.assertFalse(result["quality_equivalence_proved"])

    def test_improvements_never_cancel_regressions(self):
        result = scorer.score(
            {"one": self.task(), "two": self.task("two")},
            {"one": self.result(), "two": self.result("13")},
            {"one": self.result("13"), "two": self.result()},
        )
        self.assertFalse(result["zero_observed_regression_gate_passed"])
        suite = result["suites"]["gsm8k-test-64-hash-subset"]
        self.assertEqual(suite["accuracy_delta_percentage_points"], 0)
        self.assertEqual(suite["regressions"], ["one"])
        self.assertEqual(suite["improvements"], ["two"])

    def test_truncated_correct_answer_is_incomplete_and_fails_gate(self):
        result = scorer.score(
            {"one": self.task()},
            {"one": self.result()},
            {"one": self.result(finish_reason="length")},
        )
        self.assertFalse(result["zero_observed_regression_gate_passed"])
        self.assertFalse(result["pairs"][0]["candidate"]["correct"])

    def test_missing_baseline_cannot_pass_even_if_candidate_correct(self):
        result = scorer.score({"one": self.task()}, {}, {"one": self.result()})
        self.assertFalse(result["zero_observed_regression_gate_passed"])

    def test_server_error_with_plausible_answer_cannot_pass(self):
        candidate = self.result()
        candidate["error"] = "request failed"
        result = scorer.score(
            {"one": self.task()}, {"one": self.result()}, {"one": candidate}
        )
        self.assertFalse(result["zero_observed_regression_gate_passed"])

    def test_duplicate_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.jsonl"
            row = {"task_id": "one", **self.result()}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                scorer.load_jsonl(path)

    def test_unexpected_id_is_rejected(self):
        with self.assertRaises(ValueError):
            scorer.score(
                {"one": self.task()}, {"one": self.result()}, {"other": self.result()}
            )

    def test_zero_regressions_still_has_nonzero_uncertainty(self):
        self.assertAlmostEqual(scorer.binomial_upper(0, 64), 0.045729702330762456)
        self.assertEqual(scorer.binomial_upper(0, 0), 1.0)
        self.assertEqual(scorer.binomial_upper(64, 64), 1.0)
        self.assertGreater(scorer.binomial_upper(1, 64), scorer.binomial_upper(0, 64))


if __name__ == "__main__":
    unittest.main()
