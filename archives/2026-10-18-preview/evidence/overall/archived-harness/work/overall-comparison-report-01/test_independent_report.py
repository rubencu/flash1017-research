"""Finite synthetic-record tests only; no benchmark/runtime imports."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("independent_overall_report", HERE / "independent_report.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def save_lines(path, values):
    path.write_text("".join(json.dumps(x) + "\n" for x in values))


class RecordAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="overall-independent-cpu-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        source = root / "work" / "overall-comparison-01"
        source.mkdir(parents=True)
        self.plan_path, self.manifest_path, self.proposal_path = (source / x for x in ("plan.json", "manifest.json", "proposal.json"))
        self.supervisor_path = root / "results" / "supervisor.json"
        self.supervisor_path.parent.mkdir()
        self.roots = {"native2048": str(root / "stock" / "omlx"), "native8192": str(root / "stock" / "omlx"), "complete": str(root / "candidate" / "omlx")}
        sources = {}
        for runtime in set(self.roots.values()):
            path = Path(runtime) / "fake.py"
            path.parent.mkdir(parents=True)
            path.write_text("# synthetic CPU test identity\n")
            sources[str(path)] = reporter.sha(path)
        profiles = {arm: {"mtp": True, "depth": 5 if arm == "complete" else 3,
            "fixed": arm == "complete", "ced_prefill": False,
            "prefill_step": 2048 if arm == "native2048" else 8192,
            "engram_prefetch_bytes": (64 if arm == "complete" else 16) * 2**20}
            for arm in self.roots}
        keys = [f"case{i}" for i in range(8)]
        sizes = [16384] * 4 + [65536] * 2 + [1024] * 2
        fixtures, prompts = [], {}
        for name, length in zip(keys + ["warmup"], sizes + [16384]):
            tokens = [7] * length
            prompts[name] = {"tokens": tokens, "input_tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(), "max_tokens": 64 if name == "warmup" else 512}
            if name != "warmup":
                fixtures.append({"id": name, "cohort": "synthetic-generation" if length == 1024 else "natural-source"})
        proposal = {"fixtures": fixtures}
        save(self.proposal_path, proposal)
        processes = []
        for block, arms in enumerate((list(profiles), list(profiles)[::-1])):
            for arm in arms:
                jobs = [{"prompt": "warmup", "seed": 1, "warmup": True}]
                jobs += [{"prompt": key, "seed": 100 + block * 10 + i, "warmup": False} for i, key in enumerate(keys if block == 0 else keys[::-1])]
                processes.append({"arm": arm, "block": block, "jobs": jobs})
        self.plan = {"proposal_sha256": reporter.sha(self.proposal_path), "profiles": profiles,
            "cache_limit_bytes": 8 * 2**30, "prompts": prompts, "processes": processes, "model_execution_wall_limit_seconds": 7200}
        save(self.plan_path, self.plan)
        sources.update({str(self.plan_path): reporter.sha(self.plan_path), str(self.proposal_path): reporter.sha(self.proposal_path)})
        frozen = {"stock_is_current_main": False, "checkpoint_revision": "78456115d301eecf9fadcdcc81c0cf6f8b2c8eab", "packages": {"mlx": "test-only"}, "sources_sha256": sources}
        save(self.manifest_path, frozen)
        self.supervisor = {"benchmark_status": "all_processes_completed", "plan_sha256": reporter.sha(self.plan_path),
            "manifest_sha256": reporter.sha(self.manifest_path), "restoration_status": "restored",
            "model_execution_started_at": 90,
            "prior_state": {"loaded_models": ["prior"], "memory": {"cap": 1}},
            "restored_state": {"loaded_models": ["prior"], "memory": {"cap": 1}}, "processes": []}
        self.raw_paths = []
        for index, proc in enumerate(processes):
            arm, block = proc["arm"], proc["block"]
            path = self.supervisor_path.parent / f"{index}.jsonl"
            self.raw_paths.append(path)
            profile = profiles[arm] | {"name": arm}
            candidate = arm == "complete"
            rows = [{"type": "manifest", "process_index": index, "arm": arm, "block": block, "profile": profile,
                "plan_sha256": reporter.sha(self.plan_path), "manifest_sha256": reporter.sha(self.manifest_path),
                "runtime_root": self.roots[arm], "packages": frozen["packages"],
                "memory_policy": {"cache_limit_bytes": 8 * 2**30, "previous_wired_limit_bytes": 0}, "device": {"name": "synthetic"}}]
            rows.append({"type": "load", "seconds": 20, "resources": {"mlx_active_bytes": 200}})
            state = {"mtp_enabled": True, "dspark_enabled": True, "mtp_depth": profile["depth"], "mtp_fixed": profile["fixed"],
                "ced_prefill": False, "prefetch_bytes": profile["engram_prefetch_bytes"], "prefill_step_requested": profile["prefill_step"],
                "attention_rounding": candidate, "attention_rounding_field_present": candidate,
                "confidence_policy": candidate, "cone_attached": candidate, "row_cache_attached": candidate,
                "native_target_class": not candidate, "draft_width": 5 if candidate else None, "head_max_rows": 6 if candidate else None}
            src = {"path": str(Path(self.roots[arm]) / "fake.py"), "sha256": sources[str(Path(self.roots[arm]) / "fake.py")]}
            activation = {"type": "source_activation", "phase": "before_runs", "passed": True, "checks": {"synthetic": True},
                "runtime_root": self.roots[arm], "state": state, "sources": {"omlx.fake": src},
                "functions": {k: src for k in ("head", "target_head", "draft_head", "target_call", "target_prefill", "draft_loop", "verify_loop", "loader")}, "source_escapes": []}
            rows.append(activation)
            for job_index, job in enumerate(proc["jobs"]):
                prompt = prompts[job["prompt"]]
                n, output_n = len(prompt["tokens"]), prompt["max_tokens"]
                rate, prefill = (50, 40) if candidate else (40, 50)
                arrivals = [prefill + .5 + i / rate for i in range(output_n)]
                row = {"type": "run", "job_index": job_index, "process_index": index, "arm": arm, "block": block,
                    "profile": profile, "prompt_id": job["prompt"], "seed": job["seed"], "warmup": job["warmup"],
                    "input_tokens": n, "input_tokens_sha256": prompt["input_tokens_sha256"],
                    "generated_tokens": output_n, "token_ids": [2 if candidate else 1] * output_n,
                    "fixed_length_valid": True, "first_eos_position": None, "mtp_stats": [{"cycles": 1}],
                    "prefill_seconds": prefill, "ttft_seconds": arrivals[0], "total_seconds": arrivals[-1] + .1,
                    "arrival_seconds": arrivals, "decode_tokens_per_second": (output_n - 1) / (arrivals[-1] - arrivals[0]),
                    "resources": {"system_available_min_bytes": 20 * 2**30, "swap_start_bytes": 0, "swap_peak_bytes": 0,
                        "process_footprint_peak_bytes": 210 * 2**30, "disk_read_bytes_delta": 200}, "events": []}
                if candidate:
                    eligible = sum(min(profile["prefill_step"], n - 1 - start) >= 2668 for start in range(0, n - 1, profile["prefill_step"]))
                    row["events"] = [{"confidence_logits": [1]}]
                    row["candidate"] = {"passed": True, "checks": {"synthetic": True}, "expected_eligible_chunks": eligible,
                        "before": {"rows": {"live_tables": [{"unique_queries": 0}]}},
                        "after": {"cone": {"cone_counts": {"cone_calls": eligible}, "prime_counts": {"padded_appends": eligible, "wkv_pads": eligible * 3}, "counts": {}},
                            "rows": {"live_tables": [{"unique_queries": 1}], "retained_bytes": 512}}}
                rows.append(row)
                if candidate and job["warmup"]:
                    rows.append({"type": "row_cache_reset_after_warmup", "snapshot": {"retained_bytes": 0}})
            rows.append(activation | {"phase": "after_runs"})
            rows.append({"type": "complete", "jobs": 9})
            rows.append({"type": "cleanup", "native_close": True, "candidate_close": True if candidate else None,
                         "rows_after_native_close": {"retained_bytes": 0, "live_tables": []}})
            save_lines(path, rows)
            save_lines(path.with_suffix(".load.jsonl"), [{"stage": "before", "shard": "one", "available_bytes": 100 * 2**30}, {"stage": "after", "shard": "one", "available_bytes": 20 * 2**30}])
            self.supervisor["processes"].append({"index": index, "arm": arm, "block": block, "output": str(path),
                "exit_code": 0, "pid": index + 1, "started_at_unix": 100 + index * 100, "completed_at_unix": 150 + index * 100})
        save(self.supervisor_path, self.supervisor)

    def audit(self):
        return reporter.audit(self.plan_path, self.manifest_path, self.proposal_path, self.supervisor_path, self.roots)

    def mutate(self, index, predicate, update):
        rows = reporter.read_lines(self.raw_paths[index])
        row = next(x for x in rows if predicate(x))
        update(row)
        save_lines(self.raw_paths[index], rows)

    def test_complete_and_independent_aggregate(self):
        report = self.audit()
        self.assertEqual(report["status"], "accepted_complete_fixed_length_record_audit", report["structural_issues"])
        self.assertEqual(report["measured_requests"], 48)
        self.assertEqual(len(report["pairs"]), 32)
        expected = (40.5 + 511 / 50 + .1) / (50.5 + 511 / 40 + .1)
        self.assertTrue(all(math.isclose(x["total_seconds_ratio"], expected, rel_tol=1e-12) for x in report["aggregates"]))

    def test_early_eos_invalidates_without_dropping(self):
        self.mutate(0, lambda r: r["type"] == "run" and not r["warmup"], lambda r: r.update(first_eos_position=100, fixed_length_valid=False))
        report = self.audit()
        self.assertFalse(report["performance_claim_qualified"])
        self.assertEqual(len(report["pairs"]), 32)
        self.assertEqual(len(report["early_eos_records"]), 1)

    def test_bad_activation_hash_rejected(self):
        self.mutate(0, lambda r: r["type"] == "source_activation", lambda r: r["sources"]["omlx.fake"].update(sha256="bad"))
        self.assertTrue(any("Imported source not pinned" in x for x in self.audit()["structural_issues"]))

    def test_missing_cleanup_rejected(self):
        path = self.raw_paths[0]
        save_lines(path, [r for r in reporter.read_lines(path) if r["type"] != "cleanup"])
        self.assertFalse(self.audit()["performance_claim_qualified"])

    def test_unconfirmed_exit_rejected(self):
        self.supervisor["processes"][0]["exit_code"] = None
        save(self.supervisor_path, self.supervisor)
        self.assertTrue(any("exit not confirmed" in x for x in self.audit()["structural_issues"]))

    def test_load_trace_failure_rejected(self):
        save_lines(self.raw_paths[0].with_suffix(".load.jsonl"), [{"stage": "before", "shard": "one", "available_bytes": 100 * 2**30}])
        self.assertTrue(any("Incomplete shard-load trace" in x for x in self.audit()["structural_issues"]))

    def test_restoration_not_conflated_with_performance(self):
        self.supervisor["restoration_status"] = "failed"
        save(self.supervisor_path, self.supervisor)
        report = self.audit()
        self.assertTrue(report["performance_claim_qualified"])
        self.assertFalse(report["restoration_confirmed"])
        self.assertNotEqual(report["status"], "accepted_complete_fixed_length_record_audit")

    def test_forged_decode_rate_rejected(self):
        self.mutate(0, lambda r: r["type"] == "run", lambda r: r.update(decode_tokens_per_second=100))
        self.assertTrue(any("Decode rate arithmetic" in x for x in self.audit()["structural_issues"]))

    def test_unexpected_fallback_rejected(self):
        self.mutate(2, lambda r: r["type"] == "run", lambda r: r["candidate"]["after"]["cone"]["cone_counts"].update(cone_calls=0))
        self.assertTrue(any("Observed replay count" in x for x in self.audit()["structural_issues"]))

    def test_incomplete_json_is_preserved_as_invalid(self):
        with self.raw_paths[0].open("a") as f:
            f.write('{"incomplete":')
        self.assertTrue(any("Unreadable/incomplete process" in x for x in self.audit()["structural_issues"]))

    def test_wall_limit_rejected(self):
        self.supervisor["processes"][-1]["completed_at_unix"] = 8000
        save(self.supervisor_path, self.supervisor)
        self.assertTrue(any("wall limit exceeded" in x for x in self.audit()["structural_issues"]))


if __name__ == "__main__":
    unittest.main()
