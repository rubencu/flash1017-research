"""Read-only stdlib checks of frozen fixture identities and plan geometry."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

from build_fixtures import HERE, QUESTIONS, assert_no_inference_imports, sha, strip_gutenberg, token_sha
from freeze_plans import ARMS, BOUNDARIES, PERFORMANCE, chunks


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fm=json.loads((HERE/"fixture-manifest.json").read_text())
        cls.sm=json.loads((HERE/"source-manifest.json").read_text())
        cls.pm=json.loads((HERE/"plans-manifest.json").read_text())
        cls.plans={name:json.loads((HERE/item["path"]).read_text()) for name,item in cls.pm["plans"].items()}

    def test_no_inference_imports(self):
        assert_no_inference_imports()

    def test_source_manifest_pin(self):
        self.assertEqual(sha((HERE/"source-manifest.json").read_bytes()),self.fm["source_manifest_sha256"])

    def test_source_files_all_hash(self):
        for item in self.sm["files"]:
            self.assertEqual(sha((HERE/item["path"]).read_bytes()),item["sha256"],item["path"])

    def test_upstream_notices_retained(self):
        self.assertIn("PYTHON SOFTWARE FOUNDATION LICENSE",(HERE/"sources/cpython/LICENSE").read_text())
        self.assertIn("Permission to use",(HERE/"sources/curl/COPYING").read_text())
        for name in ("moby-dick","pride-prejudice"):
            self.assertIn("Public domain in the USA",(HERE/"sources"/name/"catalog.html").read_text())
            self.assertTrue("THE FULL PROJECT GUTENBERG™ LICENSE" in (HERE/"sources"/name/"complete.txt").read_text(), name + " retains full license")

    def test_gutenberg_wrapper_validation(self):
        with self.assertRaises(ValueError):strip_gutenberg("no legal boundaries")
        for name in ("moby-dick","pride-prejudice"):
            body,_=strip_gutenberg((HERE/"sources"/name/"complete.txt").read_text(encoding="utf-8-sig"))
            self.assertEqual(body,(HERE/self.fm["corpora"][name]["path"]).read_text())

    def test_corpora_and_stream_hashes(self):
        for item in self.fm["corpora"].values():
            self.assertGreater(item["tokens"],131200)
            self.assertEqual(sha((HERE/item["path"]).read_bytes()),item["sha256"])
            p=HERE/item["state_stream"]["path"]
            self.assertEqual(sha(p.read_bytes()),item["state_stream"]["file_sha256"])
            d=json.loads(p.read_text());self.assertEqual(len(d["tokens"]),131200)
            self.assertEqual(token_sha(d["tokens"]),d["sha256"])

    def test_source_workload_holdout(self):
        self.assertEqual(set(self.fm["corpora"]),set(QUESTIONS))
        self.assertEqual(self.fm["disjoint_from_prior"]["prior_repository"],"https://github.com/jundot/omlx")
        self.assertFalse(self.fm["disjoint_from_prior"]["source_identity_and_file_hash_overlap"])
        for item in self.fm["corpora"].values():self.assertFalse(item["old_full_source_hash_overlap"])

    def test_frozen_builders(self):
        for path,digest in self.pm["builder_files_sha256"].items():self.assertEqual(sha((HERE/path).read_bytes()),digest)
        self.assertEqual(self.fm["builder_sha256"],sha((HERE/"build_fixtures.py").read_bytes()))

    def test_fixture_manifest_pin(self):
        self.assertEqual(self.pm["fixture_manifest_sha256"],sha((HERE/"fixture-manifest.json").read_bytes()))

    def test_all_exact_prompts(self):
        self.assertEqual(len(self.fm["prompts"]),16)
        for meta in self.fm["prompts"].values():
            p=HERE/meta["path"];self.assertEqual(sha(p.read_bytes()),meta["file_sha256"])
            d=json.loads(p.read_text());self.assertEqual(len(d["tokens"]),d["prompt_tokens"])
            self.assertEqual(token_sha(d["tokens"]),d["input_tokens_sha256"])
            self.assertTrue(all(type(x) is int and 0<=x<129280 for x in d["tokens"]))

    def test_plans_still_unrun_and_pinned(self):
        for name,meta in self.pm["plans"].items():
            self.assertEqual(sha((HERE/meta["path"]).read_bytes()),meta["sha256"])
            self.assertEqual(self.plans[name]["status"],"frozen_unrun")

    def test_baseline_runtime_and_arm_identity(self):
        for plan in self.plans.values():
            self.assertEqual(plan["profiles"][ARMS[0]],plan["profiles"][ARMS[1]])
            for profile in plan["profiles"].values():
                self.assertEqual(profile,self.fm["baseline_runtime"])
                self.assertEqual(profile["engram_prefetch_bytes"],67108864)
                self.assertEqual(profile["prefill_step"],8192)
                self.assertFalse(profile["ced_prefill"])

    def test_plan_jobs_complete_balanced_adjacent(self):
        for plan in self.plans.values():
            self.assertEqual(len(plan["jobs"]),26)
            self.assertEqual(sum(bool(j.get("warmup")) for j in plan["jobs"]),2)
            orders=[];seen=set()
            for i in range(2,26,2):
                a,b=plan["jobs"][i:i+2]
                self.assertEqual(a["prompt"],b["prompt"]);self.assertEqual(a["seed"],b["seed"])
                self.assertEqual({a["profile"],b["profile"]},set(ARMS))
                self.assertNotIn(a["prompt"],seen);seen.add(a["prompt"])
                orders.append("AB" if a["profile"]==ARMS[0] else "BA")
            self.assertEqual(Counter(orders),{"AB":6,"BA":6})

    def test_warmup_not_measured_source(self):
        self.assertTrue(self.fm["warmup"]["synthetic"])
        for plan in self.plans.values():
            warm=plan["prompts"]["authored-warmup-16384"]
            self.assertEqual(len(warm["tokens"]),16384)
            self.assertNotIn(warm["input_tokens_sha256"],{p["input_tokens_sha256"] for name,p in plan["prompts"].items() if name!="authored-warmup-16384"})

    def test_prefix_minus_final_token_convention(self):
        for plan in self.plans.values():
            for name,p in plan["prompts"].items():
                if name=="authored-warmup-16384":continue
                q=p["qualification"]
                self.assertEqual(p["id"],name);self.assertEqual(len(p["tokens"]),p["prompt_tokens"])
                self.assertEqual(q["prefix_tokens"],len(p["tokens"])-1)
                self.assertEqual(q["prefix_tokens_sha256"],token_sha(p["tokens"][:-1]))
                self.assertEqual(q["last_unprefilled_token"],p["tokens"][-1])
                self.assertEqual(q["cache_only_chunk_lengths_at_requested_step"],chunks(q["prefix_tokens"]))

    def test_boundary_coverage(self):
        plan=self.plans["boundary"]
        got={(p["source"],p["qualification"]["prefix_tokens"]) for name,p in plan["prompts"].items() if name!="authored-warmup-16384"}
        self.assertEqual(got,set(BOUNDARIES))
        for n,want in [(2667,[2667]),(2668,[2668]),(8193,[8192,1]),(10859,[8192,2667]),(10860,[8192,2668]),(10861,[8192,2669])]:self.assertEqual(chunks(n),want)

    def test_performance_strata(self):
        plan=self.plans["performance"]
        counts=Counter(p["prompt_tokens"] for name,p in plan["prompts"].items() if name!="authored-warmup-16384")
        self.assertEqual(counts,{16384:4,32768:4,65536:3,131072:1})

    def test_output_lengths_and_sampling(self):
        for name,plan in self.plans.items():
            for p in plan["prompts"].values():
                self.assertEqual(p["max_tokens"],32 if name=="boundary" else 128)
                self.assertEqual(p["temperature"],1.0);self.assertTrue(p["force_length"])

    def test_seeds_distinct_across_suites(self):
        a={j["seed"] for j in self.plans["boundary"]["jobs"]}
        b={j["seed"] for j in self.plans["performance"]["jobs"]}
        self.assertFalse(a&b)

    def test_http_cases_long_shared_prefix(self):
        meta=self.fm["http_cases"];path=HERE/meta["path"]
        self.assertEqual(sha(path.read_bytes()),meta["sha256"])
        for case in json.loads(path.read_text()):
            self.assertGreater(case["exact_common_prefix_tokens"],32000)
            self.assertNotEqual(case["first_tokens_sha256"],case["second_tokens_sha256"])
            self.assertNotEqual(case["expected_first"],case["expected_second"])

    def test_vision_fixture(self):
        meta=self.fm["vision"];data=(HERE/meta["path"]).read_bytes()
        self.assertEqual(sha(data),meta["sha256"]);self.assertEqual(data[:8],b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II",data[16:24]),(512,384))
        self.assertEqual(meta["expected"],{"blue_circles":2,"green_squares":3,"red_triangles":1})

    def test_invalid_chunk_arguments(self):
        for value in (-1,1.2,True):
            with self.assertRaises(ValueError):chunks(value)
        self.assertEqual(chunks(0),[])


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures))
    out=HERE/"cpu-validation-02.json"
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps({"status":"passed" if result.wasSuccessful() else "failed","tests":result.testsRun,
                              "failures":len(result.failures),"errors":len(result.errors),"at_utc":datetime.now(timezone.utc).isoformat(),
                              "test_source_sha256":sha(Path(__file__).read_bytes()),"plans_manifest_sha256":sha((HERE/"plans-manifest.json").read_bytes()),
                              "cpu_only":True,"inference_imports":False},indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
