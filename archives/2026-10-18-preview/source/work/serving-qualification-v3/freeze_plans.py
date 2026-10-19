"""Freeze two bench-shaped plans before model outcomes; CPU tokenizer only."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random

from build_fixtures import HERE, LAB, QUESTIONS, assert_no_inference_imports, exact_prompt, sha, token_sha, write_json

ARMS = ("native64", "fastpath64")
# Every prefix length below is the cache-only prefill count. bench.run_one
# reserves one final input token, so prompt_tokens is prefix_tokens + 1.
BOUNDARIES = [
    ("cpython", 2667), ("curl", 2668), ("moby-dick", 2669),
    ("pride-prejudice", 8190), ("cpython", 8191), ("curl", 8192),
    ("moby-dick", 8193), ("pride-prejudice", 10859), ("cpython", 10860),
    ("curl", 10861), ("moby-dick", 16383), ("pride-prejudice", 16384),
]
PERFORMANCE = [(source, length) for length in (16384,32768) for source in QUESTIONS]
PERFORMANCE += [(source,65536) for source in ("cpython","curl","pride-prejudice")]
PERFORMANCE += [("moby-dick",131072)]


def chunks(count, step=8192):
    if type(count) is not int or count < 0 or type(step) is not int or step <= 0:
        raise ValueError("Invalid chunk geometry")
    return [min(step,count-start) for start in range(0,count,step)]


def paired_jobs(prompt_ids, seed_base, order_seed):
    # Equal AB/BA counts assigned before shuffling; each pair remains adjacent.
    if len(prompt_ids) % 2:
        raise ValueError("Balanced order requires an even pair count")
    pairs = []
    for i,prompt in enumerate(prompt_ids):
        order = ARMS if i % 2 == 0 else ARMS[::-1]
        pairs.append([{"profile": arm,"prompt":prompt,"repeat":0,"seed":seed_base+i,
                       "pair_id":prompt,"frozen_pair_index":i} for arm in order])
    random.Random(order_seed).shuffle(pairs)
    return [job for pair in pairs for job in pair]


def main():
    from tokenizers import Tokenizer
    assert_no_inference_imports()
    if (HERE/"plans").exists() or (HERE/"plans-manifest.json").exists():
        raise FileExistsError("Frozen plans already exist")
    fm_path = HERE/"fixture-manifest.json"
    fm = json.loads(fm_path.read_text())
    profile = json.loads((LAB/"qualified-profile.json").read_text())
    if profile["runtime"] != fm["baseline_runtime"] or profile["runtime"].get("engram_prefetch_bytes") != 64*1024*1024:
        raise ValueError("Baseline runtime changed or is not native64")
    for item in fm["tokenizer_files"].values():
        if sha(Path(item["path"]).read_bytes()) != item["sha256"]:
            raise ValueError("Tokenizer changed")
    native = Path(fm["native_encoder"])
    if sha(native.read_bytes()) != fm["native_encoder_sha256"]:
        raise ValueError("Native encoder changed")
    spec = importlib.util.spec_from_file_location("frozen_plan_native_encoding",native)
    encoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(encoder)
    tokenizer = Tokenizer.from_file(fm["tokenizer_files"]["tokenizer.json"]["path"])
    source_ids = {}
    for source, item in fm["corpora"].items():
        p=HERE/item["path"]
        if sha(p.read_bytes()) != item["sha256"]:
            raise ValueError("Corpus changed")
        source_ids[source] = tokenizer.encode(p.read_text(),add_special_tokens=False).ids
    warm_path = HERE/fm["warmup"]["path"]
    if sha(warm_path.read_bytes()) != fm["warmup"]["sha256"]:
        raise ValueError("Warmup changed")
    warmup=json.loads(warm_path.read_text())
    prepared={}
    for suite, choices, seed_base, order_seed, outputs in (
        ("boundary",BOUNDARIES,101820100,101820010,32),
        ("performance",PERFORMANCE,101820200,101820020,128)):
        prompts={"authored-warmup-16384":warmup | {"id":"authored-warmup-16384","max_tokens":outputs}}
        measured=[]
        for source,value in choices:
            length=value+1 if suite=="boundary" else value
            key=f"{suite}-{source}-{length}"
            if suite=="performance":
                meta=fm["prompts"][f"{source}-{length}"]; p=HERE/meta["path"]
                if sha(p.read_bytes()) != meta["file_sha256"]:
                    raise ValueError("Frozen prompt changed")
                prompt=json.loads(p.read_text())
            else:
                prompt=exact_prompt(tokenizer,encoder,source_ids[source],length,QUESTIONS[source])
            prompt.update(id=key,source=source,max_tokens=outputs,timeout_seconds=1200)
            prefix=length-1
            prompt["qualification"]={"prefix_tokens":prefix,"prefix_tokens_sha256":token_sha(prompt["tokens"][:-1]),
                                     "last_unprefilled_token":prompt["tokens"][-1],
                                     "cache_only_chunk_lengths_at_requested_step":chunks(prefix),
                                     "expected_cone_eligible_chunk_lengths": [n for n in chunks(prefix) if n>=2668],
                                     "expected_short_fallback_chunk_lengths": [n for n in chunks(prefix) if n<2668],
                                     "geometry_note":"Eligibility assumes ordinary singleton text, requested8192 chunks and no memory-guard shrink. Record actual chunk lengths. A conservative production guard may decline additional shapes, but must record its reason; no silent fastpath bypass can count as fastpath coverage."}
            prompts[key]=prompt;measured.append(key)
        jobs=[{"profile":arm,"prompt":"authored-warmup-16384","warmup":True,"repeat":0,"seed":seed_base-1} for arm in ARMS]
        jobs+=paired_jobs(measured,seed_base,order_seed)
        plan={"description":f"Frozen serving-qualification-v3 {suite} plan, prepared before any v3 model outcomes. Same native64 runtime in both arms; only the independently pinned runner may activate fastpath64.",
              "status":"frozen_unrun","frozen_at_utc":datetime.now(timezone.utc).isoformat(),
              "cache_limit_bytes":8*1024**3,"profiles":{arm:profile["runtime"] for arm in ARMS},"prompts":prompts,"jobs":jobs,
              "design":{"suite":suite,"measured_pairs":len(measured),"measured_jobs":len(measured)*2,"warmup_jobs":2,
                        "generation_tokens":outputs,"pair_order_seed":order_seed,"pair_orders":{"AB":6,"BA":6},
                        "temperature":1.0,"top_p":1.0,"force_length":True,"matched_seed_per_pair":True,
                        "primary_reference":"Currently qualified native64, full-depth CED-disabled attention with existing confidence/head6.",
                        "candidate_dispatch":"Runner installs persistent cone/backbone/DSpark+HotRows adaptation for fastpath64. Plain bench.main would run identical arms and is not an authorized execution of this plan.",
                        "performance_gate":"Do not begin performance plan before mandatory boundary/state tests pass or every unsupported geometry has a verified full-native fallback. Freeze runner/source manifest before the first model outcome; plans must not change afterwards.",
                        "history_policy":"Within a suite use one model process; fresh request KV; no OS cache purge. Candidate row-cache history persists; baseline bypass retains allocated history. Log per-table hits/misses/bytes and report first-use versus subsequent reuse. Separate suites may use separate loads; do not mix summaries.",
                        "timing_policy":"Boundary timings diagnostic only. Performance has one pair per source/length; report each pair and length strata, without claiming confidence intervals or independent cold replicates. Do not aggregate boundary timing into performance.",
                        "divergence_policy":"Retain all complete observations and exact first differing token/trace; disallow token-identical speed claims on divergence. A crash/earlyEOS/incomplete ordered job set yields incomplete status. Forced length may count emitted EOS; record actual IDs.",
                        "limitations":["New sources are held out from local kernel/cone tuning, not from model training.","Lengths within a source share a source prefix and are correlated.","Code/prose tasks are not a public coding or general-quality benchmark.","Memory guard may shrink chunk sizes; unsupported geometry must fall back safely and is not fastpath coverage.","One131K pair is exploratory capacity/state evidence, not a long-context benchmark."]},
              "fixture_manifest":str(fm_path),"fixture_manifest_sha256":sha(fm_path.read_bytes()),
              "source_manifest_sha256":fm["source_manifest_sha256"],"baseline_runtime_sha256":fm["baseline_runtime_sha256"]}
        path=HERE/"plans"/(suite+"-01.json");write_json(path,plan)
        prepared[suite]={"path":str(path.relative_to(HERE)),"sha256":sha(path.read_bytes()),"measured_pairs":len(measured),"jobs":len(jobs),"measured_inputs":measured}
    manifest={"status":"frozen_unrun","created_at_utc":datetime.now(timezone.utc).isoformat(),"plans":prepared,
              "fixture_manifest_sha256":sha(fm_path.read_bytes()),"source_manifest_sha256":fm["source_manifest_sha256"],
              "builder_files_sha256":{name:sha((HERE/name).read_bytes()) for name in ("build_fixtures.py","freeze_plans.py","fetch_sources.py")},
              "total_measured_pairs":24,"gpu_executed":False,"inference_imports":False,"selection":"All inputs, lengths, generated lengths, seeds and order chosen before any v3 model outcome; root pins the independent execution harness separately."}
    assert_no_inference_imports()
    write_json(HERE/"plans-manifest.json",manifest)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
