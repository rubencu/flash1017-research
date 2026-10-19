"""Freeze existing public inputs, exact source closure and executable schedule.

Loads a tokenizer only; never loads weights or executes model inference.
"""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "outputs/flash1017-lab"
APP = Path("/Applications/oMLX.app/Contents/Resources")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan_path = HERE / "plan-01.json"
    manifest_path = HERE / "manifest-01.json"
    if plan_path.exists() or manifest_path.exists():
        raise FileExistsError("Preserve frozen plans")
    proposal_path = HERE / "evidence/agent-proposed-plan-02.json"
    proposal = json.loads(proposal_path.read_text())
    candidate = json.loads((LAB / "qualified-candidate-profile.json").read_text())
    checkpoint = json.loads((LAB / "results/checkpoint-hashes.json").read_text())
    from transformers import PreTrainedTokenizerFast
    from omlx.patches.deepseek_v41.processing import Processor
    from omlx.patches.deepseek_v41.config import ModelConfig
    from flash1017.bench import prepare
    model_path = Path(checkpoint["directory"])
    processor = Processor(PreTrainedTokenizerFast.from_pretrained(model_path, local_files_only=True),
                          ModelConfig.from_dict(json.loads((model_path / "config.json").read_text())))
    source_paths = {proposal_path, LAB / "qualified-candidate-profile.json", LAB / "qualified-profile.json",
        LAB / "results/checkpoint-hashes.json", LAB / "provenance.json"}
    prompts = {}
    for fixture in proposal["fixtures"]:
        source_path = Path(fixture["source_plan"])
        if sha(source_path) != fixture["source_plan_sha256"]:
            raise ValueError("Historical fixture source changed")
        source_paths.add(source_path)
        source = json.loads(source_path.read_text())["prompts"][fixture["source_plan_prompt_key"]]
        prompt = {k: copy.deepcopy(v) for k, v in source.items() if k != "qualification"}
        prompt.update({k: fixture[k] for k in ("max_tokens", "temperature", "top_p", "force_length", "thinking", "timeout_seconds", "cohort")})
        tokens = prepare(processor, prompt)
        digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        if len(tokens) != fixture["input_tokens"] or digest != fixture["input_tokens_sha256"]:
            raise ValueError("Prepared exact input differs from recorded fixture: " + fixture["id"])
        prompt.update(tokens=tokens, input_tokens_sha256=digest)
        prompts[fixture["id"]] = prompt
    source_plan_path = Path(proposal["source_plan"])
    source_paths.add(source_plan_path)
    warm_id = proposal["warmup"]["source_plan_prompt_key"]
    warm = copy.deepcopy(json.loads(source_plan_path.read_text())["prompts"][warm_id])
    warm.update(max_tokens=proposal["warmup"]["max_tokens"], cohort="excluded-warmup", timeout_seconds=1200)
    prompts[warm_id] = warm
    profiles = {}
    for name, step in (("native2048", 2048), ("native8192", 8192)):
        profiles[name] = {"mtp": True, "depth": 3, "fixed": False, "ced_prefill": False,
            "prefill_step": step, "engram_prefetch_bytes": 16*2**20,
            "attention_rounding": False, "fused_head": False}
    profiles["complete"] = copy.deepcopy(candidate["runtime"])
    processes = []
    for block in proposal["process_blocks"]:
        for arm in block["arm_order"]:
            jobs = [{"prompt": warm_id, "seed": proposal["warmup"]["seed"], "warmup": True}]
            jobs += [{"prompt": case, "seed": block["seed_by_case"][case], "warmup": False}
                     for case in block["case_order"]]
            processes.append({"arm": arm, "block": block["block"], "jobs": jobs})
    plan = {"status": "frozen_before_any_model_run", "created_at_unix": time.time(),
        "scope": proposal["scope"], "proposal_sha256": sha(proposal_path),
        "cache_limit_bytes": 8*2**30, "profiles": profiles, "prompts": prompts,
        "processes": processes, "counts": proposal["counts"], "cache_policy": proposal["cache_policy"],
        "claim_policy": proposal["claim_policy"], "model_execution_wall_limit_seconds": 7200,
        "native_mtp_depth_basis": "Installed utils/model_loading.py default branch sets depth3; native adaptive controller enabled, not fixed.",
        "common_harness_changes": "Same direct-load residency/headroom guard, resource meter, statistics capture and bench.run_one in every arm; no stock inference arithmetic patch."}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    source_paths.add(plan_path)
    for path in HERE.rglob("*.py"):
        source_paths.add(path)
    for path in (LAB / "flash1017").glob("*.py"):
        source_paths.add(path)
    # Verify the complete qualified closure without changing it.
    for relative, digest in candidate["inference_files_sha256"].items():
        path = LAB / relative
        if sha(path) != digest:
            raise ValueError("Qualified inference source changed: " + relative)
        source_paths.add(path)
    candidate_manifest_path = ROOT / "work/serving-profile-v3/staged-01/candidate-manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    source_paths.add(candidate_manifest_path)
    for filename, digest in candidate_manifest["sources_sha256"].items():
        path = Path(filename)
        if sha(path) != digest:
            raise ValueError("Qualified candidate dependency changed: " + filename)
        source_paths.add(path)
    for file in (ROOT / "work/serving-integration-v3b").glob("*.py"):
        source_paths.add(file)
    stock_root = APP / "omlx"
    provenance = json.loads((LAB / "provenance.json").read_text())
    for relative, digest in provenance["baseline_files_sha256"].items():
        if sha(stock_root / relative) != digest:
            raise ValueError("Installed oMLX differs from original baseline: " + relative)
    for file in stock_root.rglob("*"):
        if file.is_file() and file.suffix in {".py", ".so", ".dylib", ".metallib"}:
            source_paths.add(file)
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "mlx-vlm", "numpy", "transformers")}
    manifest = {"created_at_unix": time.time(), "status": "frozen_before_any_model_run", "packages": versions,
        "stock_label": "Unmodified installed oMLX.app0.7.1rc1 inference source with native adaptive MTP",
        "stock_is_current_main": False, "checkpoint_revision": "78456115d301eecf9fadcdcc81c0cf6f8b2c8eab",
        "sources_sha256": {str(path.resolve()): sha(path) for path in sorted(source_paths)}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"plan_sha256": sha(plan_path), "manifest_sha256": sha(manifest_path),
        "sources": len(source_paths), "packages": versions, "counts": plan["counts"]}, indent=2))


if __name__ == "__main__":
    main()
