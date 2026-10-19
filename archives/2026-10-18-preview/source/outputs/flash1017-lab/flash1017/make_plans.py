"""Write fixed, inspectable experiment plans before running measurements."""
import argparse
import json
from pathlib import Path
import random


def plans():
    profiles = {
        "stock-ar": {"mtp": False},
        "stock-a3": {"mtp": True, "depth": 3},
        "reference-a3": {"mtp": True, "depth": 3, "attention_rounding": True},
        "reference-ar": {"mtp": False, "attention_rounding": True},
    }
    profiles["stock-a3-head"] = profiles["stock-a3"] | {"fused_head": True}
    profiles["reference-a3-head"] = profiles["reference-a3"] | {"fused_head": True}
    profiles["reference-a5-head6"] = {"mtp": True, "depth": 5, "attention_rounding": True,
                                       "fused_head": True, "head_max_rows": 6}
    smoke_prompts = {
        "identity": {"text": "Reply with exactly this string and nothing else: FLASH1017_OK", "max_tokens": 32},
        "arithmetic": {"text": 'Return only JSON with keys "product" and "sum" for 37 and 19.', "max_tokens": 96},
        "code": {"text": "Write a Python function stable_unique(xs) that removes duplicates while preserving first occurrence. Include a short example with a repeated value. Return only code.", "max_tokens": 256},
    }
    smoke = {"profiles": profiles, "prompts": smoke_prompts,
             "jobs": [{"profile": profile, "prompt": prompt}
                      for prompt in smoke_prompts for profile in profiles]}

    profiles = dict(profiles)
    for fixed in (False, True):
        for depth in range(1, 6):
            name = f"reference-{'f' if fixed else 'a'}{depth}"
            profiles[name] = {"mtp": True, "depth": depth, "fixed": fixed,
                              "attention_rounding": True}
            profiles[f"stock-{'f' if fixed else 'a'}{depth}"] = {
                "mtp": True, "depth": depth, "fixed": fixed,
                "attention_rounding": False,
            }
    profiles["reference-a3-ced"] = profiles["reference-a3"] | {"ced_prefill": True}
    profiles.update({name + "-head": profile | {"fused_head": True}
                     for name, profile in list(profiles.items())
                     if profile.get("mtp", True) and not profile.get("fused_head", False)
                     and (name.startswith("reference-") or name == "stock-a3")})
    profiles["reference-f5-head6"] = profiles["reference-f5"] | {"fused_head": True, "head_max_rows": 6}
    prompts = {
        "cal-code-1k": {"text": "Write a complete, extensively documented Python module implementing an LRU cache with capacity resizing, statistics, snapshots, restoration, and a large unittest suite. Aim for at least 200 lines. Begin with the implementation and then the tests.", "prompt_tokens": 1024, "max_tokens": 256, "temperature": 1.0, "force_length": True},
        "cal-prose-1k": {"text": "Write a detailed textbook chapter of at least 1500 words explaining how a household refrigerator moves heat. Cover thermodynamics, refrigerant phase changes, the compressor, thermostat control, practical energy measurements, and common failure modes. Start directly with the chapter.", "context": "A heat pump transfers thermal energy from a colder reservoir to a warmer reservoir. Work must be supplied to maintain the cycle. Measurements distinguish heat flow, electrical input and useful cooling.\n", "prompt_tokens": 1024, "max_tokens": 256, "temperature": 1.0, "force_length": True},
        "warmup": {"text": "Write a complete 200-line Python module implementing an LRU cache with tests and documentation.", "prompt_tokens": 1024, "max_tokens": 64, "temperature": 1.0, "force_length": True},
    }
    sweep_jobs = []
    for profile in profiles:
        sweep_jobs.append({"profile": profile, "prompt": "warmup", "warmup": True})
    measured = [{"profile": profile, "prompt": prompt, "repeat": repeat, "seed": 1017 + repeat}
                for repeat in range(2) for prompt in ("cal-code-1k", "cal-prose-1k")
                for profile in profiles]
    random.Random(1017).shuffle(measured)
    sweep = {"profiles": profiles, "prompts": prompts, "jobs": sweep_jobs + measured}

    prof = {f"cost-{depth}": {"mtp": True, "depth": 5, "fixed": True,
                              "attention_rounding": True, "instrument": True,
                              "draft_width": 5, "forced_depth": depth, "fused_head": True}
            for depth in range(6)}
    cost_jobs = [{"profile": profile, "prompt": "warmup", "warmup": True} for profile in prof]
    measured = [{"profile": profile, "prompt": prompt, "repeat": repeat, "seed": 1017 + repeat}
                for repeat in range(2) for prompt in ("cal-code-1k", "cal-prose-1k") for profile in prof]
    random.Random(1018).shuffle(measured)
    cost = {"profiles": prof, "prompts": prompts, "jobs": cost_jobs + measured}
    return {"smoke": smoke, "sweep": sweep, "cost-calibration": cost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, plan in plans().items():
        path = args.directory / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
