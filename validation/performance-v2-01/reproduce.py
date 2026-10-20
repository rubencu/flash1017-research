#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recalculate saved measurements, or replay the frozen audit with exact wheels.

CPU-only. No inference, generated answer execution, server access, or private
settings. The measurement mode does not substitute for the full wheel audit.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require(value, message):
    if not value:
        raise ValueError(message)


def geomean(values):
    values = list(values)
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def verify_bundle():
    manifest = read(HERE / "MANIFEST.json")
    for item in manifest["files"]:
        name = PurePosixPath(item["path"])
        require(
            not name.is_absolute() and ".." not in name.parts, "Unsafe manifest path"
        )
        data = (HERE / name).read_bytes()
        require(
            len(data) == item["bytes"]
            and hashlib.sha256(data).hexdigest() == item["sha256"],
            "Changed evidence: " + str(name),
        )


def measurements():
    report = read(HERE / "qualified/pair-summary-final-01.json")
    plan = read(HERE / "qualified/protocol.json")
    require(report["status"] == "passed", "Saved comparison did not qualify")
    lookup = {}
    for process in plan["processes"]:
        values = rows(HERE / "qualified" / process["process_id"] / "responses.jsonl")
        sequence = [process["warmup_task_id"], *process["measured_task_ids"]]
        require(
            [r["task_id"] for r in values] == sequence,
            "Missing, reordered or extra responses",
        )
        for index, row in enumerate(values):
            require(row["warmup"] is (index == 0), "Incorrect warmup marker")
            require(
                not row.get("error") and not row.get("observer_error"), "Recorded error"
            )
            lookup[(process["arm"], process["block"], row["task_id"])] = row
    case_ratios = {}
    for pair in report["pairs"]:
        task, block = pair["task_id"], pair["block"]
        a, b = (lookup[(arm, block, task)] for arm in ["baseline", "candidate"])
        for field in ["request_sha256", "text", "finish_reason"]:
            require(a[field] == b[field], "Paired response mismatch: " + field)
        for field in ["prompt_tokens", "completion_tokens"]:
            require(a["usage"][field] == b["usage"][field], "Token-count mismatch")
        ratio = {
            "client_ttft_seconds": a["client_ttft_seconds"] / b["client_ttft_seconds"]
        }
        for metric in ["prompt_tokens_per_second", "generation_tokens_per_second"]:
            ratio[metric] = b["usage"][metric] / a["usage"][metric]
        if a["finish_reason"] == b["finish_reason"] == "stop":
            ratio["completed_answer_latency_seconds"] = (
                a["wall_seconds"] / b["wall_seconds"]
            )
        require(
            ratio == pair["ratios"], "Saved pair ratio differs from raw measurements"
        )
        case_ratios.setdefault(task, {})[block] = ratio
    require(
        len(case_ratios) == 8 and all(set(v) == {0, 1} for v in case_ratios.values()),
        "Incomplete 16-pair comparison",
    )
    for task in case_ratios:
        for arm in ["baseline", "candidate"]:
            a, b = (lookup[(arm, block, task)] for block in [0, 1])
            require(
                (a["text"], a["finish_reason"]) == (b["text"], b["finish_reason"]),
                "Within-arm output variation",
            )
    per_case = {
        task: {
            metric: geomean(blocks[b][metric] for b in [0, 1])
            for metric in set(blocks[0]) & set(blocks[1])
        }
        for task, blocks in case_ratios.items()
    }
    groups = {}
    for name, group in report["aggregates"]["groups"].items():
        result = {}
        for metric, saved in group["metrics"].items():
            eligible = [task for task in group["task_ids"] if metric in per_case[task]]
            require(eligible == saved["eligible_task_ids"], "Eligibility changed")
            require(
                len(eligible) * 2 == saved["paired_observations"], "Wrong denominator"
            )
            value = (
                geomean(per_case[task][metric] for task in eligible)
                if eligible
                else None
            )
            require(value == saved["geomean_ratio"], "Group arithmetic differs")
            result[metric] = {
                "geomean_ratio": value,
                "paired_observations": len(eligible) * 2,
            }
        groups[name] = result
    return {
        "status": "recorded_measurements_reproduced",
        "scope": "Recalculated all 16 paired ratios and group aggregates from saved raw responses. Bundle hashes checked. Does not repeat wheel/source/runtime identity checks or model inference.",
        "measured_requests": 32,
        "excluded_warmups": 4,
        "pairs": 16,
        "groups": groups,
    }


def full_audit(args):
    require(
        args.baseline_wheel and args.candidate_wheel,
        "Full audit requires both exact wheel files",
    )
    source_prefix = "work/flash1017-performance-" + (
        "04" if args.attempt == "qualified" else "03"
    )
    with tempfile.TemporaryDirectory(prefix="flash1017-evidence-") as directory:
        stage = Path(directory)
        for item in read(HERE / "COPY-PROVENANCE.json")["files"]:
            name = PurePosixPath(item["source_workspace_path"])
            require(
                not name.is_absolute() and ".." not in name.parts,
                "Unsafe source mapping",
            )
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / item["bundle_path"], target)
        attempt = stage / source_prefix
        plan = read(attempt / "protocol.json")
        command = [
            sys.executable,
            "-I",
            "-B",
            str(stage / plan["analysis_harness_path"]),
            "--workspace-root",
            str(stage),
            "--plan",
            str(attempt / "protocol.json"),
            "--baseline-wheel",
            str(args.baseline_wheel.resolve()),
            "--candidate-wheel",
            str(args.candidate_wheel.resolve()),
            "--output",
            str(args.output.resolve()),
        ]
        for process in plan["processes"]:
            pid = process["process_id"]
            command.extend(["--run", pid + "=" + str(attempt / pid)])
        result = subprocess.run(command, check=False)
        if args.attempt == "qualified":
            require(result.returncode == 0, "Frozen audit failed; inspect output")
            require(
                read(args.output)
                == read(HERE / "qualified/pair-summary-final-01.json"),
                "Replayed report differs from saved report",
            )
        else:
            observed = read(args.output)
            require(
                result.returncode == 1
                and observed["status"] == "failed"
                and observed["aggregates"] is None,
                "Interrupted attempt must remain failed without aggregates",
            )
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["measurements", "full"], default="measurements"
    )
    parser.add_argument(
        "--attempt", choices=["qualified", "interrupted"], default="qualified"
    )
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        not args.output.resolve().is_relative_to(HERE),
        "Write outputs outside the frozen bundle",
    )
    verify_bundle()
    if args.mode == "full":
        return full_audit(args)
    require(
        args.attempt == "qualified",
        "Interrupted attempt has no aggregate to reproduce; use full mode",
    )
    result = measurements()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "pairs": 16, "output": str(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
