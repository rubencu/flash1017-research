#!/usr/bin/env python3
"""Freeze an outcome-independent GSM8K subset and synthetic long-context controls.

This does not load a model. Only the pinned checkpoint's tokenizer.json is read.
GSM8K is a public quality benchmark; the authored retrieval controls are an
engineering diagnostic and must not be called a RULER evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

GSM_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SELECTION_SEED = "flash1017-quality-01-gsm8k-v1"
CONTROL_SEED = "flash1017-quality-01-long-context-v1"
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def record(seed, index):
    digest = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest().upper()
    return f"Record K{digest[:16]} has value V{digest[16:32]}.\n"


def long_control(tokenizer, target, position, hops):
    task_id = f"long-{target}-{position}-{hops}hop"
    seed = f"{CONTROL_SEED}:{task_id}"
    key = "Q" + sha256(f"{seed}:query".encode())[:16].upper()
    value = "A" + sha256(f"{seed}:answer".encode())[:16].upper()
    middle = "M" + sha256(f"{seed}:middle".encode())[:16].upper()
    header = (
        "Read the following lookup records. Every key appears exactly once. "
        "Use only these records to answer the query.\n\n"
    )
    if hops == 1:
        needles = [f"Record {key} has value {value}.\n"]
        question = f"What is the value of {key}?"
    else:
        needles = [
            f"Record {key} has value {middle}.\n",
            f"Record {middle} has value {value}.\n",
        ]
        question = (
            f"Start at key {key}. Look up its value, then use that value as a key "
            "and look up the second value. What is the second value?"
        )
    footer = (
        f"\nQuery: {question}\n"
        "Reply with exactly one line in this format: #### ANSWER\n"
        "Replace ANSWER with the complete value, without explanation."
    )
    positions = {"early": 0.1, "middle": 0.5, "late": 0.9}
    # A filler record is roughly 30 tokens. Size by the actual pinned tokenizer,
    # then adjust the number of whole records; do not split a needle or a record.
    count = target // 30
    while True:
        filler = [record(seed, i) for i in range(count)]
        first = int(count * positions[position])
        insertions = {first: needles[0]}
        if hops == 2:
            # The second lookup is deliberately far from the first and can
            # precede it. Never place both in the same local context window.
            second = int(count * (0.9 if position != "late" else 0.1))
            insertions[second] = needles[1]
        body = "".join(insertions.get(i, "") + line for i, line in enumerate(filler))
        prompt = header + body + footer
        n_tokens = len(tokenizer.encode(prompt).ids)
        if n_tokens > target:
            count -= max(1, (n_tokens - target) // 28)
        elif target - n_tokens > 64:
            count += max(1, (target - n_tokens - 32) // 40)
        else:
            break
    offsets = []
    for needle in needles:
        assert prompt.count(needle) == 1
        offset = prompt.index(needle)
        offsets.append(len(tokenizer.encode(prompt[:offset]).ids))
    return {
        "task_id": task_id,
        "suite": "synthetic-long-context-controls",
        "quality_benchmark": False,
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "expected_answer": value,
        "scorer": "exact-marked-string-v1",
        "metadata": {
            "target_content_tokens": target,
            "actual_content_tokens": n_tokens,
            "needle_token_offsets": offsets,
            "position": position,
            "hops": hops,
            "selection": "complete 2 lengths x 3 positions x 2 lookup depths",
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    from tokenizers import Tokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_file = args.output_dir / "tasks.jsonl"
    if tasks_file.exists():
        raise SystemExit(
            "Refusing to overwrite frozen tasks; choose a new output directory"
        )
    source_path = args.source_dir / "gsm8k-test.jsonl"
    examples = [json.loads(line) for line in source_path.read_text().splitlines()]
    if len(examples) != 1319:
        raise SystemExit(f"Unexpected GSM8K test count: {len(examples)}")
    keyed = [
        (
            sha256(f"{SELECTION_SEED}\n{i}\n{ex['question']}".encode()),
            i,
            ex,
        )
        for i, ex in enumerate(examples)
    ]
    selected = sorted(keyed)[:64]
    tasks = []
    for selection_hash, index, ex in selected:
        match = ANSWER_RE.search(ex["answer"])
        if match is None:
            raise SystemExit(f"No official numeric answer in test row {index}")
        prompt = (
            ex["question"] + "\n\nSolve the problem. You may show your reasoning. "
            "Finish with exactly one line in the format #### NUMBER, "
            "where NUMBER is the final numerical answer."
        )
        tasks.append(
            {
                "task_id": f"gsm8k-test-{index:04d}",
                "suite": "gsm8k-test-64-hash-subset",
                "quality_benchmark": True,
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "expected_answer": match.group(1).strip().replace(",", ""),
                "scorer": "gsm8k-official-regex-string-v1",
                "metadata": {
                    "source_index": index,
                    "source_question_sha256": sha256(ex["question"].encode()),
                    "selection_sha256": selection_hash,
                },
            }
        )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for target in (16384, 65536):
        for position in ("early", "middle", "late"):
            for hops in (1, 2):
                tasks.append(long_control(tokenizer, target, position, hops))
    write_jsonl(tasks_file, tasks)
    # The request file has no answers or reference solutions, reducing accidental
    # leakage into a generic generation harness.
    requests = [
        {k: v for k, v in task.items() if k not in {"expected_answer", "scorer"}}
        for task in tasks
    ]
    write_jsonl(args.output_dir / "requests.jsonl", requests)
    write_json(
        args.output_dir / "PROTOCOL.json",
        {
            "name": "flash1017-quality-01-supplement-v1",
            "freeze_before_any_model_outputs": True,
            "gsm8k": {
                "repository": "https://github.com/openai/grade-school-math",
                "revision": GSM_REVISION,
                "dataset_url": f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REVISION}/grade_school_math/data/test.jsonl",
                "source_count": len(examples),
                "selected_count": 64,
                "selection_seed": SELECTION_SEED,
                "selection_rule": "Take the 64 smallest SHA256(seed + newline + zero-based index + newline + exact question), preserving hash order",
                "scoring": "Official ANS_RE.search and comma-removal string equality; no numeric relaxation or answer repair",
                "official_leaderboard_comparable": False,
                "reason": "This is a fixed subset with an explicit zero-shot prompt, not the complete GSM8K benchmark or a replication of a published prompting setup",
            },
            "long_context": {
                "count": 12,
                "seed": CONTROL_SEED,
                "kind": "authored deterministic engineering controls",
                "standard_benchmark": False,
                "ruler_result": False,
                "content_token_targets": [16384, 65536],
                "chat_template_tokens": "Not included in targets; generation harness records actual prompt length",
                "coverage": "single lookup and two-hop lookup at early, middle, late locations",
                "scoring": "Exactly one non-empty output line: #### followed by the exact case-sensitive expected value",
            },
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "samples_per_task_per_build": 1,
                "normal_eos": True,
                "max_tokens": "Per-task requests.jsonl value; identical for both builds",
                "no_output_based_task_filtering": True,
                "no_output_based_prompt_or_budget_changes": True,
            },
            "paired_acceptance": {
                "nonzero_regression_margin": None,
                "rule": "No baseline-correct to candidate-incorrect task, no missing/duplicate result, no request error, and no length-truncated response in either build",
                "improvements_cannot_offset_regressions": True,
                "missing_or_truncated": "Retain and report as failed/incomplete; cannot silently exclude",
                "score_means": "No observed task regressions on this frozen suite, not universal answer-quality equivalence",
                "confidence": "Report a one-sided 95% exact binomial bound on observed regression rate, with sampling assumptions and finite-suite limits; do not interpret a nonsignificant difference as equivalence",
                "engineering_gate": "Numerical/state parity and code-path activation are separate prerequisites owned by the runtime validation harness",
            },
            "sources": {
                str(path.name): {
                    "sha256": sha256(path.read_bytes()),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(args.source_dir.iterdir())
                if path.is_file()
            },
            "tokenizer_sha256": sha256(args.tokenizer.read_bytes()),
            "tasks_sha256": sha256(tasks_file.read_bytes()),
            "requests_sha256": sha256(
                (args.output_dir / "requests.jsonl").read_bytes()
            ),
            "prepare_script_sha256": sha256(Path(__file__).read_bytes()),
        },
    )
    print(json.dumps({"tasks": len(tasks), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
