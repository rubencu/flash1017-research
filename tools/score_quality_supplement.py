#!/usr/bin/env python3
"""Score fixed paired outputs without allowing improvements to erase regressions.

Input JSONL records: {task_id, text, finish_reason, error?}. Only stop or eos
finish reasons qualify as complete. A result may additionally include completion_token_ids
for exact token-sequence comparison. No model execution or generated-code exec.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

GSM_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
STRICT_ANS_RE = re.compile(r"#### ([A-Z0-9]+)")


def extract_answer(text, scorer):
    if scorer == "gsm8k-official-regex-string-v1":
        match = GSM_ANS_RE.search(text)
        return match.group(1).strip().replace(",", "") if match else None
    if scorer == "exact-marked-string-v1":
        match = STRICT_ANS_RE.fullmatch(text.strip())
        return match.group(1) if match else None
    raise ValueError(f"Unknown scorer: {scorer}")


def load_jsonl(path):
    values = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    counts = Counter(v["task_id"] for v in values)
    duplicate = sorted(k for k, v in counts.items() if v != 1)
    if duplicate:
        raise ValueError(f"Duplicate task IDs in {path}: {duplicate}")
    return {v["task_id"]: v for v in values}


def binomial_upper(successes, trials, alpha=0.05):
    """One-sided exact Clopper-Pearson upper confidence bound."""
    if trials == 0 or successes == trials:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    lower, upper = 0.0, 1.0
    for _ in range(100):
        p = (lower + upper) / 2.0
        cdf = sum(
            math.comb(trials, k) * p**k * (1.0 - p) ** (trials - k)
            for k in range(successes + 1)
        )
        if cdf > alpha:
            lower = p
        else:
            upper = p
    return (lower + upper) / 2.0


def outcome(task, result):
    if result is None:
        return {"correct": False, "complete": False, "reason": "missing"}
    text = result.get("text")
    reason = result.get("finish_reason")
    complete = (
        isinstance(text, str) and reason in {"stop", "eos"} and not result.get("error")
    )
    answer = extract_answer(text, task["scorer"]) if isinstance(text, str) else None
    return {
        "correct": complete and answer == task["expected_answer"],
        "complete": complete,
        "answer": answer,
        "finish_reason": reason,
        "error": result.get("error"),
        "reason": "complete" if complete else "error-or-incomplete",
    }


def score(tasks, baseline, candidate):
    unexpected = (set(baseline) | set(candidate)) - set(tasks)
    if unexpected:
        raise ValueError(f"Unexpected task IDs: {sorted(unexpected)}")
    pairs = []
    for task_id, task in tasks.items():
        b = outcome(task, baseline.get(task_id))
        c = outcome(task, candidate.get(task_id))
        pair = {
            "task_id": task_id,
            "suite": task["suite"],
            "baseline": b,
            "candidate": c,
            "regression": b["correct"] and not c["correct"],
            "improvement": not b["correct"] and c["correct"],
            "text_equal": (
                baseline.get(task_id, {}).get("text")
                == candidate.get(task_id, {}).get("text")
                if task_id in baseline and task_id in candidate
                else False
            ),
        }
        if "completion_token_ids" in baseline.get(
            task_id, {}
        ) and "completion_token_ids" in candidate.get(task_id, {}):
            pair["completion_token_ids_equal"] = (
                baseline[task_id]["completion_token_ids"]
                == candidate[task_id]["completion_token_ids"]
            )
        pairs.append(pair)
    summaries = {}
    for suite in sorted({p["suite"] for p in pairs}):
        rows = [p for p in pairs if p["suite"] == suite]
        n = len(rows)
        b_correct = sum(p["baseline"]["correct"] for p in rows)
        c_correct = sum(p["candidate"]["correct"] for p in rows)
        regressions = [p["task_id"] for p in rows if p["regression"]]
        improvements = [p["task_id"] for p in rows if p["improvement"]]
        incomplete = [
            p["task_id"]
            for p in rows
            if not p["baseline"]["complete"] or not p["candidate"]["complete"]
        ]
        summaries[suite] = {
            "tasks": n,
            "baseline_correct": b_correct,
            "candidate_correct": c_correct,
            "accuracy_delta_percentage_points": (c_correct - b_correct) * 100.0 / n,
            "regressions": regressions,
            "improvements": improvements,
            "incomplete_pairs": incomplete,
            "identical_text_pairs": sum(p["text_equal"] for p in rows),
            "regression_rate_one_sided_95pct_exact_upper": binomial_upper(
                len(regressions), n
            ),
            "conditional_regression_among_baseline_correct_one_sided_95pct_upper": binomial_upper(
                len(regressions), b_correct
            ),
            "bound_interpretation": (
                "Exploratory sampling uncertainty under independent representative tasks; "
                "the finite test set is not a random sample of all future prompts. "
                "For authored long-context controls this is only a descriptive calculation."
            ),
            "zero_observed_regression_gate_passed": not regressions and not incomplete,
        }
    return {
        "quality_equivalence_proved": False,
        "acceptance_rule": "zero baseline-correct/candidate-incorrect tasks and complete paired outputs; no offsetting regressions with improvements",
        "zero_observed_regression_gate_passed": all(
            s["zero_observed_regression_gate_passed"] for s in summaries.values()
        ),
        "suites": summaries,
        "pairs": pairs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = score(
        load_jsonl(args.tasks), load_jsonl(args.baseline), load_jsonl(args.candidate)
    )
    result["inputs"] = {
        key: {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in {
            "tasks": args.tasks,
            "baseline": args.baseline,
            "candidate": args.candidate,
            "scorer": Path(__file__),
        }.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
    return 0 if result["zero_observed_regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
