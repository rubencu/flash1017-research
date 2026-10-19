#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the pinned EvalPlus zero-shot chat requests without executing code."""
import argparse
import hashlib
import json
from pathlib import Path

EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
DATASETS = {
    "humaneval": {"file": "HumanEvalPlus-v0.1.10.jsonl", "tasks": 164,
                  "sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"},
    "mbpp": {"file": "MbppPlus-v0.2.0.jsonl", "tasks": 378,
             "sha256": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"},
}
# Exact v0.3.1 OpenAI/chat-mode text, from codegen.py and
# evalplus/gen/util/openai_request.py (EvalPlus Apache-2.0).
SYSTEM = "You are a helpful assistant good at coding."
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"


def load_tasks(datasets):
    result = {}
    for suite, identity in DATASETS.items():
        data = (Path(datasets) / identity["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError(f"Pinned {suite} dataset hash mismatch")
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(rows) != identity["tasks"] or len({r["task_id"] for r in rows}) != len(rows):
            raise ValueError(f"Pinned {suite} task inventory mismatch")
        for row in rows:
            if row["task_id"] in result:
                raise ValueError("Repeated task identity")
            result[row["task_id"]] = (suite, row)
    return result


def request(task_id, suite, problem):
    return {"task_id": task_id, "suite": suite, "entry_point": problem["entry_point"],
            "max_tokens": 2048,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": INSTRUCTION + "\n```python\n" + problem["prompt"].strip() + "\n```"}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = load_tasks(args.datasets)
    data = "".join(json.dumps(request(tid, suite, problem), ensure_ascii=False) + "\n"
                   for tid, (suite, problem) in tasks.items()).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(data)
    print(json.dumps({"path": str(args.output.resolve()), "tasks": len(tasks),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "contents": "messages and entry points only; no hidden inputs or canonical answers"}, indent=2))


if __name__ == "__main__":
    main()
