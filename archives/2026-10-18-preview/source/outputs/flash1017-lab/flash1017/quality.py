"""Small local coding regression suite, NOT a replacement for public benchmarks.

Test inputs are deterministic but omitted from prompts. Generated functions
run in a separate process with time/resource limits and restricted syntax and
builtins. No generated imports, file/network access, classes, or dunder access.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import heapq
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import threading


TASKS = {
    "merge_intervals": "Implement merge_intervals(intervals). Input is a list of [start, end] integer pairs with start <= end. Return a list of merged pairs sorted by start. Merge overlapping OR touching intervals. Empty input returns []. Do not mutate the input.",
    "longest_unique": "Implement longest_unique(text), returning the length of the longest contiguous substring containing no repeated characters. Handle empty strings and arbitrary Unicode characters.",
    "stable_unique": "Implement stable_unique(values), returning the first occurrence of each distinct value in order. Values may include nested lists and dictionaries, so they need not be hashable. Equality uses normal Python ==. Do not mutate the input.",
    "glob_match": "Implement glob_match(text, pattern), returning a bool. The entire text must match. '?' matches exactly one Unicode character; '*' matches zero or more characters. All other pattern characters are literal. Empty strings and consecutive stars must work.",
    "topological_order": "Implement topological_order(graph), where graph is a dict mapping string nodes to lists of outgoing neighbor names. Nodes appearing only as neighbors must be included. Return the lexicographically smallest valid topological ordering, always choosing the smallest currently available node. Duplicate edges count as one edge. Raise ValueError on a directed cycle. Empty input returns []. Do not mutate graph.",
    "shortest_path": "Implement shortest_path(edges, start, end). edges is a list of [source, destination, nonnegative_integer_weight] describing directed edges between string nodes. Return the minimum total weight from start to end, or None if unreachable. If start == end return 0 even if absent from edges. Parallel edges and zero weights are allowed.",
    "lru_simulate": "Implement lru_simulate(capacity, operations). Each operation is ['put', key, value] or ['get', key], with string keys. Return [get_results, final_keys_oldest_to_newest]. A get returns the value or None and makes a hit most recently used. Put updates/inserts and makes that key most recent, evicting oldest keys to fit capacity. Capacity is a nonnegative integer; zero stores nothing. Do not mutate the input.",
    "rotate_ragged": "Implement rotate_ragged(rows, fill=None). Pad each input row on the right to the maximum row length using fill, then rotate the resulting rectangle 90 degrees clockwise. Return new lists and do not mutate rows. Empty input or all-empty rows returns [].",
}


def oracle(name, args):
    if name == "merge_intervals":
        out = []
        for a, b in sorted(args[0]):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out
    if name == "longest_unique":
        best, begin, seen = 0, 0, {}
        for i, char in enumerate(args[0]):
            begin = max(begin, seen.get(char, -1) + 1)
            best = max(best, i - begin + 1)
            seen[char] = i
        return best
    if name == "stable_unique":
        out = []
        for value in args[0]:
            if value not in out:
                out.append(value)
        return out
    if name == "glob_match":
        text, pattern = args
        regex = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
        return re.fullmatch(regex, text, re.DOTALL) is not None
    if name == "topological_order":
        graph = {node: set(neighbors) for node, neighbors in args[0].items()}
        nodes = set(graph) | {v for values in graph.values() for v in values}
        degree = dict.fromkeys(nodes, 0)
        for neighbors in graph.values():
            for target in neighbors:
                degree[target] += 1
        ready = [node for node in nodes if degree[node] == 0]
        heapq.heapify(ready)
        out = []
        while ready:
            node = heapq.heappop(ready)
            out.append(node)
            for neighbor in graph.get(node, ()):
                degree[neighbor] -= 1
                if not degree[neighbor]:
                    heapq.heappush(ready, neighbor)
        if len(out) != len(nodes):
            raise ValueError("cycle")
        return out
    if name == "shortest_path":
        edges, start, end = args
        dist = {start: 0}
        # Independent Bellman-Ford oracle, including isolated start=end.
        for _ in range(len({v for a, b, _ in edges for v in (a, b)})):
            changed = False
            for a, b, weight in edges:
                if a in dist and dist[a] + weight < dist.get(b, float("inf")):
                    dist[b] = dist[a] + weight
                    changed = True
            if not changed:
                break
        return dist.get(end)
    if name == "lru_simulate":
        capacity, operations = args
        cache, output = OrderedDict(), []
        for op in operations:
            if op[0] == "get":
                output.append(cache.get(op[1]))
                if op[1] in cache:
                    cache.move_to_end(op[1])
            else:
                cache[op[1]] = op[2]
                cache.move_to_end(op[1])
                while len(cache) > capacity:
                    cache.popitem(last=False)
        return [output, list(cache)]
    if name == "rotate_ragged":
        rows = args[0]
        fill = args[1] if len(args) > 1 else None
        width = max(map(len, rows), default=0)
        return [[row[i] if i < len(row) else fill for row in reversed(rows)]
                for i in range(width)]
    raise ValueError(name)


def cases(name):
    rng = random.Random(7311017)
    result = []
    for _ in range(160):
        n = rng.randrange(20)
        if name == "merge_intervals":
            result.append([[sorted([rng.randrange(-12, 13), rng.randrange(-12, 13)]) for _ in range(n)]])
        elif name == "longest_unique":
            result.append(["".join(rng.choices("abAB01🦊é中\n", k=n))])
        elif name == "stable_unique":
            pool = [None, 0, False, 1, True, [1], [1, [2]], {"x": 1}, "a", []]
            result.append([[rng.choice(pool) for _ in range(n)]])
        elif name == "glob_match":
            result.append(["".join(rng.choices("ab🦊\n", k=rng.randrange(9))),
                           "".join(rng.choices("ab🦊?*", k=rng.randrange(9)))])
        elif name == "topological_order":
            nodes = list("abcdef")[:rng.randrange(7)]
            graph = {node: [rng.choice(nodes) for _ in range(rng.randrange(4))]
                     for node in nodes[:rng.randrange(len(nodes) + 1)]}
            result.append([graph])
        elif name == "shortest_path":
            nodes = list("abcdef")
            edges = [[rng.choice(nodes), rng.choice(nodes), rng.randrange(10)] for _ in range(n)]
            result.append([edges, rng.choice(nodes + ["isolated"]), rng.choice(nodes + ["isolated"])])
        elif name == "lru_simulate":
            operations = []
            for _ in range(n * 2):
                key = rng.choice("abcdef")
                operations.append(["get", key] if rng.random() < .5 else ["put", key, rng.randrange(-9, 10)])
            result.append([rng.randrange(5), operations])
        elif name == "rotate_ragged":
            result.append([[[rng.randrange(-9, 10) for _ in range(rng.randrange(8))] for _ in range(n)], rng.choice([None, 0, "x"])])
    return result


def extract_code(text):
    text = text.split("</think>")[-1].split("<｜end▁of▁sentence｜>")[0]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else text.strip()


def validate_code(source, name):
    if len(source) > 32000:
        raise ValueError("Generated source exceeds 32K character limit")
    tree = ast.parse(source)
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal,
                 ast.AsyncFunctionDef, ast.Await, ast.With, ast.AsyncWith)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"Unsupported generated syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and "__" in node.attr:
            raise ValueError("Dunder attribute access is forbidden")
        if isinstance(node, ast.Name) and "__" in node.id:
            raise ValueError("Dunder names are forbidden")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise ValueError("Decorators are forbidden")
    if not any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body):
        raise ValueError("Required top-level function is missing")
    return tree


def worker():
    import builtins
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    payload = json.load(sys.stdin)
    tree = validate_code(payload["source"], payload["name"])
    names = "abs all any bool dict enumerate filter float frozenset int isinstance iter len list map max min next ord chr range reversed round set slice sorted str sum tuple zip Exception ValueError TypeError KeyError IndexError StopIteration"
    allowed = {name: getattr(builtins, name) for name in names.split()}
    env = {"__builtins__": allowed}
    exec(compile(tree, "<generated-function>", "exec"), env)
    output = []
    for args in payload["cases"]:
        before = json.dumps(args, sort_keys=True)
        try:
            value = env[payload["name"]](*args)
            output.append({"value": value, "mutated": json.dumps(args, sort_keys=True) != before})
        except Exception as exc:
            output.append({"error": type(exc).__name__})
    print(json.dumps(output))


def judge(name, text):
    source = extract_code(text)
    try:
        validate_code(source, name)
    except (ValueError, SyntaxError) as exc:
        return {"passed": False, "error": str(exc), "source": source}
    inputs = cases(name)
    payload = {"name": name, "source": source, "cases": inputs}
    try:
        # -I removes cwd/PYTHONPATH and user site injection. Only this trusted
        # worker module is loaded by its explicit path; source arrives on stdin.
        invocation = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"]
        import psutil
        process = subprocess.Popen(invocation, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, cwd="/private/tmp")
        done = threading.Event()
        memory_exceeded = threading.Event()

        def watch_memory():
            try:
                child = psutil.Process(process.pid)
            except psutil.NoSuchProcess:
                return
            while not done.wait(.02):
                try:
                    if child.memory_info().rss > 256 * 2**20:
                        memory_exceeded.set()
                        process.kill()
                        break
                except psutil.NoSuchProcess:
                    break

        watcher = threading.Thread(target=watch_memory, daemon=True)
        watcher.start()
        try:
            stdout, stderr = process.communicate(json.dumps(payload), timeout=6)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        finally:
            done.set()
            watcher.join()
        if process.returncode:
            return {"passed": False, "error": "Memory limit" if memory_exceeded.is_set() else "Worker failed",
                    "returncode": process.returncode, "stderr": stderr[-1000:], "source": source}
        actual = json.loads(stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"passed": False, "error": type(exc).__name__, "source": source}
    failures = []
    for index, (args, value) in enumerate(zip(inputs, actual)):
        try:
            expected = {"value": oracle(name, args), "mutated": False}
        except ValueError:
            expected = {"error": "ValueError"}
        if value != expected:
            failures.append({"case": index, "input": args, "expected": expected, "actual": value})
    return {"passed": len(actual) == len(inputs) and not failures,
            "cases": len(inputs), "failed_cases": len(failures),
            "first_failures": failures[:3], "source": source}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker()
    if not args.input or not args.output:
        parser.error("--input and --output required")
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") != "run" or row["prompt_id"] not in TASKS:
            continue
        results.append({"task": row["prompt_id"], "profile": row["profile"],
                        "repeat": row["repeat"], "total_seconds": row["total_seconds"],
                        "finish_reason": row["finish_reason"],
                        **judge(row["prompt_id"], row["text"])})
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
