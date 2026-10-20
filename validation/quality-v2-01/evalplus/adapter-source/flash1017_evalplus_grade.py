#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned EvalPlus oracles in one macOS seatbelt process per task/test phase.

Generated and canonical code never executes in the host supervisor. The worker
calls upstream unsafe_execute unchanged under the documented macOS process,
temporary-directory, resource-limit, and explicit seed-policy adaptations.
"""
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import psutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("flash1017_evalplus_export", HERE / "flash1017_evalplus_export.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
MEMORY = 4 * 1024**3
CPU_SECONDS = 180
OUTPUT_LIMIT = 1024**2
MIN_TIME = 1.0
TIME_FACTOR = 4.0
WORKER_MARKER = "FLASH1017_EVALPLUS_WORKER"
PROTOCOL_ID = "flash1017-evalplus-macos-seeded-02"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_source(source):
    result = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if result != protocol.EVALPLUS_COMMIT:
        raise ValueError("EvalPlus checkout is not the pinned v0.3.1 commit")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "evalplus"], check=True, stdout=subprocess.DEVNULL)
    extra = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "evalplus"], text=True)
    if any(line.endswith(".py") for line in extra.splitlines()):
        raise ValueError("Untracked Python files in pinned EvalPlus source")


def limits(cpu=CPU_SECONDS):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    # macOS 27 rejects both AS and DATA when lowering from unlimited. Memory
    # is instead monitored by the parent; this is not a hard allocation cap.
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def profile(source, scratch):
    # No file-write rule for scratch: the supervisor creates all inputs and the
    # private cwd first. EvalPlus's tempdir context is adapted to use that cwd.
    read_roots = ["/System/Library", "/usr/lib", "/Library/Apple/System/Library",
                  sys.base_prefix, sys.prefix, str(source / "evalplus"), str(scratch)]
    reads = "\n".join(f"(subpath {json.dumps(str(Path(p).resolve()))})" for p in read_roots)
    scripts = " ".join(f"(literal {json.dumps(str(p))})" for p in
                       [Path(__file__).resolve(), HERE / "flash1017_evalplus_export.py"])
    executables = " ".join(f"(literal {json.dumps(p)})" for p in
                           {sys.executable, str(Path(sys.executable).resolve())})
    return f'''(version 1)
(deny default)
(allow file-read-metadata)
(allow file-read* (literal "/"))
(allow file-read* (literal {json.dumps(str(source))}))
(allow file-read* {reads} {scripts}
  (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))
(allow file-write-data (literal "/dev/null"))
(allow process-exec {executables})
(allow process-info* (target self))
(allow signal (target self))
(allow sysctl-read)
'''


def child_env(scratch):
    return {"PATH": "/usr/bin:/bin", "HOME": str(scratch), "TMPDIR": str(scratch),
            "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "EVALPLUS_MAX_MEMORY_BYTES": str(MEMORY), WORKER_MARKER: "1"}


def denied(operation):
    try:
        operation()
    except OSError as exc:
        if exc.errno == errno.EPERM:
            return True
        raise
    return False


def isolation_checks(job):
    """Fail closed before importing any candidate/reference source."""
    import socket
    if os.environ.get(WORKER_MARKER) != "1":
        raise RuntimeError("Worker must be started by the sandbox supervisor")
    checks = {}
    checks["seeded_startup"] = (os.environ.get("PYTHONHASHSEED") == "0" and
                                not sys.flags.ignore_environment and
                                bool(sys.flags.no_user_site) and bool(sys.flags.safe_path) and
                                "PYTHONPATH" not in os.environ)
    if not checks["seeded_startup"]:
        raise RuntimeError("Worker startup does not satisfy the seeded isolation policy")
    checks["outside_read_denied"] = denied(lambda: Path(job["canary"]).read_bytes())
    checks["outside_write_denied"] = denied(lambda: Path(job["canary"] + ".write").write_text("probe"))
    checks["scratch_write_denied"] = denied(lambda: Path("forbidden-write").write_text("probe"))
    def socket_attempt():
        with socket.socket() as client:
            client.connect(("127.0.0.1", 9))
    checks["network_denied"] = denied(socket_attempt)
    def fork_attempt():
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
    checks["fork_denied"] = denied(fork_attempt)
    checks["foreign_exec_denied"] = denied(lambda: os.posix_spawn("/bin/echo", ["echo", "unexpected-exec"], {}))
    for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE", "RLIMIT_CPU"):
        checks[name] = list(resource.getrlimit(getattr(resource, name)))
    checks["memory_policy"] = "parent RSS watchdog; 4 GiB threshold, 20 ms polling, possible overshoot"
    if not all(value for key, value in checks.items() if key.endswith("denied")):
        raise RuntimeError(f"Sandbox failed isolation probes: {checks}")
    return checks


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    # Set limits before imports or any reference/candidate code. Doing this in
    # the exec'd worker avoids unsafe preexec_fn calls in a threaded supervisor.
    limits(job.get("cpu_limit", CPU_SECONDS))
    write = sys.stdout.write
    flush = sys.stdout.flush
    def emit(value):
        write(json.dumps(value, ensure_ascii=True) + "\n")
        flush()
    try:
        checks = isolation_checks(job)
        if job["mode"] == "probe":
            emit({"event": "result", "status": "probe_pass", "isolation": checks})
            return
        if job["mode"] == "determinism_probe":
            import random
            import numpy as np
            random.seed(0)
            np.random.seed(0)
            emit({"event": "result", "status": "determinism_probe_pass",
                  "hash": hash("fixed-evaluator-probe"),
                  "random_values": [random.random() for _ in range(3)],
                  "numpy_values": np.random.random(3).tolist(),
                  "ignore_environment": sys.flags.ignore_environment,
                  "no_user_site": sys.flags.no_user_site, "safe_path": sys.flags.safe_path})
            return
        if job["mode"] == "cpu_probe":
            emit({"event": "candidate_start", "timeout": 5})
            while True:
                pass
        if job["mode"] == "memory_probe":
            emit({"event": "candidate_start", "timeout": 5})
            allocation = bytearray(128 * 1024**2)
            for offset in range(0, len(allocation), 4096):
                allocation[offset] = 1
            time.sleep(5)
            raise RuntimeError(f"RSS watchdog failed: {len(allocation)} allocated")
        if job["mode"] == "wall_probe":
            emit({"event": "candidate_start", "timeout": 0.1})
            time.sleep(5)
            raise RuntimeError("Wall watchdog failed")
        if job["mode"] == "output_probe":
            emit({"event": "candidate_start", "timeout": 5})
            write("x" * (OUTPUT_LIMIT + 1))
            flush()
            time.sleep(5)
            raise RuntimeError("Output watchdog failed")
        sys.path.insert(0, job["source"])
        # Some import-only dependencies ask gettempdir() to probe writability.
        # Its fixed answer is our existing, non-writable private directory.
        tempfile.tempdir = job["cwd"]
        import evalplus.eval as upstream
        from evalplus.data.mbpp import mbpp_deserialize_inputs
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.gen.util import trusted_exec
        from evalplus.sanitize import sanitize
        import random
        import numpy as np
        problem = job["problem"]
        inputs = problem[job["phase"] + "_input"]
        if job["suite"] == "mbpp":
            inputs = mbpp_deserialize_inputs(problem["task_id"], inputs)
        random.seed(0)
        np.random.seed(0)
        expected, reference_time = trusted_exec(
            problem["prompt"] + problem["canonical_solution"], inputs, problem["entry_point"],
            record_time=True,
            output_not_none=(job["suite"] == "mbpp" and problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS))
        time_limits = [max(MIN_TIME, TIME_FACTOR * elapsed) for elapsed in reference_time]
        raw = job["raw_completion"]
        # Sanitization parses text only. An exception yields a failed answer,
        # never a dropped task or hand-repaired solution.
        try:
            code = sanitize(raw, entrypoint=problem["entry_point"])
        except Exception as exc:
            emit({"event": "result", "status": "fail", "reason": "sanitization_error",
                  "error_type": type(exc).__name__, "tests": len(inputs)})
            return
        native_chdir = os.chdir
        cwd = os.getcwd()
        @contextlib.contextmanager
        def precreated_tempdir():
            native_chdir(job["cwd"])
            try:
                yield job["cwd"]
            finally:
                native_chdir(cwd)
        upstream.create_tempdir = precreated_tempdir
        original_guard = upstream.reliability_guard
        upstream.reliability_guard = lambda maximum_memory_bytes=None: original_guard(maximum_memory_bytes=None)
        state = SimpleNamespace(value=upstream._UNKNOWN)
        progress = SimpleNamespace(value=0)
        details = [False] * len(inputs)
        timeout = min(60.0, sum(time_limits)) + 2.0  # upstream fast-check join budget
        emit({"event": "candidate_start", "timeout": timeout,
              "solution_sha256": sha(code.encode()), "tests": len(inputs)})
        random.seed(0)
        np.random.seed(0)
        upstream.unsafe_execute(job["suite"], problem["entry_point"], code, inputs,
                                expected, time_limits, problem["atol"], True,
                                state, details, progress)
        status = upstream._mapping[state.value] or upstream.TIMEOUT
        if status == upstream.PASS and (progress.value != len(inputs) or not all(details)):
            status = upstream.FAIL
        usage = resource.getrusage(resource.RUSAGE_SELF)
        emit({"event": "result", "status": status, "tests": len(inputs),
              "tests_completed": progress.value, "solution_sha256": sha(code.encode()),
              "raw_sha256": sha(raw.encode()), "reference_time_sum": sum(reference_time),
              "cpu_seconds": usage.ru_utime + usage.ru_stime, "maxrss": usage.ru_maxrss,
              "sanitized_solution": code})
    except BaseException as exc:
        emit({"event": "result", "status": "infrastructure_error", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def run_worker(source, work, job, *, cpu=CPU_SECONDS, rss_limit=MEMORY):
    work.mkdir(parents=True, exist_ok=True)
    canary = work / "isolation-canary.txt"
    if not canary.exists():
        canary.write_text("Non-secret evaluator isolation probe.\n")
    with tempfile.TemporaryDirectory(prefix="phase-", dir=work) as temp:
        scratch = Path(temp).resolve()
        cwd = scratch / "cwd"
        cwd.mkdir()
        job = {**job, "source": str(source), "cwd": str(cwd), "canary": str(canary.resolve()),
               "cpu_limit": cpu}
        job_path = scratch / "job.json"
        job_path.write_text(json.dumps(job))
        sb = scratch / "sandbox.sb"
        sb.write_text(profile(source, scratch))
        stderr_path = scratch / "stderr.txt"
        with stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(["/usr/bin/sandbox-exec", "-f", str(sb), sys.executable,
                                     "-s", "-P", "-B", str(Path(__file__).resolve()), "--_worker", str(job_path)],
                                    cwd=cwd, env=child_env(scratch), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=stderr, close_fds=True,
                                    start_new_session=True)
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            buffer, received, events = b"", 0, []
            deadline = time.monotonic() + 180
            candidate_started, reason = False, None
            monitored = psutil.Process(proc.pid)
            peak_rss = 0
            while selector.get_map():
                try:
                    rss = monitored.memory_info().rss
                except psutil.NoSuchProcess:
                    rss = 0
                peak_rss = max(peak_rss, rss)
                if rss > rss_limit:
                    reason = "rss_watchdog_limit"
                    os.killpg(proc.pid, signal.SIGKILL)
                    break
                if time.monotonic() > deadline:
                    reason = "candidate_timeout" if candidate_started else "reference_or_setup_timeout"
                    os.killpg(proc.pid, signal.SIGKILL)
                    break
                for key, _ in selector.select(0.02):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    received += len(data)
                    if received > OUTPUT_LIMIT:
                        reason = "output_limit"
                        os.killpg(proc.pid, signal.SIGKILL)
                        break
                    buffer += data
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if isinstance(event, dict):
                            events.append(event)
                            if event.get("event") == "candidate_start" and not candidate_started:
                                candidate_started = True
                                deadline = time.monotonic() + min(62, float(event["timeout"]))
                if reason:
                    break
            selector.close()
            proc.stdout.close()
            proc.wait(timeout=5)
        results = [event for event in events if event.get("event") == "result"]
        if results and reason is None and proc.returncode == 0:
            return {**results[-1], "returncode": proc.returncode, "sampled_peak_rss": peak_rss}
        status = "timeout" if candidate_started else "infrastructure_error"
        return {"status": status, "reason": reason or "worker_exit", "returncode": proc.returncode,
                "sampled_peak_rss": peak_rss,
                "events": events, "stderr": stderr_path.read_bytes()[:16000].decode(errors="replace")}


def samples(path, tasks):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row["task_id"]
        if task_id not in tasks:
            continue  # other declared evaluation suites have a different grader
        if task_id in result:
            raise ValueError(f"More than one pass@1 sample for {task_id}")
        content = row.get("raw_completion", row.get("text"))
        if not isinstance(content, str):
            raise ValueError(f"Missing raw_completion/text for {task_id}")
        result[task_id] = content
    if set(result) != set(tasks):
        raise ValueError(f"Missing coding samples: {sorted(set(tasks) - set(result))}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_worker", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--datasets", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--canonical", action="store_true", help="Validate every canonical answer inside the same sandbox")
    parser.add_argument("--controls", action="store_true", help="Run predefined negative controls on two fixed tasks")
    parser.add_argument("--task", action="append", help="Declared debugging subset; cannot be reported as full benchmark")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2,
                        help="At most two 4 GiB RSS-monitored sandbox workers (8 GiB combined threshold)")
    args = parser.parse_args()
    if args._worker:
        worker(args._worker)
        return
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
        raise SystemExit("This adapter requires macOS seatbelt; never fall back to unsandboxed execution")
    if not args.source or not args.work:
        parser.error("--source and --work are required")
    source, work = args.source.resolve(), args.work.resolve()
    verify_source(source)
    probe = run_worker(source, work, {"mode": "probe"})
    if probe["status"] != "probe_pass":
        raise SystemExit(json.dumps(probe, indent=2))
    if args.probe:
        repeat = [run_worker(source, work, {"mode": "determinism_probe"}) for _ in range(2)]
        fingerprints = [{k: r.get(k) for k in ("status", "hash", "random_values", "numpy_values",
                                               "ignore_environment", "no_user_site", "safe_path")} for r in repeat]
        if fingerprints[0] != fingerprints[1] or fingerprints[0]["status"] != "determinism_probe_pass":
            raise SystemExit("Determinism startup probe failed: " + json.dumps(repeat))
        cpu = run_worker(source, work, {"mode": "cpu_probe"}, cpu=1)
        if cpu["returncode"] not in (-signal.SIGKILL, -signal.SIGXCPU) or cpu.get("reason") == "candidate_timeout":
            raise SystemExit("CPU resource-limit probe failed: " + json.dumps(cpu))
        memory = run_worker(source, work, {"mode": "memory_probe"}, rss_limit=64 * 1024**2)
        if memory.get("reason") != "rss_watchdog_limit":
            raise SystemExit("RSS watchdog probe failed: " + json.dumps(memory))
        wall = run_worker(source, work, {"mode": "wall_probe"})
        output = run_worker(source, work, {"mode": "output_probe"})
        if wall.get("reason") != "candidate_timeout" or output.get("reason") != "output_limit":
            raise SystemExit("Wall/output watchdog probe failed: " + json.dumps([wall, output]))
        print(json.dumps({"status": "passed", "seatbelt": probe, "cpu_limit": cpu,
                          "memory_watchdog": memory, "wall_watchdog": wall, "output_watchdog": output,
                          "determinism_repeat": repeat, "protocol_id": PROTOCOL_ID}, indent=2))
        return
    if not args.datasets or not args.output:
        parser.error("--datasets and --output are required")
    tasks = protocol.load_tasks(args.datasets)
    if args.controls:
        if args.canonical or args.samples or args.task:
            parser.error("--controls cannot be combined with other sample modes")
        records = []
        for task_id in ("HumanEval/0", "Mbpp/2"):
            suite, problem = tasks[task_id]
            entry = problem["entry_point"]
            controls = {"wrong_answer": f"def {entry}(*args):\n    return None\n",
                        "malformed": "```python\ndef broken(:\n```",
                        "infinite_loop": f"def {entry}(*args):\n    while True:\n        pass\n"}
            for name, answer in controls.items():
                for phase in ("base", "plus"):
                    result = run_worker(source, work, {"mode": "grade", "suite": suite,
                                        "phase": phase, "problem": problem, "raw_completion": answer})
                    records.append({"task_id": task_id, "control": name, "phase": phase, "outcome": result})
                    if result["status"] not in ("fail", "timeout"):
                        raise SystemExit("Negative control failed: " + json.dumps(records[-1]))
        with args.output.open("x") as output:
            json.dump({"status": "passed", "controls": records}, output, indent=2)
        print("All 12 negative control phases rejected as expected.")
        return
    if args.task:
        tasks = {key: tasks[key] for key in args.task}
    if args.canonical:
        answers = {key: task["prompt"] + task["canonical_solution"] for key, (_, task) in tasks.items()}
    elif args.samples:
        answers = samples(args.samples, tasks)
    else:
        parser.error("Use --samples or --canonical")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol_id": PROTOCOL_ID, "worker_python_flags": ["-s", "-P", "-B"],
                "seed_policy": "PYTHONHASHSEED=0; random.seed(0) and numpy.random.seed(0) before reference and candidate; independent user RNGs/entropy are not overridden",
                "evalplus_commit": protocol.EVALPLUS_COMMIT, "datasets": protocol.DATASETS,
                "adapter_sha256": sha(Path(__file__).read_bytes()),
                "export_helper_sha256": sha((HERE / "flash1017_evalplus_export.py").read_bytes()),
                "samples_sha256": sha(args.samples.read_bytes()) if args.samples else None,
                "canonical_validation": args.canonical, "tasks": list(tasks),
                "python": sys.version, "platform": sys.platform,
                "min_time_limit_seconds": MIN_TIME, "reference_time_factor": TIME_FACTOR,
                "phase_wall_budget": "min(60, sum(per-test limits)) + 2 seconds",
                "memory": "4 GiB monitored RSS threshold, 20 ms poll; not a hard allocation cap",
                "cpu_hard_limit_seconds": CPU_SECONDS, "output_limit_bytes": OUTPUT_LIMIT,
                "workers": args.jobs, "combined_rss_threshold_bytes": args.jobs * MEMORY,
                "sandbox_probe": probe}
    with args.output.with_suffix(args.output.suffix + ".metadata.json").open("x") as meta:
        json.dump(metadata, meta, indent=2)
    def grade(item):
        task_id, (suite, problem) = item
        record = {"task_id": task_id, "suite": suite, "canonical_validation": args.canonical}
        for phase in ("base", "plus"):
            outcome = run_worker(source, work, {"mode": "grade", "suite": suite, "phase": phase,
                                 "problem": problem, "raw_completion": answers[task_id]})
            record[phase] = outcome
            if outcome["status"] == "infrastructure_error":
                return record
        record["base_pass"] = record["base"]["status"] == "pass"
        record["plus_pass"] = record["base_pass"] and record["plus"]["status"] == "pass"
        return record
    with args.output.open("x") as output, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for record in pool.map(grade, tasks.items()):
            output.write(json.dumps(record) + "\n"); output.flush()
            if any(record.get(phase, {}).get("status") == "infrastructure_error" for phase in ("base", "plus")):
                pool.shutdown(wait=True, cancel_futures=True)
                raise SystemExit(f"Infrastructure error: {record}")
            print(f"{record['task_id']}: base={record['base']['status']} plus={record['plus']['status']}", flush=True)


if __name__ == "__main__":
    main()
