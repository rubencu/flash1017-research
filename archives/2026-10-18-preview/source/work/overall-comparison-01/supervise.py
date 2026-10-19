"""Run frozen arms sequentially, restoring the original app's idle state."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "outputs/flash1017-lab"
APP = Path("/Applications/oMLX.app/Contents/Resources")
PYTHON = APP / "Python/cpython-3.11/bin/python3.11"
SITE = APP / "Python/framework-mlx-base/lib/python3.11/site-packages"
OUT = LAB / "results/overall-comparison-01"
FIELDS = ("memory_guard_tier", "memory_guard_custom_ceiling_gb")


def save(path, value):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def call(path, payload=None):
    request = urllib.request.Request("http://127.0.0.1:8000" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def memory_fields():
    result = call("/admin/api/global-settings")
    memory = result.get("settings", result)["memory"]
    return {field: memory[field] for field in FIELDS}


def env_for(arm):
    env = os.environ.copy()
    paths = [str(LAB), str(LAB / "_runtime"), str(APP), str(SITE)] if arm == "complete" else [str(APP), str(LAB), str(SITE)]
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    return env


def stop_owned_child(child, report, report_path):
    """Bounded shutdown of only the benchmark child created by this supervisor."""
    if child.poll() is not None:
        return
    steps = report.setdefault("forced_shutdown", [])
    for sig, grace in ((signal.SIGINT, 20), (signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        if child.poll() is not None:
            return
        steps.append({"pid": child.pid, "signal": int(sig), "grace_seconds": grace, "at_unix": time.time()})
        save(report_path, report)
        child.send_signal(sig)
        try:
            child.wait(timeout=grace)
            steps[-1]["exit_code"] = child.returncode
            save(report_path, report)
            return
        except subprocess.TimeoutExpired:
            steps[-1]["grace_expired"] = True
    raise RuntimeError("Benchmark child exit unconfirmed; refusing another model load")


def main():
    plan_path, manifest_path = HERE / "plan-01.json", HERE / "manifest-01.json"
    plan = json.loads(plan_path.read_text())
    if OUT.exists():
        raise FileExistsError("Preserve prior attempt: " + str(OUT))
    OUT.mkdir(parents=True)
    report_path = OUT / "supervisor.json"
    report = {"status": "preflight", "started_at_unix": time.time(), "processes": [],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    save(report_path, report)
    child = None
    reserved = False
    exit_code = 1
    with (HERE / "supervisor.lock").open("a+") as owned:
        fcntl.flock(owned, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", 8001)) == 0:
                    raise RuntimeError("Candidate server is running; explicit owned stop required first")
            status = call("/api/status")
            if status.get("active_requests") or status.get("waiting_requests"):
                raise RuntimeError("Original app is busy; not changing it")
            models = status.get("loaded_models", [])
            if models != ["Qwen3.8-Flash-Next-oQ8e-mtp"]:
                raise RuntimeError("Original model state differs from reviewed preflight")
            report["prior_state"] = {"loaded_models": models, "memory": memory_fields(), "candidate_server_running": False}
            report["temporary_memory"] = {"memory_guard_tier": "custom", "memory_guard_custom_ceiling_gb": 32}
            save(report_path, report)
            reserved = True
            call("/admin/api/global-settings", report["temporary_memory"])
            if memory_fields() != report["temporary_memory"]:
                raise RuntimeError("Memory reservation failed")
            status = call("/api/status")
            if status.get("active_requests") or status.get("waiting_requests"):
                raise RuntimeError("App became busy; no model unloaded")
            for model in models:
                call("/admin/api/models/" + urllib.parse.quote(model, safe="") + "/unload", {})
            if call("/api/status").get("loaded_models"):
                raise RuntimeError("Original app still has a loaded model")
            import psutil
            if psutil.virtual_memory().available < 210*2**30:
                raise MemoryError("Memory headroom insufficient after native unload")
            report.update(status="running", model_execution_started_at=time.time())
            save(report_path, report)
            deadline = time.monotonic() + plan["model_execution_wall_limit_seconds"]
            for index, process in enumerate(plan["processes"]):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Frozen overall execution time limit reached")
                name = f"{index:02d}-{process['arm']}-block{process['block']}"
                output, log = OUT / (name + ".jsonl"), OUT / (name + ".log")
                command = [str(PYTHON), "-B", "-u", str(HERE / "worker.py"), "--plan", str(plan_path),
                    "--manifest", str(manifest_path), "--process-index", str(index), "--output", str(output)]
                item = {"index": index, "arm": process["arm"], "block": process["block"],
                    "output": str(output), "log": str(log), "started_at_unix": time.time()}
                report["processes"].append(item)
                report["current_process"] = index
                with log.open("x") as stream:
                    child = subprocess.Popen(command, env=env_for(process["arm"]), cwd=ROOT,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    item["pid"] = child.pid
                    save(report_path, report)
                    print(json.dumps({"event": "process_started", **item}), flush=True)
                    while child.poll() is None:
                        if time.monotonic() >= deadline:
                            stop_owned_child(child, report, report_path)
                            raise TimeoutError("Frozen overall execution time limit reached")
                        time.sleep(1)
                    item.update(exit_code=child.returncode, completed_at_unix=time.time())
                    child = None
                save(report_path, report)
                print(json.dumps({"event": "process_completed", **item}), flush=True)
                if item["exit_code"]:
                    raise RuntimeError("Benchmark process failed; preserve partial results and restore app")
            exit_code = 0
            report["benchmark_status"] = "all_processes_completed"
        except BaseException as exc:
            report.update(benchmark_status="incomplete", error=repr(exc))
            raise
        finally:
            child_exit_confirmed = True
            if child is not None and child.poll() is None:
                try:
                    stop_owned_child(child, report, report_path)
                except BaseException as exc:
                    child_exit_confirmed = False
                    report["child_shutdown_error"] = repr(exc)
            report["status"] = "restoring_original_state"
            save(report_path, report)
            try:
                if not child_exit_confirmed:
                    raise RuntimeError("Restoration blocked because benchmark child exit is unconfirmed")
                if reserved:
                    original = report["prior_state"]["memory"]
                    current = memory_fields()
                    if current not in (original, report["temporary_memory"]):
                        raise RuntimeError("App settings changed independently; not overwriting")
                    if current != original:
                        call("/admin/api/global-settings", original)
                    if memory_fields() != original:
                        raise RuntimeError("Original settings not restored")
                    for model in report["prior_state"]["loaded_models"]:
                        status = call("/api/status")
                        if status.get("active_requests") or status.get("waiting_requests"):
                            raise RuntimeError("App became busy during restoration; no competing load")
                        call("/admin/api/models/" + urllib.parse.quote(model, safe="") + "/load", {})
                    final = call("/api/status")
                    report["restored_state"] = {"loaded_models": final.get("loaded_models"), "memory": memory_fields()}
                    if final.get("loaded_models") != report["prior_state"]["loaded_models"]:
                        raise RuntimeError("Original loaded model state not restored")
                report["restoration_status"] = "restored" if reserved else "unchanged"
            except BaseException as exc:
                report["restoration_status"] = "failed"
                report["restoration_error"] = repr(exc)
                exit_code = 1
            report.update(status="complete" if exit_code == 0 else "incomplete_restoration_recorded", completed_at_unix=time.time())
            save(report_path, report)
            print(json.dumps({"event": "finished", "status": report["status"], "restoration_status": report["restoration_status"]}), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
