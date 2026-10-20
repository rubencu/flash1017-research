"""Run reviewed installed-wheel timing only after complete quality approval.

Only the benchmark child we create may be signaled: SIGINT/90s, TERM/30s,
then KILL/30s. The original server process is never stopped. Signals received
by this supervisor are remembered; they never interrupt handle assignment or
cleanup. All model restoration is conditional on idle/uncontested resources.
"""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUALITY = ROOT / "work/flash1017-quality-02"
state = {
    "status": "waiting-for-quality",
    "processes": [],
    "original_models": [],
    "publication": False,
    "cleanup_errors": [],
}
child = None
child_record = None
original = None
interrupt_requested = None
reviewed = None
reviewed_hash = None
plan_hash = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save():
    tmp = HERE / "state.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(HERE / "state.json")


def cleanup_save():
    try:
        save()
    except OSError as exc:
        state["cleanup_errors"].append("State write: " + repr(exc))


def interrupted(signum, frame):
    global interrupt_requested
    if interrupt_requested is None:
        interrupt_requested = signum


def check_interrupt():
    if interrupt_requested is not None:
        raise KeyboardInterrupt(f"Signal {interrupt_requested}")


def record_exit():
    if child is not None and child_record is not None:
        child_record["exit_code"] = child.poll()
        if child_record["exit_code"] is not None:
            child_record["exited_unix"] = time.time()
        cleanup_save()


def stop_child():
    global child
    if child is None:
        return
    for signum, grace in [
        (signal.SIGINT, 90),
        (signal.SIGTERM, 30),
        (signal.SIGKILL, 30),
    ]:
        if child.poll() is not None:
            break
        child_record.setdefault("shutdown_signals", []).append(
            {"signal": int(signum), "grace_seconds": grace, "sent_unix": time.time()}
        )
        child.send_signal(signum)
        cleanup_save()
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    record_exit()
    if child.poll() is None:
        raise RuntimeError("Owned benchmark child exit remains uncertain")
    child = None


def request(method, path, deadline, **kwargs):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Original-server operation deadline expired")
    response = original.request(
        method, path, timeout=min(remaining, 180 if method == "POST" else 15), **kwargs
    )
    response.raise_for_status()
    if time.monotonic() > deadline:
        raise TimeoutError("Original-server response arrived after its wall deadline")
    return response


def models(deadline):
    return request("GET", "/admin/api/models", deadline).json()["models"]


def idle(deadline):
    activity = request("GET", "/admin/api/activity", deadline).json()["active_models"]
    return (
        activity["total_active_requests"] == 0
        and activity["total_waiting_requests"] == 0
    )


def wait_headroom(minimum, deadline, interruptible):
    while time.monotonic() < deadline:
        if interruptible:
            check_interrupt()
        if psutil.virtual_memory().available >= minimum:
            return
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise MemoryError(
        f"Available memory did not recover to {minimum / 2**30:g} GiB before deadline"
    )


def verify_reviewed():
    if digest(HERE / "REVIEWED-CONFIG.json") != reviewed_hash:
        raise RuntimeError("Reviewed configuration changed")
    for path, expected in reviewed["files"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError("Reviewed source or artifact changed: " + path)
    if plan_hash is not None and digest(HERE / "protocol.json") != plan_hash:
        raise RuntimeError("Frozen performance protocol changed")
    if (
        "quality_report_sha256" in state
        and digest(QUALITY / "pair-summary-final-02.json")
        != state["quality_report_sha256"]
    ):
        raise RuntimeError("Approved quality report changed")


def restore_original_models():
    state["restoration"] = []
    if original is None:
        state["restoration_complete"] = not state["original_models"]
        return
    for model in state["original_models"]:
        outcome = {"model": model}
        state["restoration"].append(outcome)
        try:
            if child is not None and child.poll() is None:
                raise RuntimeError(
                    "Owned benchmark child may still be running; no model load"
                )
            deadline = time.monotonic() + 360
            entries = models(deadline)
            entry = next(m for m in entries if m["id"] == model)
            if not entry["loaded"]:
                wait_headroom(180 * 2**30, min(deadline, time.monotonic() + 120), False)
                if not idle(deadline):
                    raise RuntimeError(
                        "Original server is active; model restoration deferred"
                    )
                unexpected = [
                    m["id"]
                    for m in models(deadline)
                    if (m["loaded"] or m.get("is_loading"))
                    and m["id"] not in state["original_models"]
                ]
                if unexpected:
                    raise RuntimeError(
                        "New user-loaded models are present; restoration deferred"
                    )
                request("POST", f"/admin/api/models/{model}/load", deadline)
            while time.monotonic() < deadline:
                entry = next(m for m in models(deadline) if m["id"] == model)
                if entry["loaded"] and not entry["is_loading"]:
                    break
                time.sleep(min(1, max(0, deadline - time.monotonic())))
            outcome.update(loaded=entry["loaded"], is_loading=entry["is_loading"])
            if not outcome["loaded"] or outcome["is_loading"]:
                outcome["error"] = (
                    "Restoration wall deadline expired before model was ready"
                )
        except Exception as exc:
            outcome["error"] = repr(exc)
        cleanup_save()
    state["restoration_complete"] = len(state["restoration"]) == len(
        state["original_models"]
    ) and all(
        r.get("loaded") is True and r.get("is_loading") is False and "error" not in r
        for r in state["restoration"]
    )


def main():
    global child, child_record, original, reviewed, reviewed_hash, plan_hash
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if (HERE / "state.json").exists():
        raise RuntimeError("Refusing to overwrite an existing performance attempt")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        state["supervisor_sha256"] = digest(Path(__file__))
        state["shutdown_policy"] = (
            "Owned child only: SIGINT/90s, SIGTERM/30s, SIGKILL/30s; original server never stopped"
        )
        reviewed_hash = digest(HERE / "REVIEWED-CONFIG.json")
        reviewed = read(HERE / "REVIEWED-CONFIG.json")
        state["reviewed_config_sha256"] = reviewed_hash
        verify_reviewed()
        save()
        deadline = time.monotonic() + 6 * 3600
        while time.monotonic() < deadline:
            check_interrupt()
            verify_reviewed()
            final = (
                read(QUALITY / "finalization-state-02.json")
                if (QUALITY / "finalization-state-02.json").exists()
                else {}
            )
            if final.get("status") in ("failed-quality-gate", "failed-operationally"):
                state["status"] = "blocked-by-quality"
                save()
                return 1
            grading = (
                ROOT
                / "work/flash1017-quality-01/evalplus-deterministic-02/candidate-wait-status.json"
            )
            if grading.exists() and read(grading).get("status") == "failed":
                state["status"] = "blocked-by-grading"
                save()
                return 1
            if (
                final.get("status") == "passed"
                and (HERE / "REVIEWED-CONFIG.json").exists()
            ):
                break
            time.sleep(10)
        else:
            raise TimeoutError(
                "Quality/review prerequisites did not finish in six hours"
            )
        quality_report = QUALITY / "pair-summary-final-02.json"
        state["quality_report_sha256"] = final["report_sha256"]
        verify_reviewed()
        report = read(quality_report)
        if not (
            report["planned_acceptance"]["status"] == "passed"
            and report["integrity"]["status"] == "verified"
            and report["graded_pairs"] == 618
            and report["paired_responses"] == 618
            and report["observed_exact_response_parity_established"] is True
            and report["observed_task_score_preservation_established"] is True
        ):
            raise RuntimeError("Strict quality qualification has not passed")
        plan = read(ROOT / reviewed["draft_plan_path"])
        plan.update(reviewed["final_fields"])
        plan.update(
            schema="installed-http-performance-v1",
            status="frozen_before_any_model_run",
            quality_report_path=str(quality_report.relative_to(ROOT)),
            quality_report_sha256=digest(quality_report),
            execution_supervisor_sha256=digest(Path(__file__)),
            execution_supervisor_policy=state["shutdown_policy"],
        )
        plan.pop("quality_receipt_sha256", None)
        for arm in ("baseline", "candidate"):
            for key in ("version", "wheel_sha256"):
                if plan[arm][key] != report["builds"][arm][key]:
                    raise RuntimeError(
                        f"Quality and performance build mismatch: {arm}/{key}"
                    )
        with (HERE / "protocol.json").open("x") as output:
            json.dump(plan, output, indent=2)
            output.write("\n")
        plan_hash = digest(HERE / "protocol.json")
        state.update(
            status="preflight",
            plan_sha256=plan_hash,
            reviewed_config_sha256=reviewed_hash,
        )
        save()
        original = httpx.Client(
            base_url="http://127.0.0.1:8000", timeout=15, trust_env=False
        )
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            check_interrupt()
            if idle(deadline):
                break
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        else:
            raise RuntimeError(
                "Original server remained busy for 30 minutes; no model unloaded"
            )
        entries = models(time.monotonic() + 30)
        if any(m.get("is_loading") for m in entries):
            raise RuntimeError("Original server has a loading model; no model unloaded")
        state["original_models"] = [m["id"] for m in entries if m["loaded"]]
        save()
        for model in state["original_models"]:
            check_interrupt()
            deadline = time.monotonic() + 180
            if not idle(deadline):
                raise RuntimeError(
                    "Original server became busy; do not unload active requests"
                )
            request("POST", f"/admin/api/models/{model}/unload", deadline)
        for process in plan["processes"]:
            check_interrupt()
            verify_reviewed()
            deadline = time.monotonic() + 30
            if not idle(deadline) or any(
                m["loaded"] or m.get("is_loading") for m in models(deadline)
            ):
                raise RuntimeError(
                    "New original-server activity; abort timing without unloading it"
                )
            wait_headroom(222 * 2**30, time.monotonic() + 120, True)
            check_interrupt()
            verify_reviewed()
            deadline = time.monotonic() + 30
            if not idle(deadline) or any(
                m["loaded"] or m.get("is_loading") for m in models(deadline)
            ):
                raise RuntimeError(
                    "Original server changed during headroom wait; no child spawned"
                )
            process_id, arm = process["process_id"], process["arm"]
            state.update(status="running", current_process=process_id)
            save()
            python = ROOT / reviewed["environments"][arm] / "bin/python"
            output_dir = HERE / process_id
            cmd = [
                str(python),
                "-I",
                "-B",
                str(ROOT / plan["performance_harness_path"]),
                "--process-id",
                process_id,
                "--arm",
                arm,
                "--model",
                str(args.model),
                "--wheel",
                str(ROOT / reviewed["wheels"][arm]),
                "--plan",
                str(HERE / "protocol.json"),
                "--base-path",
                str(HERE / (process_id + "-state")),
                "--output",
                str(output_dir),
                "--port",
                "8017",
                "--workspace-root",
                str(ROOT),
            ]
            child_record = {"process_id": process_id, "status": "spawning"}
            state["processes"].append(child_record)
            with (HERE / (process_id + ".log")).open("x") as log:
                child = subprocess.Popen(
                    cmd, cwd="/tmp", stdout=log, stderr=subprocess.STDOUT
                )
                child_record.update(
                    pid=child.pid, started_unix=time.time(), status="running"
                )
                try:
                    child_record["process_create_time"] = psutil.Process(
                        child.pid
                    ).create_time()
                except psutil.NoSuchProcess:
                    child_record["process_create_time"] = None
                save()
                print("Started", process_id, child.pid, flush=True)
                deadline = time.monotonic() + plan["process_wall_timeout_seconds"]
                while child.poll() is None:
                    check_interrupt()
                    if time.monotonic() > deadline:
                        stop_child()
                        raise TimeoutError(
                            process_id + " exceeded frozen process wall deadline"
                        )
                    time.sleep(2)
                record_exit()
                code = child.returncode
                child = None
            # Process identity/exit are already durable before API/receipt checks.
            deadline = time.monotonic() + 30
            if not idle(deadline) or any(
                m["loaded"] or m.get("is_loading") for m in models(deadline)
            ):
                raise RuntimeError(
                    "Original server was used during timing; observations remain unqualified"
                )
            receipt = (
                read(output_dir / "receipt.json")
                if (output_dir / "receipt.json").exists()
                else {}
            )
            child_record.update(
                status=receipt.get("status"),
                measured_completed=receipt.get("measured_completed"),
            )
            save()
            if code or receipt.get("status") != "generated":
                raise RuntimeError(
                    "Performance process failed; preserve observations without retry"
                )
        check_interrupt()
        state["status"] = "generated"
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        print(state["error"], flush=True)
    finally:
        try:
            stop_child()
        except BaseException as exc:
            state["cleanup_errors"].append("Owned child cleanup: " + repr(exc))
        try:
            restore_original_models()
        except BaseException as exc:
            state["cleanup_errors"].append("Restoration: " + repr(exc))
            state["restoration_complete"] = False
        if original is not None:
            try:
                original.close()
            except Exception as exc:
                state["cleanup_errors"].append("Original client close: " + repr(exc))
        if state["cleanup_errors"] or not state.get("restoration_complete"):
            state["status"] = "failed-cleanup"
        if interrupt_requested is not None:
            state["status"] = "interrupted"
            state["interrupt_signal"] = interrupt_requested
        cleanup_save()
    if plan_hash is not None and interrupt_requested is None:
        try:
            verify_reviewed()
            cmd = [
                sys.executable,
                "-I",
                "-B",
                str(ROOT / reviewed["analysis_harness_path"]),
                "--plan",
                str(HERE / "protocol.json"),
                "--baseline-wheel",
                str(ROOT / reviewed["wheels"]["baseline"]),
                "--candidate-wheel",
                str(ROOT / reviewed["wheels"]["candidate"]),
                "--output",
                str(HERE / "pair-summary-final-01.json"),
            ]
            for process in plan["processes"]:
                cmd.extend(
                    [
                        "--run",
                        process["process_id"] + "=" + str(HERE / process["process_id"]),
                    ]
                )
            with (HERE / "analysis.log").open("x") as log:
                result = subprocess.run(
                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300
                )
            state["analysis_exit_code"] = result.returncode
            if (HERE / "pair-summary-final-01.json").exists():
                state["report_sha256"] = digest(HERE / "pair-summary-final-01.json")
            state["status"] = (
                "qualified"
                if (
                    result.returncode == 0
                    and state["status"] == "generated"
                    and interrupt_requested is None
                    and not state["cleanup_errors"]
                    and state.get("restoration_complete") is True
                )
                else "failed-performance-gate"
            )
        except BaseException:
            state["status"] = "failed-analysis"
            state["analysis_error"] = traceback.format_exc()
        if interrupt_requested is not None:
            state["status"] = "interrupted"
            state["interrupt_signal"] = interrupt_requested
        save()
    print(json.dumps(state), flush=True)
    return 0 if state["status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
