"""Run the frozen performance suite, then restore the qualified chat server.

No inference imports. The child runner owns the shared model lock; this process
waits for its exit before starting the default server. A failed benchmark also
restores chat availability. Original-app settings are restored only after the
lab model is loaded, closing the second-model reload window.
"""
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LAB=ROOT/"outputs/flash1017-lab"


def main():
    report_path=HERE/"performance-supervisor-01.json"
    if report_path.exists():
        raise FileExistsError("Preserve prior supervisor evidence")
    boundary=json.loads((LAB/"results/serving-v3-boundary-01.qualification.json").read_text())
    if (boundary.get("status") != "complete_validated" or len(boundary.get("pairs",[])) != 13
            or not all(row["passed"] for row in boundary["pairs"])
            or not boundary.get("instance_hooks_restored") or not boundary.get("model_cleanup_completed")):
        raise RuntimeError("Boundary qualification has not completed successfully")
    report={"status":"running", "started_at_unix":time.time()}
    def save():
        report_path.write_text(json.dumps(report,indent=2)+"\n")
    save()
    result=1
    try:
        with (HERE/"performance-01.log").open("x") as log:
            child=subprocess.Popen([str(LAB/"run.sh"),"-u",str(HERE/"run_qualification.py"),
                "--mode","performance","--plan",str(ROOT/"work/serving-qualification-v3/plans/performance-01.json"),
                "--manifest",str(HERE/"execution-manifest-01.json"),
                "--output",str(LAB/"results/serving-v3-performance-01.jsonl")],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            report["benchmark_pid"]=child.pid
            save()
            while True:
                try:
                    result=child.wait()
                    break
                except KeyboardInterrupt:
                    # Ask the runner to drain its admitted worker before
                    # cleanup; never close/unload a still-running model here.
                    import signal
                    child.send_signal(signal.SIGINT)
            report["benchmark_exit_code"]=result
    finally:
        report["status"]="restoring_chat"
        save()
        with (HERE/"restore-chat-01.log").open("x") as log:
            launch=subprocess.run([str(LAB/"run.sh"),"-u","-m","flash1017.launch","--no-browser"],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            report["chat_launch_exit_code"]=launch.returncode
            if launch.returncode == 0:
                restore=subprocess.run([str(LAB/"run.sh"),str(ROOT/"work/serving-integration-v3/reserve_memory.py"),"restore"],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                report["original_memory_restore_exit_code"]=restore.returncode
        report["status"]=("completed_chat_restored" if report.get("chat_launch_exit_code") == 0
            and report.get("original_memory_restore_exit_code") == 0 else "restoration_failed")
        report["completed_at_unix"]=time.time()
        save()
    raise SystemExit(result if report["status"] == "completed_chat_restored" else 1)


if __name__ == "__main__":
    main()
