"""Start a specific reviewed candidate and verify its identity before opening chat."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"outputs/flash1017-lab"
SERVER=ROOT/"work/serving-server-v3/serve_candidate.py"
STAMP=LAB/"server-state/server-process.json"
MODEL="DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import psutil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--log",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--no-browser",action="store_true")
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError("Use a new launch report")
    args.profile=args.profile.resolve()
    args.manifest=args.manifest.resolve()
    profile=json.loads(args.profile.read_text())
    namespace=hashlib.sha256(json.dumps(profile["runtime"],sort_keys=True).encode()).hexdigest()[:12]
    report={"status":"validating","profile":str(args.profile),"profile_sha256":sha(args.profile),
        "manifest":str(args.manifest),"manifest_sha256":sha(args.manifest),"namespace":namespace,
        "launcher_sha256":sha(__file__),"started_at_unix":time.time()}
    def save():
        args.report.write_text(json.dumps(report,indent=2)+"\n")
    save()
    child=None
    # This is the frozen server's stdlib-only gate; it imports no inference
    # runtime, acquires no model and checks the entire supplied source closure.
    validation=subprocess.run([str(LAB/"run.sh"),str(SERVER),"--profile",str(args.profile),
        "--manifest",str(args.manifest),"--validate-only"],cwd=ROOT,capture_output=True,text=True)
    report["validation_exit_code"]=validation.returncode
    if validation.returncode:
        report.update(status="refused",validation_error=validation.stderr[-4000:]);save()
        raise RuntimeError("Candidate source/profile validation failed; see launch report")
    def verify_profile_files():
        if sha(args.profile)!=report["profile_sha256"] or sha(args.manifest)!=report["manifest_sha256"]:
            raise RuntimeError("Profile/manifest changed during startup")
        for relative,digest in profile["inference_files_sha256"].items():
            if sha(LAB/relative)!=digest:
                raise RuntimeError("Pinned inference source changed: "+relative)
    def owned():
        try:data=json.loads(STAMP.read_text())
        except (FileNotFoundError,json.JSONDecodeError):return None
        try:
            process=psutil.Process(data["pid"])
            created=process.create_time()
        except psutil.NoSuchProcess:return None
        if created!=data["created"]:return None
        if data["port"]!=8001 or Path(data["profile"]).resolve()!=args.profile:
            raise RuntimeError("Another lab profile is running; stop it before starting this candidate")
        return data
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):raise RuntimeError("Loopback readiness redirect refused")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def get(path):
        with opener.open("http://127.0.0.1:8001"+path,timeout=2) as response:return json.load(response)
    try:
        verify_profile_files()
        existing=owned()
        if existing is None:
            if args.log.exists():raise FileExistsError("Use a new candidate server log")
            with socket.socket() as check:
                check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                check.bind(("127.0.0.1",8001))
            with args.log.open("x") as log:
                child=subprocess.Popen([str(LAB/"run.sh"),"-u",str(SERVER),"--profile",str(args.profile),
                    "--manifest",str(args.manifest),"--port","8001"],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            report.update(status="loading",child_pid=child.pid);save()
        deadline=time.monotonic()+300
        while True:
            if child is not None and child.poll() is not None:
                raise RuntimeError("Candidate exited during startup; inspect its saved log")
            current=owned()
            ready=False
            if current is not None:
                if child is not None and current["pid"]!=child.pid:
                    raise RuntimeError("Startup stamp belongs to a different process than the launched candidate")
                try:
                    health,status,telemetry=get("/health"),get("/api/status"),get("/api/flash1017/candidate")
                    models=telemetry.get("models",[])
                    model=models[0] if len(models)==1 else None
                    ready=((health.get("engine_pool") or {}).get("loaded_count")==1
                        and status.get("loaded_models")==[MODEL] and telemetry.get("namespace")==namespace
                        and telemetry.get("manifest_sha256")==report["manifest_sha256"]
                        and telemetry.get("candidate")==profile["runtime"]["serving_candidate"]["name"]
                        and telemetry.get("loaded_bindings")==1 and model is not None
                        and model.get("status")=="loaded" and model.get("enabled") is True
                        and model.get("row_status")=="active"
                        and telemetry.get("failed_unloads")==0 and telemetry.get("collected_without_native_unload")==0)
                except (OSError,ValueError):pass
            if time.monotonic()>deadline:raise TimeoutError("Candidate not ready within300seconds; no second model was started")
            if ready:
                verify_profile_files()
                if owned()!=current:
                    raise RuntimeError("Server stamp/process changed during readiness checks")
                if time.monotonic()>deadline:raise TimeoutError("Readiness verification exceeded300seconds")
                report.update(status="ready",server=current,telemetry=telemetry);break
            time.sleep(1)
        report["completed_at_unix"]=time.time();save()
        print("Ready: http://127.0.0.1:8001/admin/chat",flush=True)
        if not args.no_browser:webbrowser.open("http://127.0.0.1:8001/admin/chat")
    except BaseException as exc:
        report.update(status="failed",error=repr(exc),completed_at_unix=time.time());save()
        # A timed-out child may still own weights/work. Leave it observable;
        # callers must use the ordinary drained stop path before any restore.
        raise


if __name__=="__main__":main()
