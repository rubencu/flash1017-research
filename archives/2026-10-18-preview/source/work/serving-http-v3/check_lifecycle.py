"""Exercise native unload/reload and inspect candidate ownership cleanup."""
import argparse
import json
from pathlib import Path
import time
from common import Client,MODEL,content,delta,envelope,write


def released(telemetry,previous_unloads):
    if (telemetry.get("loaded_bindings") != 0 or telemetry.get("models")
            or telemetry.get("collected_without_native_unload") != 0
            or telemetry.get("failed_unloads") != 0
            or len(telemetry.get("unloads",[])) != previous_unloads + 1):
        return False
    last=telemetry["unloads"][-1]
    final=last.get("final",{})
    rows,cone=final.get("rows",{}),final.get("cone",{})
    return (last.get("status")=="unloaded" and last.get("owner_closed") is True
        and all(last.get(key) is True for key in
            ("native_close_returned","hook_restore_returned","candidate_close_returned"))
        and rows.get("status")=="closed" and rows.get("retained_bytes")==0
        and not rows.get("live_tables") and rows.get("enabled") is False
        and len(rows.get("retired_tables",[]))==2
        and all(t.get("closed") is True and t.get("nbytes")==0 for t in rows["retired_tables"])
        and cone.get("closed") is True and cone.get("owned_instances")==0
        and cone.get("model_references_released") is True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    client=Client(args.profile,"candidate")
    report=envelope(client,"native-unload-and-reload")
    before=client.telemetry()
    report["before_unload"]=before
    if before["models"][0].get("retained_row_cache_bytes_observed",0)<=0:
        raise RuntimeError("Run serving requests first so release tests a populated row cache")
    write(args.output,report)
    try:
        start=time.monotonic()
        with client.call("/v1/models/"+MODEL+"/unload",{},timeout=120) as response:
            report["unload_response"]={"http_status":response.status,"body":json.load(response)}
        # Native route awaits scheduler shutdown and resource release for an
        # idle model. Any queued unload must settle before another load starts.
        deadline=time.monotonic()+120
        while True:
            status=client.status()
            after=client.get("/api/flash1017/candidate")
            report["after_unload"]={"status":status,"telemetry":after,
                                    "elapsed_seconds":time.monotonic()-start}
            write(args.output,report)
            if not status["loaded_models"] and not status["active_requests"] and not status["waiting_requests"]:
                break
            if time.monotonic()>deadline:raise TimeoutError("Native unload did not finish")
            time.sleep(.2)
        if not released(after,len(before["unloads"])):
            raise RuntimeError("Native unload failed complete candidate restoration/release checks")
        report["tests"].append({"test":"native-unload-release","passed":True})
        write(args.output,report)
        # Only a proven successful release permits loading the next instance.
        start=time.monotonic()
        with client.call("/v1/models/"+MODEL+"/load",{},timeout=300) as response:
            report["reload_response"]={"http_status":response.status,"body":json.load(response)}
        report["reload_seconds"]=time.monotonic()-start
        client.idle()
        fresh=client.telemetry()
        report["after_reload"]=fresh
        model=fresh["models"][0]
        if (client.status()["loaded_models"] != [MODEL] or model.get("status")!="loaded"
                or model.get("enabled") is not True or model.get("row_status")!="active"
                or model.get("retained_row_cache_bytes_observed")!=0
                or any(model.get(key) for key in ("cone_counts","shape_counts","prime_counts","call_counts"))
                or fresh["failed_unloads"] or fresh["collected_without_native_unload"]):
            raise RuntimeError("Reloaded candidate ownership is not healthy")
        response=client.chat([{"role":"user","content":"Reply exactly: CANDIDATE_V3_RELOADED"}],max_tokens=40)
        report["recovery"]=response
        passed=content(response["response"])=="CANDIDATE_V3_RELOADED"
        report["tests"].append({"test":"reload-generation","passed":passed})
        if not passed:raise RuntimeError("Reloaded candidate failed exact recovery")
        client.verify()
        report["status"]="complete_validated"
    except BaseException as exc:
        report.update(status="failed",exception=repr(exc))
        raise
    finally:
        report["completed_at_unix"]=time.time()
        write(args.output,report)


if __name__=="__main__":main()
