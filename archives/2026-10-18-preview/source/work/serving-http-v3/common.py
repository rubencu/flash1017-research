"""Loopback-only HTTP qualification helpers; never import an inference runtime."""
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"outputs/flash1017-lab"
FIX=ROOT/"work/serving-qualification-v3"
MODEL="DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        raise RuntimeError("Qualification HTTP requests must not follow redirects")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+".pending")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def content(response):
    return (response["choices"][0]["message"].get("content") or "").strip()


class Client:
    def __init__(self,profile_path,kind):
        import psutil
        self.profile_path=Path(profile_path).resolve()
        self.profile=json.loads(self.profile_path.read_text())
        if self.profile.get("qualified") is not True:
            raise RuntimeError("Full-model qualification is required before HTTP qualification")
        self.profile_hash=sha(self.profile_path)
        self.kind=kind
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        self.stamp_path=LAB/"server-state/server-process.json"
        self.stamp=json.loads(self.stamp_path.read_text())
        if (self.stamp["port"] != 8001 or Path(self.stamp["profile"]).resolve() != self.profile_path
                or psutil.Process(self.stamp["pid"]).create_time() != self.stamp["created"]):
            raise RuntimeError("Expected the owned local model/profile on port8001")
        marker=self.profile["runtime"].get("serving_candidate")
        if (kind == "candidate") != bool(marker):
            raise RuntimeError("Profile does not match requested native/candidate arm")
        self.verify()
        initial=self.status()
        if initial["loaded_models"] != [MODEL] or initial["active_requests"] or initial["waiting_requests"]:
            raise RuntimeError("Expected one loaded, idle exact1017 model")

    def verify(self):
        import psutil
        if sha(self.profile_path) != self.profile_hash or json.loads(self.stamp_path.read_text()) != self.stamp:
            raise RuntimeError("Server/profile changed during HTTP qualification")
        if psutil.Process(self.stamp["pid"]).create_time() != self.stamp["created"]:
            raise RuntimeError("Owned server PID changed during HTTP qualification")
        for relative,expected in self.profile["inference_files_sha256"].items():
            if sha(LAB/relative) != expected:
                raise RuntimeError("Qualified inference source changed: "+relative)

    def call(self,path,payload=None,timeout=600):
        parsed=urllib.parse.urlsplit(path)
        if not path.startswith("/") or parsed.scheme or parsed.netloc or parsed.fragment:
            raise ValueError("Only local absolute API paths are accepted")
        request=urllib.request.Request("http://127.0.0.1:8001"+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type":"application/json"})
        return self.opener.open(request,timeout=timeout)

    def get(self,path,timeout=10):
        with self.call(path,timeout=timeout) as response:
            return json.load(response)

    def status(self):
        raw=self.get("/api/status")
        return {key:raw[key] for key in ("active_requests","waiting_requests","loaded_models")}

    def idle(self,deadline=60):
        start=time.monotonic()
        while time.monotonic()-start < deadline:
            status=self.status()
            elapsed=time.monotonic()-start
            if elapsed>=deadline:
                break
            if not status["active_requests"] and not status["waiting_requests"]:
                return {"seconds":elapsed,"status":status,
                        "scope":"API queue idle only; not an independent GPU/executor fence."}
            time.sleep(.1)
        raise TimeoutError("Server did not report an idle queue within the declared deadline")

    def telemetry(self):
        if self.kind == "native":
            return None
        result=self.get("/api/flash1017/candidate")
        marker=self.profile["runtime"]["serving_candidate"]
        namespace=digest(self.profile["runtime"])[:12]
        if (result.get("loaded_bindings") != 1 or result.get("manifest_sha256") !=
                marker["manifest_sha256"] or result.get("candidate")!=marker["name"]
                or result.get("namespace")!=namespace):
            raise RuntimeError("Candidate telemetry does not match the pinned profile")
        models=result.get("models",[])
        if (len(models)!=1 or models[0].get("status")!="loaded" or models[0].get("enabled") is not True
                or models[0].get("row_status")!="active" or result.get("failed_unloads")!=0
                or result.get("collected_without_native_unload")!=0):
            raise RuntimeError("Candidate telemetry reports degraded ownership or lifecycle")
        for group in ("cone_counts","shape_counts","prime_counts","call_counts"):
            values=models[0].get(group)
            if not isinstance(values,dict) or any(type(v) is not int or v<0 for v in values.values()):
                raise RuntimeError("Candidate telemetry has invalid counters: "+group)
        return result

    def chat(self,messages,**extra):
        payload={"model":MODEL,"messages":messages,"temperature":0,"max_tokens":96,
                 "chat_template_kwargs":{"enable_thinking":False},**extra}
        before=self.telemetry()
        started=time.monotonic()
        with self.call("/v1/chat/completions",payload) as response:
            result=json.load(response)
        elapsed=time.monotonic()-started
        idle=self.idle()
        after=self.telemetry()
        return {"response":result,"seconds":elapsed,"request_sha256":digest(payload),
                "queue_idle":idle,"before":before,"after":after}


def counters(telemetry,group="cone_counts"):
    return telemetry["models"][0].get(group,{})


def delta(row,group="cone_counts"):
    if row["before"] is None:
        return {}
    a,b=counters(row["before"],group),counters(row["after"],group)
    return {key:b.get(key,0)-a.get(key,0) for key in a.keys()|b.keys()}


def envelope(client,case):
    return {"status":"running","case":case,"kind":client.kind,"server":client.stamp,
            "profile_sha256":client.profile_hash,"started_at_unix":time.time(),
            "timing_scope":"Functional serving qualification; not a controlled speed comparison.",
            "source_sha256":{str(p):sha(p) for p in Path(__file__).parent.glob("*.py")},"tests":[]}
