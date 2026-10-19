"""Separate opt-in launcher; delegates lock, port and lifecycle to qualified serve.

Importing this file is CPU/stdlib only. No profile is created or qualified here.
Actual startup requires an explicitly qualified candidate profile and frozen
source manifest. The baseline qualified launcher and runtime are unchanged.
"""
import argparse
from collections import deque
import copy
import hashlib
import importlib.util
import ipaddress
import json
from pathlib import Path
import sys
import threading
import time
import weakref


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "outputs/flash1017-lab"
NAME = "cone-hotrows-v3"
PROFILE_KIND = "flash1017-serving-candidate-v3"
TELEMETRY_PATH = "/api/flash1017/candidate"
_MISSING = object()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_profile(profile_path, manifest_path, *, lab=LAB):
    """Read-only startup gate. Qualification itself belongs to the root audit."""
    profile = json.loads(Path(profile_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    baseline_path = Path(lab) / "qualified-profile.json"
    baseline = json.loads(baseline_path.read_text())
    if profile.get("qualified") is not True or profile.get("profile_kind") != PROFILE_KIND:
        raise ValueError("An explicitly qualified candidate profile is required")
    marker = profile["runtime"].get("serving_candidate")
    expected = {"name": NAME, "schema": 1, "row_cache_bytes_per_table": 256*2**20,
                "manifest_sha256": sha(manifest_path)}
    if marker != expected:
        raise ValueError("Candidate profile namespace/manifest does not match")
    if {k:v for k,v in profile["runtime"].items() if k != "serving_candidate"} != baseline["runtime"]:
        raise ValueError("Candidate runtime must preserve the qualified native64 settings")
    if profile.get("packages") != baseline.get("packages"):
        raise ValueError("Candidate dependencies differ from qualified native64")
    for relative, digest in baseline["inference_files_sha256"].items():
        if profile.get("inference_files_sha256", {}).get(relative) != digest:
            raise ValueError("Candidate profile must retain all baseline inference pins")
    if manifest.get("baseline", {}).get("profile_sha256") != sha(baseline_path):
        raise ValueError("Candidate manifest was frozen against a different baseline")
    sources = manifest.get("sources_sha256", {})
    required = [Path(__file__).resolve(), ROOT / "work/serving-integration-v3/runtime.py",
                ROOT / "work/serving-integration-v3/evidence.py",
                ROOT / "work/serving-cone-v3/cone_serving.py",
                ROOT / "work/cone-experiment/cone.py", ROOT / "work/cone-v2/stabilization.py",
                ROOT / "work/cone-dspark-stable/stabilization.py",
                ROOT / "work/serving-cache-v3/serving_cache.py",
                ROOT / "work/serving-cache-v3/hot_rows_upstream.py"]
    if any(str(path) not in sources for path in required):
        raise ValueError("Server candidate/dependency source is not pinned")
    for path, digest in sources.items():
        if sha(path) != digest:
            raise ValueError("Candidate source changed: " + path)
    identity = hashlib.sha256(json.dumps(profile["runtime"],sort_keys=True).encode()).hexdigest()[:12]
    native_identity = hashlib.sha256(json.dumps(baseline["runtime"],sort_keys=True).encode()).hexdigest()[:12]
    if identity == native_identity:
        raise ValueError("Candidate must have a separate prefix-cache namespace")
    return profile, identity


def _note(primary, secondary):
    if hasattr(primary, "add_note"):
        primary.add_note("Additional candidate cleanup failure: " + repr(secondary))


def observed_cache_bytes(cache):
    """Read immutable NumPy array-size metadata without HotRows.nbytes's lock.

    Hold local references while reading metadata, never copy array contents.
    Initialization/clear can race us; report unknown instead of partial totals.
    """
    tags, freq, data = cache.tags, cache.freq, cache.data
    if tags is None and freq is None and data is None:
        return 0
    if tags is None or freq is None or data is None:
        return None
    total = int(tags.nbytes) + int(freq.nbytes) + sum(int(raw.nbytes) for raw,_ in data)
    if cache.tags is not tags or cache.freq is not freq or cache.data is not data:
        return None
    return total


class Binding:
    """One loaded candidate; attach cleanup to the existing row-owner hook.

    Native model.close already dispatches through rows._model_close. Replacing
    only that controller attribute keeps the row owner's model.close patch
    identity intact, so its own teardown can restore it successfully.
    """
    def __init__(self, registry, owner, key):
        self.registry, self.owner, self.key = registry, owner, key
        self.created = time.time()
        rows = owner.rows
        self._rows = rows
        self._previous = vars(rows).get("_model_close", _MISSING)
        self._native = rows._model_close
        self._closing = False
        self._lock = threading.Lock()
        self._hook = lambda: self._unload()
        object.__setattr__(rows, "_model_close", self._hook)

    def observe(self):
        """Never take inference locks or evaluate/copy any tensor or row array."""
        owner = self.owner
        if owner is None:
            return {"status":"unloaded"}
        try:
            cone, rows = owner.cone, owner.rows
            scopes = (getattr(cone,"cone",None),getattr(cone,"shape",None),getattr(cone,"prime",None))
            tables = tuple(getattr(rows,"_states",()))
            table_observations = [{"index":state.index,"retained_bytes":observed_cache_bytes(state.cache),
                                   "unique_queries":state.cache.queries,"unique_hits":state.cache.hits,
                                   "counts":dict(state.counts)} for state in tables]
            sizes = [item["retained_bytes"] for item in table_observations]
            return {"status":"closing" if self._closing else "loaded",
                    "enabled":bool(cone.enabled), "python_call_active_hint":cone._owner_thread is not None,
                    "cone_counts":dict(scopes[0].counts) if scopes[0] else {},
                    "shape_counts":dict(scopes[1].counts) if scopes[1] else {},
                    "prime_counts":dict(scopes[2].counts) if scopes[2] else {},
                    "call_counts":dict(cone.counts), "row_status":rows.status,
                    "retained_row_cache_bytes_observed":sum(sizes) if all(n is not None for n in sizes) else None,
                    "tables":table_observations}
        except (AttributeError, RuntimeError):
            # Teardown can race observation. Do not block or turn an optional
            # telemetry GET into an inference failure or an idle assertion.
            return {"status":"transitioning"}

    def _restore_hook(self):
        if vars(self._rows).get("_model_close",_MISSING) is not self._hook:
            raise RuntimeError("Candidate row-controller unload hook was replaced")
        if self._previous is _MISSING:
            object.__delattr__(self._rows,"_model_close")
        else:
            object.__setattr__(self._rows,"_model_close",self._previous)

    def _unload(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent/reentrant candidate unload")
        owner, native = self.owner, self._native
        if owner is None or native is None:
            self._lock.release()
            raise RuntimeError("Candidate unload hook is no longer active")
        self._closing = True
        error, result = None, None
        report = {"created_at_unix":self.created,"closed_at_unix":None,
                  "native_close_returned":False,"candidate_close_returned":False}
        try:
            try:
                result = native()
                report["native_close_returned"] = True
            except BaseException as exc:
                error = exc
                report["native_close_error"] = repr(exc)
            # Always try both operations. The row controller may have already
            # released its tables even when native close raised.
            for name, action in (("hook_restore",self._restore_hook),("candidate_close",owner.close)):
                try:
                    action()
                    report[name + "_returned"] = True
                except BaseException as exc:
                    report[name + "_error"] = repr(exc)
                    if error is None:
                        error = exc
                    else:
                        _note(error,exc)
            try:
                report["final"] = owner.snapshot()
            except BaseException as exc:
                report["snapshot_error"] = repr(exc)
            report["owner_closed"] = bool(owner.closed)
            report["status"] = ("cleanup_failed" if not owner.closed else
                                "unloaded" if error is None else "unloaded_with_error")
            report["closed_at_unix"] = time.time()
            self.registry.finished(self,report)
        finally:
            # No registry-held object or stored traceback may keep an unloaded
            # model alive, including when native/owner cleanup raises.
            self.owner = self._native = self._rows = self._previous = self._hook = None
            self._lock.release()
        if error is not None:
            raise error
        return result


class Registry:
    def __init__(self, *, namespace, manifest_sha256):
        self.namespace, self.manifest_sha256 = namespace, manifest_sha256
        self._active = {}
        self._history = deque(maxlen=16)
        self._missed_unloads = 0
        self._failed_unloads = 0
        # GC may invoke a weakref callback on this same thread during an
        # allocation inside a registry operation. It must be able to reenter.
        self._lock = threading.RLock()
        self._app = self._route = None

    def already_attached(self, model):
        with self._lock:
            reference = self._active.get(id(model))
            return reference is not None and reference() is not None

    def attach(self, owner):
        key = id(owner.model)
        with self._lock:
            reference = self._active.get(key)
            if reference is not None and reference() is not None:
                raise RuntimeError("Model already has a server candidate binding")
            binding = Binding(self,owner,key)
            # The model-owned row hook retains this binding while loaded. The
            # telemetry registry must never be the reason an orphaned model
            # survives a failed load or a native path which forgot to close.
            self._active[key] = weakref.ref(binding,lambda ref:self._collected(key,ref))
            return binding

    def _collected(self, key, reference):
        with self._lock:
            if self._active.get(key) is reference:
                del self._active[key]
                self._missed_unloads += 1
                self._history.append({"status":"collected_without_native_unload",
                                      "observed_at_unix":time.time(),"owner_closed":False})

    def finished(self, binding, report):
        plain = copy.deepcopy(report)
        with self._lock:
            reference = self._active.get(binding.key)
            if reference is not None and reference() is binding:
                del self._active[binding.key]
            if report.get("status") != "unloaded" or not report.get("owner_closed",False):
                self._failed_unloads += 1
            self._history.append(plain)

    def telemetry(self):
        with self._lock:
            references = tuple(self._active.values())
            plain_history = list(self._history)
            missed = self._missed_unloads
            failed = self._failed_unloads
        active = tuple(binding for ref in references if (binding := ref()) is not None)
        history = copy.deepcopy(plain_history)
        return {"candidate":NAME,"namespace":self.namespace,"manifest_sha256":self.manifest_sha256,
                "observation":"Nonblocking, non-atomic CPU counters; not a GPU/worker-idle proof.",
                "loaded_bindings":len(active),"models":[binding.observe() for binding in active],
                "collected_without_native_unload":missed,"failed_unloads":failed,"unloads":history}

    def install_route(self, server):
        if self._route is not None:
            return
        app = server.app
        if any(getattr(route,"path",None) == TELEMETRY_PATH for route in app.routes):
            raise RuntimeError("Candidate telemetry route already exists")
        async def endpoint(request):
            host = getattr(getattr(request,"client",None),"host",None)
            try:
                local = ipaddress.ip_address(host).is_loopback
            except (TypeError,ValueError):
                local = False
            if not local:
                raise server.HTTPException(status_code=403,detail="Loopback only")
            return self.telemetry()
        endpoint.__annotations__["request"] = server.FastAPIRequest
        app.add_api_route(TELEMETRY_PATH,endpoint,methods=["GET"],include_in_schema=False)
        self._app, self._route = app, app.routes[-1]

    def remove_route(self):
        if self._route is not None:
            if self._route in self._app.routes:
                self._app.routes.remove(self._route)
            self._route = self._app = None


def configure_wrapper(original, candidate_type, registry, expected_runtime):
    def configure(model, runtime):
        if runtime != expected_runtime:
            raise ValueError("Server candidate runtime differs from its explicit profile")
        if registry.already_attached(model):
            raise RuntimeError("Candidate configure is fresh-load-only")
        owner = binding = None
        try:
            experiment = original(model,runtime)
            owner = candidate_type(model,record_events=False)
            binding = registry.attach(owner)
            owner.set_enabled(True)
            server = sys.modules.get("omlx.server")
            if server is not None:
                registry.install_route(server)
            return experiment
        except BaseException as exc:
            # Original load already created the model. A failed attachment must
            # close those resources rather than escaping the native load hook.
            if owner is not None and binding is None:
                try:
                    owner.close()
                except BaseException as cleanup:
                    _note(exc,cleanup)
            try:
                model.close()
            except BaseException as cleanup:
                _note(exc,cleanup)
            raise
    return configure


def _candidate_type():
    directory = ROOT / "work/serving-integration-v3"
    existing = sys.modules.get("evidence")
    if existing is not None and Path(getattr(existing,"__file__","")).resolve() != directory/"evidence.py":
        raise RuntimeError("An unrelated evidence module is already imported")
    sys.path.insert(0,str(directory))
    try:
        spec = importlib.util.spec_from_file_location("_flash1017_server_v3_runtime",directory/"runtime.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.Candidate
    finally:
        sys.path.remove(str(directory))


def require_complete_shutdown(status):
    if (status["loaded_bindings"] or status["collected_without_native_unload"]
            or status["failed_unloads"]):
        raise RuntimeError("Native server did not complete every candidate unload")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--port",type=int,default=8001)
    parser.add_argument("--validate-only",action="store_true")
    args = parser.parse_args()
    profile, namespace = validate_profile(args.profile,args.manifest)
    if args.validate_only:
        print(json.dumps({"status":"profile_validated","namespace":namespace,
                          "model_loaded":False,"mlx_imported":"mlx.core" in sys.modules,"server_started":False}))
        return
    registry = Registry(namespace=namespace,manifest_sha256=sha(args.manifest))
    # Actual server startup only. The unchanged qualified launcher retains its
    # process lock, port check, checkpoint/package checks, state root and CLI.
    from flash1017 import bench, serve
    original, argv = bench.configure, sys.argv
    wrapper = configure_wrapper(original,_candidate_type(),registry,profile["runtime"])
    bench.configure = wrapper
    sys.argv = [argv[0],"--profile",str(args.profile),"--port",str(args.port)]
    primary_error = None
    try:
        serve.main()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if bench.configure is wrapper:
            bench.configure = original
        sys.argv = argv
        registry.remove_route()
        # Native server shutdown is responsible for draining its workers and
        # calling model.close. Never force close a possibly live model here.
        status = registry.telemetry()
        print(json.dumps({"candidate_server_final":status}),flush=True)
        try:
            require_complete_shutdown(status)
        except RuntimeError as exc:
            if primary_error is None:
                raise
            _note(primary_error,exc)


if __name__ == "__main__":
    main()
