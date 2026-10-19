"""Lifecycle/profile/route tests using only stdlib fakes; no HTTP or MLX."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import serve_candidate as server


class Model:
    def __init__(self):
        self.calls=[]
        self.fail=False
    def close(self):
        self.calls.append("native-model-close")
        if self.fail: raise ValueError("native failure")


class Rows:
    def __init__(self,model):
        self._model=model
        self._original=model.close
        self.status="active"
        self._states=[]
        self._saved=getattr(model,"__dict__").get("close",server._MISSING)
        self._installed=lambda:self._model_close()
        model.close=self._installed
    def _model_close(self):
        model=self._model
        model.calls.append("row-model-close")
        try: self._original()
        finally:
            self._restore()
            self.status="closed"
    def _restore(self):
        model=self._model
        if model is None:return
        if model.close is not self._installed:raise RuntimeError("foreign model.close")
        if self._saved is server._MISSING:del model.close
        else:model.close=self._saved
        self._model=self._original=None
    def close(self):
        if self._model is not None:
            self._model.calls.append("row-detach")
            self._restore()
            self.status="detached"


class Candidate:
    def __init__(self,model,record_events=False):
        self.model=self.target=model
        self.closed=False
        self.rows=Rows(model)
        self.cone=SimpleNamespace(enabled=False,_owner_thread=None,counts=Counter(),
                                  cone=SimpleNamespace(counts=Counter()),
                                  shape=SimpleNamespace(counts=Counter()),
                                  prime=SimpleNamespace(counts=Counter()))
        self.fail_close=False
        model.calls.append("candidate-install")
    def set_enabled(self,value):
        self.model.calls.append("enable")
        self.cone.enabled=value
    def close(self):
        if self.closed:return
        model=self.model
        model.calls.append("candidate-close")
        self.rows.close()
        model.calls.append("cone-restore")
        self.model=self.target=None
        self.closed=True
        if self.fail_close:raise RuntimeError("candidate failure")
    def snapshot(self):
        return {"closed":self.closed,"row_status":self.rows.status,"retained_bytes":0}


class FakeHTTPException(Exception):
    def __init__(self,**kwargs):self.status_code=kwargs["status_code"]


class App:
    def __init__(self):self.routes=[]
    def add_api_route(self,path,endpoint,**kw):
        self.routes.append(SimpleNamespace(path=path,endpoint=endpoint,**kw))


class Lifecycle(unittest.TestCase):
    def registry(self):return server.Registry(namespace="test-candidate",manifest_sha256="a"*64)
    def bind(self,model=None):
        model=model or Model()
        registry=self.registry()
        owner=Candidate(model)
        binding=registry.attach(owner)
        return model,registry,owner,binding

    def test_success_existing_model_hook_identity_order_and_repeated_native_close(self):
        model,registry,owner,binding=self.bind()
        self.assertIs(model.close,owner.rows._installed)
        model.close()
        self.assertEqual(model.calls,["candidate-install","row-model-close","native-model-close","candidate-close","cone-restore"])
        self.assertTrue(owner.closed)
        self.assertNotIn("_model_close",vars(owner.rows))
        self.assertEqual(registry.telemetry()["loaded_bindings"],0)
        model.close()
        self.assertEqual(model.calls[-1],"native-model-close")

    def test_native_error_still_restores_and_releases(self):
        model,registry,owner,binding=self.bind()
        model.fail=True
        with self.assertRaisesRegex(ValueError,"native failure"):model.close()
        self.assertTrue(owner.closed)
        summary=registry.telemetry()["unloads"][-1]
        self.assertTrue(summary["candidate_close_returned"])
        self.assertEqual(summary["status"],"unloaded_with_error")
        self.assertIsNone(binding.owner)

    def test_primary_exception_preserved_when_cleanup_also_fails(self):
        model,registry,owner,binding=self.bind()
        model.fail=owner.fail_close=True
        with self.assertRaisesRegex(ValueError,"native failure") as caught:model.close()
        if hasattr(caught.exception,"__notes__"):
            self.assertTrue(any("candidate failure" in s for s in caught.exception.__notes__))
        self.assertIn("candidate failure",registry.telemetry()["unloads"][-1]["candidate_close_error"])
        self.assertEqual(registry.telemetry()["loaded_bindings"],0)

    def test_incomplete_owner_cleanup_remains_final_failure_after_registry_release(self):
        model,registry,owner,binding=self.bind()
        def fail():raise RuntimeError("incomplete owner cleanup")
        owner.close=fail
        with self.assertRaisesRegex(RuntimeError,"incomplete"):model.close()
        summary=registry.telemetry()
        self.assertEqual(summary["loaded_bindings"],0)
        self.assertEqual(summary["failed_unloads"],1)
        self.assertEqual(summary["unloads"][-1]["status"],"cleanup_failed")
        with self.assertRaises(RuntimeError):server.require_complete_shutdown(summary)
        Candidate.close(owner)

    def test_final_shutdown_accepts_only_clean_completed_unloads(self):
        model,registry,owner,binding=self.bind()
        with self.assertRaises(RuntimeError):server.require_complete_shutdown(registry.telemetry())
        model.close()
        server.require_complete_shutdown(registry.telemetry())
        for name in ("failed_unloads","collected_without_native_unload"):
            summary=registry.telemetry();summary[name]=1
            with self.assertRaises(RuntimeError):server.require_complete_shutdown(summary)

    def test_controller_own_method_restored(self):
        model=Model();owner=Candidate(model);registry=self.registry()
        inherited=owner.rows._model_close
        prior=lambda:inherited()
        owner.rows._model_close=prior
        registry.attach(owner)
        model.close()
        self.assertIs(owner.rows._model_close,prior)

    def test_registry_and_binding_do_not_retain_model_after_close(self):
        model,registry,owner,binding=self.bind()
        reference=weakref.ref(model)
        model.close()
        del model
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(binding.owner)
        self.assertTrue(registry.telemetry()["unloads"][0]["owner_closed"])

    def test_reload_has_independent_lifetime(self):
        registry=self.registry()
        for _ in range(2):
            model=Model();owner=Candidate(model);registry.attach(owner)
            self.assertEqual(registry.telemetry()["loaded_bindings"],1)
            model.close()
        self.assertEqual(len(registry.telemetry()["unloads"]),2)

    def test_missed_native_close_cannot_make_registry_leak_model(self):
        model,registry,owner,binding=self.bind()
        reference=weakref.ref(model)
        del model,owner,binding
        gc.collect()
        self.assertIsNone(reference())
        summary=registry.telemetry()
        self.assertEqual(summary["loaded_bindings"],0)
        self.assertEqual(summary["collected_without_native_unload"],1)
        self.assertEqual(summary["unloads"][-1]["status"],"collected_without_native_unload")

    def test_orphan_collection_reenters_registry_lock(self):
        model,registry,owner,binding=self.bind()
        with registry._lock:
            del model,owner,binding
            gc.collect()
            self.assertEqual(registry._missed_unloads,1)

    def test_configure_retains_result_enables_only_after_original(self):
        registry=self.registry();model=Model();profile={"test":1}
        def original(m,p):m.calls.append("original-configure");return "experiment"
        configure=server.configure_wrapper(original,Candidate,registry,profile)
        self.assertEqual(configure(model,profile),"experiment")
        self.assertEqual(model.calls[:3],["original-configure","candidate-install","enable"])
        with self.assertRaises(RuntimeError):configure(model,profile)
        model.close()

    def test_initial_configure_failure_closes_native_model(self):
        registry=self.registry();model=Model()
        def original(m,p):raise ValueError("configure failure")
        configure=server.configure_wrapper(original,Candidate,registry,{})
        with self.assertRaisesRegex(ValueError,"configure failure"):configure(model,{})
        self.assertEqual(model.calls,["native-model-close"])

    def test_enable_failure_triggers_both_owners_cleanup(self):
        class Failing(Candidate):
            def set_enabled(self,value):raise ValueError("enable failure")
        registry=self.registry();model=Model()
        configure=server.configure_wrapper(lambda m,p:None,Failing,registry,{})
        with self.assertRaisesRegex(ValueError,"enable failure"):configure(model,{})
        self.assertIn("cone-restore",model.calls)
        self.assertEqual(registry.telemetry()["loaded_bindings"],0)

    def test_telemetry_never_takes_inference_lock(self):
        model,registry,owner,binding=self.bind()
        owner.cone._lock=threading.Lock()
        owner.cone._lock.acquire()
        owner.rows._lock=threading.Lock()
        owner.rows._lock.acquire()
        owner.cone._owner_thread=123
        try:
            self.assertTrue(registry.telemetry()["models"][0]["python_call_active_hint"])
        finally:
            owner.cone._lock.release();owner.rows._lock.release()
        model.close()

    def test_live_hotrows_lock_held_by_other_thread_does_not_block_telemetry(self):
        class Cache:
            def __init__(self):
                self.lock=threading.Lock();self.tags=SimpleNamespace(nbytes=8)
                self.freq=SimpleNamespace(nbytes=2);self.data=((SimpleNamespace(nbytes=80),"fake"),)
                self.queries=11;self.hits=5
            @property
            def nbytes(self):
                with self.lock:raise AssertionError("Locked nbytes property must not be used")
        model,registry,owner,binding=self.bind()
        cache=Cache();owner.rows._states=[SimpleNamespace(index=0,cache=cache,counts=Counter())]
        cache.lock.acquire()
        with ThreadPoolExecutor(1) as worker:
            future=worker.submit(registry.telemetry)
            try:
                value=future.result(timeout=.5)
                self.assertEqual(value["models"][0]["retained_row_cache_bytes_observed"],90)
            finally:cache.lock.release()
        model.close()

    def test_partial_hotrows_initialization_is_unknown(self):
        self.assertIsNone(server.observed_cache_bytes(SimpleNamespace(tags=None,freq=SimpleNamespace(nbytes=2),data=None)))

    def test_route_loopback_only_read_only_and_removal(self):
        registry=self.registry();app=App()
        module=SimpleNamespace(app=app,FastAPIRequest=type("Request",(),{}),HTTPException=FakeHTTPException)
        registry.install_route(module);registry.install_route(module)
        self.assertEqual(len(app.routes),1)
        route=app.routes[0]
        self.assertEqual(route.methods,["GET"])
        self.assertFalse(route.include_in_schema)
        for host in ("127.0.0.1","::1"):
            response=asyncio.run(route.endpoint(SimpleNamespace(client=SimpleNamespace(host=host))))
            self.assertEqual(response["candidate"],server.NAME)
        for host in ("192.0.2.1",None,"localhost"):
            with self.assertRaises(FakeHTTPException):
                asyncio.run(route.endpoint(SimpleNamespace(client=SimpleNamespace(host=host))))
        registry.remove_route();self.assertEqual(app.routes,[])

    def test_no_mlx_import(self):self.assertNotIn("mlx.core",sys.modules)


class Profile(unittest.TestCase):
    def files(self,directory):
        here=Path(directory);lab=here/"lab";lab.mkdir()
        baseline={"qualified":True,"runtime":{"ced_prefill":False,"engram_prefetch_bytes":64*2**20},
                  "packages":{"test":"1"},"inference_files_sha256":{"source.py":"abc"}}
        (lab/"qualified-profile.json").write_text(json.dumps(baseline))
        paths=[Path(server.__file__).resolve(),server.ROOT/"work/serving-integration-v3/runtime.py",
               server.ROOT/"work/serving-integration-v3/evidence.py",server.ROOT/"work/serving-cone-v3/cone_serving.py",
               server.ROOT/"work/serving-cache-v3/serving_cache.py",server.ROOT/"work/serving-cache-v3/hot_rows_upstream.py",
               server.ROOT/"work/cone-experiment/cone.py",server.ROOT/"work/cone-v2/stabilization.py",
               server.ROOT/"work/cone-dspark-stable/stabilization.py"]
        manifest={"baseline":{"profile_sha256":server.sha(lab/"qualified-profile.json")},
                  "sources_sha256":{str(p):server.sha(p) for p in paths}}
        mp=here/"manifest.json";mp.write_text(json.dumps(manifest))
        profile=json.loads(json.dumps(baseline));profile["profile_kind"]=server.PROFILE_KIND
        profile["runtime"]["serving_candidate"]={"name":server.NAME,"schema":1,
               "row_cache_bytes_per_table":256*2**20,"manifest_sha256":server.sha(mp)}
        pp=here/"profile.json";pp.write_text(json.dumps(profile))
        return lab,pp,mp,profile
    def test_explicit_namespace_and_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            lab,pp,mp,profile=self.files(directory)
            actual,namespace=server.validate_profile(pp,mp,lab=lab)
            self.assertEqual(actual,profile);self.assertEqual(len(namespace),12)
    def test_reject_unqualified_native_marker_runtime_and_manifest_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            lab,pp,mp,profile=self.files(directory)
            for key,value in (("qualified",False),("profile_kind","native")):
                changed=json.loads(json.dumps(profile));changed[key]=value;pp.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):server.validate_profile(pp,mp,lab=lab)
            for mutation in (lambda p:p["runtime"].pop("serving_candidate"),
                             lambda p:p["runtime"].update(ced_prefill=True),
                             lambda p:p["runtime"]["serving_candidate"].update(manifest_sha256="wrong")):
                changed=json.loads(json.dumps(profile));mutation(changed);pp.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):server.validate_profile(pp,mp,lab=lab)


if __name__ == "__main__":unittest.main()
