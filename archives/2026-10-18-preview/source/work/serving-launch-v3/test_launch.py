"""Startup refusal/reuse contracts; all network/process operations are mocked."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
import launch


class Response(io.BytesIO):
    def __init__(self,value):super().__init__(json.dumps(value).encode())


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.profile=self.root/"profile.json"
        self.profile.write_text(json.dumps({"runtime":{"serving_candidate":{"name":"cone-hotrows-v3"}},"inference_files_sha256":{}}))
        self.manifest=self.root/"manifest.json";self.manifest.write_text("{}")
        self.report=self.root/"report.json";self.log=self.root/"server.log";self.stamp=self.root/"stamp.json"
        self.identity=launch.hashlib.sha256(json.dumps(json.loads(self.profile.read_text())["runtime"],sort_keys=True).encode()).hexdigest()[:12]
        self.telemetry={"candidate":"cone-hotrows-v3","namespace":self.identity,"manifest_sha256":launch.sha(self.manifest),"loaded_bindings":1,
            "models":[{"status":"loaded","enabled":True,"row_status":"active"}],
            "failed_unloads":0,"collected_without_native_unload":0}
        self.payloads={"/health":{"engine_pool":{"loaded_count":1}},"/api/status":{"loaded_models":[launch.MODEL]},
                       "/api/flash1017/candidate":self.telemetry}
        self.opener=Mock()
        self.opener.open.side_effect=lambda url,timeout:Response(self.payloads[url.removeprefix("http://127.0.0.1:8001")])
        self.child=Mock(pid=9182);self.child.poll.return_value=None
        self.validation=types.SimpleNamespace(returncode=0,stderr="",stdout="{}")
        self.process=types.SimpleNamespace(create_time=lambda:123.0)
        self.run=Mock(return_value=self.validation)
        def spawn(*a,**kw):
            self.write_stamp(pid=self.child.pid)
            return self.child
        self.spawn=Mock(side_effect=spawn)
        self.socket=Mock();self.socket.__enter__=Mock(return_value=self.socket);self.socket.__exit__=Mock(return_value=False)
        self.argv=["launch.py","--profile",str(self.profile),"--manifest",str(self.manifest),
            "--log",str(self.log),"--report",str(self.report),"--no-browser"]

    def write_stamp(self,**changes):
        record={"pid":9181,"created":123.0,"profile":str(self.profile),"port":8001,**changes}
        self.stamp.write_text(json.dumps(record))

    @contextlib.contextmanager
    def mocks(self,clock=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(launch,"STAMP",self.stamp))
            stack.enter_context(patch.object(launch,"LAB",self.root))
            stack.enter_context(patch("sys.argv",self.argv))
            stack.enter_context(patch.object(launch.subprocess,"run",self.run))
            stack.enter_context(patch.object(launch.subprocess,"Popen",self.spawn))
            stack.enter_context(patch.object(launch.urllib.request,"build_opener",return_value=self.opener))
            stack.enter_context(patch.object(launch.socket,"socket",return_value=self.socket))
            stack.enter_context(patch.dict("sys.modules",{"psutil":types.SimpleNamespace(Process=lambda pid:self.process,NoSuchProcess=ProcessLookupError)}))
            stack.enter_context(patch.object(launch.webbrowser,"open",side_effect=AssertionError("No browser expected")))
            stack.enter_context(patch.object(launch.time,"sleep",return_value=None))
            if clock is not None:stack.enter_context(patch.object(launch.time,"monotonic",side_effect=clock))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def test_existing_exact_candidate_is_reused(self):
        self.write_stamp()
        with self.mocks():launch.main()
        self.spawn.assert_not_called()
        self.assertEqual(json.loads(self.report.read_text())["status"],"ready")

    def test_live_other_profile_refused_before_spawn(self):
        self.write_stamp(profile=str(self.root/"native.json"))
        with self.mocks(),self.assertRaisesRegex(RuntimeError,"Another lab profile"):launch.main()
        self.spawn.assert_not_called();self.opener.open.assert_not_called()

    def test_validation_failure_never_starts_or_polls(self):
        self.validation.returncode=1;self.validation.stderr="bad source pin"
        with self.mocks(),self.assertRaisesRegex(RuntimeError,"validation failed"):launch.main()
        self.spawn.assert_not_called();self.opener.open.assert_not_called()
        self.assertEqual(json.loads(self.report.read_text())["status"],"refused")

    def test_new_candidate_ready_uses_detached_process(self):
        with self.mocks():launch.main()
        self.assertTrue(self.spawn.call_args.kwargs["start_new_session"])
        self.assertTrue(self.spawn.call_args.kwargs["close_fds"])
        self.assertEqual(json.loads(self.report.read_text())["server"]["pid"],self.child.pid)

    def test_occupied_port_never_spawns(self):
        self.socket.bind.side_effect=OSError("already used")
        with self.mocks(),self.assertRaises(OSError):launch.main()
        self.spawn.assert_not_called()

    def test_namespace_mismatch_times_out_without_killing_or_second_spawn(self):
        self.telemetry["namespace"]="wrong"
        with self.mocks(clock=[0,301]),self.assertRaises(TimeoutError):launch.main()
        self.assertEqual(self.spawn.call_count,1)
        self.child.terminate.assert_not_called();self.child.kill.assert_not_called()

    def test_failed_unload_telemetry_not_ready(self):
        self.write_stamp();self.telemetry["failed_unloads"]=1
        with self.mocks(clock=[0,301]),self.assertRaises(TimeoutError):launch.main()
        self.spawn.assert_not_called()

    def test_profile_change_during_poll_is_refused(self):
        self.write_stamp()
        def changed(url,timeout):
            self.profile.write_text(self.profile.read_text()+"\n")
            return Response(self.payloads[url.removeprefix("http://127.0.0.1:8001")])
        self.opener.open.side_effect=changed
        with self.mocks(),self.assertRaisesRegex(RuntimeError,"changed during startup"):launch.main()
        self.spawn.assert_not_called()

    def test_exited_child_never_treated_as_ready(self):
        self.child.poll.return_value=1
        with self.mocks(),self.assertRaisesRegex(RuntimeError,"exited during startup"):launch.main()
        self.opener.open.assert_not_called()

    def test_existing_report_is_preserved(self):
        self.report.write_text("original")
        with self.mocks(),self.assertRaises(FileExistsError):launch.main()
        self.assertEqual(self.report.read_text(),"original");self.run.assert_not_called()

    def test_slow_success_is_rejected_after_deadline(self):
        self.write_stamp()
        with self.mocks(clock=[0,301]),self.assertRaises(TimeoutError):launch.main()
        self.assertEqual(json.loads(self.report.read_text())["status"],"failed")

    def test_replaced_stamp_during_http_is_rejected(self):
        self.write_stamp()
        def replaced(url,timeout):
            self.write_stamp(pid=9911)
            return Response(self.payloads[url.removeprefix("http://127.0.0.1:8001")])
        self.opener.open.side_effect=replaced
        with self.mocks(),self.assertRaisesRegex(RuntimeError,"stamp/process changed"):launch.main()

    def test_launched_child_must_own_stamp(self):
        def wrong_spawn(*args,**kwargs):
            self.write_stamp(pid=9911)
            return self.child
        self.spawn.side_effect=wrong_spawn
        with self.mocks(),self.assertRaisesRegex(RuntimeError,"different process"):launch.main()
        self.opener.open.assert_not_called()

    def test_model_list_shape_and_candidate_identity(self):
        for mutate in (lambda t:t.update(candidate="foreign"),lambda t:t["models"].append(dict(t["models"][0]))):
            with self.subTest(mutate=mutate):
                self.report.unlink(missing_ok=True);self.write_stamp()
                original=json.loads(json.dumps(self.telemetry))
                mutate(self.telemetry)
                with self.mocks(clock=[0,301]),self.assertRaises(TimeoutError):launch.main()
                self.telemetry.clear();self.telemetry.update(original)

    def test_changed_inference_source_blocks_reuse(self):
        self.write_stamp()
        source=self.root/"source.py";source.write_text("modified")
        profile=json.loads(self.profile.read_text());profile["inference_files_sha256"]={"source.py":"0"*64}
        self.profile.write_text(json.dumps(profile))
        with self.mocks(),self.assertRaisesRegex(RuntimeError,"inference source changed"):launch.main()
        self.spawn.assert_not_called();self.opener.open.assert_not_called()


if __name__=="__main__":unittest.main()
