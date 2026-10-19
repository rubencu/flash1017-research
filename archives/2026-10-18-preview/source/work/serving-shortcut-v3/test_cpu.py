"""CPU-only shortcut tests; delegated processes and all server activity mocked."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import shortcut as s


class ShortcutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        def write(relative, value):
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) if not isinstance(value, str) else value); return path
        self.write = write
        write('outputs/flash1017-lab/run.sh', 'synthetic run script; never executed')
        write(s.LAUNCHER, 'synthetic launcher; never executed')
        native_source = write('outputs/flash1017-lab/native.py', 'native source')
        baseline = {'qualified': True, 'runtime': {'ced_prefill': False, 'prefill_step': 8192, 'engram_prefetch_bytes': 64 * 2**20},
                    'packages': {'mlx': 'fixture'}, 'inference_files_sha256': {'native.py': s.sha(native_source)}}
        baseline_path = write('outputs/flash1017-lab/qualified-profile.json', baseline)
        extra = write('work/dependency.py', 'reviewed helper')
        manifest = {'baseline': {'profile_sha256': s.sha(baseline_path)}, 'max_context_window': 131072,
                    'sources_sha256': {str(extra): s.sha(extra)}}
        manifest_path = write(s.MANIFEST, manifest)
        write('work/serving-launch-v3/manifest-01.json', {'sources_sha256': {str(self.root / s.LAUNCHER): s.sha(self.root / s.LAUNCHER)}})
        self.profile = baseline | {'qualified': True, 'qualification_http_status': 'complete_validated',
                                   'profile_kind': 'flash1017-serving-candidate-v3', 'max_context_window': 131072,
                                   'runtime': baseline['runtime'] | {'serving_candidate': {'name': 'cone-hotrows-v3', 'schema': 1,
                                      'row_cache_bytes_per_table': 256 * 2**20, 'manifest_sha256': s.sha(manifest_path)}}}
        self.profile_path = write(s.PROFILE, self.profile)
        self.pins = {relative: s.sha(self.root / relative) for relative in s.BOOTSTRAP_SHA}
        self.calls = []

    def run(self, result=None):
        # unittest calls TestCase.run; preserve that API, using invoke below.
        return super().run(result)

    def invoke(self, runner=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            code, folder = s.invoke(root=self.root, pins=self.pins, runner=runner, **kwargs)
        return code, folder, s.read(folder / 'shortcut-report.json')

    def success(self, command, **kwargs):
        self.calls.append((command, kwargs))
        data = s.validate(self.root, pins=self.pins) | {'status': 'ready'}
        s.save(command[command.index('--report') + 1], data)
        kwargs['stdout'].write('mock launcher ready\n'); kwargs['stdout'].flush()
        return SimpleNamespace(returncode=0)

    def fail(self, error, child=False):
        def run(command, **kwargs):
            self.calls.append((command, kwargs))
            data = {'status': 'failed', 'error': error}
            if child: data['child_pid'] = 999
            s.save(command[command.index('--report') + 1], data)
            return SimpleNamespace(returncode=1)
        return run

    def change_profile(self, **values):
        self.profile.update(values); self.profile_path.write_text(json.dumps(self.profile))

    def test_source_and_complete_http_gate(self):
        self.assertEqual(s.validate(self.root, pins=self.pins)['qualification_http_status'], 'complete_validated')

    def test_validate_only_cannot_delegate(self):
        runner = Mock(side_effect=AssertionError('must not execute'))
        code, folder, report = self.invoke(runner, validate_only=True)
        self.assertEqual(code, 0); self.assertEqual(report['status'], 'validation_passed')
        runner.assert_not_called(); self.assertFalse((folder / 'server.log').exists())

    def test_pending_http_refused_without_delegate(self):
        self.change_profile(qualification_http_status='pending'); runner = Mock()
        code, folder, report = self.invoke(runner)
        self.assertEqual(code, 2); self.assertEqual(report['status'], 'refused'); runner.assert_not_called()

    def test_inexact_http_spelling_refused(self):
        self.change_profile(qualification_http_status='passed')
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_unqualified_refused(self):
        self.change_profile(qualified=False)
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_missing_final_profile_friendly(self):
        self.profile_path.unlink(); code, _, report = self.invoke(Mock())
        self.assertEqual(code, 2); self.assertIn('not installed yet', report['message'])

    def test_changed_launcher_bootstrap_refused(self):
        (self.root / s.LAUNCHER).write_text('changed')
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_changed_manifest_bootstrap_refused(self):
        (self.root / s.MANIFEST).write_text('{}')
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_changed_transitive_source_refused(self):
        (self.root / 'work/dependency.py').write_text('changed')
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_changed_native_source_refused(self):
        (self.root / 'outputs/flash1017-lab/native.py').write_text('changed')
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_changed_runtime_refused(self):
        self.profile['runtime']['prefill_step'] = 4096; self.change_profile()
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_inherited_unvalidated_one_million_context_refused(self):
        self.change_profile(max_context_window=1048576)
        with self.assertRaises(s.Refused): s.validate(self.root, pins=self.pins)

    def test_success_delegates_exact_frozen_launcher_with_unique_paths(self):
        first = self.invoke(self.success, no_browser=True); second = self.invoke(self.success, no_browser=True)
        self.assertEqual(first[0], 0); self.assertEqual(second[0], 0); self.assertNotEqual(first[1], second[1])
        for code, folder, report in (first, second):
            self.assertEqual(report['status'], 'ready'); self.assertTrue((folder / 'launcher.log').exists())
            self.assertTrue((folder / 'launch-report.json').exists()); self.assertEqual(report['launcher_exit_code'], 0)
        command, kwargs = self.calls[0]
        self.assertEqual(command[:3], [str(self.root / 'outputs/flash1017-lab/run.sh'), '-u', str(self.root / s.LAUNCHER)])
        self.assertEqual(command[-1], '--no-browser'); self.assertFalse(kwargs['check']); self.assertNotIn('shell', kwargs)

    def test_other_profile_message_does_not_retry(self):
        code, _, report = self.invoke(self.fail('RuntimeError: Another lab profile is running; stop it first'))
        self.assertEqual(code, 2); self.assertEqual(len(self.calls), 1)
        self.assertIn('No second model', report['message']); self.assertIn('Stop Flash1017', report['message'])

    def test_port_busy_message_does_not_retry(self):
        _, _, report = self.invoke(self.fail('OSError: [Errno 48] Address already in use'))
        self.assertIn('No second model', report['message']); self.assertEqual(len(self.calls), 1)

    def test_timeout_keeps_started_server_observable(self):
        _, folder, report = self.invoke(self.fail('TimeoutError: candidate not ready', child=True))
        self.assertIn('may still be loading', report['message']); self.assertEqual(len(self.calls), 1)
        self.assertTrue((folder / 'launch-report.json').exists())

    def test_active_shortcut_lock_refuses_second_delegate(self):
        log_root = self.root / 'outputs/flash1017-lab/server-state/candidate-launches'; log_root.mkdir(parents=True)
        runner = Mock()
        with s.startup_lock(log_root / 'startup.lock'):
            code, _, report = self.invoke(runner)
        self.assertEqual(code, 2); runner.assert_not_called(); self.assertIn('already running', report['message'])
        self.assertEqual(self.invoke(self.success)[0], 0)

    def test_exit_zero_without_ready_report_is_refused(self):
        code, _, report = self.invoke(lambda *a, **k: SimpleNamespace(returncode=0))
        self.assertEqual(code, 2); self.assertEqual(report['status'], 'failed')

    def test_ready_wrong_namespace_refused(self):
        def wrong(command, **kwargs):
            result = self.success(command, **kwargs); path = command[command.index('--report') + 1]
            data = s.read(path); data['namespace'] = 'wrong'; s.save(path, data); return result
        self.assertEqual(self.invoke(wrong)[0], 2)

    def test_profile_changed_during_launch_refused(self):
        def changed(command, **kwargs):
            result = self.success(command, **kwargs); self.change_profile(name='changed'); return result
        self.assertEqual(self.invoke(changed)[0], 2)

    def test_interrupt_is_durable_without_retry_or_stop(self):
        runner = Mock(side_effect=KeyboardInterrupt)
        code, _, report = self.invoke(runner)
        self.assertEqual(code, 130); self.assertEqual(report['status'], 'interrupted'); self.assertEqual(runner.call_count, 1)
        self.assertIn('may still be loading', report['message'])

    def test_existing_invocation_reports_are_never_overwritten(self):
        _, folder, _ = self.invoke(validate_only=True); original = (folder / 'shortcut-report.json').read_bytes()
        self.invoke(validate_only=True); self.assertEqual((folder / 'shortcut-report.json').read_bytes(), original)

    def test_native_profile_and_sources_unchanged(self):
        paths = [self.root / p for p in ('outputs/flash1017-lab/qualified-profile.json', 'outputs/flash1017-lab/run.sh', s.LAUNCHER, s.MANIFEST)]
        before = [p.read_bytes() for p in paths]; self.invoke(self.success)
        self.assertEqual(before, [p.read_bytes() for p in paths])

    def test_no_inference_imports(self):
        self.assertFalse(any(k in ('mlx', 'omlx', 'torch') or k.startswith(('mlx.', 'omlx.', 'torch.')) for k in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
