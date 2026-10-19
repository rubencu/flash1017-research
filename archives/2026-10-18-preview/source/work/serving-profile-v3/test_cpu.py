"""Synthetic file-only profile preparation tests; no inference imports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare as p


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(p.encoded(value) if not isinstance(value, str) else value.encode())
    return path


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.lab = self.root / 'outputs/flash1017-lab'
        self.folder = self.root / 'work/serving-profile-v3'
        self.out = self.folder / 'staged-01'
        for name in (*p.REQUIRED, 'work/serving-profile-v3/prepare.py', 'work/serving-profile-v3/test_cpu.py',
                     'work/serving-profile-v3/PLAN.txt', 'work/serving-cache-v3/NOTICE.txt',
                     'work/serving-analysis-v3/analyze_performance_v2.py', 'work/serving-state-review-v3/validate_boundary.py'):
            put(self.root / name, '# synthetic ' + name)
        native = put(self.lab / 'flash1017/serve.py', '# native fixture')
        self.baseline_path = self.lab / 'qualified-profile.json'
        self.baseline = {'qualified': True, 'runtime': {'ced_prefill': False, 'prefill_step': 8192,
                         'engram_prefetch_bytes': 64 * 2**20, 'mtp': True}, 'max_context_window': 1048576,
                         'packages': {'mlx': 'fixture'}, 'inference_files_sha256': {'flash1017/serve.py': p.sha(native)},
                         'qualification_http_status': 'complete'}
        put(self.baseline_path, self.baseline)
        self.baseline_bytes = self.baseline_path.read_bytes()
        self.baseline_sha = p.sha(self.baseline_path)
        self.server_manifest = self.root / 'work/serving-server-v3/manifest-01.json'
        server = {'status': 'frozen_for_server_qualification', 'baseline': {'profile_sha256': self.baseline_sha},
                  'sources_sha256': {str(self.root / f): p.sha(self.root / f) for f in p.REQUIRED}}
        server['sources_sha256'][str(self.baseline_path)] = self.baseline_sha
        put(self.server_manifest, server)
        analyzer = self.root / 'work/serving-analysis-v3/analyze_performance_v2.py'
        put(self.root / 'work/serving-analysis-v3/manifest-v2.json', {'sources_sha256': {str(analyzer): p.sha(analyzer)}})
        self.boundary = self.make_run('boundary')
        self.performance = self.make_run('performance')
        dependency = put(self.root / 'site/numpy/core.so', 'synthetic binary')
        self.numpy = {'name': 'numpy', 'version': 'fixture', 'site_packages': str(dependency.parent),
                      'sources_sha256': {str(dependency): p.sha(dependency)}}

    def make_run(self, mode):
        raw = put(self.lab / f'results/{mode}.jsonl', '{"synthetic":true}\n')
        folder = raw.with_suffix('.sources')
        originals = [self.baseline_path, *[self.root / f for f in p.MEASURED]]
        # Exercise the byte-identical corrected harness path accepted for serving.
        if mode == 'performance':
            for i, original in enumerate(originals):
                if '/serving-integration-v3/' in str(original):
                    other = Path(str(original).replace('/serving-integration-v3/', '/serving-integration-v3b/'))
                    put(other, original.read_text())
                    originals[i] = other
        plan = put(self.root / f'plans/{mode}.json', {'mode': mode})
        originals.append(plan)
        execution = put(self.root / f'plans/{mode}-execution.json', {
            'baseline': {'profile_sha256': self.baseline_sha},
            'sources_sha256': {str(f): p.sha(f) for f in originals}})
        index = {}
        for i, original in enumerate([*originals, execution]):
            snapshot = folder / f'{i:03d}-{original.name}'
            put(snapshot, original.read_text())
            index[str(original)] = {'snapshot': snapshot.name, 'sha256': p.sha(snapshot)}
        ipath = put(folder / 'manifest.json', index)
        qpath = put(raw.with_suffix('.qualification.json'), {
            'status': 'complete_validated', 'mode': mode, 'model_cleanup_completed': True,
            'instance_hooks_restored': True, 'source_hashes_reverified': True, 'mismatch_pending': False,
            'pairs': [{'passed': True} for _ in range(13)], 'raw_output_sha256': p.sha(raw),
            'plan_sha256': p.sha(plan), 'manifest_sha256': p.sha(execution),
            'row_cache_after_model_close': {'status': 'closed', 'retained_bytes': 0, 'live_tables': []}})
        pins = {'plan_sha256': p.sha(plan), 'execution_manifest_sha256': p.sha(execution)}
        if mode == 'boundary':
            analyzer = self.root / 'work/serving-state-review-v3/validate_boundary.py'
            pins['source_snapshot_index_sha256'] = p.sha(ipath)
            report = {'status': 'accepted_independent_raw_state_audit', 'raw': str(raw), 'raw_sha256': p.sha(raw),
                      'qualification_sha256': p.sha(qpath), 'provenance': pins, 'measured_pairs': 12,
                      'measured_output_ids_compared': 384, 'measured_target_slot_comparisons': 4320,
                      'measured_physical_draft_ring_comparisons': 36, 'measured_full_vocabulary_logit_values_compared': 1551360,
                      'analyzer': str(analyzer), 'analyzer_sha256': p.sha(analyzer)}
        else:
            analyzer = self.root / 'work/serving-analysis-v3/analyze_performance_v2.py'
            pins.update({'raw_sha256': p.sha(raw), 'run_source_snapshot_manifest_sha256': p.sha(ipath),
                         'analyzer': str(analyzer), 'analyzer_sha256': p.sha(analyzer)})
            report = {'status': 'accepted_complete_token_and_functional_trace_identical', 'raw': str(raw), 'provenance': pins,
                      'measured': {'pairs': 12, 'output_tokens_compared': 1536},
                      'warmup_separate': {'pairs': 1, 'output_tokens_compared': 128}, 'all_measured_orders': {'AB': 6, 'BA': 6}}
        return put(self.root / f'reports/{mode}.json', report)

    def run_prepare(self):
        with patch.object(p, 'numpy_pins', return_value=copy.deepcopy(self.numpy)):
            return p.prepare(self.boundary, self.performance, self.out, root=self.root)

    def modify(self, path, fn):
        value = p.read(path)
        fn(value)
        put(path, value)

    def reject(self):
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            self.run_prepare()
        self.assertFalse(self.out.exists())
        self.assertEqual(self.baseline_path.read_bytes(), self.baseline_bytes)

    def test_complete_stages_unqualified_and_preserves_native(self):
        result = self.run_prepare()
        self.assertFalse(result['qualified'])
        profile = p.read(self.out / 'candidate-profile.unqualified.json')
        manifest = p.read(self.out / 'candidate-manifest.json')
        self.assertFalse(profile['qualified'])
        self.assertEqual(profile['qualification_http_status'], 'pending')
        self.assertEqual(profile['max_context_window'], 131072)
        self.assertEqual(profile['packages'], self.baseline['packages'])
        runtime = profile['runtime'].copy()
        marker = runtime.pop('serving_candidate')
        self.assertEqual(runtime, self.baseline['runtime'])
        self.assertEqual(marker['manifest_sha256'], p.sha(self.out / 'candidate-manifest.json'))
        self.assertNotEqual(result['candidate_namespace'], result['native_namespace'])
        self.assertEqual((self.out / 'native-profile.preserved.json').read_bytes(), self.baseline_bytes)
        self.assertEqual(self.baseline_path.read_bytes(), self.baseline_bytes)
        self.assertEqual(profile['inference_files_sha256'], self.baseline['inference_files_sha256'])
        self.assertTrue(set(self.numpy['sources_sha256']) <= set(manifest['sources_sha256']))
        for path, digest in manifest['sources_sha256'].items():
            self.assertEqual(p.sha(path), digest)

    def test_no_qualification_switch_or_inference_import(self):
        self.assertNotIn('mlx.core', sys.modules)
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('omlx', sys.modules)

    def test_frozen_launcher_contract_with_synthetic_files_only(self):
        self.run_prepare()
        source = Path(__file__).resolve().parents[1] / 'serving-server-v3/serve_candidate.py'
        spec = importlib.util.spec_from_file_location('_profile_fixture_launcher', source)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        profile_path = self.out / 'candidate-profile.unqualified.json'
        manifest_path = self.out / 'candidate-manifest.json'
        with patch.object(launcher, 'ROOT', self.root), patch.object(launcher, '__file__', str(self.root / p.REQUIRED[0])):
            # Unqualified real output must fail. A synthetic temp fixture alone
            # exercises the deliberate qualification transition, with no model.
            with self.assertRaises(ValueError):
                launcher.validate_profile(profile_path, manifest_path, lab=self.lab)
            self.modify(profile_path, lambda d: d.update(qualified=True))
            profile, identity = launcher.validate_profile(profile_path, manifest_path, lab=self.lab)
        self.assertTrue(profile['qualified'])
        self.assertEqual(identity, p.namespace(profile['runtime']))

    def test_numpy_distribution_requires_unique_installation(self):
        with patch.object(p.importlib.metadata, 'distributions', return_value=[]):
            with self.assertRaises(ValueError):
                p.numpy_pins(self.root / 'site')

    def test_numpy_distribution_file_selection(self):
        class Distribution:
            metadata, version = {'Name': 'numpy'}, 'test1'
            files = ['numpy/__init__.py', 'numpy/core.so', 'numpy/a.pyc', 'numpy-test.dist-info/METADATA', 'numpy-test.dist-info/RECORD']
            def locate_file(other, name):
                return self.root / 'site' / name
        for name in Distribution.files:
            put(self.root / 'site' / name, 'synthetic')
        with patch.object(p.importlib.metadata, 'distributions', return_value=[Distribution()]):
            result = p.numpy_pins(self.root / 'site')
        self.assertEqual(result['version'], 'test1')
        self.assertEqual(len(result['sources_sha256']), 4)
        self.assertFalse(any(k.endswith('.pyc') for k in result['sources_sha256']))

    def test_incomplete_performance_refused(self):
        path = Path(p.read(self.performance)['raw']).with_suffix('.qualification.json')
        self.modify(path, lambda d: d.update(status='running'))
        self.reject()

    def test_unaccepted_performance_refused(self):
        self.modify(self.performance, lambda d: d.update(status='rejected_no_speed_claim'))
        self.reject()

    def test_failed_boundary_refused(self):
        self.modify(self.boundary, lambda d: d.update(status='not_accepted'))
        self.reject()

    def test_partial_pair_count_refused(self):
        path = Path(p.read(self.performance)['raw']).with_suffix('.qualification.json')
        self.modify(path, lambda d: d['pairs'].pop())
        self.reject()

    def test_cleanup_failure_refused(self):
        path = Path(p.read(self.performance)['raw']).with_suffix('.qualification.json')
        self.modify(path, lambda d: d.update(instance_hooks_restored=False))
        self.reject()

    def test_residual_cache_refused(self):
        path = Path(p.read(self.performance)['raw']).with_suffix('.qualification.json')
        self.modify(path, lambda d: d['row_cache_after_model_close'].update(retained_bytes=1))
        self.reject()

    def test_raw_tamper_refused(self):
        Path(p.read(self.performance)['raw']).write_text('tampered')
        self.reject()

    def test_snapshot_tamper_refused(self):
        folder = Path(p.read(self.performance)['raw']).with_suffix('.sources')
        next(folder.glob('000-*')).write_text('tampered')
        self.reject()

    def test_source_tamper_refused(self):
        (self.root / p.REQUIRED[0]).write_text('changed launcher')
        self.reject()

    def test_qualified_native_pin_tamper_refused(self):
        (self.lab / 'flash1017/serve.py').write_text('changed native')
        self.reject()

    def test_analyzer_tamper_refused(self):
        (self.root / 'work/serving-analysis-v3/analyze_performance_v2.py').write_text('changed validator')
        self.reject()

    def test_wrong_order_balance_refused(self):
        self.modify(self.performance, lambda d: d.update(all_measured_orders={'AB': 7, 'BA': 5}))
        self.reject()

    def test_wrong_scope_refused(self):
        self.modify(self.boundary, lambda d: d.update(measured_full_vocabulary_logit_values_compared=1))
        self.reject()

    def test_output_escape_refused(self):
        self.out = self.root / 'outputs/not-allowed'
        self.reject()

    def test_existing_output_refused_without_overwrite(self):
        self.run_prepare()
        before = (self.out / 'candidate-manifest.json').read_bytes()
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertEqual((self.out / 'candidate-manifest.json').read_bytes(), before)

    def test_no_performance_threshold(self):
        # Advantage is a parent decision. Accepted evidence may report any speed.
        self.modify(self.performance, lambda d: d['measured'].update(prefill_seconds={'latency_reduction_percent': -20}))
        self.assertEqual(self.run_prepare()['status'], 'staged_unqualified_http_pending')


if __name__ == '__main__':
    unittest.main(verbosity=2)
