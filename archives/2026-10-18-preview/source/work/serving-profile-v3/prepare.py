"""Stage an unqualified candidate profile from completed, pinned evidence.

Stdlib only. Never imports the model/runtime/NumPy, reads weight data, contacts
a server, acquires the inference lock, or grants qualification. The reviewed
analyzers own raw numerical/statistical validation; this builder verifies their
accepted reports, linked execution evidence, and the sources about to be used.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = Path('/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages')
REQUIRED = (
    'work/serving-server-v3/serve_candidate.py',
    'work/serving-integration-v3/runtime.py',
    'work/serving-integration-v3/evidence.py',
    'work/serving-cone-v3/cone_serving.py',
    'work/cone-experiment/cone.py',
    'work/cone-v2/stabilization.py',
    'work/cone-dspark-stable/stabilization.py',
    'work/serving-cache-v3/serving_cache.py',
    'work/serving-cache-v3/hot_rows_upstream.py',
)
MEASURED = REQUIRED[1:]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def namespace(runtime):
    return hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()[:12]


def numpy_pins(site=SITE):
    """Inspect distribution metadata without importing NumPy or its extension."""
    matches = [d for d in importlib.metadata.distributions(path=[str(site)])
               if d.metadata.get('Name', '').lower() == 'numpy']
    require(len(matches) == 1, 'Need exactly one NumPy distribution in the serving site-packages')
    dist = matches[0]
    paths = sorted({Path(dist.locate_file(f)).resolve() for f in (dist.files or [])
                    if str(f).endswith(('.py', '.so', '.dylib', '/METADATA', '/RECORD'))})
    require(paths and any(p.suffix == '.so' for p in paths), 'NumPy compiled runtime files are missing')
    require(all(p.is_relative_to(Path(site).resolve()) for p in paths), 'NumPy runtime path escaped site-packages')
    return {'name': 'numpy', 'version': dist.version, 'site_packages': str(Path(site).resolve()),
            'scope': 'Python, compiled runtime files, METADATA and RECORD; no imports or weight reads.',
            'sources_sha256': {str(p): sha(p) for p in paths}}


def verify_report(report_path, mode, root, baseline_sha):
    report_path = Path(report_path).resolve()
    report = read(report_path)
    accepted = ('accepted_independent_raw_state_audit' if mode == 'boundary' else
                'accepted_complete_token_and_functional_trace_identical')
    require(report.get('status') == accepted, mode + ' report has not been accepted')
    raw = Path(report['raw']).resolve()
    qpath = raw.with_suffix('.qualification.json')
    q = read(qpath)
    require(q.get('status') == 'complete_validated' and q.get('mode') == mode, 'Incomplete/wrong qualification')
    require(all(q.get(k) is True for k in ('model_cleanup_completed', 'instance_hooks_restored', 'source_hashes_reverified')),
            'Execution cleanup/source verification incomplete')
    require(q.get('mismatch_pending') is False and len(q['pairs']) == 13 and
            all(p.get('passed') is True for p in q['pairs']), 'Execution pairs incomplete or divergent')
    cache = q['row_cache_after_model_close']
    require(cache['status'] == 'closed' and cache['retained_bytes'] == 0 and cache['live_tables'] == [], 'Retained row cache not released')
    pins = report['provenance']
    raw_sha = sha(raw)
    require(raw_sha == q['raw_output_sha256'] == (report['raw_sha256'] if mode == 'boundary' else pins['raw_sha256']),
            'Raw report hash differs')
    if mode == 'boundary':
        require(report['qualification_sha256'] == sha(qpath), 'Boundary qualification hash differs')
        require((report['measured_pairs'], report['measured_output_ids_compared'], report['measured_target_slot_comparisons'],
                 report['measured_physical_draft_ring_comparisons'], report['measured_full_vocabulary_logit_values_compared']) ==
                (12, 384, 4320, 36, 1551360), 'Boundary scope differs')
        analyzer, analyzer_sha = Path(report['analyzer']).resolve(), report['analyzer_sha256']
        index_sha = pins['source_snapshot_index_sha256']
    else:
        require((report['measured']['pairs'], report['measured']['output_tokens_compared'],
                 report['warmup_separate']['pairs'], report['warmup_separate']['output_tokens_compared']) ==
                (12, 1536, 1, 128), 'Performance scope differs')
        require(report['all_measured_orders'] == {'AB': 6, 'BA': 6}, 'Performance order balance differs')
        analyzer, analyzer_sha = Path(pins['analyzer']).resolve(), pins['analyzer_sha256']
        require(analyzer == root / 'work/serving-analysis-v3/analyze_performance_v2.py', 'Expected frozen analyzer v2')
        frozen = read(root / 'work/serving-analysis-v3/manifest-v2.json')
        require(frozen['sources_sha256'][str(analyzer)] == analyzer_sha, 'Performance analyzer is not the reviewed revision')
        index_sha = pins['run_source_snapshot_manifest_sha256']
    require(sha(analyzer) == analyzer_sha, 'Accepted analyzer source changed')
    folder, index_path = raw.with_suffix('.sources'), raw.with_suffix('.sources') / 'manifest.json'
    require(sha(index_path) == index_sha, 'Source snapshot index changed')
    index = read(index_path)
    archives = {}
    for original, item in index.items():
        require(Path(item['snapshot']).name == item['snapshot'], 'Unsafe snapshot filename')
        path = folder / item['snapshot']
        require(sha(path) == item['sha256'], 'Archived source changed: ' + original)
        archives[original] = path
    require(pins['plan_sha256'] == q['plan_sha256'] and pins['execution_manifest_sha256'] == q['manifest_sha256'],
            'Report/qualification provenance differs')
    manifests = [path for original, path in archives.items() if index[original]['sha256'] == q['manifest_sha256']]
    require(len(manifests) == 1, 'Execution manifest is not uniquely archived')
    execution = read(manifests[0])
    manifest_original = next(k for k, p in archives.items() if p == manifests[0])
    require(set(index) == set(execution['sources_sha256']) | {manifest_original}, 'Archived source closure differs')
    require(all(index[p]['sha256'] == h for p, h in execution['sources_sha256'].items()), 'Execution source pins differ')
    baseline_path = root / 'outputs/flash1017-lab/qualified-profile.json'
    require(index[str(baseline_path)]['sha256'] == baseline_sha == execution['baseline']['profile_sha256'], 'Run used a different baseline')
    require(q['plan_sha256'] in {v['sha256'] for v in index.values()}, 'Frozen plan missing')
    for relative in MEASURED:
        path = root / relative
        # The corrected performance harness retained identical runtime/evidence
        # bytes in v3b. The server intentionally loads their original v3 paths.
        alternatives = (str(path), str(path).replace('/serving-integration-v3/', '/serving-integration-v3b/'))
        require(any(p in index and index[p]['sha256'] == sha(path) for p in alternatives), 'Serving dependency differs from measured bytes: ' + relative)
    evidence = {str(p): sha(p) for p in [report_path, raw, qpath, index_path, analyzer, *archives.values()]}
    return {'report': str(report_path), 'report_sha256': sha(report_path), 'raw': str(raw), 'raw_sha256': raw_sha,
            'qualification': str(qpath), 'qualification_sha256': sha(qpath), 'status': accepted,
            'plan_sha256': q['plan_sha256'], 'execution_manifest_sha256': q['manifest_sha256']}, evidence


def prepare(boundary_audit, performance_analysis, output_directory, *, root=ROOT, site=SITE):
    root = Path(root).resolve()
    lab = root / 'outputs/flash1017-lab'
    output_directory = Path(output_directory).resolve()
    require(output_directory.is_relative_to(root / 'work/serving-profile-v3'), 'Write only under new serving-profile-v3 staging')
    require(not output_directory.exists(), 'Use a fresh output directory')
    baseline_path = lab / 'qualified-profile.json'
    baseline, baseline_sha = read(baseline_path), sha(baseline_path)
    require(baseline.get('qualified') is True and baseline['runtime']['ced_prefill'] is False and
            baseline['runtime']['prefill_step'] == 8192 and baseline['runtime']['engram_prefetch_bytes'] == 64 * 2**20,
            'Expected qualified native64 baseline')
    server_manifest = root / 'work/serving-server-v3/manifest-01.json'
    server = read(server_manifest)
    require(server['status'] == 'frozen_for_server_qualification' and server['baseline']['profile_sha256'] == baseline_sha,
            'Server freeze differs from baseline')
    sources = dict(server['sources_sha256'])
    require(all(str(root / p) in sources for p in REQUIRED), 'Missing transitive serving source')
    sources[str(server_manifest)] = sha(server_manifest)
    for relative, digest in baseline['inference_files_sha256'].items():
        path = (lab / relative).resolve()
        require(path.is_relative_to(lab), 'Baseline inference pin escaped lab')
        sources[str(path)] = digest
    for path, digest in sources.items():
        require(sha(path) == digest, 'Current serving/native source differs: ' + path)
    boundary, boundary_sources = verify_report(boundary_audit, 'boundary', root, baseline_sha)
    performance, performance_sources = verify_report(performance_analysis, 'performance', root, baseline_sha)
    numpy = numpy_pins(site)
    sources.update(boundary_sources)
    sources.update(performance_sources)
    sources.update(numpy.pop('sources_sha256'))
    for name in ('prepare.py', 'test_cpu.py', 'PLAN.txt'):
        path = root / 'work/serving-profile-v3' / name
        sources[str(path)] = sha(path)
    for path in (root / 'work/serving-analysis-v3/manifest-v2.json', root / 'work/serving-cache-v3/NOTICE.txt'):
        sources[str(path)] = sha(path)
    manifest = {'schema': 'flash1017-serving-profile-preparation-v3/1', 'status': 'staged_http_qualification_pending',
                'baseline': {'profile_sha256': baseline_sha}, 'sources_sha256': dict(sorted(sources.items())),
                'additional_dependency': numpy, 'evidence': {'boundary': boundary, 'performance': performance},
                'qualification_http_status': 'pending', 'qualification_granted_by_builder': False,
                'max_context_window': 131072}
    manifest_bytes = encoded(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    runtime = copy.deepcopy(baseline['runtime'])
    runtime['serving_candidate'] = {'name': 'cone-hotrows-v3', 'schema': 1,
                                  'row_cache_bytes_per_table': 256 * 2**20, 'manifest_sha256': manifest_sha}
    require(namespace(runtime) != namespace(baseline['runtime']), 'Candidate namespace equals native namespace')
    profile = {'qualified': False, 'profile_kind': 'flash1017-serving-candidate-v3',
               'name': 'Flash1017 cone + HotRows candidate, native64, 131K interim context',
               'runtime': runtime, 'packages': copy.deepcopy(baseline['packages']),
               'inference_files_sha256': copy.deepcopy(baseline['inference_files_sha256']),
               'max_context_window': 131072, 'qualification_http_status': 'pending',
               'qualification_scope': 'Full-model boundary and strict paired generation evidence accepted. Candidate HTTP, prefix reuse, cancellation, multimodal fallback, and unload/reload remain pending. This file cannot start the server until root explicitly qualifies it for that validation.',
               'baseline': {'profile': str(baseline_path), 'sha256': baseline_sha, 'preserved': True},
               'evidence': manifest['evidence'],
               'limits': ['Finite tests do not establish universal bitwise equivalence.',
                          '131072 is an interim total context cap, not a claim of long-context benchmark quality.',
                          'Native archived1M results are not candidate1M evidence.',
                          '512MiB bounds retained row-cache arrays; temporaries and model memory are additional.',
                          'NumPy serving dependency is file-pinned separately; baseline package-version contract remains unchanged.']}
    require(sha(baseline_path) == baseline_sha, 'Baseline changed during preparation')
    # Check all closure bytes once more immediately before committing artifacts.
    for path, digest in sources.items():
        require(sha(path) == digest, 'Source/evidence changed during preparation: ' + path)
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / 'candidate-manifest.json').write_bytes(manifest_bytes)
    (output_directory / 'candidate-profile.unqualified.json').write_bytes(encoded(profile))
    (output_directory / 'native-profile.preserved.json').write_bytes(baseline_path.read_bytes())
    return {'status': 'staged_unqualified_http_pending', 'directory': str(output_directory),
            'manifest_sha256': manifest_sha, 'candidate_namespace': namespace(runtime),
            'native_namespace': namespace(baseline['runtime']), 'sources_pinned': len(sources),
            'max_context_window': 131072, 'qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary-audit', type=Path, required=True)
    parser.add_argument('--performance-analysis', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.boundary_audit, args.performance_analysis, args.output_directory), indent=2))


if __name__ == '__main__':
    main()
