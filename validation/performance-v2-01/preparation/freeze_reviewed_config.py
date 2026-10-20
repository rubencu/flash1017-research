"""Bind reviewed local sources; the final quality report is attached only after it passes."""
from pathlib import Path
import copy
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
quality = json.loads((ROOT / 'work/flash1017-quality-02/protocol.json').read_text())
draft = json.loads((HERE / 'plan-draft-01.json').read_text())
reference = json.loads((HERE / 'shared-environment-reference-01.json').read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
relative = lambda p: str(p.relative_to(ROOT))
frozen = HERE / 'frozen-source'
frozen.mkdir(exist_ok=False)
for name in ['run_installed_performance.py', 'run_installed_quality.py', 'summarize_installed_performance.py']:
    (frozen / name).write_bytes((ROOT / 'repositories/flash1017-research/tools' / name).read_bytes())
if digest(frozen / 'run_installed_quality.py') != quality['harness_sha256']:
    raise RuntimeError('Quality support source differs from the qualified-generation source')
fields = {'shared_environment': reference['shared_environment'],
          'shared_environment_reference_sha256': digest(HERE / 'shared-environment-reference-01.json'),
          'performance_harness_path': relative(frozen / 'run_installed_performance.py'),
          'performance_harness_sha256': digest(frozen / 'run_installed_performance.py'),
          'quality_harness_path': relative(frozen / 'run_installed_quality.py'),
          'quality_harness_sha256': digest(frozen / 'run_installed_quality.py'),
          'analysis_harness_sha256': digest(frozen / 'summarize_installed_performance.py'),
          'analysis_harness_path': relative(frozen / 'summarize_installed_performance.py'),
          'baseline': copy.deepcopy(quality['baseline']),
          'candidate': copy.deepcopy(quality['candidate']),
          'memory_observer': {'method': 'Read-only MLX active/cache/cumulative peak snapshots outside HTTP timing; no resets',
                              'scope': 'Peak includes model load and warmup, not full system resident memory or request-only peak'},
          'qualification_stage': 'Freeze only after exact 618-pair quality and fresh isolated grading pass',
          'review_items_before_freeze': []}
fields['baseline']['expected_runtime'] = {
    'prefill_step_size': 2048, 'max_num_seqs': 1,
    'attention_output_quantization': False, 'flash1017_enabled': False,
    'confidence_enabled': False, 'replay_installed': False,
    'engram_prefetch_bytes': None, 'mtp_depth': 3, 'mtp_depth_fixed': False}
fields['baseline']['prefetch_observer_note'] = 'Native default is 16 MiB in pinned source. This release exposes no budget_bytes observer, so the snapshot is unavailable (null), not zero.'
artifacts = copy.deepcopy(draft['artifacts'])
for key, name in [('performance-source', 'run_installed_performance.py'),
                  ('quality-source', 'run_installed_quality.py'),
                  ('analysis-source', 'summarize_installed_performance.py')]:
    artifacts[key] = {'path': relative(frozen / name), 'sha256': digest(frozen / name)}
fields['artifacts'] = artifacts
wheels = {
    'baseline': 'work/flash1017-fork-validation/downloads/omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl',
    'candidate': 'work/flash1017-quality-02/dist/omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl'}
for arm, wheel in wheels.items():
    if digest(ROOT / wheel) != quality[arm]['wheel_sha256']:
        raise RuntimeError('Wheel differs from quality protocol')
files = {record['path']: record['sha256'] for record in artifacts.values()}
for path in [HERE / 'run_when_qualified.py', HERE / 'plan-draft-01.json',
             HERE / 'shared-environment-reference-01.json',
             ROOT / 'work/flash1017-quality-02/protocol.json',
             *[ROOT / p for p in wheels.values()]]:
    files[relative(path)] = digest(path)
reviewed = {
    'schema': 'flash1017-performance-local-review/1',
    'status': 'sources-reviewed-pending-complete-quality',
    'scope': 'Local parent source review and CPU checks, not user publication approval or external replication',
    'draft_plan_path': relative(HERE / 'plan-draft-01.json'),
    'files': files, 'final_fields': fields, 'wheels': wheels,
    'analysis_harness_path': relative(frozen / 'summarize_installed_performance.py'),
    'environments': {'baseline': 'work/flash1017-quality-01/baseline-env',
                     'candidate': 'work/flash1017-quality-02/candidate-env'},
    'cpu_checks': {'research_tools': 79, 'supervisor_lifecycle': 10,
                   'actual_installed_environment_preflight': 'passed both arms; 110 packages and 10 native assets match'},
    'no_publication': True}
with (HERE / 'REVIEWED-CONFIG.json').open('x') as output:
    json.dump(reviewed, output, indent=2);output.write('\n')
print(json.dumps({'reviewed_config_sha256': digest(HERE / 'REVIEWED-CONFIG.json'),
                  'performance_harness_sha256': fields['performance_harness_sha256'],
                  'analysis_harness_sha256': fields['analysis_harness_sha256'],
                  'status': 'pending complete quality; no model run started'}))
