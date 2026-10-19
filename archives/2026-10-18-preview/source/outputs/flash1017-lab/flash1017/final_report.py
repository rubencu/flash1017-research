"""Collect completed qualification evidence without performing inference."""
import datetime
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results = root/'results'
    def read(name):
        return json.loads((results/name).read_text())
    http = read('server-validation-01.json')
    long = read('long-context-http-01.json')
    restart = read('server-restart-02.json')
    finished = read('server-finalization-02.json')
    cached = read('long-context-cache-01.json')
    original = json.loads((root/'server-state/original-server-memory-reservation.json').read_text())
    assert len(http) == 7 and all(t['passed'] for t in http)
    assert len(long['tests']) == 2 and all(t['status'] == 'complete' and t['passed'] for t in long['tests'])
    assert restart['status'] == 'complete' and restart['passed']
    assert finished['status'] == 'complete' and original['status'] == 'restored'
    assert cached['status'] == 'complete' and cached['passed']
    projection = read('validation-summary-01.json')
    long_summary = []
    for test in long['tests']:
        resources = test['resources']
        long_summary.append({
            'name': test['name'], 'input_tokens': test['actual_input_tokens'],
            'correct_keys': sum(test['correct'].values()), 'total_keys': len(test['correct']),
            'ttft_seconds': test['ttft_seconds'], 'total_seconds': test['total_seconds'],
            'api_usage': test['usage'],
            'peak_footprint_gib': resources['peak_process_footprint_bytes']/2**30,
            'minimum_system_available_gib': resources['minimum_system_available_bytes']/2**30,
            'additional_system_swap_bytes': max(0, resources['swap_peak_bytes']-resources['swap_start_bytes']),
        })
    evidence_names = [
        'heldout-01.jsonl', 'confidence-16k-01.jsonl',
        'head-full-numerics-01.json', 'head-single-row-numerics-01.json',
        'quality-grades-01.json', 'deployment-quality-grades-01.json',
        'server-validation-01.json', 'long-context-http-01.json',
        'server-restart-01.json', 'server-finalization-01.json',
        'server-restart-02.json', 'server-finalization-02.json',
        'long-context-cache-01.json', 'startup-port-fix-01.json',
        'final-source-archive-report.json', 'final-integrity-02.json',
    ]
    report = {
        'status': 'complete',
        'compiled_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model': 'DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp',
        'hardware': 'M5 Ultra, 80 GPU cores, 256 GiB',
        'primary_kernel_result': {
            'paired_geometric_generation_speedup': projection['primary_kernel_pair']['paired_geometric_speedup'],
            'matched_streams': projection['primary_kernel_pair']['identical_token_streams'],
            'compared_output_tokens': 10240,
            'scope': 'Reference arithmetic, fixed draft depth two; 1K and 16K synthetic inputs, 512 generated tokens, ten matched pairs per context.',
        },
        'http_checks': [{'test': t['test'], 'passed': t['passed']} for t in http],
        'long_context': long_summary,
        'long_context_scope': long['scope'],
        'cached_changed_question': {k: cached[k] for k in [
            'input_tokens', 'cached_tokens', 'ttft_seconds', 'total_seconds', 'correct', 'scope']},
        'stop_restart_recovery_passed': True,
        'original_server_memory_settings_restored': True,
        'limitations': [
            'One machine, one quantization, synthetic throughput prompts, and small local coding checks.',
            'Confidence scheduling has different sampled streams and is separate from the token-identical kernel comparison.',
            'Six-row projection matches original single-row arithmetic, but differs numerically from the stock six-row matrix-multiply fallback.',
            'No general long-context reasoning, full-precision quality, global-fastest, or first-Mac claim.',
        ],
        'evidence_sha256': {name: hashlib.sha256((results/name).read_bytes()).hexdigest() for name in evidence_names},
    }
    target = results/'final-report-01.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'evidence_sha256'}, indent=2))


if __name__ == '__main__':
    main()
