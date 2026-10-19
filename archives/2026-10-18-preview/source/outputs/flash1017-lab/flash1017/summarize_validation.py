"""Produce a compact audit of the frozen full-model validation experiments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from .analyze import analyze, geometric_mean, paired_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    evidence = {}

    def read(name, lines=False):
        path = root / 'results' / name
        data = path.read_bytes()
        evidence[name] = hashlib.sha256(data).hexdigest()
        return [json.loads(s) for s in data.splitlines()] if lines else json.loads(data)

    rows = read('heldout-01.jsonl', True)
    runs = [r for r in rows if r.get('type') == 'run']
    if len(runs) != len(rows[0]['plan']['jobs']):
        raise ValueError('Held-out experiment is incomplete')
    usable = [r for r in runs if not r.get('warmup') and r.get('fixed_length_valid') is not False]
    selection = json.loads((root / 'plans/selection.json').read_text())
    baseline, candidate = selection['matched_baseline'], selection['candidate']
    report = {'scope': 'One M5 Ultra 80-GPU-core Mac, exact 1017 oQ2.5e checkpoint, single-request synthetic throughput tests.',
              'heldout': analyze(rows), 'measured_runs': sum(not r.get('warmup') for r in runs),
              'primary_kernel_pair': paired_summary(usable, baseline, candidate),
              'by_context': {}}
    for length in sorted({r['input_tokens'] for r in usable}):
        selected = [r for r in usable if r['input_tokens'] == length]
        profiles = defaultdict(list)
        for row in selected:
            profiles[row['profile']['name']].append(row)
        report['by_context'][str(length)] = {
            'profiles': {name: {'runs': len(values),
                'geometric_mean_decode_tps': geometric_mean([r['decode_tokens_per_second'] for r in values]),
                'median_ttft_seconds': median(r['ttft_seconds'] for r in values),
                'peak_footprint_gib': max(r['resources']['process_footprint_peak_bytes'] for r in values)/2**30,
                'minimum_available_gib': min(r['resources']['system_available_min_bytes'] for r in values)/2**30,
                'maximum_swap_increase_bytes': max(r['resources']['swap_peak_bytes']-r['resources']['swap_start_bytes'] for r in values)}
                for name, values in profiles.items()},
            'kernel_pair': paired_summary(selected, baseline, candidate)}
    short = [r for r in usable if r['input_tokens'] == 1024]
    report['short_context_comparisons'] = [paired_summary(short, a, b) for a, b in [
        (selection['best_stock'], candidate),
        (candidate, selection['best_native_head6']),
        (candidate, 'confidence-head6'),
        (selection['best_native_head6'], 'confidence-head6')]]
    report['coding_regressions'] = {}
    for stem in ['quality', 'ced-quality']:
        raw = read(stem+'-01.jsonl', True)
        measured = [r for r in raw if r.get('type') == 'run']
        if len(measured) != len(raw[0]['plan']['jobs']):
            raise ValueError(stem+' experiment is incomplete')
        grades = read(stem+'-grades-01.json')
        groups = defaultdict(list)
        for grade in grades:
            groups[grade['profile']['name']].append(grade)
        report['coding_regressions'][stem] = {
            name: {'tasks': len(values), 'passed': sum(v['passed'] for v in values),
                   'failed_tasks': [v['task'] for v in values if not v['passed']],
                   'scope': 'Eight local tasks with 160 hidden cases each; not a public benchmark.'}
            for name, values in groups.items()}
    events = [e for r in usable if r['profile']['name'] == 'confidence-head6' for e in r.get('events', [])]
    report['confidence_event_count'] = len(events)
    report['numerics'] = read('head-full-numerics-01.json')
    report['evidence_sha256'] = evidence
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'kernel_speedup': report['primary_kernel_pair']['paired_geometric_speedup'],
                      'identical_streams': report['primary_kernel_pair']['identical_token_streams'],
                      'total_pairs': report['primary_kernel_pair']['total_pairs'],
                      'coding_regressions': report['coding_regressions']}, indent=2))


if __name__ == '__main__':
    main()
