#!/usr/bin/env python3
"""Compare matched limits within each fixed prefill geometry."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
plan_bytes = (HERE / 'plan.json').read_bytes()
plan = json.loads(plan_bytes)
pairs, sources = [], {}
for block in (0, 1):
    arms = {}
    for arm in ('baseline', 'candidate'):
        file = HERE / f'{arm}-{block}.jsonl'
        data = file.read_bytes()
        sources[file.name] = hashlib.sha256(data).hexdigest()
        rows = [json.loads(line) for line in data.splitlines()]
        assert rows[0]['plan_sha256'] == hashlib.sha256(plan_bytes).hexdigest()
        assert rows[-1] == {'type': 'closed', 'pending': False}
        arms[arm] = {x['id']: x for x in rows if x['type'] == 'run'}
        assert len(arms[arm]) == len(plan['prompts'])
    for prompt in plan['prompts']:
        b, c = (arms[a][prompt['id']] for a in ('baseline', 'candidate'))
        assert b['input_sha256'] == c['input_sha256'] == prompt['sha256']
        assert len(b['token_ids']) == len(c['token_ids']) == plan['decode_tokens']
        assert b['prefill_step'] == c['prefill_step'] == prompt['prefill_step']
        assert b['io']['submit_calls'] == c['io']['submit_calls']
        if prompt['prefill_step'] == 2048:
            assert b['io']['admitted'] == c['io']['admitted'] == b['io']['submit_calls']
            assert not b['io'].get('rejected') and not c['io'].get('rejected')
        else:
            assert b['io']['foreground_over16_calls'] == c['io']['background_over16_calls'] == 8
            assert b['io']['rejected'] == 8 and not c['io'].get('rejected')
        pairs.append({
            'block': block, 'id': prompt['id'], 'prefill_step': prompt['prefill_step'],
            'tokens_equal': b['token_ids'] == c['token_ids'],
            'output_tokens': len(b['token_ids']),
            'baseline_total_seconds': b['total_seconds'], 'candidate_total_seconds': c['total_seconds'],
            'total_time_ratio': c['total_seconds'] / b['total_seconds'],
            'ttft_ratio': c['ttft_seconds'] / b['ttft_seconds'],
            'decode_throughput_ratio': c['decode_tps'] / b['decode_tps'],
            'baseline_ttft_seconds': b['ttft_seconds'], 'candidate_ttft_seconds': c['ttft_seconds'],
            'baseline_decode_tps': b['decode_tps'], 'candidate_decode_tps': c['decode_tps'],
            'peak_footprint_delta_bytes': c['resources']['process_footprint_peak_bytes'] - b['resources']['process_footprint_peak_bytes'],
            'baseline_io': b['io'], 'candidate_io': c['io'],
            'baseline_disk_read_bytes': b['resources'].get('disk_read_bytes_delta'),
            'candidate_disk_read_bytes': c['resources'].get('disk_read_bytes_delta'),
            'baseline_swap_growth_bytes': b['resources']['swap_peak_bytes'] - b['resources']['swap_start_bytes'],
            'candidate_swap_growth_bytes': c['resources']['swap_peak_bytes'] - c['resources']['swap_start_bytes'],
        })
summary = {'pairs': pairs, 'source_sha256': sources, 'groups': {}}
for step in (2048, 8192):
    group = [x for x in pairs if x['prefill_step'] == step]
    gmean = lambda key: math.exp(sum(math.log(x[key]) for x in group) / len(group))
    summary['groups'][str(step)] = {
        'paired_requests': len(group),
        'equal_output_pairs': sum(x['tokens_equal'] for x in group),
        'compared_output_tokens': sum(x['output_tokens'] for x in group),
        'total_time_reduction_percent': 100 * (1 - gmean('total_time_ratio')),
        'ttft_reduction_percent': 100 * (1 - gmean('ttft_ratio')),
        'decode_throughput_gain_percent': 100 * (gmean('decode_throughput_ratio') - 1),
        'faster_total_pairs': sum(x['total_time_ratio'] < 1 for x in group),
        'peak_footprint_delta_range_MiB': [min(x['peak_footprint_delta_bytes'] for x in group) / 2**20, max(x['peak_footprint_delta_bytes'] for x in group) / 2**20],
    }
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['groups'], indent=2))
