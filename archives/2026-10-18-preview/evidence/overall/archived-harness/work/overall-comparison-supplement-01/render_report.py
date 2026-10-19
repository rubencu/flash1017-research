"""Render a local review report from completed independently audited observations."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--supplement', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    a = json.loads(args.audit.read_text())
    s = json.loads(args.supplement.read_text())
    if s['audit_sha256'] != hashlib.sha256(args.audit.read_bytes()).hexdigest():
        raise ValueError('Supplement and independent audit do not match')
    if not s['prefill_ttft']['qualified_separately']:
        raise ValueError('Require a completed, structurally valid comparison and restoration')
    arms = {(r['arm'], r['group']): r['geometric_means'] for r in s['per_arm_group_summaries']}
    pairs = {(r['baseline'], r['group']): r['metrics'] for r in s['pair_group_summaries']}
    prefix_summary = ('Post-hoc time to token 128 fell '
        f'{pairs[("native2048", "all")]["prefix128_delivery_seconds"]["improvement_percent"]:.2f}%, using all 48 requests.'
        if s['prefix128']['available_for_all48'] else 'No valid all-48 first-128 supplement is available.')
    lines = [
        'DeepSeek-V4.1-Flash-1017: complete runtime versus installed oMLX',
        'Local review copy. No publication or upstream submission made.', '',
        'RESULT',
        f'Across this fixed suite, time to first token fell {pairs[("native2048", "all")]["ttft_seconds"]["improvement_percent"]:.2f}% versus stock 2K:',
        f'{arms[("native2048", "all")]["ttft_seconds"]:.3f}s to {arms[("complete", "all")]["ttft_seconds"]:.3f}s (geometric means).',
        'The complete package improves the aggregate, with workload-dependent gains.',
        f'Original full-512 useful-answer qualification: {"passed" if a["performance_claim_qualified"] else "failed"};',
        f'{s["forced512"]["early_eos_count"]} requests reached EOS early. Every observation is retained.',
        prefix_summary,
        'This is a warm-runtime performance result; answer-quality equivalence is not established.', '',
        'WHAT WAS COMPARED',
        'Same M5 Ultra Mac Studio, 80 GPU cores, 256 GiB memory; same downloaded model files.',
        'Model: DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp, mlx-community revision',
        '78456115d301eecf9fadcdcc81c0cf6f8b2c8eab. No new quantization or weight changes.',
        'Stock 2K: untouched installed oMLX 0.7.1rc1, native adaptive MTP max depth 3,',
        '2048-token prefill chunks, native 16 MiB Engram prefetch budget.',
        'Stock 8K: the same untouched runtime and MTP settings, with 8192-token chunks.',
        'Complete: the full qualified local runtime, with all existing accepted changes:',
        '  - BF16 output-head reuse for multiple rows, up to six rows;',
        '  - reference attention rounding after inverse RoPE, in target and draft;',
        '  - confidence-based speculative prefix selection, draft width five;',
        '  - 8192-token chunks and a 64 MiB Engram prefetch budget;',
        '  - repaired dependency replay and arithmetic-shape preservation;',
        '  - scoped HotRows integration, 256 MiB per table / 512 MiB total retention;',
        '  - model ownership and cleanup changes.',
        'This measures the whole package, including its configuration. It does not assign',
        'causal credit to individual patches. Historical component gains are not multiplied.',
        'The denominator is the original installed version, NOT current upstream main.', '',
        'METHOD',
        '48 measured requests: three configurations x two fresh processes x eight cases.',
        'Six additional warmups were excluded. The second block reversed configuration',
        'and case order. Per-case seeds matched across configurations within each block.',
        'Inputs: two 1K synthetic code/prose cases; four 16K natural-source cases;',
        'two 64K natural-source cases. Natural sources: CPython, curl, Moby-Dick,',
        'and Pride and Prejudice. Exact input token IDs were frozen before execution.',
        'Each measured request asked the generator for 512 tokens with stopping disabled,',
        'temperature 1, top-p 1, and thinking disabled. Native MTP was active in both',
        'stock configurations. All three used the same embedded Python and dependencies.',
        'Common memory policy: 8 GiB MLX allocator cache; a 224 GiB wired limit and',
        'allocation guideline, 32 GiB below physical memory. This is not a hard OS cap.',
        'The same per-shard loading headroom checks applied to all configurations.',
        'Timings start before prefill and exclude model loading and HTTP transport.',
        'Every request had fresh conversation state. Models stayed loaded within each',
        'process. OS caches were not purged. Candidate row caches were cleared after',
        'warmup and retained through the eight measured requests, as declared in advance.',
        'Aggregate numbers are geometric means over matched cases, with equal case',
        'weights. Whiskers in figures show the two block means, not confidence intervals.', '',
        'QUALIFICATION',
        f'All 48 requests completed; independent structural issues: {len(a["structural_issues"])}.',
        f'Original service restoration confirmed: {a["restoration_confirmed"]}.',
        f'Original fixed-512 response qualification: {a["performance_claim_qualified"]}.',
        f'Early end-of-answer tokens occurred in {s["forced512"]["early_eos_count"]} of 48 requests.',
        'These observations are retained. The frozen acceptance rule is unchanged.',
        'Prefill and first-token timings qualify separately and contain no post-EOS continuation.',
        'Full-512 timing/rate numbers below are descriptive forced-continuation measurements,',
        'including post-EOS work. They are not validated useful-answer completion times.',
    ]
    if s['prefix128']['available_for_all48']:
        lines += ['A supplementary first-128-token analysis is available for all 48 requests.',
                  'It was chosen after observing early EOS and is explicitly POST HOC.',
                  'Every measured prefix precedes EOS. Its endpoint is the saved delivery',
                  'timestamp of token 128 within the ORIGINAL 512-token execution.',
                  'It does not simulate a new max_tokens=128 request or include terminal',
                  'synchronization/cleanup at 128. Speculative batches may compute further',
                  'tokens before delivery; these are delivery observations, not kernel timings.']
    else:
        lines += ['The all-48 first-128-token supplement is unavailable; no cases were removed',
                  'and no shorter horizon was selected to rescue it.']

    def table(metric, title, suffix):
        lines.extend(['', title, 'Group             Stock 2K     Stock 8K     Complete     Change vs 2K    Change vs 8K'])
        for group in ['all', '1K', '16K', '64K', 'natural', 'synthetic', 'block0', 'block1']:
            values = [arms[(arm, group)][metric] for arm in ['native2048', 'native8192', 'complete']]
            changes = [pairs[(base, group)][metric]['change_percent'] for base in ['native2048', 'native8192']]
            lines.append(f'{group:14s} {values[0]:9.3f}{suffix} {values[1]:9.3f}{suffix} {values[2]:9.3f}{suffix} {changes[0]:13.2f}% {changes[1]:14.2f}%')
        lines.append('Negative change is less time; positive change is a higher rate.')

    table('prefill_seconds', 'PROMPT PROCESSING — qualified separately', 's')
    table('ttft_seconds', 'TIME TO FIRST TOKEN — qualified separately', 's')
    if s['prefix128']['available_for_all48']:
        table('prefix128_delivery_seconds', 'TIME TO TOKEN 128 — post-hoc observed delivery in the original runs', 's')
        table('prefix128_post_first_tokens_per_second', 'DELIVERY RATE, TOKENS 2–128 — post hoc; tokens per second', ' ')
    table('total_seconds', 'FORCED 512-TOKEN ELAPSED TIME — descriptive, includes post-EOS work', 's')
    table('decode_tokens_per_second', 'FORCED 512-TOKEN DECODE RATE — descriptive; tokens per second', ' ')
    lines += ['', 'PER-CASE FIRST-TOKEN TIMES — geometric means of the two blocks',
              'Case                                  Stock 2K    Stock 8K    Complete']
    by_case = {(r['baseline'], r['fixture']): r for r in s['per_case_two_block_summaries']}
    for fixture in sorted({r['fixture'] for r in s['per_case_two_block_summaries']}):
        x = by_case[('native2048', fixture)]['metrics']['ttft_seconds']
        y = by_case[('native8192', fixture)]['metrics']['ttft_seconds']
        lines.append(f'{fixture:38s} {x["baseline_geometric_mean"]:9.3f}s {y["baseline_geometric_mean"]:9.3f}s {x["candidate_geometric_mean"]:9.3f}s')
    if s['prefix128']['available_for_all48']:
        lines += ['', 'PER-CASE TIME TO TOKEN 128 — post hoc; geometric means of both blocks',
                  'Case                                  Stock 2K    Stock 8K    Complete    Change vs 2K']
        for fixture in sorted({r['fixture'] for r in s['per_case_two_block_summaries']}):
            x = by_case[('native2048', fixture)]['metrics']['prefix128_delivery_seconds']
            y = by_case[('native8192', fixture)]['metrics']['prefix128_delivery_seconds']
            lines.append(f'{fixture:38s} {x["baseline_geometric_mean"]:9.3f}s {y["baseline_geometric_mean"]:9.3f}s {x["candidate_geometric_mean"]:9.3f}s {x["change_percent"]:13.2f}%')
    lines += ['', 'RESOURCES — measured requests; sampled peaks can miss brief transients']
    for r in a['resource_summary']:
        lines.append(f'{r["arm"]}: footprint peak {r["max_process_footprint_gib"]:.2f} GiB; minimum available {r["minimum_system_available_gib"]:.2f} GiB; maximum within-request swap growth {r["max_request_swap_growth_gib"]:.3f} GiB; summed process disk reads {r["process_disk_read_gib_sum"]:.2f} GiB.')
    lines += ['Disk-read counters cover the whole process request, not exclusively Engram',
              'or a claim about physical SSD traffic. Cache history is part of this procedure.',
              '', 'EARLY EOS — zero-based positions; all observations retained']
    for r in a['early_eos_records']:
        lines.append(json.dumps(r, sort_keys=True))
    lines += ['', 'LIMITS',
              'Outputs can differ. This performance comparison does not establish answer',
              'quality equivalence, correctness on a task suite, or faster equivalent work.',
              'Two process blocks on one machine are not a confidence interval or evidence',
              'for other hardware, checkpoints, concurrency levels, cold starts, or tail latency.',
              'This is an existing local-runtime comparison, not portable-release qualification.',
              'Current upstream main and its changed native build layout were not benchmarked.',
              'No additional optimization experiments were introduced for this comparison.',
              '', 'EVIDENCE',
              f'Independent audit: {args.audit.resolve()}',
              f'Supplement: {args.supplement.resolve()}',
              f'Frozen executable plan SHA256: {s["plan_sha256"]}',
              f'Frozen source manifest SHA256: {a["source_digests"]["manifest"]}',
              'Raw per-request tokens, timing traces, activation checks and cleanup records:']
    lines += [f'{r["path"]}\n  SHA256 {r["sha256"]}' for r in s['raw_bindings']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
