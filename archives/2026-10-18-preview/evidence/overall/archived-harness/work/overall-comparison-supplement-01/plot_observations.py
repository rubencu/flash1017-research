"""Plot qualified observations without changing the original EOS acceptance rule."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def gm(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--supplement', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.supplement.read_text())
    if not report['prefill_ttft']['qualified_separately']:
        raise RuntimeError('Require the complete independent audit and separate TTFT qualification')
    rows = []
    for source in report['raw_bindings']:
        data = Path(source['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise RuntimeError('Raw data changed since supplementary audit')
        rows.extend(r for r in map(json.loads, data.splitlines()) if r['type'] == 'run' and not r['warmup'])
    if len(rows) != 48:
        raise RuntimeError('All 48 measured requests are required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    arms = ['native2048', 'native8192', 'complete']
    labels = ['Stock · 2K chunks', 'Stock · 8K chunks', 'Complete runtime']
    colors = ['#8C98A5', '#547F9C', '#187B69']
    charts = [('first-token', 'Time to the first token', lambda r: r['ttft_seconds'],
               'All 48 measurements retained. Early answer endings do not affect this endpoint.')]
    if report['prefix128']['available_for_all48']:
        charts.append(('first-128-tokens', 'Time to the first 128 tokens', lambda r: r['arrival_seconds'][127],
                       'Post-hoc delivery measurement within the original 512-token runs; every prefix precedes EOS.'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, title, metric, caveat in charts:
        for ext in ['png', 'pdf', 'json']:
            if (args.output_dir / f'{name}.{ext}').exists():
                raise FileExistsError('Refusing to replace an existing figure')
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharey=True)
        plotted = {'title': title, 'supplement_sha256': hashlib.sha256(args.supplement.read_bytes()).hexdigest(),
                   'caveat': caveat, 'scope': 'Geometric means; bars span the two block means, not confidence intervals', 'groups': []}
        for ax, size, cohort in zip(axes, [1024, 16384, 65536], ['1K · code / prose', '16K · four natural sources', '64K · two natural sources']):
            subsets = [[r for r in rows if r['input_tokens'] == size and r['arm'] == arm] for arm in arms]
            means = [gm([metric(r) for r in subset]) for subset in subsets]
            blocks = [[gm([metric(r) for r in subset if r['block'] == block]) for block in [0, 1]] for subset in subsets]
            low = [max(0, m-min(b)) for m, b in zip(means, blocks)]
            high = [max(0, max(b)-m) for m, b in zip(means, blocks)]
            ax.barh(range(3), means, height=.55, color=colors, xerr=[low, high],
                    error_kw={'ecolor': '#28323B', 'capsize': 3, 'elinewidth': 1})
            maximum = max(max(b) for b in blocks)
            for y, m in enumerate(means):
                ax.text(max(m, max(blocks[y])) + .035*maximum, y, f'{m:.2f}s' if m < 10 else f'{m:.1f}s', va='center', fontsize=11)
            ax.set_xlim(0, maximum*1.31)
            ax.set_yticks(range(3), labels)
            ax.set_title(cohort, fontsize=11, pad=14)
            ax.set_xlabel('Seconds · lower is better')
            ax.grid(axis='x', alpha=.15)
            ax.set_axisbelow(True)
            plotted['groups'].append({'input_tokens': size, 'arms': [
                {'arm': a, 'requests': len(s), 'geomean_seconds': m, 'block_means_seconds': b}
                for a, s, m, b in zip(arms, subsets, means, blocks)]})
        axes[0].invert_yaxis()
        fig.suptitle(title, x=.045, y=.97, ha='left', fontsize=21, fontweight='bold')
        fig.text(.045, .88, 'All Flash1017 changes together vs untouched installed oMLX 0.7.1rc1 · native speculative decoding enabled', fontsize=10.5)
        fig.text(.045, .16, caveat, fontsize=9)
        fig.text(.045, .11, 'M5 Ultra · 80 GPU cores · 256 GiB · same oQ2.5e-mtp checkpoint · two fresh processes per configuration', fontsize=9)
        fig.text(.045, .06, 'Warm-model timings exclude loading and HTTP. Cache history persists as documented. Whiskers show block means, not confidence intervals.', fontsize=8)
        fig.subplots_adjust(left=.17, right=.985, top=.75, bottom=.30, wspace=.31)
        for ext in ['png', 'pdf']:
            fig.savefig(args.output_dir / f'{name}.{ext}', dpi=180, facecolor='white')
        (args.output_dir / f'{name}.json').write_text(json.dumps(plotted, indent=2) + '\n')
        plt.close(fig)
        print(name)


if __name__ == '__main__':
    main()
