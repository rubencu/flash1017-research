"""Render the frozen, measured results; requires matplotlib, not MLX."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/'results/validation-summary-01.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#344256', 'text.color': '#172c42',
                         'xtick.color': '#344256', 'ytick.color': '#344256',
                         'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.8),
                                     gridspec_kw={'width_ratios': [1, 1.1]})
    fig.subplots_adjust(top=.71, bottom=.23, left=.08, right=.97, wspace=.32)
    fig.text(.08, .94, 'DeepSeek-V4.1-Flash-1017', fontsize=22, weight='bold')
    fig.text(.08, .88, '7.3% faster token generation with the same measured outputs',
             fontsize=14, color='#007f74')
    fig.text(.08, .81, 'M5 Ultra · 80 GPU cores · 256 GiB · oQ2.5e-mtp · one request at a time',
             fontsize=10, color='#607184')
    colors = ['#bcc9d6', '#007f74']
    for i, (name, label) in enumerate([('reference-f2', 'Original projection'),
                                     ('reference-f2-head', 'Weight-reuse projection')]):
        values = [report['by_context'][c]['profiles'][name]['geometric_mean_decode_tps']
                  for c in ['1024', '16384']]
        positions = [x+(i-.5)*.34 for x in range(2)]
        bars = left.bar(positions, values, width=.32, color=colors[i], label=label)
        left.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
    left.set(xticks=[0, 1], xticklabels=['1K input', '16K input'],
             ylabel='Generated tokens / second', ylim=(0, 46))
    left.grid(axis='y', alpha=.15)
    left.set_axisbelow(True)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.075, .77),
               frameon=False, fontsize=9, ncol=2)
    for y, context in enumerate(['16384', '1024']):
        pair = report['by_context'][context]['kernel_pair']
        ratios = [p['ratio']-1 for p in pair['pairs']]
        jitter = [(i-4.5)*.028 for i in range(len(ratios))]
        right.scatter(ratios, [y+j for j in jitter], s=27, color='#94a9b9', alpha=.8)
        mean = pair['paired_geometric_speedup']-1
        low, high = [x-1 for x in pair['bootstrap_95_percent_interval']]
        right.errorbar(mean, y, xerr=[[mean-low], [high-mean]],
                       color='#007f74', fmt='D', markersize=8, capsize=5, lw=2)
        right.text(mean, y+.23, f'+{mean:.2%}', ha='center', color='#007f74', weight='bold')
    right.axvline(0, color='#bcc9d6', lw=1)
    right.set(yticks=[0, 1], yticklabels=['16K input', '1K input'],
              xlabel='Paired generation-rate improvement', xlim=(-.01, .19), ylim=(-.45, 1.55))
    right.xaxis.set_major_formatter(PercentFormatter(1))
    right.grid(axis='x', alpha=.15)
    fig.text(.08, .045,
             '20 matched pairs · 512 output tokens each · all 10,240 compared tokens identical\n'
             'Same reference arithmetic and fixed draft depth 2. Fresh KV; warmed OS/kernel caches.\n'
             'Dots: individual pairs. Diamonds: geometric means. Bars: simple paired-bootstrap 95% intervals.',
             fontsize=9, color='#607184', linespacing=1.6)
    destination = root/'figures'
    destination.mkdir(exist_ok=True)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(destination/f'kernel-heldout.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
