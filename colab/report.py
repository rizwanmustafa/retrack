"""
REPORT.md (and figures/) from the evaluations in <results>/eval/<dataset>_<variant>_s<seed>/.

    python colab/report.py --results /content/drive/MyDrive/retrack_results_r2

Per dataset: all-camera results (mean ± std over seeds), MODA / MOTA / IDF1 with k cameras removed (mean over every
combination of k cameras, then over seeds), R2 - B0 paired by seed, and a robustness figure.
"""
import argparse
import glob
import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd

ORDER = ['B0', 'R2']
NAMES = {'B0': 'B0: TrackTacular (dense BEV decoder)', 'R2': 'R2: sparse person queries'}
COLS = [('moda', 'MODA'), ('modp', 'MODP'), ('precision', 'prec.'), ('recall', 'recall'), ('mota', 'MOTA'),
        ('idf1', 'IDF1'), ('id_switches', 'ID sw.'), ('det_fp', 'FP'), ('det_fn', 'FN')]


def load(root):
    rows, summaries = [], []
    for path in sorted(glob.glob(osp.join(root, 'eval', '*', 'metrics.csv'))):
        name = osp.basename(osp.dirname(path))
        m = re.match(r'(\w+?)_(B0|R2)_s(\d+)$', name)
        if not m:
            continue
        dataset, variant, seed = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(path)
        df['dataset'], df['variant'], df['seed'] = dataset, variant, seed
        rows.append(df)
        s = osp.join(osp.dirname(path), 'summary.json')
        if osp.exists(s):
            j = json.load(open(s))
            summaries.append(dict(dataset=dataset, variant=variant, seed=seed, params=j.get('params_M'),
                                  ms=j.get('ms_per_frame')))
    if not rows:
        return None, None
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def fmt(vals, digits=2):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return '–'
    if len(vals) == 1:
        return f'{vals[0]:.{digits}f}'
    return f'{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}'


def figure(df, dataset, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, label) in zip(axes, [('moda', 'MODA'), ('mota', 'MOTA'), ('idf1', 'IDF1')]):
        for variant in ORDER:
            d = df[df.variant == variant]
            if d.empty:
                continue
            per_seed = d.groupby(['seed', 'k'])[key].mean().unstack('k')  # seeds x k
            ks = per_seed.columns.values
            mean, std = per_seed.mean().values, per_seed.std(ddof=1).fillna(0).values
            ax.plot(ks, mean, marker='o', label=f'{variant} ({len(per_seed)} seed{"s" if len(per_seed) > 1 else ""})')
            ax.fill_between(ks, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel('cameras removed (k)')
        ax.set_title(label)
        ax.set_xticks(sorted(df.k.unique()))
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f'{dataset}: mean over every combination of k removed cameras (band: ± 1 std over seeds)')
    fig.tight_layout()
    path = osp.join(out_dir, f'robustness_{dataset}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    args = p.parse_args()
    df, sm = load(args.results)
    out = [f'# TrackTacular vs R2 — results\n',
           'Test split: last 40 frames. Detection threshold 0.5; MODA/MODP at 0.5 m; MOTA/IDF1 at 1 m with '
           "TrackTacular's tracker for both variants. One MODA point is about 9–10 errors on WILDTRACK and about 15 "
           'on MultiviewX. With several seeds: mean ± sample std.\n']
    if df is None:
        out.append('No evaluations found yet.\n')
    else:
        fig_dir = osp.join(args.results, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        for dataset in sorted(df.dataset.unique()):
            d = df[df.dataset == dataset]
            allc = d[d.k == 0]
            out.append(f'\n## {dataset}: all cameras\n')
            out.append('| variant | seeds | ' + ' | '.join(c[1] for c in COLS) + ' | params M | ms/frame |')
            out.append('|---|---|' + '---|' * (len(COLS) + 2))
            for v in ORDER:
                a = allc[allc.variant == v]
                if a.empty:
                    continue
                s = sm[(sm.dataset == dataset) & (sm.variant == v)] if sm is not None and len(sm) else pd.DataFrame()
                cells = [fmt(a[c].values, 0 if c in ('id_switches', 'det_fp', 'det_fn') else 2) for c, _ in COLS]
                params = fmt(s.params.values) if len(s) else '–'
                ms = fmt(s.ms.values, 0) if len(s) else '–'
                out.append(f'| {NAMES[v]} | {", ".join(map(str, sorted(a.seed)))} | ' + ' | '.join(cells) +
                           f' | {params} | {ms} |')

            out.append(f'\n### {dataset}: cameras removed\n')
            out.append('Mean over every combination of k removed cameras (and over seeds).\n')
            ks = sorted(d.k.unique())
            head = [f'{m} k={k}' for m in ('MODA', 'MOTA', 'IDF1') for k in ks]
            out.append('| variant | ' + ' | '.join(head) + ' |')
            out.append('|---|' + '---|' * len(head))
            for v in ORDER:
                dv = d[d.variant == v]
                if dv.empty:
                    continue
                per = dv.groupby(['seed', 'k'])[['moda', 'mota', 'idf1']].mean().groupby('k').mean()
                out.append(f'| {v} | ' + ' | '.join(f'{per.loc[k, m]:.2f}' for m in ('moda', 'mota', 'idf1')
                                                    for k in ks) + ' |')

            both = sorted(set(d[d.variant == 'B0'].seed) & set(d[d.variant == 'R2'].seed))
            if both:
                out.append(f'\n### {dataset}: R2 − B0, paired by seed ({", ".join(map(str, both))})\n')
                out.append('| k | ΔMODA | ΔMOTA | ΔIDF1 | ΔFP | ΔFN |')
                out.append('|---|---|---|---|---|---|')
                g = d[d.seed.isin(both)].groupby(['variant', 'seed', 'k'])[
                    ['moda', 'mota', 'idf1', 'det_fp', 'det_fn']].mean()
                for k in ks:
                    diff = [g.loc[('R2', s, k)] - g.loc[('B0', s, k)] for s in both]
                    cells = [fmt([x[c] for x in diff], 1 if c.startswith('det') else 2)
                             for c in ('moda', 'mota', 'idf1', 'det_fp', 'det_fn')]
                    out.append(f'| {k} | ' + ' | '.join(cells) + ' |')
                if len(both) < 3:
                    out.append('\nFewer than 3 seeds: differences below about 1 point are within seed noise.')
            path = figure(d, dataset, fig_dir)
            out.append(f'\n![{dataset} robustness](figures/{osp.basename(path)})\n')
    with open(osp.join(args.results, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'-> {osp.join(args.results, "REPORT.md")}')


if __name__ == '__main__':
    main()
