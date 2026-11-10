"""
Report for research methodology 2: the new runs against the TrackTacular paper, GRACE and the attempt-1 baselines.

    python research-methodology-2/code/m2_report.py --results <results>/m2

Primary protocol = the TrackTacular paper's own: last epoch, detection threshold 0.5, TrackTacular's tracker, nothing
tuned on the test frames (cell `thr0.5|tt`). The cell `thr*|tt` (threshold tuned on the test frames) is reported
separately as an upper bound.
"""
import argparse
import glob
import os.path as osp

import numpy as np
import pandas as pd
from scipy import stats

HERE = osp.dirname(osp.abspath(__file__))
METRICS = ['moda', 'modp', 'precision', 'recall', 'mota', 'idf1', 'motp']
PRIMARY = ['moda', 'mota', 'idf1']

# Published single runs on WILDTRACK, 7 cameras. Each number was read from the PDF in research-methodology-2/papers/.
PUBLISHED = [
    # TrackTacular, bilinear sampling: Tab. 1 (detection) and Tab. 2 (tracking) of 01_tracktacular_2403.12573.pdf
    dict(name='TrackTacular paper (bilinear sampling)', source='[P01] Tab. 1-2',
         moda=92.1, modp=76.2, precision=97.0, recall=95.1, idf1=95.3, mota=91.8, motp=85.4),
    # GRACE, 7 cameras, VGF + Ray + BTR: Tab. 1 of 02_grace_2609.16872.pdf
    dict(name='GRACE, 7 cameras', source='[P02] Tab. 1',
         moda=92.58, modp=80.33, precision=96.42, recall=96.15, idf1=94.53, mota=91.04, motp=86.26),
    # TrackTacular retrained by the GRACE authors, 7 cameras: Tab. 1 of 02_grace_2609.16872.pdf
    dict(name='TrackTacular retrained by GRACE authors', source='[P02] Tab. 1',
         moda=88.50, modp=78.56, precision=97.42, recall=91.39, idf1=94.00, mota=89.71, motp=84.90),
]
BASELINE_NAMES = {'base': 'attempt 1: retrained TrackTacular (E0)',
                  'A2': 'attempt 1: + scene aug (batch-order bug)',
                  'A3': 'attempt 1: + temporal history',
                  'A1A2A3': 'attempt 1: A1+A2+A3 (batch-order bug)'}
NEW_NAMES = {'M2': 'M2 (proposed): scene aug + VGF + rays + diversity prior',
             'M2c': 'M2c: as M2 with GRACE coverage prior',
             'A2f': 'A2f: scene aug only, bug fixed',
             'VD': 'VD: VGF + rays + diversity prior, no scene aug'}


def md_table(df):
    """Markdown table without the optional `tabulate` dependency of DataFrame.to_markdown."""
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in r) + ' |')
    return '\n'.join(out)


def fmt(x, s=None):
    return f'{x:.2f}' if s is None or np.isnan(s) else f'{x:.2f} ± {s:.2f}'


def summary_rows(df, names, kind):
    rows = []
    for variant, g in df.groupby('variant', sort=False):
        r = {'configuration': names.get(variant, variant), 'kind': kind, 'seeds': len(g),
             'seed list': ','.join(str(s) for s in sorted(g.seed))}
        for m in METRICS:
            r[m] = fmt(g[m].mean(), g[m].std() if len(g) > 1 else None)
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='<results>/m2')
    args = p.parse_args()

    files = glob.glob(osp.join(args.results, 'eval', '*', 'metrics.csv'))
    if not files:
        print('no evaluations yet')
        return
    new = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base = pd.read_csv(osp.join(HERE, 'baselines_attempt1.csv'))
    paper = PUBLISHED[0]
    lines = ['# Research methodology 2 — results', '',
             'Method: `research-methodology-2/METHODOLOGY.md`. All numbers in %, WILDTRACK test frames 360–399, '
             '952 ground-truth detections, so one error moves a metric by about 0.105 points.', '']

    for cell, title in (('thr0.5|tt', 'Primary protocol: threshold 0.5, TrackTacular tracker, nothing tuned on test'),
                        ('thr*|tt', 'Upper bound: detection threshold tuned on the test frames (oracle)')):
        rows = [dict(configuration=r['name'], kind=f'published {r["source"]}', seeds=1, **{
            'seed list': '-', **{m: fmt(r[m]) for m in METRICS}}) for r in PUBLISHED] if cell == 'thr0.5|tt' else []
        rows += summary_rows(base[base.cell == cell], BASELINE_NAMES, 'attempt 1')
        rows += summary_rows(new[new.cell == cell], NEW_NAMES, 'this run')
        lines += [f'## {title}', '', md_table(pd.DataFrame(rows)), '']

    lines += ['## Against the TrackTacular paper (primary protocol)', '']
    prim = new[new.cell == 'thr0.5|tt']
    for variant, g in prim.groupby('variant', sort=False):
        parts = []
        for m in PRIMARY:
            d = g[m].mean() - paper[m]
            test = ''
            if len(g) >= 2:
                test = f', one-sample t-test p = {stats.ttest_1samp(g[m], paper[m]).pvalue:.3f}'
            parts.append(f'{m.upper()} {g[m].mean():.2f} vs {paper[m]} ({d:+.2f}{test})')
        beats = [m.upper() for m in PRIMARY if g[m].mean() > paper[m]]
        lines.append(f'- **{NEW_NAMES.get(variant, variant)}** ({len(g)} seed(s)): ' + '; '.join(parts) +
                     f'. Above the paper on: {", ".join(beats) if beats else "none"}.')
    lines += ['', 'The paper reports single runs without seed variance. Attempt 1 measured a seed spread of '
              '0.3–1.5 points on these metrics, so a one-seed lead below about 1 point is not conclusive.', '']

    lines += ['## Same seed, paired against attempt 1 (primary protocol)', '']
    for variant, g in prim.groupby('variant', sort=False):
        for seed in sorted(g.seed):
            row = g[g.seed == seed].iloc[0]
            for b in ('base', 'A2'):
                ref = base[(base.cell == 'thr0.5|tt') & (base.variant == b) & (base.seed == seed)]
                if len(ref):
                    ref = ref.iloc[0]
                    lines.append(f'- {variant} s{seed} vs attempt-1 {b} s{seed}: ' + ', '.join(
                        f'{m.upper()} {row[m] - ref[m]:+.2f}' for m in PRIMARY) +
                        f' (FN {int(row.det_fn)} vs {int(ref.det_fn)}, FP {int(row.det_fp)} vs {int(ref.det_fp)})')
    lines += ['', '## Error counts (primary protocol)', '',
              prim[['variant', 'seed', 'det_fn', 'det_fp', 'id_switches', 'moda', 'mota', 'idf1']]
              .sort_values(['variant', 'seed']).pipe(md_table), '']

    text = '\n'.join(lines)
    open(osp.join(args.results, 'REPORT.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
