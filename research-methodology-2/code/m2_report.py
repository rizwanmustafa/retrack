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
             'VD': 'VD: VGF + rays + diversity prior, no scene aug',
             'M2cx': 'M2cx: M2c + view mixing (old batch order on purpose)',
             'A2x': 'A2x: SA-VM, scene aug + view mixing of cameras 1,3,5 (deliberate)',
             'RVM': 'RVM: scene aug + random view mixing (p = 3/7 per camera)',
             'CD': 'CD: scene aug + camera dropout (control)',
             'FGx': 'FGx: SA-VM + foreground-gated lifting (methodology 4)',
             'FG0': 'FG0: foreground-gated lifting alone (methodology 4)',
             'E0c': 'E0c: retrained TrackTacular, current code'}
ROBUST_NAMES = {'attempt1_E0': 'attempt 1: retrained TrackTacular (E0)', 'attempt1_A2': 'attempt 1: SA-VM (A2)'}
CONDITIONS = ['clean', 'tta', 'calib_low', 'calib_high', 'drop1', 'drop2']


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


def robust_section(root):
    """Methodology 3: clean, mirror TTA, calibration noise and camera failures (evaluation/robust_eval.py)."""
    files = glob.glob(osp.join(root, 'robust', '*.csv'))
    if not files:
        return []
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    per_ckpt = r.groupby(['variant', 'seed', 'condition'], as_index=False)[['moda', 'mota', 'idf1']].mean()
    names = {**NEW_NAMES, **ROBUST_NAMES}
    out = ['## Methodology 3: test-time mirrors and robustness (protocol thr0.5|tt)', '',
           'Mean ± std over seeds. calib = Gaussian extrinsic noise per camera (low 0.5 deg / 0.05 m, high 2.0 deg / '
           '0.2 m; ATLASFusion Tab. 5 magnitudes), 3 draws, identical for every model. drop1 = each single camera off, '
           'drop2 = 7 camera pairs off.', '']
    for m in PRIMARY:
        rows = []
        for variant, g in per_ckpt.groupby('variant', sort=True):
            row = {'configuration': names.get(variant, variant), 'seeds': g.seed.nunique()}
            for c in CONDITIONS:
                v = g[g.condition == c][m]
                row[c] = fmt(v.mean(), v.std() if len(v) > 1 else None) if len(v) else '-'
            rows.append(row)
        out += [f'### {m.upper()}', '', md_table(pd.DataFrame(rows)), '']
    # paired by seed: every variant against attempt-1 E0, per condition
    ref = per_ckpt[per_ckpt.variant == 'attempt1_E0']
    if len(ref):
        out += ['### Paired against attempt-1 E0, same seed (mean difference over common seeds)', '']
        rows = []
        for variant, g in per_ckpt[per_ckpt.variant != 'attempt1_E0'].groupby('variant', sort=True):
            j = g.merge(ref, on=['seed', 'condition'], suffixes=('', '_e0'))
            if not len(j):
                continue
            row = {'configuration': names.get(variant, variant), 'seeds': j.seed.nunique()}
            for c in CONDITIONS:
                jc = j[j.condition == c]
                row[c] = ' / '.join(f'{(jc[m] - jc[m + "_e0"]).mean():+.2f}' for m in PRIMARY) if len(jc) else '-'
            rows.append(row)
        out += ['Cells: MODA / MOTA / IDF1 difference.', '', md_table(pd.DataFrame(rows)), '']
    return out


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
    lines += ['', '## Ablations, same seed (primary protocol, METHODOLOGY.md §4)', '']
    for a, b, h in (('M2', 'M2c', 'H3: view-diversity prior vs GRACE coverage prior'),
                    ('M2', 'A2f', 'H4: VGF + rays on top of scene augmentation'),
                    ('M2', 'VD', 'share of scene augmentation'),
                    ('M2c', 'A2f', 'VGF + rays (coverage prior) on top of scene augmentation'),
                    ('M2cx', 'M2c', 'view mixing on top of M2c'),
                    ('A2x', 'A2f', 'H5: deliberate view mixing vs none'),
                    ('RVM', 'A2x', 'H7: random vs fixed view mixing'),
                    ('RVM', 'A2f', 'random view mixing vs none'),
                    ('CD', 'A2f', 'camera dropout vs none'),
                    ('A2x', 'CD', 'H11: view mixing vs camera dropout'),
                    ('RVM', 'CD', 'H11: random view mixing vs camera dropout'),
                    ('FGx', 'A2x', 'M4-H1: foreground gating on top of SA-VM'),
                    ('FG0', 'E0c', 'M4-H2: foreground gating on the plain baseline'),
                    ('A2x', 'E0c', 'SA-VM vs baseline, both in the current code')):
        for seed in sorted(set(prim[prim.variant == a].seed) & set(prim[prim.variant == b].seed)):
            ra = prim[(prim.variant == a) & (prim.seed == seed)].iloc[0]
            rb = prim[(prim.variant == b) & (prim.seed == seed)].iloc[0]
            lines.append(f'- {h}: {a} - {b}, s{seed}: ' + ', '.join(
                f'{m.upper()} {ra[m] - rb[m]:+.2f}' for m in PRIMARY) +
                f' (FN {int(ra.det_fn)} vs {int(rb.det_fn)}, FP {int(ra.det_fp)} vs {int(rb.det_fp)})')
    lines += ['', '## Error counts (primary protocol)', '',
              prim[['variant', 'seed', 'det_fn', 'det_fp', 'trk_misses', 'trk_fp', 'id_switches', 'moda', 'mota', 'idf1']]
              .sort_values(['variant', 'seed']).pipe(md_table), '']

    lines += robust_section(args.results)
    text = '\n'.join(lines)
    open(osp.join(args.results, 'REPORT.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
