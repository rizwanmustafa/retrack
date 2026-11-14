"""
Research methodology 5 report: <results>/m5/eval/*/metrics.csv and summary.json -> <results>/m5/REPORT.md

  1. all cameras, per dataset: every variant (mean +- std over seeds when there are several), deltas vs B0
  2. camera removal: mean over the combinations of k removed cameras
  3. factorial effects of the 2^3 design: main effects of R1, R2, R4 and their interactions
"""
import argparse
import glob
import json
import os.path as osp

import numpy as np
import pandas as pd

ORDER = ['B0', 'R1', 'R2', 'R4', 'R12', 'R14', 'R24', 'R124']
FACTORS = {'R1': '1', 'R2': '2', 'R4': '4'}
METRICS = ['moda', 'modp', 'precision', 'recall', 'mota', 'idf1', 'id_switches', 'base_mota', 'base_idf1']


def has(variant, f):
    return variant != 'B0' and FACTORS[f] in variant[1:]


def fmt(m, s=None, n=1):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return '–'
    return f'{m:.2f}' if n <= 1 or s is None or np.isnan(s) else f'{m:.2f} ± {s:.2f}'


def effects(cell):
    """cell: {variant: value} for all 8 variants -> Yates effects (mean with - mean without)."""
    terms = [('R1',), ('R2',), ('R4',), ('R1', 'R2'), ('R1', 'R4'), ('R2', 'R4'), ('R1', 'R2', 'R4')]
    out = {}
    for t in terms:
        sign = lambda v: np.prod([1 if has(v, f) else -1 for f in t])
        out[' × '.join(t)] = sum(sign(v) * cell[v] for v in ORDER) / 4
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='<results>/m5')
    args = p.parse_args()
    files = sorted(glob.glob(osp.join(args.results, 'eval', '*', 'metrics.csv')))
    lines = ['# Research methodology 5 — results', '',
             'Method: `research-methodology-5/METHODOLOGY.md`. Test split: last 40 frames of each dataset; '
             'detection threshold 0.5; MODA/MODP at 0.5 m, MOTA/IDF1 at 1 m. One MODA point is about 9–10 errors '
             'on WILDTRACK\'s 40 test frames. `MOTA`/`IDF1` use each variant\'s own tracker (R4: track queries); '
             '`base` columns use TrackTacular\'s tracker on the same detections.', '']
    if not files:
        lines.append('No finished evaluations yet.')
        open(osp.join(args.results, 'REPORT.md'), 'w').write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    summ = {}
    for f in glob.glob(osp.join(args.results, 'eval', '*', 'summary.json')):
        s = json.load(open(f))
        summ[(s['dataset'], s['variant'], s['seed'])] = s

    for dataset in [d for d in ('wildtrack', 'multiviewx') if d in set(df['dataset'])]:
        d0 = df[(df['dataset'] == dataset) & (df['k'] == 0)]
        name = {'wildtrack': 'WILDTRACK', 'multiviewx': 'MultiviewX'}[dataset]
        lines += [f'## {name}: all cameras', '',
                  '| variant | R1 | R2 | R4 | seeds | MODA | ΔMODA | MODP | prec. | recall | MOTA | ΔMOTA | IDF1 | ΔIDF1 '
                  '| ID sw. | base MOTA | base IDF1 | trainable M | frozen M | ms/frame |',
                  '|' + '---|' * 20]
        b0 = d0[d0['variant'] == 'B0'][METRICS].mean() if (d0['variant'] == 'B0').any() else None
        cell = {}
        for v in ORDER:
            r = d0[d0['variant'] == v]
            if r.empty:
                continue
            m, s, n = r[METRICS].mean(), r[METRICS].std(), len(r)
            cell[v] = m
            ss = [summ.get((dataset, v, int(seed))) for seed in r['seed']]
            ss = [x for x in ss if x]
            tp = np.mean([x['trainable_params'] for x in ss]) / 1e6 if ss else np.nan
            fp = np.mean([x['frozen_params'] for x in ss]) / 1e6 if ss else np.nan
            ms = np.mean([x['ms_per_frame'] for x in ss]) if ss else np.nan
            dl = lambda k: '–' if b0 is None or v == 'B0' else f'{m[k] - b0[k]:+.2f}'
            lines.append(f'| {v} | {"✓" if has(v, "R1") else ""} | {"✓" if has(v, "R2") else ""} | '
                         f'{"✓" if has(v, "R4") else ""} | {n} | {fmt(m.moda, s.moda, n)} | {dl("moda")} | '
                         f'{fmt(m.modp)} | {fmt(m.precision)} | {fmt(m.recall)} | {fmt(m.mota, s.mota, n)} | '
                         f'{dl("mota")} | {fmt(m.idf1, s.idf1, n)} | {dl("idf1")} | {m.id_switches:.1f} | '
                         f'{fmt(m.base_mota)} | {fmt(m.base_idf1)} | {tp:.2f} | {fp:.2f} | {ms:.0f} |')
        lines.append('')

        dk = df[df['dataset'] == dataset]
        ks = sorted(dk['k'].unique())
        lines += [f'### {name}: cameras removed (mean over all combinations of k cameras)', '',
                  '| variant | ' + ' | '.join(f'MODA k={k}' for k in ks) + ' | ' +
                  ' | '.join(f'MOTA k={k}' for k in ks) + ' | ' + ' | '.join(f'IDF1 k={k}' for k in ks) + ' |',
                  '|' + '---|' * (1 + 3 * len(ks))]
        cell_all = {}
        for v in ORDER:
            r = dk[dk['variant'] == v]
            if r.empty:
                continue
            g = r.groupby('k')[['moda', 'mota', 'idf1']].mean()
            cell_all[v] = r[['moda', 'mota', 'idf1']].mean()
            lines.append(f'| {v} | ' + ' | '.join(f'{g.loc[k, m]:.2f}' if k in g.index else '–'
                                                   for m in ('moda', 'mota', 'idf1') for k in ks) + ' |')
        lines.append('')

        if all(v in cell for v in ORDER):
            lines += [f'### {name}: factorial effects (mean with − mean without; interactions as Yates contrasts)', '',
                      'Positive = the component (or the combination beyond the sum of its parts) helps. '
                      'With one seed these are single-run differences, not significance tests.', '',
                      '| term | MODA (all cams) | MOTA (all cams) | IDF1 (all cams) | MODA (all conditions) | '
                      'MOTA (all conditions) | IDF1 (all conditions) |', '|---|---|---|---|---|---|---|']
            eff = {m: effects({v: cell[v][m] for v in ORDER}) for m in ('moda', 'mota', 'idf1')}
            eff_all = {m: effects({v: cell_all[v][m] for v in ORDER}) for m in ('moda', 'mota', 'idf1')}
            for t in eff['moda']:
                lines.append(f'| {t} | ' + ' | '.join(f'{eff[m][t]:+.2f}' for m in ('moda', 'mota', 'idf1')) + ' | '
                             + ' | '.join(f'{eff_all[m][t]:+.2f}' for m in ('moda', 'mota', 'idf1')) + ' |')
            lines.append('')
        else:
            missing = [v for v in ORDER if v not in cell]
            lines += [f'Factorial effects need all 8 variants; missing: {", ".join(missing)}.', '']

    lines += ['## Caveats', '',
              '- Single seed unless the table says otherwise; differences below about 1 MODA point are within the '
              'seed-to-seed spread seen in the earlier methodologies.',
              '- R4 is trained with jittered previous-frame ground truth as tracks (not end-to-end over sequences).',
              '- ms/frame is the median forward time on this GPU with all cameras, batch 1 (decoding and tracking '
              'not included); it changes with GPU load from parallel runs.', '']
    open(osp.join(args.results, 'REPORT.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
