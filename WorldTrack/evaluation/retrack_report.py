"""
Builds the Retrack comparison report from every results/eval/*/metrics.csv.

    python -m evaluation.retrack_report --results /content/drive/MyDrive/retrack_results

Writes results/report/REPORT.md, one CSV per table and the robustness plots, and prints the main tables.
"""
import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

VARIANTS = ['concat', 'concat_drop', 'masked_drop', 'masked_cov_drop', 'weighted_cov_drop']
VARIANT_NAMES = {
    'official': 'Official TrackTacular checkpoint',
    'concat': 'Concat (baseline)',
    'concat_drop': 'Concat + cam-dropout',
    'masked_drop': 'Masked + cam-dropout',
    'masked_cov_drop': 'Masked + coverage + cam-dropout',
    'weighted_cov_drop': 'Weighted + coverage + cam-dropout (Retrack)',
    'baseline': 'TrackTacular baseline',
    'retrack': 'Retrack (full)',
}
METHOD_NAMES = {
    'S0_baseline': 'S0 baseline (thr 0.5, TrackTacular tracker)',
    'S1_threshold': 'S1 + tuned threshold',
    'S2_byte': 'S2 + ByteTrack-style association',
    'S3_byte_reid': 'S3 + camera-dropout-aware re-ID',
}
# validated categorical order (dataviz reference palette, light mode)
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

DET = ['moda', 'modp', 'precision', 'recall']
TRK = ['mota', 'idf1', 'mt', 'ml', 'id_switches', 'fragmentations']
COUNTS = ['det_fn', 'det_fp', 'trk_misses', 'trk_fp', 'id_switches', 'fragmentations']
ALL = list(dict.fromkeys(DET + TRK + COUNTS))


def load(results):
    frames = [pd.read_csv(p) for p in sorted(glob.glob(osp.join(results, 'eval', '*', 'metrics.csv')))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(df, keys, metrics):
    """
    Mean over removed-camera combinations per seed, then mean +- std over seeds.
    Also reports the spread over camera combinations (mean over seeds of the per-seed std).
    """
    per_seed = df.groupby(keys + ['seed'])[metrics].agg(['mean', 'std'])
    means = per_seed.xs('mean', axis=1, level=1)
    spreads = per_seed.xs('std', axis=1, level=1).fillna(0)
    out = means.groupby(keys).mean()
    seed_std = means.groupby(keys).std().fillna(0)
    combo_std = spreads.groupby(keys).mean()
    n = means.groupby(keys).size().rename('n_seeds')
    out = out.join(seed_std.add_suffix('_sd_seeds')).join(combo_std.add_suffix('_sd_cams')).join(n)
    return out.reset_index()


def fmt(row, m, digits=1):
    v = row[m]
    sd = row.get(f'{m}_sd_seeds', 0)
    s = f'{v:.{digits}f}'
    if row.get('n_seeds', 1) > 1:
        s += f' ± {sd:.{digits}f}'
    return s


def md_table(rows, columns):
    head = '| ' + ' | '.join(columns) + ' |\n|' + '---|' * len(columns) + '\n'
    return head + ''.join('| ' + ' | '.join(str(r[c]) for c in columns) + ' |\n' for r in rows)


def slope(ks, ys):
    return float(np.polyfit(ks, ys, 1)[0]) if len(ks) > 1 else float('nan')


def plot_curves(table, metric, title, path, label_col, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for i, name in enumerate(order):
        t = table[table[label_col] == name].sort_values('k')
        if t.empty:
            continue
        sd = t.get(f'{metric}_sd_seeds', pd.Series(0, index=t.index)).values
        ax.plot(t['k'], t[metric], color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], lw=2, ms=8,
                label=VARIANT_NAMES.get(name, METHOD_NAMES.get(name, name)))
        if (sd > 0).any():
            ax.fill_between(t['k'], t[metric] - sd, t[metric] + sd, color=COLORS[i % len(COLORS)], alpha=0.12, lw=0)
    ax.set_xlabel('number of removed cameras')
    ax.set_ylabel(metric.upper())
    ax.set_title(title, fontsize=10, loc='left')
    ax.set_xticks(sorted(table['k'].unique()))
    ax.grid(axis='y', color='#e4e3df', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sweep(sweep, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for i, m in enumerate(['moda', 'recall', 'precision', 'mota']):
        ax.plot(sweep['thr'], sweep[m], color=COLORS[i], marker=MARKERS[i], lw=2, ms=6, label=m.upper())
    ax.axvline(0.5, color='#8a8984', lw=1, ls='--')
    ax.set_xlabel('detection confidence threshold')
    ax.set_ylabel('%')
    ax.set_title('Stage 1: threshold sweep (official checkpoint, all cameras)', fontsize=10, loc='left')
    ax.grid(axis='y', color='#e4e3df', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_report(results):
    df = load(results)
    out = osp.join(results, 'report')
    os.makedirs(out, exist_ok=True)
    if df.empty:
        print('no evaluation results yet')
        return
    md = ['# Retrack vs TrackTacular — results\n',
          'Generated by `evaluation/retrack_report.py`. Detection: 0.5 m match radius; tracking: 1 m. '
          '"± x" is the std over training seeds; "cam spread" is the std over the combinations of removed cameras. '
          'k = number of removed cameras; every combination of k cameras is evaluated and averaged. '
          'Count columns are totals over the 40 test frames, averaged over combinations.\n']

    # ---------------- sanity check
    sanity_path = osp.join(results, 'sanity_lightning_test.json')
    if osp.exists(sanity_path):
        lt = json.load(open(sanity_path))
        ref = df[(df.tag == lt.get('tag')) & (df.method == 'S0_baseline') & (df.k == 0)]
        if not ref.empty:
            s0 = ref.iloc[0]
            rows = [{'metric': name, 'Lightning test (original code path)': f'{lt.get(key, float("nan")):.2f}',
                     'retrack_eval S0': f'{s0[col]:.2f}'}
                    for name, key, col in [('MODA', 'detect/moda', 'moda'), ('MODP', 'detect/modp', 'modp'),
                                           ('Recall', 'detect/recall', 'recall'), ('MOTA', 'track/mota', 'mota'),
                                           ('IDF1', 'track/idf1', 'idf1')]]
            md += [f'\n## 0. Sanity check: the evaluation engine reproduces TrackTacular\'s own test ({lt["tag"]})\n',
                   md_table(rows, list(rows[0].keys()))]

    # ---------------- Stages 1-3 on the trained TrackTacular baseline (concat), averaged over seeds
    base = df[df.variant == 'concat']
    if not base.empty:
        tuned = {}
        for tag in sorted(base.tag.unique()):
            info = osp.join(results, 'eval', tag, 'info.json')
            if osp.exists(info):
                tuned[tag] = json.load(open(info))['tuned_thr']
        sweep = base[(base.method == 'sweep') & (base.k == 0)]
        sweep = aggregate(sweep, ['thr'], ALL).sort_values('thr')
        sweep.to_csv(osp.join(out, 'stage1_threshold_sweep.csv'), index=False)
        plot_sweep(sweep, osp.join(out, 'stage1_threshold_sweep.png'))
        rows = [{'thr': f"{r['thr']:.2f}" + (' (default)' if abs(r['thr'] - 0.5) < 1e-9 else ''),
                 'Recall': fmt(r, 'recall'), 'Precision': fmt(r, 'precision'), 'MODA': fmt(r, 'moda'),
                 'MODP': fmt(r, 'modp'), 'misses': fmt(r, 'det_fn', 0), 'FP': fmt(r, 'det_fp', 0),
                 'MOTA': fmt(r, 'mota'), 'IDF1': fmt(r, 'idf1')} for r in sweep.to_dict('records')]
        md += ['\n## 1. Stage 1 — detection threshold sweep (trained TrackTacular baseline, all cameras)\n',
               'Tuned threshold (best MODA) per baseline model: ' +
               ', '.join(f'{t}: **{v}**' for t, v in tuned.items()) +
               '. It is selected on the test frames (WILDTRACK has no separate validation split), so read it as an '
               'upper bound; the full curve is shown.\n\n',
               md_table(rows, list(rows[0].keys())), '\n![sweep](stage1_threshold_sweep.png)\n']

        agg = aggregate(base[base.method.isin(METHOD_NAMES)], ['method', 'k'], ALL)
        agg.to_csv(osp.join(out, 'stages1-3_baseline_by_k.csv'), index=False)
        rows = []
        for r in agg.sort_values(['k', 'method']).to_dict('records'):
            rows.append({'k': r['k'], 'method': METHOD_NAMES[r['method']], 'MODA': fmt(r, 'moda'),
                         'Recall': fmt(r, 'recall'), 'Prec.': fmt(r, 'precision'), 'MOTA': fmt(r, 'mota'),
                         'IDF1': fmt(r, 'idf1'), 'MT%': fmt(r, 'mt'), 'ML%': fmt(r, 'ml'),
                         'IDsw': fmt(r, 'id_switches'), 'Frag': fmt(r, 'fragmentations'),
                         'misses': fmt(r, 'trk_misses', 0), 'FP': fmt(r, 'trk_fp', 0),
                         'cam spread (MOTA)': f"{r['mota_sd_cams']:.1f}"})
        md += ['\n## 2. Stages 1–3 on the trained TrackTacular model (no retraining), by removed cameras\n',
               'Each stage adds to the previous one. MODA changes only with the threshold (Stage 1): '
               'detections are scored before tracking.\n\n', md_table(rows, list(rows[0].keys()))]
        plot_curves(agg.rename(columns={'method': 'name'}), 'idf1', 'Stages 1–3 (trained baseline): IDF1',
                    osp.join(out, 'stages1-3_idf1.png'), 'name', list(METHOD_NAMES))
        plot_curves(agg.rename(columns={'method': 'name'}), 'mota', 'Stages 1–3 (trained baseline): MOTA',
                    osp.join(out, 'stages1-3_mota.png'), 'name', list(METHOD_NAMES))
        md += ['\n![idf1](stages1-3_idf1.png) ![mota](stages1-3_mota.png)\n']

    for grid, title in [('grid_S2', '3. Stage 2 grid — ByteTrack-style association parameters'),
                        ('grid_S3', '4. Stage 3 grid — re-ID weight λ and match threshold')]:
        g = df[df.method == grid]
        if g.empty:
            continue
        g = g.assign(kgroup=np.where(g.k == 0, 'all cams', 'cams removed (k≥1)'))
        ga = g.groupby(['tracker', 'kgroup'])[['mota', 'idf1', 'id_switches', 'fragmentations', 'mt']].mean()
        ga = ga.unstack('kgroup')
        ga.to_csv(osp.join(out, f'{grid}.csv'))
        rows = []
        for tracker, r in ga.iterrows():
            row = {'tracker': tracker}
            for kg in ['all cams', 'cams removed (k≥1)']:
                if ('mota', kg) in r.index:
                    row[f'MOTA {kg}'] = f"{r[('mota', kg)]:.1f}"
                    row[f'IDF1 {kg}'] = f"{r[('idf1', kg)]:.1f}"
                    row[f'IDsw {kg}'] = f"{r[('id_switches', kg)]:.1f}"
            rows.append(row)
        md += [f"\n## {title} ({', '.join(sorted(g.tag.unique()))})\n", md_table(rows, list(rows[0].keys()))]

    # ---------------- Stage 4 fusion ablation (retrained models)
    tr = df[df.variant.isin(VARIANTS)]
    if not tr.empty:
        s0 = tr[tr.method == 'S0_baseline']
        agg = aggregate(s0, ['variant', 'k'], ALL)
        agg.to_csv(osp.join(out, 'stage4_fusion_ablation.csv'), index=False)
        rows = []
        for v in VARIANTS:
            t = agg[agg.variant == v].sort_values('k')
            if t.empty:
                continue
            row = {'fusion variant': VARIANT_NAMES[v], 'seeds': int(t['n_seeds'].max())}
            for r in t.to_dict('records'):
                row[f"MODA k={r['k']}"] = fmt(r, 'moda')
            row['MODA slope / cam'] = f"{slope(t['k'], t['moda']):+.2f}"
            for r in t.to_dict('records'):
                row[f"MOTA k={r['k']}"] = fmt(r, 'mota')
            row['MOTA slope / cam'] = f"{slope(t['k'], t['mota']):+.2f}"
            rows.append(row)
        cols = list(dict.fromkeys(c for r in rows for c in r))
        rows = [{c: r.get(c, '') for c in cols} for r in rows]
        md += ['\n## 5. Stage 4 — robust view fusion ablation (retrained, threshold 0.5, TrackTacular tracker)\n',
               'Only the fusion changes between rows; detection threshold and tracker are fixed so the effect of '
               'fusion is isolated. Slope = metric points lost per removed camera (linear fit).\n\n',
               md_table(rows, cols)]
        for m in ['moda', 'mota', 'idf1']:
            plot_curves(agg, m, f'Stage 4 fusion ablation: {m.upper()} vs removed cameras',
                        osp.join(out, f'stage4_{m}.png'), 'variant', VARIANTS)
        md += ['\n![moda](stage4_moda.png) ![mota](stage4_mota.png) ![idf1](stage4_idf1.png)\n']

        # all methods x all variants (full grid)
        full = aggregate(tr[tr.method.isin(METHOD_NAMES)], ['variant', 'method', 'k'], ALL)
        full.to_csv(osp.join(out, 'all_variants_all_methods.csv'), index=False)

        # ---------------- final comparison
        base = full[(full.variant == 'concat') & (full.method == 'S0_baseline')].assign(name='baseline')
        best_method = 'S3_byte_reid' if (full.method == 'S3_byte_reid').any() else 'S2_byte'
        ours = full[(full.variant == 'weighted_cov_drop') & (full.method == best_method)].assign(name='retrack')
        if not base.empty and not ours.empty:
            rows = []
            for k in sorted(set(base.k) & set(ours.k)):
                b = base[base.k == k].iloc[0].to_dict()
                o = ours[ours.k == k].iloc[0].to_dict()
                for m, label, d in [('moda', 'MODA', 1), ('recall', 'Recall', 1), ('precision', 'Precision', 1),
                                    ('mota', 'MOTA', 1), ('idf1', 'IDF1', 1), ('mt', 'MT%', 1), ('ml', 'ML%', 1),
                                    ('id_switches', 'ID switches', 1), ('fragmentations', 'Fragmentations', 1),
                                    ('trk_misses', 'Misses', 0), ('trk_fp', 'False positives', 0)]:
                    rows.append({'k': k, 'metric': label, 'TrackTacular baseline': fmt(b, m, d),
                                 'Retrack (full)': fmt(o, m, d), 'Δ': f'{o[m] - b[m]:+.{max(d, 1)}f}'})
            md += ['\n## 6. Final comparison — TrackTacular baseline vs full Retrack\n',
                   'Baseline = retrained concatenation model, threshold 0.5, TrackTacular tracker. '
                   'Retrack = weighted fusion + coverage map + camera-dropout training, tuned threshold, '
                   'ByteTrack-style association and camera-dropout-aware re-ID. Same training schedule and seeds.\n\n',
                   md_table(rows, list(rows[0].keys()))]
            both = pd.concat([base, ours])
            for m in ['moda', 'mota', 'idf1']:
                plot_curves(both, m, f'TrackTacular baseline vs Retrack: {m.upper()}',
                            osp.join(out, f'final_{m}.png'), 'name', ['baseline', 'retrack'])
            md += ['\n![moda](final_moda.png) ![mota](final_mota.png) ![idf1](final_idf1.png)\n']
            pd.concat([base, ours]).to_csv(osp.join(out, 'final_comparison.csv'), index=False)

    off = df[df.variant == 'official']
    if not off.empty:
        agg = aggregate(off[off.method == 'S0_baseline'], ['k'], ALL)
        rows = [{'k': r['k'], 'MODA': fmt(r, 'moda'), 'Recall': fmt(r, 'recall'), 'MOTA': fmt(r, 'mota'),
                 'IDF1': fmt(r, 'idf1')} for r in agg.to_dict('records')]
        md += ['\n## Appendix: released TrackTacular checkpoint\n',
               'The released WILDTRACK weights do not reproduce the paper with the public code (TrackTacular issues '
               '#2 and #7 report these same numbers), which is why the baseline is retrained.\n\n',
               md_table(rows, list(rows[0].keys()))]

    text = ''.join(md)
    with open(osp.join(out, 'REPORT.md'), 'w') as f:
        f.write(text)
    print(text)
    print(f'\nreport written to {out}/REPORT.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    build_report(p.parse_args().results)
