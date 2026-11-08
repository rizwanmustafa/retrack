"""
ARL report: is every addition worth it?

Reads <results>/eval/*/metrics.csv (written by evaluation/arl_eval.py) and writes <results>/report/:
  REPORT.md         ladder E0-E12, then one verdict per addition
  ladder.csv, verdicts.csv, all_cells.csv
  additions.png     effect of removing each addition from the full method (mean +- std over seeds)
and copies the best checkpoints (+ their tuned parameters and a model card) to <results>/best/.

Additions and how each is judged (METHODOLOGY.md §5):
  training-time (one trained variant per on/off combination; seeds are independent -> Welch t-test)
    A1 3D random occlusion, A2 scene augmentation, A3 temporal history, A5 weight averaging
  post-hoc (switched on the cached detections of the same model; seeds are paired -> paired t-test)
    B1 tuned threshold, C1 ByteTrack-style online tracker, R appearance, C2 offline linking,
    C2b detection-level min-cost flow (instead of C1 + C2), C3 Gaussian-smoothed interpolation
  every addition is reported three ways:
    add-one        baseline + the addition vs the baseline
    leave-one-out  the full method vs the full method without the addition
    marginal       (post-hoc only) average effect over every combination of the other switches
  decision rule (fixed before running): keep iff a target metric improves by more than one standard deviation of
  the baseline over seeds and no primary metric (MODA, MOTA, IDF1) drops by more than one standard deviation.
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
import shutil

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY = ['moda', 'mota', 'idf1']
TARGET = {'A1': ['moda'], 'A2': ['moda'], 'A3': ['moda', 'idf1'], 'A5': ['moda'], 'B1': ['moda'],
          'C1': ['mota', 'idf1'], 'R': ['idf1'], 'C2': ['idf1'], 'C2b': ['idf1'], 'C3': ['mota']}
NAMES = {'A1': '3D random occlusion', 'A2': 'scene (BEV) augmentation', 'A3': 'temporal BEV history',
         'A5': 'weight averaging (last 5 epochs)', 'B1': 'tuned detection threshold',
         'C1': 'ByteTrack-style online tracker', 'R': 'OSNet appearance in C1/C2', 'C2': 'offline tracklet linking',
         'C2b': 'detection-level min-cost flow (replaces C1+C2)', 'C3': 'Gaussian-smoothed interpolation'}
FULL_TRAIN = frozenset({'A1', 'A2', 'A3'})
S0 = 'thr0.5|tt'  # TrackTacular evaluation: threshold 0.5, TrackTacular's online tracker
FULL_POST = 'thr*|byte+link|gsi'  # full ARL post-hoc stages (B1 + C1 + C2 + C3)

ALL_ASSOC = ['tt', 'tt+link', 'byte', 'byte+link', 'byte_r', 'byte_r+link', 'flow']
# post-hoc switches as pairs of cells (without, with) over the other switches
ASSOC_PAIRS = {'C1': [('tt', 'byte'), ('tt+link', 'byte+link')],
               'C2': [('tt', 'tt+link'), ('byte', 'byte+link'), ('byte_r', 'byte_r+link')],
               'R': [('byte', 'byte_r'), ('byte+link', 'byte_r+link')]}
ADD_ONE = {'B1': ('thr0.5|tt', 'thr*|tt'), 'C1': ('thr0.5|tt', 'thr0.5|byte'), 'C2': ('thr0.5|tt', 'thr0.5|tt+link'),
           'C3': ('thr0.5|tt', 'thr0.5|tt|gsi'), 'R': ('thr0.5|byte', 'thr0.5|byte_r'),
           'C2b': ('thr0.5|tt', 'thr0.5|flow')}
LEAVE_ONE_OUT = {'B1': ('thr0.5|byte+link|gsi', FULL_POST), 'C1': ('thr*|tt+link|gsi', FULL_POST),
                 'C2': ('thr*|byte|gsi', FULL_POST), 'C3': ('thr*|byte+link', FULL_POST),
                 'R': (FULL_POST, 'thr*|byte_r+link|gsi'), 'C2b': (FULL_POST, 'thr*|flow|gsi')}
LADDER = [('E0', 'baseline (retrained TrackTacular)', 'base', S0),
          ('E1', '+ A1 occlusion', 'A1', S0), ('E2', '+ A2 scene augmentation', 'A2', S0),
          ('E3', '+ A3 temporal history', 'A3', S0), ('E4', 'A1 + A2 + A3', 'A1A2A3', S0),
          ('E5', 'E4 + B1 threshold', 'A1A2A3', 'thr*|tt'), ('E6', 'E5 + C1 online tracker', 'A1A2A3', 'thr*|byte'),
          ('E7', 'E6 + C2 offline linking', 'A1A2A3', 'thr*|byte+link'),
          ('E8', 'E7 + C3 interpolation = full ARL', 'A1A2A3', FULL_POST),
          ('E9', 'E8 with C2b flow instead of C1 + C2', 'A1A2A3', 'thr*|flow|gsi'),
          ('E10', 'E8 + R appearance', 'A1A2A3', 'thr*|byte_r+link|gsi'),
          ('E12', 'E8 + A5 weight averaging', 'A1A2A3A5', FULL_POST),
          ('E0+C', 'baseline detections + all post-hoc stages', 'base', FULL_POST)]


def components(variant):
    return frozenset(re.findall(r'A\d', variant))


def canonical(comps):
    return ''.join(sorted(comps)) or 'base'


# ------------------------------------------------------------------ helpers
def fmt(m, s):
    if not np.isfinite(m):
        return '—'
    return f'{m:.2f}' if not np.isfinite(s) else f'{m:.2f} ± {s:.2f}'


def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in r.values) + ' |')
    return '\n'.join(lines)


def values(df, variant, cell, metric):
    d = df[(df.variant == variant) & (df.cell == cell)].sort_values('seed')
    return d.set_index('seed')[metric]


def compare(without, with_, paired):
    """without / with_: Series indexed by seed. Returns (delta, std of the per-seed deltas or pooled, p)."""
    if paired:
        common = without.index.intersection(with_.index)
        d = (with_[common] - without[common]).values
        if len(d) == 0:
            return np.nan, np.nan, np.nan
        p = stats.ttest_rel(with_[common], without[common]).pvalue if len(d) >= 2 and np.std(d) > 0 else np.nan
        return float(d.mean()), float(d.std(ddof=1)) if len(d) >= 2 else np.nan, p
    if len(without) == 0 or len(with_) == 0:
        return np.nan, np.nan, np.nan
    p = stats.ttest_ind(with_, without, equal_var=False).pvalue if min(len(without), len(with_)) >= 2 else np.nan
    sd = np.sqrt(with_.var(ddof=1) / len(with_) + without.var(ddof=1) / len(without)) \
        if min(len(without), len(with_)) >= 2 else np.nan
    return float(with_.mean() - without.mean()), float(sd), p


def verdict(name, deltas, sigma):
    if any(not np.isfinite(deltas[m]) for m in PRIMARY):
        return 'not run'
    if any(not np.isfinite(sigma[m]) for m in PRIMARY):
        return 'inconclusive (needs >= 2 seeds)'
    gain = any(deltas[m] > sigma[m] for m in TARGET[name])
    harm = any(deltas[m] < -sigma[m] for m in PRIMARY)
    if gain and not harm:
        return 'KEEP'
    if harm:
        return 'DROP (hurts)'
    return 'DROP (no clear gain)'


# ------------------------------------------------------------------ verdict rows
def training_rows(df, sigma, cells=(S0, FULL_POST)):
    have = {components(v): v for v in df.variant.unique()}
    rows = []
    for name in ['A1', 'A2', 'A3', 'A5']:
        tests = []
        if name != 'A5':
            tests.append(('add-one', frozenset(), frozenset({name})))
            tests.append(('leave-one-out', FULL_TRAIN - {name}, FULL_TRAIN))
        else:
            tests.append(('add to full', FULL_TRAIN, FULL_TRAIN | {'A5'}))
        for kind, a, b in tests:
            if a not in have or b not in have:
                continue
            for cell in cells:
                deltas, sds, ps = {}, {}, {}
                for m in PRIMARY:
                    deltas[m], sds[m], ps[m] = compare(values(df, have[a], cell, m), values(df, have[b], cell, m),
                                                       paired=False)
                rows.append(dict(addition=name, what=NAMES[name], test=kind, compared=f'{canonical(a)} -> '
                                 f'{canonical(b)}', post_hoc=cell, **{f'd_{m}': deltas[m] for m in PRIMARY},
                                 **{f'p_{m}': ps[m] for m in PRIMARY}, verdict=verdict(name, deltas, sigma)))
    return rows


def post_hoc_rows(df, sigma, variant):
    d = df[df.variant == variant]
    if d.empty:
        return []
    rows = []

    def add(name, kind, pairs):
        deltas, ps = {}, {}
        for m in PRIMARY:
            diffs = []
            for a, b in pairs:
                va, vb = values(d, variant, a, m), values(d, variant, b, m)
                if len(va) and len(vb):
                    diffs.append((vb - va).dropna())
            if not diffs:
                return
            per_seed = pd.concat(diffs, axis=1).mean(axis=1)  # average over contexts, per seed
            deltas[m] = float(per_seed.mean())
            ps[m] = stats.ttest_1samp(per_seed, 0).pvalue if len(per_seed) >= 2 and per_seed.std() > 0 else np.nan
        rows.append(dict(addition=name, what=NAMES[name], test=kind, compared=variant,
                         post_hoc=' / '.join(f'{a} -> {b}' for a, b in pairs[:2]) + (' ...' if len(pairs) > 2 else ''),
                         **{f'd_{m}': deltas[m] for m in PRIMARY}, **{f'p_{m}': ps[m] for m in PRIMARY},
                         verdict=verdict(name, deltas, sigma)))

    cells = set(d.cell.unique())
    for name in ['B1', 'C1', 'R', 'C2', 'C2b', 'C3']:
        for kind, pair in (('add-one', ADD_ONE[name]), ('leave-one-out', LEAVE_ONE_OUT[name])):
            if pair[0] in cells and pair[1] in cells:
                add(name, kind, [pair])
        # marginal: every pair of cells that differ only in this switch
        thrs, gsis = ('thr0.5', 'thr*'), ('', '|gsi')
        if name == 'B1':
            pairs = [(f'thr0.5|{a}{g}', f'thr*|{a}{g}') for a in ALL_ASSOC for g in gsis]
        elif name == 'C3':
            pairs = [(f'{t}|{a}', f'{t}|{a}|gsi') for t in thrs for a in ALL_ASSOC]
        elif name in ASSOC_PAIRS:
            pairs = [(f'{t}|{a}{g}', f'{t}|{b}{g}') for t in thrs for g in gsis for a, b in ASSOC_PAIRS[name]]
        else:
            pairs = []
        pairs = sorted({p for p in pairs if p[0] in cells and p[1] in cells})
        if pairs and name != 'C2b':
            add(name, 'marginal', pairs)
    return rows


# ------------------------------------------------------------------ best checkpoints
def save_best(df, results, protocol):
    eval_info = {}
    for info in glob.glob(osp.join(results, 'eval', '*', 'info.json')):
        i = json.load(open(info))
        if i.get('role') == 'test':
            eval_info[i['tag']] = i
    d = df[df.protocol == protocol].copy()
    if d.empty or not eval_info:
        return None
    d['score'] = d[PRIMARY].mean(axis=1)
    best_dir = osp.join(results, 'best')
    os.makedirs(osp.join(best_dir, 'per_variant'), exist_ok=True)
    cards = {}
    for variant, dv in list(d.groupby('variant')) + [('OVERALL', d)]:
        r = dv.loc[dv.score.idxmax()]
        info = eval_info.get(r.tag)
        if info is None or not osp.exists(info['ckpt']):
            continue
        dst = osp.join(best_dir, 'best_model.ckpt') if variant == 'OVERALL' else \
            osp.join(best_dir, 'per_variant', f'{variant}.ckpt')
        same = osp.exists(dst) and osp.exists(dst + '.json') and json.load(open(dst + '.json')).get('ckpt') == info['ckpt']
        if not same:
            shutil.copy2(info['ckpt'], dst + '.part')
            os.replace(dst + '.part', dst)
        params = osp.join(osp.dirname(info['metrics']), 'params_oracle.json') if protocol == 'oracle' \
            else info.get('clean_params')
        card = {'variant': r.variant, 'seed': int(r.seed), 'tag': r.tag, 'post_hoc': r.cell, 'protocol': protocol,
                'ckpt': info['ckpt'], 'copied_to': dst, 'tuned_params': params,
                'moda': r.moda, 'mota': r.mota, 'idf1': r.idf1, 'modp': r.modp, 'motp': r.motp,
                'note': 'selected by (MODA + MOTA + IDF1) / 3 on the test frames: use it, but report the mean over '
                        'seeds from REPORT.md, not this maximum'}
        json.dump(card, open(dst + '.json', 'w'), indent=1)
        if params and osp.exists(params):
            shutil.copy2(params, dst.replace('.ckpt', '_params.json'))
        cards[variant] = card
    json.dump(cards, open(osp.join(best_dir, 'MODEL_CARDS.json'), 'w'), indent=1)
    return cards.get('OVERALL')


# ------------------------------------------------------------------ main
def main(results):
    paths = sorted(glob.glob(osp.join(results, 'eval', '*', 'metrics.csv')))
    out = osp.join(results, 'report')
    os.makedirs(out, exist_ok=True)
    if not paths:
        print('no ARL evaluations yet')
        return
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df['variant'] = df.variant.map(lambda v: canonical(components(str(v))))
    df.to_csv(osp.join(out, 'all_cells.csv'), index=False)

    lines = ['# ARL — is every addition worth it?', '',
             'Generated by `evaluation/arl_report.py`. Method: `research_methodology_attempt_1/METHODOLOGY.md`.',
             'All numbers in %: MODA (0.5 m), MOTA / IDF1 (1 m), mean ± std over training seeds.', '',
             '**Protocols.** *clean*: thresholds and tracker parameters chosen on a validation model (frames 0-323, '
             'tuned on 324-359), then frozen. *oracle*: tuned on the test frames themselves (allowed, but an upper '
             'bound). Compare additions within one protocol.', '']
    all_verdicts, ladder_rows = [], []
    protocols = [p for p in ('clean', 'oracle') if p in df.protocol.unique()]
    for prot in protocols:
        d = df[df.protocol == prot]
        base = d[(d.variant == 'base') & (d.cell == S0)]
        sigma = {m: base[m].std(ddof=1) if len(base) >= 2 else np.nan for m in PRIMARY}
        n_seeds = d.groupby('variant').seed.nunique().to_dict()

        # ladder
        rows = []
        for eid, desc, variant, cell in LADDER:
            x = d[(d.variant == variant) & (d.cell == cell)]
            if x.empty:
                continue
            rows.append({'ID': eid, 'configuration': desc, 'variant': variant, 'post-hoc': cell, 'seeds': len(x),
                         **{m.upper(): fmt(x[m].mean(), x[m].std(ddof=1) if len(x) > 1 else np.nan)
                            for m in PRIMARY},
                         'FN': f'{x.det_fn.mean():.0f}', 'FP': f'{x.det_fp.mean():.0f}',
                         'IDSW': f'{x.id_switches.mean():.0f}'})
            ladder_rows.append(dict(protocol=prot, ID=eid, configuration=desc, variant=variant, cell=cell,
                                    seeds=len(x), **{f'{m}_mean': x[m].mean() for m in PRIMARY},
                                    **{f'{m}_std': x[m].std(ddof=1) for m in PRIMARY}))
        lines += [f'## Ladder ({prot})', '', md_table(pd.DataFrame(rows)) if rows else '_no rows yet_', '',
                  f'FN / FP are detection errors at the row\'s threshold; IDSW are identity switches. Seeds per '
                  f'variant: {n_seeds}. Baseline std (the "one std" of the decision rule): '
                  + ', '.join(f'{m.upper()} {sigma[m]:.2f}' for m in PRIMARY) + '.', '']

        # verdicts
        v_rows = training_rows(d, sigma)
        detector = 'A1A2A3' if 'A1A2A3' in set(d.variant) else 'base'
        v_rows += post_hoc_rows(d, sigma, detector)
        if detector != 'base':
            v_rows += [dict(r, compared='base (detections held fixed)') for r in post_hoc_rows(d, sigma, 'base')]
        for r in v_rows:
            r['protocol'] = prot
        all_verdicts += v_rows
        if v_rows:
            t = pd.DataFrame(v_rows)
            show = pd.DataFrame({
                'addition': t.addition + ' — ' + t.what, 'test': t.test, 'compared': t.compared,
                'post-hoc cells': t.post_hoc,
                **{f'Δ{m.upper()}': [f'{x:+.2f}' if np.isfinite(x) else '—' for x in t[f'd_{m}']] for m in PRIMARY},
                **{f'p({m.upper()})': [f'{x:.3f}' if np.isfinite(x) else '—' for x in t[f'p_{m}']] for m in PRIMARY},
                'verdict': t.verdict})
            lines += [f'## Is every addition worth it? ({prot})', '',
                      'Training-time additions are compared between independently trained variants (Welch t-test; '
                      f'post-hoc cell `{S0}` = TrackTacular evaluation and `{FULL_POST}` = full ARL post-hoc '
                      f'stages). Post-hoc additions are switched on the cached detections of variant `{detector}` '
                      'and of `base` (paired over seeds). Target metrics: '
                      + ', '.join(f'{k}: {"/".join(m.upper() for m in v)}' for k, v in TARGET.items()) + '.', '',
                      md_table(show), '']

    pd.DataFrame(ladder_rows).to_csv(osp.join(out, 'ladder.csv'), index=False)
    pd.DataFrame(all_verdicts).to_csv(osp.join(out, 'verdicts.csv'), index=False)
    plot_additions(pd.DataFrame(all_verdicts), osp.join(out, 'additions.png'))

    best = save_best(df, results, protocols[0])
    if best:
        lines += ['## Best checkpoint (saved to `best/`)', '',
                  f'`{best["copied_to"]}`: variant `{best["variant"]}`, seed {best["seed"]}, post-hoc `{best["post_hoc"]}` '
                  f'({best["protocol"]} parameters `{best["tuned_params"]}`): MODA {best["moda"]:.2f}, MOTA '
                  f'{best["mota"]:.2f}, IDF1 {best["idf1"]:.2f}. The best checkpoint of every variant is in '
                  '`best/per_variant/` with a `.json` model card. ' + best['note'] + '.', '']
    open(osp.join(out, 'REPORT.md'), 'w').write('\n'.join(lines))
    print(f'report -> {osp.join(out, "REPORT.md")}')


def plot_additions(v, path):
    if v.empty:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    prot = 'clean' if 'clean' in set(v.protocol) else v.protocol.iloc[0]
    x = v[(v.protocol == prot) & (v.test.isin(['leave-one-out', 'add to full']))
          & (~v.compared.str.startswith('base (')) & (v.post_hoc != S0)]
    if x.empty:
        return
    x = x.drop_duplicates('addition', keep='last')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    w = 0.27
    pos = np.arange(len(x))
    for i, m in enumerate(PRIMARY):
        ax.bar(pos + (i - 1) * w, x[f'd_{m}'], w, label=m.upper())
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(pos, x.addition)
    ax.set_ylabel('Δ vs full method without it (pts)')
    ax.set_title(f'Contribution of each addition to full ARL ({prot} protocol)')
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    main(p.parse_args().results)
