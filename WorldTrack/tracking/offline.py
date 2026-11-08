"""
ARL Stage C: offline association on cached BEV detections (research_methodology_attempt_1/METHODOLOGY.md, §4.3).

  online_tracklets   C1: run an online tracker (TrackTacular's or Retrack's ByteBEVTracker) and keep its output as
                     tracklets
  link_tracklets     C2: globally optimal tracklet linking. Every tracklet has at most one predecessor and one
                     successor and costs add up along a chain, so the min-cost-flow problem of KSP / muSSP reduces
                     to a rectangular assignment with a "no link" option, solved exactly with the Hungarian method
  flow_tracks        C2b: detection-level min-cost flow (KSP / muSSP-style network with entry, exit, detection and
                     transition edges; MVFlow's motion term), solved as a linear program
  gsi                C3: Gaussian-smoothed interpolation (StrongSORT, Eq. 12-15) inside each track

A track is a dict {'t': int array of frame indices (0..T-1), 'xy': (n, 2) reference coordinates, 'score': (n,),
'feat': (D,) mean appearance descriptor or None}. Reference coordinates are 2.5 cm ground-grid cells.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy import sparse

from tracking.multitracker import build_tracker

CELL_M = 0.025  # metres per reference (ground-grid) cell on WILDTRACK and MultiviewX
BIG = 1e6


# ------------------------------------------------------------------ C1
def online_tracklets(frames, thr, tracker_kwargs, use_feats=False):
    """Runs the online tracker frame by frame. Returns tracks (one per output track id)."""
    import torch
    tracker = build_tracker(conf_thres=thr, **tracker_kwargs)
    acc = {}
    for t, f in enumerate(frames):
        if tracker_kwargs.get('use_reid', False):
            stracks = tracker.update(f['xy'], f['xy_prev'], f['score'], feats=f['feat'])
        else:
            stracks = tracker.update(torch.from_numpy(f['xy']), torch.from_numpy(f['xy_prev']),
                                     torch.from_numpy(f['score']))
        for s in stracks:
            d = acc.setdefault(s.track_id, {'t': [], 'xy': [], 'score': [], 'feats': []})
            d['t'].append(t)
            d['xy'].append(np.asarray(s.xy, dtype=np.float64))
            d['score'].append(float(np.asarray(s.score)))
            if use_feats and getattr(s, 'curr_feat', None) is not None:
                d['feats'].append(np.asarray(s.curr_feat, dtype=np.float64))
    return [_finish(d) for d in acc.values()]


def _finish(d):
    feats = [x for x in d.get('feats', []) if x is not None and np.all(np.isfinite(x))]
    feat = None
    if feats:
        feat = np.mean(feats, axis=0)
        feat /= max(np.linalg.norm(feat), 1e-12)
    return {'t': np.asarray(d['t'], dtype=int), 'xy': np.asarray(d['xy'], dtype=np.float64).reshape(-1, 2),
            'score': np.asarray(d['score'], dtype=np.float64), 'feat': feat}


# ------------------------------------------------------------------ C2
def _velocity(t, xy, at_end, k=3):
    """Least-squares velocity (cells / frame) over the last (or first) k points of a track."""
    if len(t) < 2:
        return np.zeros(2)
    sl = slice(-k, None) if at_end else slice(0, k)
    tt, pp = t[sl].astype(np.float64), xy[sl]
    tt = tt - tt.mean()
    den = (tt ** 2).sum()
    return np.zeros(2) if den == 0 else (tt[:, None] * (pp - pp.mean(0))).sum(0) / den


def link_cost_matrix(tracks, max_gap=5, gate=1.0, gate_growth=0.5, w_m=1.0, w_t=0.1, w_a=0.0):
    """Cost of linking the end of track i to the start of track j (BIG where not allowed).

    Allowed if 1 <= start_j - end_i <= max_gap and the motion-extrapolation distance is inside the gate
    gate + gate_growth * (dt - 1) metres. Cost = w_m * d / gate_dt + w_t * (dt - 1) + w_a * (1 - cos(f_i, f_j)).
    """
    n = len(tracks)
    C = np.full((n, n), BIG)
    ends = np.array([tr['t'][-1] for tr in tracks])
    starts = np.array([tr['t'][0] for tr in tracks])
    v_end = [_velocity(tr['t'], tr['xy'], True) for tr in tracks]
    v_start = [_velocity(tr['t'], tr['xy'], False) for tr in tracks]
    for i, ti in enumerate(tracks):
        for j, tj in enumerate(tracks):
            dt = starts[j] - ends[i]
            if i == j or dt < 1 or dt > max_gap:
                continue
            fwd = ti['xy'][-1] + v_end[i] * dt  # i extrapolated to j's first frame
            bwd = tj['xy'][0] - v_start[j] * dt  # j extrapolated back to i's last frame
            d = 0.5 * (np.linalg.norm(fwd - tj['xy'][0]) + np.linalg.norm(bwd - ti['xy'][-1])) * CELL_M
            g = gate + gate_growth * (dt - 1)
            if d > g:
                continue
            c = w_m * d / g + w_t * (dt - 1)
            if w_a > 0 and ti['feat'] is not None and tj['feat'] is not None:
                c += w_a * (1 - float(ti['feat'] @ tj['feat']))
            C[i, j] = c
    return C


def link_tracklets(tracks, c0=1.0, **cost_kwargs):
    """Minimum-cost path cover: links i -> j iff that lowers the total cost, where leaving a tracklet end and a
    tracklet start unlinked costs c0 in total. Exact (assignment on the standard 2n x 2n augmented matrix)."""
    n = len(tracks)
    if n < 2:
        return tracks
    C = link_cost_matrix(tracks, **cost_kwargs)
    M = np.full((2 * n, 2 * n), BIG)
    M[:n, :n] = C
    M[:n, n:][np.diag_indices(n)] = c0 / 2  # end i stays unlinked
    M[n:, :n][np.diag_indices(n)] = c0 / 2  # start j stays unlinked
    M[n:, n:] = 0
    rows, cols = linear_sum_assignment(M)
    succ = {}
    for r, c in zip(rows, cols):
        if r < n and c < n and C[r, c] < BIG:
            succ[r] = c
    has_pred = set(succ.values())
    out = []
    for i in range(n):
        if i in has_pred:
            continue
        chain = [i]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        out.append(_merge([tracks[k] for k in chain]))
    return out


def _merge(parts):
    feats = [p['feat'] for p in parts if p['feat'] is not None]
    feat = None
    if feats:
        feat = np.mean(feats, axis=0)
        feat /= max(np.linalg.norm(feat), 1e-12)
    return {'t': np.concatenate([p['t'] for p in parts]), 'xy': np.concatenate([p['xy'] for p in parts]),
            'score': np.concatenate([p['score'] for p in parts]), 'feat': feat}


# ------------------------------------------------------------------ C2b
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def flow_tracks(frames, thr=0.5, low_thr=0.1, max_skip=2, gate=1.0, gate_growth=0.5, c_b=1.0, w_t=0.2):
    """
    Detection-level min-cost flow over the whole sequence (KSP / muSSP-style network).
      detection edge   cost logit(thr) - logit(score): negative (worth tracking) iff score > thr
      entry + exit     c_b in total, so a track must be worth more than c_b to exist
      transition i->j  (j up to max_skip frames after i) d / gate_dt + w_t * (dt - 1), where d (metres) is the
                       distance between i and j's learned previous-frame position (MVFlow's motion term), allowed
                       if d <= gate + gate_growth * (dt - 1)
    Solved as an LP; the constraint matrix is a network matrix, so the simplex solution is integral.
    """
    nodes = []  # (t, index in frame)
    for t, f in enumerate(frames):
        for k in np.where(f['score'] >= low_thr)[0]:
            nodes.append((t, k))
    n = len(nodes)
    if n == 0:
        return []
    by_t = {}
    for u, (t, k) in enumerate(nodes):
        by_t.setdefault(t, []).append(u)
    xy = np.array([frames[t]['xy'][k] for t, k in nodes], dtype=np.float64)
    xy_prev = np.array([frames[t]['xy_prev'][k] for t, k in nodes], dtype=np.float64)
    score = np.array([frames[t]['score'][k] for t, k in nodes], dtype=np.float64)

    edges, e_cost = [], []
    for u, (t, _) in enumerate(nodes):
        for dt in range(1, max_skip + 1):
            for v in by_t.get(t + dt, []):
                d = np.linalg.norm(xy[u] - xy_prev[v]) * CELL_M
                g = gate + gate_growth * (dt - 1)
                if d <= g:
                    edges.append((u, v))
                    e_cost.append(d / g + w_t * (dt - 1))
    E = len(edges)
    # variables: det (n), entry (n), exit (n), transitions (E)
    c = np.concatenate([_logit(thr) - _logit(score), np.full(n, c_b / 2), np.full(n, c_b / 2),
                        np.asarray(e_cost, dtype=np.float64)])
    rows, cols, vals = [], [], []
    for u in range(n):  # inflow: entry_u + sum_in tr - det_u = 0 ; outflow: det_u - exit_u - sum_out tr = 0
        rows += [u, u, n + u, n + u]
        cols += [n + u, u, u, 2 * n + u]
        vals += [1, -1, 1, -1]
    for e, (u, v) in enumerate(edges):
        rows += [v, n + u]
        cols += [3 * n + e, 3 * n + e]
        vals += [1, -1]
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * n, 3 * n + E))
    res = linprog(c, A_eq=A, b_eq=np.zeros(2 * n), bounds=(0, 1), method='highs-ds')
    if res.status != 0:
        raise RuntimeError(f'min-cost flow LP failed: {res.message}')
    x = res.x
    succ = {u: v for e, (u, v) in enumerate(edges) if x[3 * n + e] > 0.5}
    has_pred = set(succ.values())
    tracks = []
    for u in range(n):
        if x[u] < 0.5 or u in has_pred:
            continue
        chain = [u]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        tracks.append({'t': np.array([nodes[w][0] for w in chain]), 'xy': xy[chain], 'score': score[chain],
                       'feat': None})
    return tracks


# ------------------------------------------------------------------ C3
def gsi(track, max_gap=2, tau=10.0, noise=0.01):
    """
    Gaussian-smoothed interpolation (StrongSORT Eq. 12-15): fill gaps of <= max_gap frames by linear interpolation,
    then replace all positions by the Gaussian-process posterior mean P* = K(F*,F) (K(F,F) + s^2 I)^-1 P with an
    RBF kernel of length scale lambda = tau * log(tau^3 / l) frames (l = track length). The paper gives no lower
    bound for lambda (it is negative for l > tau^3), so it is clipped to >= 1 frame here. noise = s^2 in m^2.
    Gaps longer than max_gap stay gaps; nothing is extrapolated beyond the track's ends.
    """
    t, xy = track['t'], track['xy']
    order = np.argsort(t)
    t, xy, sc = t[order], xy[order], track['score'][order]
    ts, ps, ss = [t[0]], [xy[0]], [sc[0]]
    for k in range(1, len(t)):
        gap = t[k] - t[k - 1] - 1
        if 0 < gap <= max_gap:
            for g in range(1, gap + 1):
                a = g / (gap + 1)
                ts.append(t[k - 1] + g)
                ps.append((1 - a) * xy[k - 1] + a * xy[k])
                ss.append(min(sc[k - 1], sc[k]))
        ts.append(t[k])
        ps.append(xy[k])
        ss.append(sc[k])
    ts, ps = np.asarray(ts, dtype=np.float64), np.asarray(ps, dtype=np.float64)
    if tau and len(ts) >= 3:
        lam = max(tau * np.log(tau ** 3 / len(ts)), 1.0)
        K = np.exp(-(ts[:, None] - ts[None]) ** 2 / (2 * lam ** 2))
        P = ps * CELL_M
        mu = P.mean(0)
        P = K @ np.linalg.solve(K + noise * np.eye(len(ts)), P - mu) + mu
        ps = P / CELL_M
    return {'t': ts.astype(int), 'xy': ps, 'score': np.asarray(ss), 'feat': track['feat']}


def tracks_to_hyps(tracks, num_frames):
    """per-frame hypotheses [(track_id, xy)] for evaluation; ids are 1..len(tracks)"""
    hyps = [[] for _ in range(num_frames)]
    for tid, tr in enumerate(tracks, start=1):
        for t, p in zip(tr['t'], tr['xy']):
            if 0 <= t < num_frames:
                hyps[int(t)].append((tid, p))
    return hyps
