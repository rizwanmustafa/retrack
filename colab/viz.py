"""
Visual tools for any TrackTacular (B0) or R2 checkpoint, for Colab (they also run in a local Jupyter).

    !git clone -q -b feat/r2-sparse-queries https://github.com/rizwanmustafa/retrack.git /content/retrack  # once
    %run -i /content/retrack/colab/viz.py

The %run mounts Google Drive and prints every checkpoint it finds (numbered). Then, in new cells:

    explore(3)                 interactive viewer: frame slider / play, switch cameras on and off, threshold, layers
    compare(1, 3)              two checkpoints side by side (e.g. B0 vs R2), same frame and cameras
    camera_evidence(3, 10)     what each camera adds to the BEV heatmap in frame 10
    tracks(3, drop=[0, 4])     trajectories over the 40 test frames, ID switches marked (cameras 1 and 5 off)
    robustness(1, 3)           MODA / MOTA / IDF1 / FP / FN against the number of removed cameras
    video(3, drop=[2])         MP4 (or GIF) of the viewer over all test frames, saved to Drive
    checkpoints()              list again (e.g. after a training finished); checkpoints(roots=[...]) for other folders

A checkpoint can be given by its number in the list, its path, or a unique part of its path ('multiviewx_R2_s1').
Cameras are numbered from 0 in arguments (drop=[0] removes C1) and shown as C1..Cn in the figures.

Colours (BEV and camera images): green = detected person (true positive), orange = missed person (false negative),
red = false detection, white dots = ground truth. The number next to a missed person is how many working cameras
see that person (annotation boxes): the baseline needs about 3.
"""
import glob
import io
import json
import os
import os.path as osp
import re
import sys
import time

_HERE = osp.dirname(osp.abspath(__file__)) if '__file__' in globals() else '/content/retrack/colab'
sys.path.insert(0, _HERE)
sys.path.insert(0, osp.join(osp.dirname(_HERE), 'WorldTrack'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import DATA, DRIVE, in_colab, mount_drive, pip_install, ensure_dataset  # noqa: E402

pip_install()
from evaluation import r2_eval  # noqa: E402

DATA_ROOT = globals().get('DATA_ROOT', DATA)
ROOTS = globals().get('ROOTS', None)  # folders to search for checkpoints (default: Drive's retrack_results*)
VIDEO_DIR = globals().get('VIDEO_DIR', f'{DRIVE}/retrack_videos')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = globals().get('NUM_WORKERS', 2)  # data-loading processes for the test frames
CAM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
C_TP, C_FN, C_FP, C_GT = '#2ecc40', '#ff9f1a', '#ff2d2d', 'white'

_LIST = []  # last checkpoint listing
_HP_CACHE = {}  # path -> (mtime, hparams, info)
_SESSIONS = {}  # path -> Session
_DATA_CACHE = {}  # (dataset, resolution) -> (items, dataset object, visibility)


# ------------------------------------------------------------------ checkpoints
def _default_roots():
    roots = sorted(glob.glob(f'{DRIVE}/retrack_results*')) + [f'{DRIVE}/checkpoints', '/content/checkpoints']
    return [r for r in roots if osp.isdir(r)]


def _seed_of(path):
    m = re.search(r'_s(\d+)(?:/|$)', path)
    return int(m.group(1)) if m else None


def _eval_dir(path):
    """<results>/eval/<tag> for a checkpoint at <results>/runs/<tag>/lightning_logs/version_*/checkpoints/x.ckpt"""
    parts = path.split(os.sep)
    if 'runs' in parts:
        i = len(parts) - 1 - parts[::-1].index('runs')
        if i + 1 < len(parts):
            return os.sep.join(parts[:i] + ['eval', parts[i + 1]])
    return None


def checkpoints(roots=None, include_preflight=False, refresh=False):
    """Finds checkpoints (*.ckpt) under `roots` and prints a numbered list. Returns the list."""
    global _LIST
    roots = roots or ROOTS or _default_roots()
    if isinstance(roots, str):
        roots = [roots]
    paths = []
    for r in roots:
        if r.endswith('.ckpt') and osp.exists(r):
            paths.append(r)
        else:
            paths += glob.glob(osp.join(r, '**', '*.ckpt'), recursive=True)
    paths = sorted({p for p in paths if include_preflight or '/preflight/' not in p})
    entries = []
    for p in paths:
        mtime = osp.getmtime(p)
        if refresh or p not in _HP_CACHE or _HP_CACHE[p][0] != mtime:
            try:
                hp, info = r2_eval.read_hparams(p)
            except Exception as e:  # noqa: BLE001
                hp, info = {}, {'error': str(e)[:80]}
            _HP_CACHE[p] = (mtime, hp, info)
        _, hp, info = _HP_CACHE[p]
        ev = _eval_dir(p)
        entries.append(dict(path=p, variant=r2_eval.variant_of(hp) if hp else 'unreadable',
                            dataset=r2_eval.dataset_of(hp) if hp else '?', seed=_seed_of(p),
                            epoch=info.get('epoch'), mb=osp.getsize(p) / 2 ** 20,
                            evaluated=bool(ev and osp.exists(osp.join(ev, 'metrics.csv'))),
                            root=next((r for r in roots if p.startswith(r)), osp.dirname(p))))
    entries.sort(key=lambda e: (e['root'], e['dataset'], e['variant'].startswith('unsupported'), e['variant'],
                                e['seed'] or 0, e['path']))
    _LIST = entries
    if not entries:
        print('No checkpoints found in:\n  ' + '\n  '.join(roots or ['(no folders)']) +
              "\nTrain some with colab/train_eval.py, or call checkpoints(roots=['/path/to/folder']).")
        return entries
    root = None
    usable = 0
    for i, e in enumerate(entries):
        if e['root'] != root:
            root = e['root']
            print(f'\n{root}')
            print(f'  {"#":>3}  {"variant":8} {"dataset":11} {"seed":>6} {"epoch":>5} {"MB":>5}  {"eval":4}  file')
        ok = not e['variant'].startswith(('unsupported', 'unreadable'))
        usable += ok
        rel = osp.relpath(e['path'], root)
        v = e['variant'] if ok else '–'
        print(f'  {i:>3}  {v:8} {e["dataset"]:11} {e["seed"] if e["seed"] is not None else "":>6} '
              f'{"" if e["epoch"] is None else e["epoch"]:>5} {e["mb"]:5.0f}  {"yes" if e["evaluated"] else "":4}  '
              f'{rel}' + ('' if ok else f'   [{e["variant"]}]'))
    print(f'\n{usable} usable checkpoint(s). Open one with explore(<#>), e.g. explore('
          f'{next(i for i, e in enumerate(entries) if not e["variant"].startswith(("unsup", "unread"))) if usable else 0})')
    return entries


def _resolve(which):
    if isinstance(which, Session):
        return which.path
    if isinstance(which, (int, np.integer)):
        if not _LIST:
            checkpoints()
        assert 0 <= which < len(_LIST), f'no checkpoint #{which}; see checkpoints()'
        return _LIST[which]['path']
    if osp.exists(str(which)):
        return str(which)
    if not _LIST:
        checkpoints()
    hits = [e['path'] for e in _LIST if str(which) in e['path']]
    assert len(hits) == 1, f'"{which}" matches {len(hits)} checkpoints: {hits[:5]}'
    return hits[0]


# ------------------------------------------------------------------ data and model
def _visibility(data_dir, frame_step, frames):
    folder = osp.join(data_dir, 'annotations_positions')
    files = {int(f.split('.')[0]): osp.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')}
    vis = {}
    for fr in frames:
        with open(files[fr * frame_step]) as fh:
            for p in json.load(fh):
                vis[(fr, int(p['personID']))] = {i for i, v in enumerate(p['views'])
                                                 if not (v['xmin'] == -1 and v['xmax'] == -1 and
                                                         v['ymin'] == -1 and v['ymax'] == -1)}
    return vis


class Session:
    """A loaded checkpoint, its test frames and cached runs (one per set of switched-off cameras)."""

    def __init__(self, path):
        t0 = time.time()
        self.path = path
        print(f'loading {path}', flush=True)
        self.model, self.hp = r2_eval.load_model(path, DEVICE)
        self.dataset, self.variant = r2_eval.dataset_of(self.hp), r2_eval.variant_of(self.hp)
        self.seed = _seed_of(path)
        self.label = f'{self.variant} {self.dataset}' + (f' s{self.seed}' if self.seed is not None else '')
        key = (self.dataset, tuple(self.model.resolution))
        if key not in _DATA_CACHE:
            data_dir = ensure_dataset(self.dataset, DATA_ROOT)
            print(f'loading the {self.dataset} test frames from {data_dir} ...', flush=True)
            items, ds = r2_eval.load_test_items(data_dir, self.model, num_workers=NUM_WORKERS)
            vis = _visibility(data_dir, ds.base.frame_step, [int(it['frame'][0]) for it in items])
            _DATA_CACHE[key] = (items, ds, vis)
        self.items, self.ds, self.vis = _DATA_CACHE[key]
        it = self.items[0]
        self.S, _, self.H, self.W = it['img'].shape[1:]
        self.Y, self.Z, self.X = self.model.Y, self.model.Z, self.model.X
        b = self.model.bounds
        self.z_head = b[4] + 0.9 * (b[5] - b[4])  # 1.8 m on both datasets (WILDTRACK in cm, MultiviewX in m)
        self.z_mid = b[4] + 0.45 * (b[5] - b[4])
        self.z_sign = self.hp.get('z_sign', 1)
        K, E, R = it['intrinsic'][0].double(), it['extrinsic'][0].double(), it['ref_T_global'][0].double()
        self.cams_T_ref = (E @ torch.inverse(R)).numpy()  # S,4,4 (cameras are static; test frames are not shifted)
        self.pix_T_ref = (K @ torch.from_numpy(self.cams_T_ref)).numpy()
        cam_ref = np.stack([np.linalg.inv(T)[:3, 3] for T in self.cams_T_ref])  # S,3 camera centres
        self.cam_mem = self.ref2mem(cam_ref[:, :2])
        ys, xs = np.meshgrid(np.arange(self.Y), np.arange(self.X), indexing='ij')
        cells = self.mem2ref(np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float64))
        _, ok = self.project(np.c_[cells, np.full(len(cells), self.z_mid)])
        self.fov = ok.reshape(self.S, self.Y, self.X)  # cell (at mid-body height) inside camera s's image
        self.runs = {}
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f'{self.label}: {n:.2f} M parameters, {len(self.items)} test frames, {self.S} cameras, '
              f'on {DEVICE} ({time.time() - t0:.0f}s)', flush=True)

    # geometry
    def ref2mem(self, xy):
        xyz = torch.zeros(1, len(xy), 3, dtype=torch.float32)
        xyz[0, :, :2] = torch.as_tensor(np.asarray(xy, dtype=np.float32))
        return self.model.vox_util.Ref2Mem(xyz, self.Y, self.Z, self.X)[0, :, :2].numpy()

    def mem2ref(self, xy):
        xyz = torch.zeros(1, len(xy), 3, dtype=torch.float32)
        xyz[0, :, :2] = torch.as_tensor(np.asarray(xy, dtype=np.float32))
        return self.model.vox_util.Mem2Ref(xyz, self.Y, self.Z, self.X)[0, :, :2].numpy().astype(np.float64)

    def project(self, xyz_ref):
        """N,3 reference points -> S,N,2 pixel coordinates in the 720x1280 test images, and S,N 'in view'."""
        X = np.c_[xyz_ref, np.ones(len(xyz_ref))].T  # 4,N
        pix = self.pix_T_ref @ X  # S,4,N
        cam_z = (self.cams_T_ref @ X)[:, 2]  # S,N
        d = np.where(np.abs(pix[:, 2]) < 1e-9, 1e-9, pix[:, 2])
        uv = np.stack([pix[:, 0] / d, pix[:, 1] / d], -1)
        ok = (self.z_sign * cam_z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < self.W) & \
             (uv[..., 1] >= 0) & (uv[..., 1] < self.H)
        return uv, ok

    # runs
    def run(self, drop=(), maps=True, quiet=False):
        drop = tuple(sorted(int(c) for c in drop))
        key = (drop, maps)
        if key in self.runs:
            return self.runs[key]
        t0 = time.time()
        if not quiet:
            print(f'running {self.label} on {len(self.items)} frames with '
                  f'{"cameras " + ", ".join(f"C{c + 1}" for c in drop) + " off" if drop else "all cameras"} ...',
                  end=' ', flush=True)
        frames = r2_eval.run_inference(self.model, self.items, drop, keep_maps=maps)
        hyps = r2_eval.run_tracker(frames)
        trk, acc = r2_eval.tracking_metrics(frames, hyps)
        rec = dict(drop=drop, frames=frames, hyps=hyps, acc=acc, det=r2_eval.detection_metrics(frames), trk=trk)
        if maps:  # keep the cache small: maps only for the most recent camera sets
            for k in [k for k in self.runs if k[1]][:-5]:
                del self.runs[k]
        self.runs[key] = rec
        if not quiet:
            print(f'{time.time() - t0:.1f}s', flush=True)
        return rec

    def summary(self, rec):
        d, t = rec['det'], rec['trk']
        return (f"MODA {d['moda']:.2f}  MOTA {t['mota']:.2f}  IDF1 {t['idf1']:.2f}  "
                f"(FP {d['det_fp']}, FN {d['det_fn']}, ID switches {t['id_switches']})")


def session(which):
    """The loaded checkpoint (cached) for a number from checkpoints(), a path or a unique part of a path."""
    path = _resolve(which)
    if path not in _SESSIONS:
        _SESSIONS[path] = Session(path)
    return _SESSIONS[path]


# ------------------------------------------------------------------ per-frame analysis
def frame_analysis(sess, rec, i, thr=0.5):
    """Matching of frame i at threshold thr: dict with gt, det (ref coordinates) and the TP / FN / FP indices."""
    f = rec['frames'][i]
    gt = f['gt'][:, :2]
    keep = f['score'] > thr
    det, score = f['xy'][keep].astype(np.float64), f['score'][keep]
    gi, di, _ = r2_eval.match_frame(gt, det)
    fn = np.setdiff1d(np.arange(len(gt)), gi)
    fp = np.setdiff1d(np.arange(len(det)), di)
    pids = f['gt'][:, 2].astype(int)
    working = set(range(sess.S)) - set(rec['drop'])
    seen = np.array([len(sess.vis.get((f['frame'], p), set()) & working) for p in pids])
    best = np.array([f['score'][np.linalg.norm(f['xy'] - g, axis=1) < r2_eval.MOD_DIST].max(initial=0.0)
                     for g in gt]) if len(gt) else np.zeros(0)
    return dict(f=f, gt=gt, pids=pids, det=det, score=score, tp_gt=gi, tp_det=di, fn=fn, fp=fp, seen=seen, best=best)


def _report_html(sess, rec, i, thr):
    a = frame_analysis(sess, rec, i, thr)
    lines = [f"<b>{sess.label}</b>, all 40 frames with these cameras: {sess.summary(rec)}",
             f"frame {a['f']['frame']}: {len(a['gt'])} people, {len(a['tp_gt'])} found, "
             f"<span style='color:{C_FN}'>{len(a['fn'])} missed</span>, "
             f"<span style='color:{C_FP}'>{len(a['fp'])} false detections</span> (threshold {thr:.2f})"]
    for j in a['fn']:
        cams = sorted(sess.vis.get((a['f']['frame'], a['pids'][j]), set()) - set(rec['drop']))
        lines.append(f"&nbsp;&nbsp;missed person {a['pids'][j]}: seen by {len(cams)} working camera(s) "
                     f"({', '.join(f'C{c + 1}' for c in cams) or 'none'}), best score nearby {a['best'][j]:.2f}")
    for j in a['fp']:
        d = np.linalg.norm(a['gt'] - a['det'][j], axis=1).min() * 0.025 if len(a['gt']) else np.inf
        lines.append(f"&nbsp;&nbsp;false detection: score {a['score'][j]:.2f}, nearest person {d:.2f} m away")
    return '<br>'.join(lines)


# ------------------------------------------------------------------ drawing
def _bev_axes_limits(sess, pad=0.15):
    x0, x1, y0, y1 = -0.5, sess.X - 0.5, -0.5, sess.Y - 0.5
    px, py = pad * sess.X, pad * sess.Y
    cx = np.clip(sess.cam_mem[:, 0], x0 - px, x1 + px)
    cy = np.clip(sess.cam_mem[:, 1], y0 - py, y1 + py)
    return (min(x0, cx.min() - 3), max(x1, cx.max() + 3), max(y1, cy.max() + 3), min(y0, cy.min() - 3)), cx, cy


def draw_bev(ax, sess, rec, i, thr=0.5, background='heatmap', ids=True, queries=False, fov=True, title=True):
    a = frame_analysis(sess, rec, i, thr)
    f = a['f']
    drop = set(rec['drop'])
    ext = (-0.5, sess.X - 0.5, sess.Y - 0.5, -0.5)
    if background == 'heatmap' and 'heat' in f:
        im = ax.imshow(f['heat'], cmap='magma', vmin=0, vmax=1, extent=ext, interpolation='nearest')
        cbar_label = 'R2 proposal heatmap' if 'queries' in f else 'centre heatmap'
    elif background == 'coverage':
        cov = sum(sess.fov[s] for s in range(sess.S) if s not in drop)
        im = ax.imshow(cov, cmap='viridis', vmin=0, vmax=sess.S, extent=ext, interpolation='nearest')
        cbar_label = 'working cameras that see the cell'
    else:
        im = ax.imshow(np.zeros((sess.Y, sess.X)), cmap='gray', vmin=0, vmax=1, extent=ext)
        cbar_label = None
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    lim, cx, cy = _bev_axes_limits(sess)
    if fov:
        for s in range(sess.S):
            off = s in drop
            ax.contour(np.arange(sess.X), np.arange(sess.Y), sess.fov[s].astype(float), levels=[0.5],
                       colors=['#888888' if off else CAM_COLORS[s % len(CAM_COLORS)]], linewidths=0.8,
                       linestyles='--' if off else '-', alpha=0.5 if off else 0.8)
    for s in range(sess.S):
        off = s in drop
        ax.plot(cx[s], cy[s], marker='^', ms=9, color='#888888' if off else CAM_COLORS[s % len(CAM_COLORS)],
                mec='black', clip_on=False)
        ax.annotate(f'C{s + 1}' + (' off' if off else ''), (cx[s], cy[s]), xytext=(4, 4), textcoords='offset points',
                    fontsize=8, color='#888888' if off else 'black', clip_on=False)
    if queries and 'queries' in f:
        q = f['queries'][-1]
        ax.scatter(f['anchors'][:, 0], f['anchors'][:, 1], s=6, marker='+', c='#bbbbbb', linewidths=0.6,
                   label='query start (proposal)')
        ax.scatter(q['xy'][:, 0], q['xy'][:, 1], s=10, c=q['score'], cmap='cool', vmin=0, vmax=1,
                   label='query after 3 rounds (colour = score)')
    gt_m = sess.ref2mem(a['gt']) if len(a['gt']) else np.zeros((0, 2))
    det_m = sess.ref2mem(a['det']) if len(a['det']) else np.zeros((0, 2))
    if len(gt_m):
        ax.scatter(gt_m[:, 0], gt_m[:, 1], s=8, c=C_GT, edgecolors='black', linewidths=0.3, zorder=4)
    for g, d in zip(a['tp_gt'], a['tp_det']):
        ax.plot([gt_m[g, 0], det_m[d, 0]], [gt_m[g, 1], det_m[d, 1]], color=C_TP, lw=0.8, zorder=4)
    if len(a['tp_det']):
        ax.scatter(det_m[a['tp_det'], 0], det_m[a['tp_det'], 1], s=28, c=C_TP, edgecolors='black', linewidths=0.4,
                   zorder=5, label='detected')
    if len(a['fn']):
        ax.scatter(gt_m[a['fn'], 0], gt_m[a['fn'], 1], s=90, facecolors='none', edgecolors=C_FN, linewidths=1.8,
                   zorder=5, label='missed (number = working cameras that see them)')
        for j in a['fn']:
            ax.annotate(str(a['seen'][j]), gt_m[j], xytext=(5, -9), textcoords='offset points', fontsize=8,
                        color=C_FN, fontweight='bold')
    if len(a['fp']):
        ax.scatter(det_m[a['fp'], 0], det_m[a['fp'], 1], s=60, marker='x', c=C_FP, linewidths=2, zorder=5,
                   label='false detection')
    if ids:
        hyp = rec['hyps'][i]
        if hyp:
            hm = sess.ref2mem(np.array([h[1] for h in hyp]))
            for (tid, _), p in zip(hyp, hm):
                ax.annotate(str(tid), p, xytext=(-4, 6), textcoords='offset points', fontsize=7, color='#00e5ff')
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#222222')
    if title:
        on = [f'C{s + 1}' for s in range(sess.S) if s not in drop]
        ax.set_title(f"{sess.label} | frame {f['frame']} | cameras on: {' '.join(on)} | "
                     f"found {len(a['tp_gt'])}/{len(a['gt'])}, missed {len(a['fn'])}, false {len(a['fp'])}",
                     fontsize=9)
    return a


def draw_cameras(axes, sess, rec, i, a, step=3):
    f = a['f']
    drop = set(rec['drop'])
    img = sess.items[i]['img'][0]  # S,3,H,W uint8
    people = [(a['gt'][j], C_TP if j in set(a['tp_gt']) else C_FN) for j in range(len(a['gt']))]
    people += [(a['det'][j], C_FP) for j in a['fp']]
    if people:
        xy = np.array([p[0] for p in people])
        foot, ok_f = sess.project(np.c_[xy, np.full(len(xy), sess.model.bounds[4])])
        head, ok_h = sess.project(np.c_[xy, np.full(len(xy), sess.z_head)])
    for s, ax in enumerate(axes):
        im = img[s].permute(1, 2, 0).numpy()[::step, ::step]
        off = s in drop
        ax.imshow((im * 0.3).astype(np.uint8) if off else im)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#888888' if off else CAM_COLORS[s % len(CAM_COLORS)])
            sp.set_linewidth(2)
        n_seen = sum(1 for p in a['pids'] if s in sess.vis.get((f['frame'], p), set()))
        ax.set_title(f'C{s + 1}' + (' OFF' if off else f'  ({n_seen} people annotated)'), fontsize=8,
                     color='#888888' if off else 'black')
        if off:
            ax.text(0.5, 0.5, 'OFF', transform=ax.transAxes, ha='center', va='center', fontsize=22,
                    color='white', alpha=0.8)
            continue
        for k, (_, color) in enumerate(people):
            if not (ok_f[s, k] or ok_h[s, k]):
                continue
            (uf, vf), (uh, vh) = foot[s, k] / step, head[s, k] / step
            h = abs(vf - vh)
            w = 0.38 * h
            ax.add_patch(Rectangle(((uf + uh) / 2 - w / 2, min(vf, vh)), w, h, fill=False, ec=color,
                                   lw=1.4 if color != C_TP else 0.9))
        ax.set_xlim(0, im.shape[1])
        ax.set_ylim(im.shape[0], 0)


def render(sess, rec, i, thr=0.5, background='heatmap', ids=True, queries=False, fov=True, cameras=True,
           width=16, dpi=80):
    """Figure of frame i: BEV on top, camera images below."""
    aspect = sess.X / sess.Y
    bev_h = min(6.5, width * 0.9 / aspect)
    cols = 4 if sess.S > 6 else 3
    rows = int(np.ceil(sess.S / cols)) if cameras else 0
    cam_h = width / cols * 0.5625 + 0.25
    fig = Figure(figsize=(width, bev_h + rows * cam_h), dpi=dpi)
    gs = fig.add_gridspec(1 + rows, cols, height_ratios=[bev_h] + [cam_h] * rows, hspace=0.25, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    a = draw_bev(ax, sess, rec, i, thr, background, ids, queries, fov)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower left', fontsize=7, framealpha=0.8, bbox_to_anchor=(0, -0.02))
    if cameras:
        axes = [fig.add_subplot(gs[1 + s // cols, s % cols]) for s in range(sess.S)]
        draw_cameras(axes, sess, rec, i, a)
    return fig


def _show(fig, dpi=None):
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi or fig.dpi, bbox_inches='tight')
    display(Image(buf.getvalue()))


def _parse_drop(drop):
    if drop is None:
        return ()
    if isinstance(drop, (int, np.integer)):
        return (int(drop),)
    return tuple(int(c) for c in drop)


# ------------------------------------------------------------------ interactive tools
def explore(which, frame=0, drop=(), cameras=True):
    """Interactive viewer. Frame slider (or Play), one button per camera (click to switch it off: the model is
    re-run on all 40 frames with that set of cameras, a few seconds on a GPU), detection threshold, background
    (model heatmap / how many working cameras see each cell), layers. The text under the figure lists every miss
    and false detection of the frame."""
    import ipywidgets as W
    from IPython.display import display, HTML
    sess = session(which)
    n = len(sess.items)
    w_frame = W.IntSlider(value=min(frame, n - 1), min=0, max=n - 1, description='frame', continuous_update=False,
                          layout=W.Layout(width='420px'))
    w_play = W.Play(value=w_frame.value, min=0, max=n - 1, interval=1500, description='play')
    W.link((w_play, 'value'), (w_frame, 'value'))
    w_cams = [W.ToggleButton(value=s not in _parse_drop(drop), description=f'C{s + 1}',
                             layout=W.Layout(width='52px')) for s in range(sess.S)]
    w_thr = W.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description='threshold', continuous_update=False)
    w_bg = W.Dropdown(options=[('model heatmap', 'heatmap'), ('working-camera coverage', 'coverage'),
                               ('none', 'none')], value='heatmap', description='background')
    w_img = W.Checkbox(value=cameras, description='camera images', indent=False)
    w_ids = W.Checkbox(value=True, description='track IDs', indent=False)
    w_fov = W.Checkbox(value=True, description='fields of view', indent=False)
    w_q = W.Checkbox(value=False, description='R2 queries', indent=False, disabled=sess.variant != 'R2')
    w_all = W.Button(description='all cameras on', layout=W.Layout(width='130px'))
    w_status = W.HTML()
    out, info = W.Output(), W.HTML()

    def drop_now():
        return tuple(s for s, b in enumerate(w_cams) if not b.value)

    def redraw(*_):
        for s, b in enumerate(w_cams):
            b.button_style = '' if b.value else 'danger'
        d = drop_now()
        if len(d) == sess.S:
            w_status.value = '<span style="color:red">switch at least one camera on</span>'
            return
        if (d, True) not in sess.runs:
            w_status.value = f'<i>running the model with {", ".join(f"C{c + 1}" for c in d) or "all cameras"} ' \
                             f'{"off" if d else "on"} on all {n} frames ...</i>'
        rec = sess.run(d, quiet=True)
        w_status.value = ''
        fig = render(sess, rec, w_frame.value, w_thr.value, w_bg.value, w_ids.value, w_q.value, w_fov.value,
                     w_img.value)
        with out:
            out.clear_output(wait=True)
            _show(fig)
        info.value = _report_html(sess, rec, w_frame.value, w_thr.value)

    def all_on(_):
        for b in w_cams:
            b.unobserve(redraw, names='value')
            b.value = True
            b.observe(redraw, names='value')
        redraw()

    for w in [w_frame, w_thr, w_bg, w_img, w_ids, w_fov, w_q] + w_cams:
        w.observe(redraw, names='value')
    w_all.on_click(all_on)
    display(W.VBox([W.HBox([w_play, w_frame, w_thr]), W.HBox([W.Label('cameras:')] + w_cams + [w_all]),
                    W.HBox([w_bg, w_img, w_ids, w_fov, w_q]), w_status, out, info]))
    display(HTML('<small>green = found, orange circle = missed (number: working cameras that see the person), '
                 'red x = false detection, cyan numbers = track IDs, triangles = cameras '
                 '(coloured outline = what each camera sees at mid-body height)</small>'))
    redraw()


def compare(a, b, frame=0, drop=()):
    """Two checkpoints of the same dataset side by side (same frame, cameras and threshold)."""
    import ipywidgets as W
    from IPython.display import display
    sa, sb = session(a), session(b)
    assert sa.dataset == sb.dataset, f'different datasets: {sa.dataset} vs {sb.dataset}'
    n = len(sa.items)
    w_frame = W.IntSlider(value=min(frame, n - 1), min=0, max=n - 1, description='frame', continuous_update=False,
                          layout=W.Layout(width='420px'))
    w_play = W.Play(value=w_frame.value, min=0, max=n - 1, interval=1500)
    W.link((w_play, 'value'), (w_frame, 'value'))
    w_cams = [W.ToggleButton(value=s not in _parse_drop(drop), description=f'C{s + 1}',
                             layout=W.Layout(width='52px')) for s in range(sa.S)]
    w_thr = W.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description='threshold', continuous_update=False)
    w_bg = W.Dropdown(options=[('model heatmap', 'heatmap'), ('working-camera coverage', 'coverage'),
                               ('none', 'none')], value='heatmap', description='background')
    out, info, status = W.Output(), W.HTML(), W.HTML()

    def redraw(*_):
        for b_ in w_cams:
            b_.button_style = '' if b_.value else 'danger'
        d = tuple(s for s, b_ in enumerate(w_cams) if not b_.value)
        if len(d) == sa.S:
            status.value = '<span style="color:red">switch at least one camera on</span>'
            return
        status.value = '<i>running ...</i>'
        ra, rb = sa.run(d, quiet=True), sb.run(d, quiet=True)
        status.value = ''
        wide = sa.X / sa.Y > 2
        fig = Figure(figsize=(16, 11 if wide else 7), dpi=80)
        axes = fig.subplots(2, 1) if wide else fig.subplots(1, 2)
        for ax, s_, r_ in zip(axes, (sa, sb), (ra, rb)):
            draw_bev(ax, s_, r_, w_frame.value, w_thr.value, w_bg.value)
        with out:
            out.clear_output(wait=True)
            _show(fig)
        info.value = f'<b>{sa.label}</b>: {sa.summary(ra)}<br><b>{sb.label}</b>: {sb.summary(rb)}<br>' \
                     '(metrics over all test frames with these cameras)'

    for w in [w_frame, w_thr, w_bg] + w_cams:
        w.observe(redraw, names='value')
    display(W.VBox([W.HBox([w_play, w_frame, w_thr, w_bg]), W.HBox([W.Label('cameras:')] + w_cams), status, out,
                    info]))
    redraw()


def camera_evidence(which, frame=0):
    """What each camera contributes in one frame: heatmap with all cameras minus heatmap without that camera
    (red: evidence only this camera provides; a person who turns dark when a camera is removed depended on it)."""
    sess = session(which)
    i = int(frame)
    item = [sess.items[i]]
    full = r2_eval.run_inference(sess.model, item, (), keep_maps=True)[0]
    rec = dict(drop=(), frames=[full], hyps=[[]])
    cols = 2 if sess.X / sess.Y > 2 else 3
    rows = int(np.ceil((sess.S + 1) / cols))
    fig = Figure(figsize=(16, rows * 16 / cols * sess.Y / sess.X * 1.15 + 0.5), dpi=80)
    axes = fig.subplots(rows, cols).ravel()
    draw_bev(axes[0], sess, rec, 0, background='heatmap', ids=False, title=False)
    axes[0].set_title(f'{sess.label} | frame {full["frame"]} | all cameras: heatmap', fontsize=9)
    gt_m = sess.ref2mem(full['gt'][:, :2]) if len(full['gt']) else np.zeros((0, 2))
    ext = (-0.5, sess.X - 0.5, sess.Y - 0.5, -0.5)
    for s in range(sess.S):
        ax = axes[s + 1]
        without = r2_eval.run_inference(sess.model, item, (s,), keep_maps=True)[0]
        diff = full['heat'] - without['heat']
        ax.imshow(diff, cmap='RdBu_r', vmin=-0.8, vmax=0.8, extent=ext, interpolation='nearest')
        ax.contour(np.arange(sess.X), np.arange(sess.Y), sess.fov[s].astype(float), levels=[0.5],
                   colors=[CAM_COLORS[s % len(CAM_COLORS)]], linewidths=1)
        if len(gt_m):
            ax.scatter(gt_m[:, 0], gt_m[:, 1], s=10, facecolors='none', edgecolors='black', linewidths=0.6)
        lim, cx, cy = _bev_axes_limits(sess)
        ax.plot(cx[s], cy[s], marker='^', ms=9, color=CAM_COLORS[s % len(CAM_COLORS)], mec='black', clip_on=False)
        a = frame_analysis(sess, dict(drop=(s,), frames=[without], hyps=[[]]), 0)
        ax.set_title(f'C{s + 1}: all − without C{s + 1}  (without it: {len(a["fn"])} missed, {len(a["fp"])} false)',
                     fontsize=9)
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[sess.S + 1:]:
        ax.axis('off')
    fig.suptitle('red = heatmap evidence that disappears without the camera, blue = what the camera suppresses; '
                 'circles = people', fontsize=10)
    _show(fig)


def tracks(which, drop=()):
    """Trajectories of all test frames: ground truth (grey), tracks (coloured by ID), ID switches (red stars),
    misses (small orange dots)."""
    sess = session(which)
    rec = sess.run(_parse_drop(drop))
    fig = Figure(figsize=(16, min(9, 16 / (sess.X / sess.Y)) + 0.6), dpi=80)
    ax = fig.subplots()
    ax.imshow(np.zeros((sess.Y, sess.X)), cmap='gray', vmin=0, vmax=1, extent=(-0.5, sess.X - 0.5, sess.Y - 0.5, -0.5))
    gt_pos = {}
    traj = {}
    for f in rec['frames']:
        for x, y, pid in f['gt']:
            gt_pos[(f['frame'], int(pid))] = (x, y)
            traj.setdefault(('gt', int(pid)), []).append((x, y))
    for (_, pid), pts in traj.items():
        m = sess.ref2mem(np.array(pts))
        ax.plot(m[:, 0], m[:, 1], color='#777777', lw=1)
    htraj = {}
    for f, hyp in zip(rec['frames'], rec['hyps']):
        for tid, xy in hyp:
            htraj.setdefault(tid, []).append(xy)
    import matplotlib
    cmap = matplotlib.colormaps['tab20']
    for tid, pts in htraj.items():
        m = sess.ref2mem(np.array(pts))
        ax.plot(m[:, 0], m[:, 1], color=cmap(tid % 20), lw=1.6)
        ax.annotate(str(tid), m[-1], fontsize=7, color=cmap(tid % 20))
    ev = rec['acc'].mot_events.reset_index()
    sw = ev[ev.Type == 'SWITCH']
    if len(sw):
        pts = np.array([gt_pos[(int(r.FrameId), int(r.OId))] for r in sw.itertuples()
                        if (int(r.FrameId), int(r.OId)) in gt_pos])
        if len(pts):
            m = sess.ref2mem(pts)
            ax.scatter(m[:, 0], m[:, 1], marker='*', s=160, c=C_FP, edgecolors='white', zorder=6, label='ID switch')
    miss = ev[ev.Type == 'MISS']
    if len(miss):
        pts = np.array([gt_pos[(int(r.FrameId), int(r.OId))] for r in miss.itertuples()
                        if (int(r.FrameId), int(r.OId)) in gt_pos])
        if len(pts):
            m = sess.ref2mem(pts)
            ax.scatter(m[:, 0], m[:, 1], s=10, c=C_FN, zorder=5, label='missed by the tracker')
    lim, cx, cy = _bev_axes_limits(sess)
    for s in range(sess.S):
        off = s in rec['drop']
        ax.plot(cx[s], cy[s], marker='^', ms=9, color='#888888' if off else CAM_COLORS[s % len(CAM_COLORS)],
                mec='black', clip_on=False)
        ax.annotate(f'C{s + 1}' + (' off' if off else ''), (cx[s], cy[s]), xytext=(4, 4), textcoords='offset points',
                    fontsize=8, clip_on=False)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='lower left', fontsize=8)
    ax.set_title(f"{sess.label} | {'cameras ' + ' '.join(f'C{c + 1}' for c in rec['drop']) + ' off' if rec['drop'] else 'all cameras'}"
                 f" | {sess.summary(rec)} | grey = ground truth, colours = track IDs", fontsize=9)
    _show(fig)


def robustness(*which, max_drop=3):
    """MODA, MOTA, IDF1, false detections and misses against the number of removed cameras (mean over every
    combination of k cameras). Uses the pipeline's evaluation (eval/<run>/metrics.csv) when there is one, otherwise
    computes it (about 4 minutes per checkpoint on a GPU for WILDTRACK with up to 3 cameras removed)."""
    import pandas as pd
    curves = []
    for w in which:
        path = _resolve(w)
        ev = _eval_dir(path)
        csv = osp.join(ev, 'metrics.csv') if ev else None
        label = None
        if csv and osp.exists(csv):
            df = pd.read_csv(csv)
            _, hp, _ = _HP_CACHE.get(path, (None, r2_eval.read_hparams(path)[0], None))
            label = f'{r2_eval.variant_of(hp)} {r2_eval.dataset_of(hp)}' + \
                (f' s{_seed_of(path)}' if _seed_of(path) is not None else '')
            print(f'{label}: from {csv}')
        else:
            sess = session(w)
            label = sess.label
            rows = []
            conds = [c for c in r2_eval.camera_conditions(sess.S, max_drop)]
            print(f'{label}: computing {len(conds)} camera conditions ...', flush=True)
            t0 = time.time()
            for c in conds:
                rec = sess.run(c, maps=False, quiet=True)
                rows.append(dict(k=len(c), **rec['det'], **rec['trk']))
            print(f'  done in {time.time() - t0:.0f}s')
            df = pd.DataFrame(rows)
        curves.append((label, df.groupby('k')[['moda', 'mota', 'idf1', 'det_fp', 'det_fn']].mean()))
    fig = Figure(figsize=(17, 3.6), dpi=80)
    axes = fig.subplots(1, 5)
    for ax, (key, name) in zip(axes, [('moda', 'MODA'), ('mota', 'MOTA'), ('idf1', 'IDF1'),
                                      ('det_fp', 'false detections'), ('det_fn', 'missed people')]):
        for label, c in curves:
            ax.plot(c.index, c[key], marker='o', label=label)
        ax.set_title(name)
        ax.set_xlabel('cameras removed')
        ax.set_xticks(list(curves[0][1].index))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    _show(fig)
    return {label: c for label, c in curves}


def video(which, drop=(), path=None, fps=2, thr=0.5, cameras=True, background='heatmap'):
    """Renders the viewer for every test frame into an MP4 (GIF if ffmpeg is missing) and shows it."""
    from IPython.display import Video, Image, display
    sess = session(which)
    rec = sess.run(_parse_drop(drop))
    tag = f"{sess.label.replace(' ', '_')}_{'off' + ''.join(str(c + 1) for c in rec['drop']) if rec['drop'] else 'all'}"
    if path is None:
        os.makedirs(VIDEO_DIR if osp.isdir(osp.dirname(VIDEO_DIR)) else '/content/videos', exist_ok=True)
        base = VIDEO_DIR if osp.isdir(VIDEO_DIR) else '/content/videos'
        path = osp.join(base, f'{tag}.mp4')
    frames = []
    print(f'rendering {len(rec["frames"])} frames ...', end=' ', flush=True)
    for i in range(len(rec['frames'])):
        fig = render(sess, rec, i, thr, background, cameras=cameras, width=14, dpi=70)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        from PIL import Image as PILImage
        im = PILImage.open(buf).convert('RGB')
        frames.append(im.resize((im.width // 2 * 2, im.height // 2 * 2)))
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, [np.asarray(f) for f in frames], fps=fps, macro_block_size=1)
        print(f'-> {path}')
        display(Video(path, embed=True, width=900))
    except Exception as e:  # noqa: BLE001  (no ffmpeg backend)
        path = osp.splitext(path)[0] + '.gif'
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        print(f'(mp4 failed: {str(e)[:60]}) -> {path}')
        display(Image(path))
    return path


def help():  # noqa: A001
    print(__doc__)


# ------------------------------------------------------------------ on %run
if __name__ == '__main__':
    mount_drive()
    print(f'device: {DEVICE}' + ('' if DEVICE == 'cuda' else '  (no GPU: switching cameras takes ~1 min per change)'))
    checkpoints()
    print('\nexplore(#)  compare(#, #)  camera_evidence(#, frame)  tracks(#, drop=[...])  robustness(#, #)  '
          'video(#)  checkpoints()  help()')
