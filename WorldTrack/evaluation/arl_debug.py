"""
Sanity checks of the ARL training additions (about a minute on Colab, no training).

  1. A1 + A2, visual: for a few training samples with occlusion and scene augmentation forced on, saves the 7
     augmented views with the BEV ground truth projected back into every view (green dots at the feet, using the
     sample's own augmented calibration) and the BEV target heatmap. If the scene augmentation is consistent, the
     dots sit on people's feet in every view (also on people hidden by a grey occluder); grey boxes are occluders.
     The dots are also checked numerically: the fraction that falls inside an annotated person box of that view.
  2. A3, numeric: align_prev_bev must reproduce an exact BEV flip, and must leave BEV(t-1) unchanged when the two
     frames share the same transform.

    python -m evaluation.arl_debug --data_dir /content/data/Wildtrack --out <results>/arl_debug
"""
import argparse
import json
import os
import os.path as osp
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.pedestrian_dataset import PedestrianDataset
from datasets.wildtrack_dataset import Wildtrack
from datasets.multiviewx_dataset import MultiviewX
from utils import vox

RESOLUTION = {'wildtrack': ((120, 4, 360), (0, 1440, 0, 480, 0, 200)),
              'multiviewx': ((160, 2, 250), (0, 1000, 0, 640, 0, 2))}


def bev_points_ref(target, dataset):
    """BEV ground-truth positions (augmented reference coordinates) from the sample's own targets."""
    valid = target['valid_bev'][0].nonzero(as_tuple=False)  # N,2 (row y, col x)
    if len(valid) == 0:
        return torch.zeros(0, 3)
    ys, xs = valid[:, 0].float(), valid[:, 1].float()
    off = target['offset_bev'][:2, valid[:, 0], valid[:, 1]].T  # N,2 (dx, dy)
    mem = torch.stack([xs + off[:, 0], ys + off[:, 1], torch.zeros_like(xs)], -1).unsqueeze(0)
    return dataset.vox_util.Mem2Ref(mem, dataset.Y, dataset.Z, dataset.X)[0]


def project(item, ref_pts):
    """ref points (N,3) -> pixel coords per camera (S,N,2) and in-front mask (S,N)"""
    N = len(ref_pts)
    pts = torch.cat([ref_pts[:, :2], torch.zeros(N, 1), torch.ones(N, 1)], 1).T  # 4,N  (feet at z = 0)
    global_T_ref = torch.inverse(item['ref_T_global'])
    out, front = [], []
    for K, E in zip(item['intrinsic'], item['extrinsic']):
        p = K @ E @ global_T_ref @ pts
        z = p[2]
        out.append((p[:2] / torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)).T)
        front.append(z > 0)
    return torch.stack(out), torch.stack(front)


def check_views(dataset, idx, out_dir):
    item, target = dataset[idx]
    ref = bev_points_ref(target, dataset)
    uv, front = project(item, ref)
    S, _, H, W = item['img'].shape
    tiles, hits, total = [], 0, 0
    for s in range(S):
        img = Image.fromarray((item['img'][s].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        # annotated person boxes of this view (image GT centres/sizes are on the H/4 grid)
        valid = target['valid_img'][s, 0].nonzero(as_tuple=False)
        boxes = []
        for (r, c) in valid.tolist():
            cx = (c + target['offset_img'][s, 0, r, c].item()) * 4
            by = (r + target['offset_img'][s, 1, r, c].item()) * 4  # foot point (bottom of the box)
            w, h = (target['size_img'][s, :, r, c] * 4).tolist()
            boxes.append((cx - w / 2, by - h, cx + w / 2, by + h))  # tolerant to which box edge is the 'foot'
        for (x, y), f in zip(uv[s].tolist(), front[s].tolist()):
            if not f or not (0 <= x < W and 0 <= y < H):
                continue
            total += 1
            inside = any(b[0] - 8 <= x <= b[2] + 8 and b[1] <= y <= b[3] + 8 for b in boxes)
            hits += inside
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 255, 0) if inside else (255, 0, 0), width=3)
        tiles.append(img.resize((W // 2, H // 2)))
    heat = target['center_bev'][0].numpy()
    heat = Image.fromarray((255 * heat / max(heat.max(), 1e-6)).astype(np.uint8)).convert('RGB')
    heat = heat.resize((W // 2, int(W // 2 * heat.height / heat.width)))
    cols, tw, th = 4, W // 2, H // 2
    canvas = Image.new('RGB', (cols * tw, 2 * th + heat.height), (40, 40, 40))
    for k, t in enumerate(tiles):
        canvas.paste(t, ((k % cols) * tw, (k // cols) * th))
    canvas.paste(heat, (0, 2 * th))
    flipped = not torch.allclose(item['scene_T'][:2, :2], torch.eye(2))
    path = osp.join(out_dir, f'sample_{idx:03d}{"_flipped" if flipped else ""}.jpg')
    canvas.save(path, quality=85)
    return {'sample': idx, 'scene_T': item['scene_T'].tolist(), 'flipped': flipped, 'projected_feet': total,
            'inside_a_person_box': hits, 'fraction': hits / total if total else float('nan'), 'image': path}


def check_alignment(Y, Z, X, bounds):
    from world_track import WorldTrackModel
    vu = vox.VoxelUtil(Y, Z, X, scene_centroid=torch.zeros(1, 3), bounds=bounds)
    me = SimpleNamespace(vox_util=vu, Y=Y, Z=Z, X=X)
    xmin, xmax, ymin, ymax = bounds[:4]
    prev = torch.randn(1, 8, Y, X)
    eye = torch.eye(3).unsqueeze(0)
    fx = torch.tensor([[[-1., 0, xmin + xmax], [0, 1, 0], [0, 0, 1]]])
    fy = torch.tensor([[[1., 0, 0], [0, -1, ymin + ymax], [0, 0, 1]]])
    res = {}
    res['identity_unchanged'] = float((WorldTrackModel.align_prev_bev(me, prev, eye, eye) - prev).abs().max())
    res['x_flip_error'] = float((WorldTrackModel.align_prev_bev(me, prev, eye, fx) - prev.flip(-1)).abs().max())
    res['y_flip_error'] = float((WorldTrackModel.align_prev_bev(me, prev, eye, fy) - prev.flip(-2)).abs().max())
    res['flip_then_unflip_error'] = float(
        (WorldTrackModel.align_prev_bev(me, prev.flip(-1), fx, eye) - prev).abs().max())
    res['ok'] = all(v < 1e-4 for k, v in res.items())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--scene_mode', default='flip', choices=['flip', 'flip_affine'])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = 'wildtrack' if 'wildtrack' in osp.basename(osp.normpath(args.data_dir)).lower() else 'multiviewx'
    (Y, Z, X), bounds = RESOLUTION[name]
    base = Wildtrack(args.data_dir) if name == 'wildtrack' else MultiviewX(args.data_dir)

    report = {'alignment (A3)': check_alignment(Y, Z, X, bounds)}
    print('A3 cache alignment:', report['alignment (A3)'])

    rows = {}
    for label, kw in [('none', {}), ('A1_occlusion', dict(occlusion_prob=1.0)),
                      ('A2_scene', dict(scene_aug_prob=1.0, scene_aug_mode=args.scene_mode)),
                      ('A1A2', dict(occlusion_prob=1.0, scene_aug_prob=1.0, scene_aug_mode=args.scene_mode))]:
        np.random.seed(0)
        torch.manual_seed(0)
        ds = PedestrianDataset(base, is_train=True, resolution=(Y, Z, X), bounds=bounds, **kw)
        d = osp.join(args.out, label)
        os.makedirs(d, exist_ok=True)
        idx = np.linspace(0, len(ds) - 1, args.samples).astype(int)
        res = [check_views(ds, int(i), d) for i in idx]
        frac = np.nanmean([r['fraction'] for r in res])
        rows[label] = {'mean_fraction_of_feet_inside_a_person_box': float(frac), 'samples': res}
        print(f'{label:13s} projected GT feet inside an annotated person box: {frac:.1%}  -> {d}')
    report['views'] = rows
    ref = rows['none']['mean_fraction_of_feet_inside_a_person_box']
    report['verdict'] = {k: ('OK' if abs(v['mean_fraction_of_feet_inside_a_person_box'] - ref) < 0.05 else
                             'CHECK: differs from the un-augmented samples')
                         for k, v in rows.items()}
    print('verdict (should all be OK; occlusion does not move people):', report['verdict'])
    json.dump(report, open(osp.join(args.out, 'arl_debug.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
