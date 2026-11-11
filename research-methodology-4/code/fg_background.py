"""
Research methodology 4: per-camera background images, and a check of the foreground masks they give.

    python research-methodology-4/code/fg_background.py --data_dir /content/data/Wildtrack --out <results>/m4/fg_background.npz

Writes, next to --out:
  fg_background.npz  per-camera median of the training frames (the first 90 % of the sequence, as the standard split;
                     no test frame is used), at 1/4 resolution, with the original intrinsics and image size
  fg_check.csv       per camera, on every 12th training frame at the test-time input size (720 x 1280): mean mask
                     inside the annotated person boxes, outside them, and the share of the image marked foreground
  fg_preview.jpg     one training frame per camera: image | background | mask
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.normpath(osp.join(HERE, '..', '..', 'WorldTrack')))
from datasets.wildtrack_dataset import Wildtrack  # noqa: E402
from models.foreground import foreground_mask, LO, HI  # noqa: E402

SCALE = 4  # background stored at 1/4 of 1080 x 1920
TRAIN_FRAC = 0.9  # PedestrianDataset's standard split
INPUT_HW = (720, 1280)  # test-time input size (configs/d_wildtrack.yml final_dim)
CHECK_EVERY = 12


def boxes(root, frame, cam):
    """Annotated person boxes of one camera in original pixels, x1 y1 x2 y2."""
    with open(osp.join(root, 'annotations_positions', f'{frame:08d}.json')) as f:
        peds = json.load(f)
    out = [[p['views'][cam][k] for k in ('xmin', 'ymin', 'xmax', 'ymax')] for p in peds]
    return np.array([b for b in out if b != [-1, -1, -1, -1]], dtype=np.float32).reshape(-1, 4)


def mask_for(path, cam, bg, K0, size0):
    """Foreground mask of one original image at the test-time input size, as the model computes it."""
    W0, H0 = size0
    img = Image.open(path).convert('RGB').resize(INPUT_HW[::-1], Image.NEAREST)
    rgb = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()[None] / 255
    K = torch.eye(4)[None]
    K[0, :3, :3] = torch.from_numpy(K0[cam])
    K[0, 0] *= INPUT_HW[1] / W0
    K[0, 1] *= INPUT_HW[0] / H0
    m = foreground_mask(rgb, K, bg[cam:cam + 1], torch.from_numpy(K0[cam:cam + 1]), torch.tensor(size0),
                        (INPUT_HW[0] // 4, INPUT_HW[1] // 4))
    return np.asarray(img), m[0, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out_dir = osp.dirname(osp.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    base = Wildtrack(args.data_dir)
    H0, W0 = base.img_shape
    train = range(0, int(base.num_frame * TRAIN_FRAC))
    paths = base.get_image_fpaths(train)
    if not osp.exists(args.out):
        bgs = []
        for cam in range(base.num_cam):
            frames = sorted(paths[cam])
            stack = np.stack([np.asarray(Image.open(paths[cam][f]).convert('RGB').resize(
                (W0 // SCALE, H0 // SCALE), Image.BILINEAR)) for f in frames])
            bgs.append(np.median(stack, axis=0).astype(np.uint8).transpose(2, 0, 1))
            print(f'camera {cam}: median of {len(frames)} training frames (frames {frames[0]}-{frames[-1]})',
                  flush=True)
        K0 = np.stack([np.asarray(k, dtype=np.float32) for k in base.intrinsic_matrices])
        np.savez_compressed(args.out + '.part.npz', background=np.stack(bgs), intrinsic=K0,
                            image_size=np.array([W0, H0], dtype=np.float32))
        os.replace(args.out + '.part.npz', args.out)
    print(f'background -> {args.out}')

    d = np.load(args.out)
    bg = torch.from_numpy(d['background']).float() / 255
    K0, size0 = d['intrinsic'], tuple(float(x) for x in d['image_size'])
    rows, preview = [], []
    for cam in range(base.num_cam):
        inside, outside, cover = [], [], []
        frames = sorted(paths[cam])[::CHECK_EVERY]
        for f in frames:
            img, m = mask_for(paths[cam][f], cam, bg, K0, size0)
            sy, sx = m.shape[0] / H0, m.shape[1] / W0
            person = np.zeros_like(m, dtype=bool)
            for x1, y1, x2, y2 in boxes(args.data_dir, f, cam):
                person[max(int(y1 * sy), 0):int(np.ceil(y2 * sy)), max(int(x1 * sx), 0):int(np.ceil(x2 * sx))] = True
            if person.any():
                inside.append(m[person].mean())
            outside.append(m[~person].mean())
            cover.append((m > 0.5).mean())
            if f == frames[len(frames) // 2]:
                small = lambda a: np.asarray(Image.fromarray(a).resize((320, 180), Image.BILINEAR))
                bimg = (d['background'][cam].transpose(1, 2, 0)).astype(np.uint8)
                mimg = np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=2)
                preview.append(np.concatenate([small(img), small(bimg), small(mimg)], axis=1))
        rows.append(dict(camera=cam, frames=len(frames), mask_inside_persons=float(np.mean(inside)),
                         mask_outside_persons=float(np.mean(outside)), share_marked_foreground=float(np.mean(cover))))
        r = rows[-1]
        print(f'camera {cam}: mean mask inside person boxes {r["mask_inside_persons"]:.2f}, outside '
              f'{r["mask_outside_persons"]:.2f}, share of image foreground {r["share_marked_foreground"]:.2f}',
              flush=True)
    with open(osp.join(out_dir, 'fg_check.csv'), 'w') as f:
        f.write(f'# soft threshold LO={LO} HI={HI}; training frames only\n')
        f.write(','.join(rows[0]) + '\n')
        for r in rows:
            f.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r.values()) + '\n')
    Image.fromarray(np.concatenate(preview, axis=0)).save(osp.join(out_dir, 'fg_preview.jpg'), quality=85)
    print(f'check -> {out_dir}/fg_check.csv, preview -> {out_dir}/fg_preview.jpg')


if __name__ == '__main__':
    main()
