"""
Diagnoses why a checkpoint scores badly: data/calibration alignment, then the model under each float32
matmul precision setting.

    python -m evaluation.diagnose --ckpt <official.ckpt> --data_dir /content/data/Wildtrack
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch


def calibration_check(data_dir):
    """Projects annotated ground positions into every camera and compares them with the annotated box feet."""
    from datasets.wildtrack_dataset import Wildtrack
    base = Wildtrack(data_dir)
    ann_dir = osp.join(data_dir, 'annotations_positions')
    files = sorted(os.listdir(ann_dir))[-40:]
    errs = {c: [] for c in range(base.num_cam)}
    for fname in files:
        for ped in json.load(open(osp.join(ann_dir, fname))):
            gx, gy = base.get_worldgrid_from_pos(ped['positionID'])
            wx, wy, _ = base.worldcoord_from_worldgrid_mat @ np.array([gx, gy, 1.])
            for c in range(base.num_cam):
                v = ped['views'][c]
                if v['xmax'] == -1:
                    continue
                cam = base.extrinsic_matrices[c] @ np.array([wx, wy, 0., 1.])
                uv = base.intrinsic_matrices[c] @ cam
                if uv[2] <= 0:
                    continue
                foot = np.array([(v['xmin'] + v['xmax']) / 2, v['ymax']])
                errs[c].append(np.linalg.norm(uv[:2] / uv[2] - foot))
    print('\n== 1. calibration vs annotations (median |projected ground point - box foot| in pixels, 1920x1080)')
    for c, e in errs.items():
        print(f'   camera C{c + 1}: {np.median(e):7.1f} px over {len(e)} boxes')
    med = np.median([np.median(e) for e in errs.values() if e])
    print('   ->', 'OK' if med < 40 else 'MISALIGNED: calibration or annotations do not match the images/each other')
    from PIL import Image
    first = sorted(os.listdir(osp.join(data_dir, 'Image_subsets', 'C1')))
    print(f'   images: {len(first)} per camera, first {first[:2]}, size {Image.open(osp.join(data_dir, "Image_subsets", "C1", first[0])).size}')
    print(f'   annotations: {len(os.listdir(ann_dir))}, last {files[-2:]}')


def model_check(ckpt, data_dir):
    from evaluation.retrack_eval import load_model, load_test_items, run_inference, detection_metrics
    print('\n== 2. official checkpoint under each float32 matmul precision (all 7 cameras, threshold 0.5)')
    print(f'   torch {torch.__version__}, cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}')
    model, _ = load_model(ckpt, 'cuda' if torch.cuda.is_available() else 'cpu')
    items = load_test_items(data_dir, model)
    for precision in ['highest', 'high', 'medium']:
        torch.set_float32_matmul_precision(precision)
        frames = run_inference(model, items)
        m = detection_metrics(frames, 0.5)
        top = np.mean([f['score'].max() for f in frames])
        print(f'   {precision:8s} MODA {m["moda"]:6.1f}  recall {m["recall"]:5.1f}  precision {m["precision"]:5.1f}  '
              f'mean max heatmap score {top:.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', required=True)
    args = p.parse_args()
    calibration_check(args.data_dir)
    model_check(args.ckpt, args.data_dir)
