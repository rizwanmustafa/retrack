# This branch: TrackTacular + R2 (sparse person queries)

`feat/r2-sparse-queries` is the original TrackTacular code (below) with one architectural change and the tools to
train, evaluate and look at it on Google Colab:

| | |
|---|---|
| **R2** | `--model.sparse_queries true` (`WorldTrack/configs/m_r2.yml`). TrackTacular's dense U-Net BEV decoder is replaced by ~100 person queries: they start at the strongest cells of a BEV heatmap, look at each other (self-attention), read the image features of every camera at 4 heights above their ground position, and refine position, score and motion over 3 rounds (`WorldTrack/models/query_decoder.py`, loss `sparse_loss` in `WorldTrack/world_track.py`). Encoder, lifting, camera fusion, tracker and metrics are TrackTacular's. |
| **two fixes** | previous-frame people are shifted with the same training augmentation as the current frame (`datasets/pedestrian_dataset.py`); cameras get the reference frame of their own sample in a batch (`repeat_interleave` in `models/segnet.py`) |
| **camera removal** | `item['cam_mask']` switches cameras off at evaluation time (their lifted features become zeros) |
| **Colab** | `colab/train_eval.py` (train + evaluate B0 and R2, report), `colab/viz.py` (interactive visual tools for any B0/R2 checkpoint); see **[colab/README.md](colab/README.md)** (quick start) and **[colab/TOOLS.md](colab/TOOLS.md)** (full documentation) |
| **evaluation** | `WorldTrack/evaluation/r2_eval.py`: MODA/MODP/precision/recall and MOTA/IDF1 with all cameras and every combination of up to 3 removed cameras |

---

# TrackTacular :octopus:

**Lifting Multi-View Detection and Tracking to the Bird’s Eye View**

[Torben Teepe](https://github.com/tteepe),
[Philipp Wolters](https://github.com/phi-wol),
[Johannes Gilg](https://github.com/Blueblue4),
[Fabian Herzog](https://github.com/fubel),
Gerhard Rigoll

[![arxiv](https://img.shields.io/badge/arXiv-2403.12573-red)](https://arxiv.org/abs/2403.12573)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lifting-multi-view-detection-and-tracking-to/multi-object-tracking-on-wildtrack)](https://paperswithcode.com/sota/multi-object-tracking-on-wildtrack?p=lifting-multi-view-detection-and-tracking-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lifting-multi-view-detection-and-tracking-to/multi-object-tracking-on-multiviewx)](https://paperswithcode.com/sota/multi-object-tracking-on-multiviewx?p=lifting-multi-view-detection-and-tracking-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lifting-multi-view-detection-and-tracking-to/multiview-detection-on-multiviewx)](https://paperswithcode.com/sota/multiview-detection-on-multiviewx?p=lifting-multi-view-detection-and-tracking-to)

> [!TIP]
> This work is an extension of your previous work [EarlyBird  🦅](https://github.com/tteepe/EarlyBird).
> Feel free to check it out and extend our multi-view object detection and tracking pipeline on other datasets!

![Overview](imgs/overview.svg)

## Usage

### Getting Started
1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support
    ```shell
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Install [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip) with CUDA support
   ```shell
   pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
   ```
3. Install remaining dependencies
   ```shell
   pip install -r requirements.txt
   ```

### Training
```shell
python world_track.py fit -c configs/t_fit.yml \
    -c configs/d_{multiviewx,wildtrack,synthehicle}.yml \
    -c configs/m_{mvdet,segnet,liftnet,bevformer}.yml
```
### Testing
```shell
python world_track.py test -c model_weights/config.yaml \
    --ckpt model_weights/model-epoch=35-val_loss=6.50.ckpt
```

## Acknowledgement
- [Simple-BEV](https://simple-bev.github.io): Adam W. Harley
- [MVDeTr](https://github.com/hou-yz/MVDeTr): Yunzhong Hou

## Citation
```bibtex
@article{teepe2023lifting,
      title={Lifting Multi-View Detection and Tracking to the Bird's Eye View}, 
      author={Torben Teepe and Philipp Wolters and Johannes Gilg and Fabian Herzog and Gerhard Rigoll},
      year={2024},
      eprint={2403.12573},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
