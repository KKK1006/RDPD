# RDPD + SDAF — VisDrone Small Object Detection

A YOLO11-based detector for small objects in the VisDrone dataset, with two custom modules:

- **RDPD** (Residual Detail-Preserving Downsampling) — a backbone block that keeps fine detail during downsampling.
- **SDAF** (Semantic–Detail Adaptive Frequency Fusion) — a neck block that fuses features in the frequency domain.

> Note: a modified copy of `ultralytics` is bundled under `./ultralytics`. Do **not** `pip install ultralytics`.

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `torch` / `torchvision` / `torchaudio` (CUDA 12.6), `einops`, `torch-dct`.

## Dataset

Config is in [data/VisDrone.yaml](data/VisDrone.yaml) (10 classes). Update `path` to point to your VisDrone dataset.

## Train

python train.py                    # default settings
python train.py --epochs 120        # override epochs
python train.py --batch 8          # override batch size


The model is defined in [ultralytics/cfg/models/yolo11m_p2_rdpd_sdaf.yaml](ultralytics/cfg/models/yolo11m_p2_rdpd_sdaf.yaml) (YOLO11m-P2, 4 scales).
