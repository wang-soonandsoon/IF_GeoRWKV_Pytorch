<div align="center">

# GeoRWKV

**Imposing Geometric Constraints on Spectral Sequences via LiDAR-Informed State Evolution**

[![IGARSS 2026](https://img.shields.io/badge/IGARSS%202026-Oral-2f6fed)](https://2026.ieeeigarss.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-HSI--LiDAR%20Classification-0f766e)](#)
[![Status](https://img.shields.io/badge/Code-Available-brightgreen)](#)

Official PyTorch implementation of **GeoRWKV**, a LiDAR-guided RWKV framework for hyperspectral-LiDAR land-cover classification.

</div>

## Overview

Linear-time sequence models such as RWKV are efficient backbones for hyperspectral image classification, but flattening a 2D patch into a 1D sequence can propagate recurrent states across physical object boundaries. GeoRWKV addresses this issue by using LiDAR geometry as a control signal for RWKV state evolution.

The model serializes local HSI-LiDAR patches with a center-out spiral order, derives a LiDAR gradient gate along the token sequence, and injects this gate into the RWKV decay term to suppress cross-boundary spectral leakage.

## Key Features

| Component | Purpose |
| --- | --- |
| Center-out spiral serialization | Preserves local continuity around the target pixel. |
| LiDAR-informed dynamic decay | Attenuates recurrent state carry-over across geometric boundaries. |
| Bidirectional WKV fusion | Aggregates context in forward and backward spiral directions. |
| Soft router | Adapts between a local MLP path and a geometry-aware fusion path. |
| RWKV-style TimeMix/ChannelMix | Provides efficient sequence and channel mixing for multimodal patches. |
| Fast implementation | Uses a sequence-native dataflow and optional CUDA WKV forward kernel. |

## Model Path

```text
HSI patch ── 1x1 stem ── spiral tokens ──┐
                                          ├── GeoRWKV blocks ── center token ── classifier
LiDAR patch ─ 1x1 stem ─ spiral tokens ──┘
        │
        └── LiDAR gradient log-gate ── dynamic RWKV decay
```

Inside each GeoRWKV block:

```text
TokenShift2D
  ├── local MLP path
  └── LiDAR-gated bidirectional RWKV fusion path
          ↓
      soft router
          ↓
      RWKV ChannelMix
```

## Repository Structure

```text
.
├── model/
│   ├── GeoRwkvV2_MixSigmoid_WeakBeta.py   # optimized sequence-native GeoRWKV
│   ├── reference_slow.py                   # slower reference implementation
│   └── ops/
│       ├── wkv_dynamic.py                  # dynamic WKV dispatcher
│       └── csrc/                           # optional CUDA forward kernel
├── setting/
│   ├── dataLoader.py                       # HSI-LiDAR dataloaders and PCA cache
│   ├── options.py                          # CLI options
│   └── utils.py
├── tests/
│   └── parity_test.py                      # fast/reference parity check
├── train.py
├── infer.py
├── benchmark.py
├── dataset_info.yaml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/wang-soonandsoon/IF_GeoRWKV_Pytorch.git
cd IF_GeoRWKV_Pytorch
pip install -r requirements.txt
```

The CUDA WKV extension is JIT-compiled automatically when CUDA and a compatible compiler are available. If the extension cannot be built, GeoRWKV falls back to the PyTorch WKV implementation.

Useful flags:

```bash
GEORWKV_DISABLE_CUDA_EXT=1     # force PyTorch reference WKV
GEORWKV_WKV_VERBOSE_BUILD=1    # show CUDA extension build logs
GEORWKV_WKV_BUILD_DIR=<path>   # custom extension build/cache directory
```

## Data Preparation

Datasets are not included in this repository. Place `.mat` files under `<data_root>`:

```text
<data_root>/
  Houston2013/
    houston_gt.mat
    houston_hsi.mat
    houston_index.mat
    houston_lidar.mat
  Houston2018/
    houston_gt.mat
    houston_hsi.mat
    houston_index.mat
    houston_lidar.mat
  Trento/
    trento_gt.mat
    trento_hsi.mat
    trento_index.mat
    trento_lidar.mat
```

Dataset metadata is configured in `dataset_info.yaml`, including class counts, PCA dimensions, patch sizes, and `.mat` keys.

## Quick Start

Run one forward/backward step:

```bash
python train.py \
  --model GeoRwkvV2_MixSigmoid_WeakBeta \
  --dataset Houston2013 \
  --data_root <data_root> \
  --dry_run
```

Check fast/reference numerical parity:

```bash
python tests/parity_test.py
```

Expected output:

```text
All parity checks passed.
```

## Training

Paper-style Houston2013 run:

```bash
python train.py \
  --model GeoRwkvV2_MixSigmoid_WeakBeta \
  --dataset Houston2013 \
  --data_root <data_root> \
  --epoch 200 \
  --lr 5e-4 \
  --batchsize 128 \
  --weight_decay 1e-2 \
  --label_smoothing 0.05 \
  --amp 1 \
  --amp_dtype bf16 \
  --cache_pca 1
```

Paper-style Trento run:

```bash
python train.py \
  --model GeoRwkvV2_MixSigmoid_WeakBeta \
  --dataset Trento \
  --data_root <data_root> \
  --epoch 200 \
  --lr 5e-4 \
  --batchsize 128 \
  --weight_decay 1e-2 \
  --label_smoothing 0.05 \
  --amp 1 \
  --amp_dtype bf16 \
  --cache_pca 1
```

Training outputs are written under `checkpoints/` by default. PCA and standardized LiDAR arrays are cached under:

```text
<data_root>/<dataset>/.cache/
```

## Inference

```bash
python infer.py \
  --model GeoRwkvV2_MixSigmoid_WeakBeta \
  --dataset Houston2013 \
  --data_root <data_root> \
  --weights <path_to_checkpoint.pth> \
  --split test \
  --amp 1
```

## Benchmark

Compare the optimized model with the slower reference implementation:

```bash
python benchmark.py \
  --device cuda:0 \
  --patch 11 \
  --batch 128 \
  --iters 50 \
  --amp 1
```

## Implementation Notes

- The optimized model keeps the same learnable parameter names as `model/reference_slow.py`, so compatible checkpoints can be loaded with `strict=True`.
- The CUDA kernel accelerates the dynamic WKV forward pass. Backward remains autograd-safe through PyTorch recomputation.
- Set `GEORWKV_DISABLE_CUDA_EXT=1` for maximum portability.
- No datasets, checkpoints, generated maps, or reports are included in this repository.

## Citation

If this code is useful for your research, please cite:

```bibtex
@inproceedings{wang2026georwkv,
  title={GeoRWKV: Imposing Geometric Constraints on Spectral Sequences via LiDAR-Informed State Evolution},
  author={Wang, Zhenduo and Liu, Xu and Li, Lingling and Zhang, Dan and Tang, Xu and Jiao, Licheng and Liu, Fang},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  year={2026}
}
```

## Acknowledgements

This repository builds on the RWKV-style sequence modeling idea and targets multimodal remote sensing classification with HSI-LiDAR data.

## License

The license for this repository has not been finalized yet. Please contact the authors before redistributing or reusing the code.
