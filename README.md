# GeoRWKV

Official PyTorch implementation of **GeoRWKV: Imposing Geometric Constraints on Spectral Sequences via LiDAR-Informed State Evolution**.

GeoRWKV is a LiDAR-guided RWKV framework for hyperspectral-LiDAR classification. It serializes local HSI patches with a center-out spiral order and uses LiDAR-derived geometric discontinuities to modulate RWKV state decay, reducing cross-boundary spectral leakage.

**IGARSS 2026 Oral**

## Highlights

- Center-out spiral serialization for patch tokens.
- LiDAR-informed log-gate injected into RWKV state decay.
- Geometry-aware bidirectional WKV fusion.
- Soft router between a local MLP path and the geometry-aware fusion path.
- RWKV-style TimeMix/ChannelMix with sigmoid-parameterized mixing ratios.
- Fast sequence-native implementation with a PyTorch reference path and optional CUDA WKV forward kernel.

## Repository Structure

```text
.
├── model/
│   ├── GeoRwkvV2_MixSigmoid_WeakBeta.py
│   ├── reference_slow.py
│   └── ops/
│       └── wkv_dynamic.py
├── setting/
│   ├── dataLoader.py
│   ├── options.py
│   └── utils.py
├── tests/
│   └── parity_test.py
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

The CUDA WKV extension is JIT-compiled automatically when available. If the extension cannot be built, the code falls back to the PyTorch implementation.

Useful environment flags:

```bash
GEORWKV_DISABLE_CUDA_EXT=1     # force PyTorch reference WKV
GEORWKV_WKV_VERBOSE_BUILD=1    # show CUDA extension build logs
GEORWKV_WKV_BUILD_DIR=<path>   # custom extension build/cache directory
```

## Data Layout

Datasets are not included in this repository. Place the `.mat` files as follows:

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

Dataset-specific keys, class counts, PCA dimensions, and patch sizes are configured in `dataset_info.yaml`.

## Quick Check

Run a dry training step:

```bash
python train.py \
  --model GeoRwkvV2_MixSigmoid_WeakBeta \
  --dataset Houston2013 \
  --data_root <data_root> \
  --dry_run
```

Check numerical parity between the optimized implementation and the slower reference implementation:

```bash
python tests/parity_test.py
```

## Training

Example command matching the main paper setting:

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

For Trento:

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

```bash
python benchmark.py --device cuda:0 --patch 11 --batch 128 --iters 50 --amp 1
```

## Notes

- The optimized model keeps the same learnable parameter names as `model/reference_slow.py`, so compatible checkpoints can be loaded with `strict=True`.
- Cached PCA and standardized LiDAR arrays are written to `<data_root>/<dataset>/.cache/`.
- No checkpoints, datasets, or generated reports are included in this repository.

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

## License

The license for this repository has not been finalized yet. Please contact the authors before redistributing or reusing the code.
