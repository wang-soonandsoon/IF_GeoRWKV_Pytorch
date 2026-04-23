from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import IncrementalPCA, PCA
from torch.utils import data
import yaml


def applyPCA(X, numComponents, incremental_threshold=500_000, batch_size=100_000):
    """
    Apply PCA to the image to reduce dimensionality.
    """
    numComponents = int(min(numComponents, X.shape[2]))
    X_2d = np.reshape(X, (-1, X.shape[2]))
    n_pixels = X_2d.shape[0]

    if n_pixels >= incremental_threshold:
        ipca = IncrementalPCA(n_components=numComponents, whiten=True, batch_size=batch_size)
        for start in range(0, n_pixels, batch_size):
            batch = X_2d[start : start + batch_size].astype(np.float32, copy=False)
            ipca.partial_fit(batch)
        newX = np.empty((n_pixels, numComponents), dtype=np.float32)
        for start in range(0, n_pixels, batch_size):
            batch = X_2d[start : start + batch_size].astype(np.float32, copy=False)
            newX[start : start + batch.shape[0]] = ipca.transform(batch).astype(np.float32, copy=False)
    else:
        pca = PCA(n_components=numComponents, whiten=True, svd_solver="randomized")
        newX = pca.fit_transform(X_2d.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    return np.reshape(newX, (X.shape[0], X.shape[1], numComponents))


def _standardize_2d(x2d: np.ndarray) -> np.ndarray:
    x2d = x2d.astype(np.float32, copy=False)
    mean = x2d.mean(axis=0, dtype=np.float64)
    std = x2d.std(axis=0, dtype=np.float64)
    std = np.where(std == 0, 1.0, std)
    return ((x2d - mean) / std).astype(np.float32, copy=False)


def min_max(x):
    min_v = np.min(x)
    max_v = np.max(x)
    return (x - min_v) / (max_v - min_v)


def _dataset_info_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dataset_info.yaml"


def _load_dataset_info() -> dict:
    with _dataset_info_path().open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("dataset_info.yaml must contain a mapping at the top level")
    return loaded


def _cache_dir(data_root: str, dataset: str) -> Path:
    p = Path(data_root) / dataset / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_tag(source_path: str, extra: str = "") -> str:
    st = os.stat(source_path)
    stem = Path(source_path).stem
    parts = [stem, f"m{int(st.st_mtime_ns)}", f"s{int(st.st_size)}"]
    if extra:
        parts.append(extra)
    return "_".join(parts)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def _load_or_compute_standardized_pca(
    hsi: np.ndarray,
    hsi_path: str,
    data_root: str,
    dataset: str,
    pca_components: int,
    use_cache: bool,
) -> np.ndarray:
    cache_path = _cache_dir(data_root, dataset) / f"{_cache_tag(hsi_path, f'pca{int(pca_components)}_std')}.npy"
    if use_cache and cache_path.exists():
        return np.load(cache_path, mmap_mode=None).astype(np.float32, copy=False)

    hsi_pca = applyPCA(hsi, pca_components)
    hsi_pca_all = _standardize_2d(hsi_pca.reshape(np.prod(hsi_pca.shape[:2]), hsi_pca.shape[2]))
    hsi_pca_std = hsi_pca_all.reshape(hsi_pca.shape[0], hsi_pca.shape[1], hsi_pca.shape[2]).astype(np.float32, copy=False)

    if use_cache:
        _atomic_save_npy(cache_path, hsi_pca_std)
    return hsi_pca_std


def _load_or_compute_standardized_x(
    xdata: np.ndarray,
    x_path: str,
    data_root: str,
    dataset: str,
    use_cache: bool,
) -> np.ndarray:
    suffix = "xstd_2d" if xdata.ndim == 2 else f"xstd_{int(xdata.shape[2])}c"
    cache_path = _cache_dir(data_root, dataset) / f"{_cache_tag(x_path, suffix)}.npy"
    if use_cache and cache_path.exists():
        return np.load(cache_path, mmap_mode=None).astype(np.float32, copy=False)

    if xdata.ndim == 2:
        xdata_all = _standardize_2d(xdata.reshape(np.prod(xdata.shape[:2]), 1)).reshape(-1)
        x_std = xdata_all.reshape(xdata.shape[0], xdata.shape[1]).astype(np.float32, copy=False)
    else:
        xdata_all = _standardize_2d(xdata.reshape(np.prod(xdata.shape[:2]), xdata.shape[2]))
        x_std = xdata_all.reshape(xdata.shape[0], xdata.shape[1], xdata.shape[2]).astype(np.float32, copy=False)

    if use_cache:
        _atomic_save_npy(cache_path, x_std)
    return x_std


class HXDataset(data.Dataset):
    def __init__(
        self,
        Xdata: np.ndarray,
        hsi_pca: np.ndarray,
        index: np.ndarray,
        gt: np.ndarray,
        windowSize: int,
        category: str | None = None,
        return_full_hsi: bool = False,
        hsi: np.ndarray | None = None,
    ):
        modes = ["symmetric", "reflect"]
        self.pad = windowSize // 2
        self.windowSize = int(windowSize)
        self.category = category
        self.return_full_hsi = bool(return_full_hsi)
        self.empty_hsi = torch.empty(0, dtype=torch.float32)

        self.hsi = None
        if self.return_full_hsi:
            if hsi is None:
                raise ValueError("return_full_hsi=True requires raw hsi")
            self.hsi = np.pad(
                np.asarray(hsi, dtype=np.float32),
                ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                mode=modes[windowSize % 2],
            )

        self.hsi_pca = np.pad(
            np.asarray(hsi_pca, dtype=np.float32),
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode=modes[windowSize % 2],
        )
        if Xdata.ndim == 2:
            self.Xdata = np.pad(
                np.asarray(Xdata, dtype=np.float32),
                ((self.pad, self.pad), (self.pad, self.pad)),
                mode=modes[windowSize % 2],
            )
        else:
            self.Xdata = np.pad(
                np.asarray(Xdata, dtype=np.float32),
                ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                mode=modes[windowSize % 2],
            )

        self.pos = np.asarray(index, dtype=np.int64)
        self.gt = np.asarray(gt, dtype=np.int64)

    @staticmethod
    def _to_tensor_chw(arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 2:
            arr = arr[:, :, None]
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(chw)

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        h = int(h)
        w = int(w)

        if self.return_full_hsi and self.hsi is not None:
            hsi_patch = self.hsi[h : h + self.windowSize, w : w + self.windowSize]
            hsi = self._to_tensor_chw(hsi_patch)
        else:
            hsi = self.empty_hsi

        hsi_pca_patch = self.hsi_pca[h : h + self.windowSize, w : w + self.windowSize]
        xdata_patch = self.Xdata[h : h + self.windowSize, w : w + self.windowSize]

        hsi_pca = self._to_tensor_chw(hsi_pca_patch)
        Xdata = self._to_tensor_chw(xdata_patch)

        if self.category == "train" and random.random() < 0.5:
            t = random.randint(1, 2)
            if hsi.numel() > 0:
                hsi = torch.flip(hsi, dims=[-t])
            Xdata = torch.flip(Xdata, dims=[-t])
            hsi_pca = torch.flip(hsi_pca, dims=[-t])

        gt = torch.tensor(int(self.gt[h, w]) - 1, dtype=torch.long)
        return hsi, Xdata, hsi_pca, gt, h, w

    def __len__(self):
        return int(self.pos.shape[0])


def _as_shape(v):
    try:
        return list(v.shape)
    except Exception:
        return None


def _as_dtype(v):
    return str(getattr(v, "dtype", None))


def _log_and_save_shapes(dataset, data_root, data_info, shapes, shape_log_path=None):
    payload = {
        "dataset": dataset,
        "data_root": data_root,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "config": {
            "window_size": int(data_info["window_size"]),
            "pca_num": int(data_info["pca_num"]),
        },
        "shapes": shapes,
    }
    msg = f"[{dataset}] data shapes: " + ", ".join(
        f"{k}={v.get('shape')}, dtype={v.get('dtype')}" for k, v in shapes.items()
    )
    print(msg)
    try:
        logging.info(msg)
    except Exception:
        pass

    if not shape_log_path:
        return
    dir_name = os.path.dirname(shape_log_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(shape_log_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def get_loader(
    dataset,
    batchsize,
    num_workers=0,
    useval=0,
    pin_memory=True,
    data_root="./data",
    shape_log_path=None,
    cache_pca=True,
    return_full_hsi=False,
    persistent_workers=False,
    prefetch_factor=2,
):
    data_info = _load_dataset_info()[dataset]
    windowSize = int(data_info["window_size"])

    # Optional override: force PCA components without editing dataset_info.yaml.
    # Useful for ablations / quick experiments (must match model in_hsi).
    try:
        pca_override = int(os.environ.get("PCA_NUM_OVERRIDE", "0") or 0)
    except Exception:
        pca_override = 0

    hsi_path = os.path.join(data_root, dataset, data_info["info"][1])
    X_path = os.path.join(data_root, dataset, data_info["info"][3])
    gt_path = os.path.join(data_root, dataset, data_info["info"][0])
    index_path = os.path.join(data_root, dataset, data_info["info"][2])

    hsi = loadmat(hsi_path)[data_info["keys"][0]]
    Xdata = loadmat(X_path)[data_info["keys"][1]]
    gt = loadmat(gt_path)[data_info["keys"][2]]
    train_index = loadmat(index_path)[data_info["keys"][3]]
    test_index = loadmat(index_path)[data_info["keys"][4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[data_info["keys"][5]]

    random_indices = np.random.choice(test_index.shape[0], size=test_index.shape[0] // 2, replace=False)
    val_index = test_index[random_indices]
    requested_pca = int(pca_override if pca_override > 0 else data_info["pca_num"])
    pca_components = int(min(requested_pca, hsi.shape[2]))

    hsi_pca = _load_or_compute_standardized_pca(
        hsi=hsi,
        hsi_path=hsi_path,
        data_root=data_root,
        dataset=dataset,
        pca_components=pca_components,
        use_cache=bool(cache_pca),
    )
    Xdata = _load_or_compute_standardized_x(
        xdata=Xdata,
        x_path=X_path,
        data_root=data_root,
        dataset=dataset,
        use_cache=bool(cache_pca),
    )

    _log_and_save_shapes(
        dataset=dataset,
        data_root=data_root,
        data_info={
            **data_info,
            "pca_num": pca_components,
            "pca_num_requested": requested_pca,
            "return_full_hsi": bool(return_full_hsi),
            "cache_pca": bool(cache_pca),
        },
        shapes={
            "hsi_raw": {"shape": _as_shape(hsi), "dtype": _as_dtype(hsi)},
            "Xdata_std": {"shape": _as_shape(Xdata), "dtype": _as_dtype(Xdata)},
            "gt": {"shape": _as_shape(gt), "dtype": _as_dtype(gt)},
            "train_index": {"shape": _as_shape(train_index), "dtype": _as_dtype(train_index)},
            "test_index": {"shape": _as_shape(test_index), "dtype": _as_dtype(test_index)},
            "all_index": {"shape": _as_shape(all_index), "dtype": _as_dtype(all_index)},
            "hsi_pca_std": {"shape": _as_shape(hsi_pca), "dtype": _as_dtype(hsi_pca)},
        },
        shape_log_path=shape_log_path,
    )

    dataset_kwargs = {
        "Xdata": Xdata,
        "hsi_pca": hsi_pca,
        "gt": gt,
        "windowSize": windowSize,
        "return_full_hsi": bool(return_full_hsi),
        "hsi": hsi if return_full_hsi else None,
    }

    HXtrainset = HXDataset(index=train_index, category="train", **dataset_kwargs)
    HXallset = HXDataset(index=all_index, category="all", **dataset_kwargs)
    if int(useval) == 0:
        HXtestset = HXDataset(index=test_index, category="test", **dataset_kwargs)
    else:
        HXtestset = HXDataset(index=val_index, category="val", **dataset_kwargs)
    HXtrntstset = HXDataset(index=trntst_index, category="trntst", **dataset_kwargs)

    loader_kwargs = {
        "batch_size": batchsize,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = data.DataLoader(dataset=HXtrainset, shuffle=True, **loader_kwargs)
    test_loader = data.DataLoader(dataset=HXtestset, shuffle=False, **loader_kwargs)
    trntst_loader = data.DataLoader(dataset=HXtrntstset, shuffle=False, **loader_kwargs)
    all_loader = data.DataLoader(dataset=HXallset, shuffle=False, **loader_kwargs)

    print("num_workers=" + str(num_workers))

    return (
        train_loader,
        test_loader,
        trntst_loader,
        all_loader,
        int(train_index.shape[0]),
        int(test_index.shape[0]),
        int(trntst_index.shape[0]),
    )
