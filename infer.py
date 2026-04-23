import argparse
import os
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np

from tools.paper_palette import get_class_names, get_label_color_map


def _select_loader(split, train_loader, test_loader, trntst_loader, all_loader):
    if split == "train":
        return train_loader
    if split == "test":
        return test_loader
    if split == "trntst":
        return trntst_loader
    if split == "all":
        return all_loader
    raise ValueError(f"Unknown split: {split}")

def _aa_and_each_class_accuracy(confusion: np.ndarray):
    diag = np.diag(confusion).astype(np.float64)
    row_sum = np.sum(confusion, axis=1).astype(np.float64)
    each_acc = np.nan_to_num(diag / np.where(row_sum == 0, 1.0, row_sum))
    aa = float(np.mean(each_acc))
    return each_acc, aa

def _save_confusion_png(confusion: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _make_label_cmap(num_classes: int):
    import matplotlib.colors as mcolors
    import matplotlib
    import matplotlib.cm as cm

    num_classes = int(num_classes)
    colors = [(0.0, 0.0, 0.0, 1.0)]  # background = 0
    if num_classes <= 20:
        if hasattr(matplotlib, "colormaps"):
            base = matplotlib.colormaps.get_cmap("tab20").resampled(num_classes)
            colors += [base(i) for i in range(num_classes)]
        else:
            base = cm.get_cmap("tab20", num_classes)
            colors += [base(i) for i in range(num_classes)]
    else:
        if hasattr(matplotlib, "colormaps"):
            base = matplotlib.colormaps.get_cmap("hsv").resampled(num_classes)
            colors += [base(i) for i in range(num_classes)]
        else:
            base = cm.get_cmap("hsv", num_classes)
            colors += [base(i) for i in range(num_classes)]
    return mcolors.ListedColormap(colors)


def _get_paper_label_palette(dataset: str | None, num_classes: int) -> np.ndarray | None:
    if not dataset:
        return None
    color_map = get_label_color_map(dataset)
    if not color_map or len(color_map) != int(num_classes):
        return None
    cmap = np.asarray(color_map, dtype=np.uint8)
    if cmap.ndim != 2 or cmap.shape[1] != 3:
        return None
    return cmap


def _get_target_names(dataset: str | None, num_classes: int) -> list[str]:
    if dataset:
        names = get_class_names(dataset)
        if isinstance(names, list) and len(names) == int(num_classes):
            return [str(n) for n in names]
    return [f"Class{i}" for i in range(int(num_classes))]


def _label_map_to_rgb(label_map: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """
    label_map: (H,W) with 0=unlabeled, 1..K=classes
    cmap: (K,3) uint8 for labels 1..K
    """
    if label_map.ndim != 2:
        raise ValueError(f"label_map must be 2D (H,W), got shape={label_map.shape}")
    if cmap.ndim != 2 or cmap.shape[1] != 3:
        raise ValueError(f"cmap must be (K,3), got shape={cmap.shape}")

    h, w = int(label_map.shape[0]), int(label_map.shape[1])
    out = np.zeros((h, w, 3), dtype=np.uint8)

    labels = label_map.astype(np.int64, copy=False)
    mask = labels > 0
    if not np.any(mask):
        return out

    idx = labels[mask] - 1
    idx = np.clip(idx, 0, int(cmap.shape[0]) - 1)
    out[mask] = cmap[idx]
    return out


def _save_label_map_png(label_map: np.ndarray, num_classes: int, out_path: str, title: str, dataset: str | None = None):
    import matplotlib.pyplot as plt
    paper_cmap = _get_paper_label_palette(dataset, num_classes=num_classes)
    if paper_cmap is not None and label_map.ndim == 2:
        plt.imsave(out_path, _label_map_to_rgb(label_map, paper_cmap))
        return
    cmap = _make_label_cmap(num_classes)
    plt.imsave(out_path, label_map, cmap=cmap, vmin=0, vmax=int(num_classes))

def _percentile_vmin_vmax(x: np.ndarray, pmin: float, pmax: float):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(x, [pmin, pmax]).astype(np.float64)
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax

def _save_colorbar(out_path: str, cmap, vmin: float, vmax: float, orientation: str = "vertical", label: str = ""):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if isinstance(cmap, str):
        cmap = mpl.colormaps.get_cmap(cmap) if hasattr(mpl, "colormaps") else plt.get_cmap(cmap)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if orientation == "horizontal":
        fig = plt.figure(figsize=(3.2, 0.55), dpi=300)
        cax = fig.add_axes([0.05, 0.40, 0.90, 0.25])
    else:
        fig = plt.figure(figsize=(0.65, 3.2), dpi=300)
        cax = fig.add_axes([0.40, 0.05, 0.25, 0.90])

    cbar = fig.colorbar(sm, cax=cax, orientation=orientation)
    if label:
        cbar.set_label(label)
    cbar.outline.set_linewidth(0.6)
    cax.tick_params(length=2.0, width=0.6, labelsize=8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

def _save_gray_image_and_colorbar(
    img2d: np.ndarray,
    out_img_path: str,
    out_cbar_path: str,
    cmap: str,
    pmin: float,
    pmax: float,
    orientation: str,
):
    import matplotlib.pyplot as plt

    vmin, vmax = _percentile_vmin_vmax(img2d, pmin=pmin, pmax=pmax)
    plt.imsave(out_img_path, img2d, cmap=cmap, vmin=vmin, vmax=vmax)
    _save_colorbar(out_cbar_path, cmap=cmap, vmin=vmin, vmax=vmax, orientation=orientation)
    return {"vmin": vmin, "vmax": vmax, "cmap": cmap, "pmin": float(pmin), "pmax": float(pmax)}

def _parse_bands(arg: str, max_bands: int):
    if not arg or arg.lower() == "auto":
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    bands = [int(p) for p in parts]
    bands = [b for b in bands if 0 <= b < max_bands]
    if len(bands) != 3:
        raise ValueError("--hsi_rgb_bands must provide 3 band indices (0-based), e.g. 10,40,100")
    return bands

def _auto_hsi_rgb_bands(num_bands: int):
    if num_bands <= 3:
        return list(range(num_bands)) + [0] * (3 - num_bands)
    b2 = int(round(0.20 * (num_bands - 1)))
    b1 = int(round(0.50 * (num_bands - 1)))
    b0 = int(round(0.80 * (num_bands - 1)))
    bands = [b0, b1, b2]
    # ensure unique while preserving order
    uniq = []
    for b in bands:
        if b not in uniq:
            uniq.append(b)
    while len(uniq) < 3:
        uniq.append(min(num_bands - 1, uniq[-1] + 1))
    return uniq[:3]

def _save_input_visualizations(
    dataset: str,
    di: dict,
    data_root: str,
    out_dir: str,
    vis_cmap: str,
    vis_pmin: float,
    vis_pmax: float,
    colorbar_orientation: str,
    hsi_rgb_bands: str,
    gt_full: np.ndarray | None,
    vis_mask_gt: bool,
):
    from scipy.io import loadmat

    vis_dir = os.path.join(out_dir, "vis")
    cbar_dir = os.path.join(out_dir, "colorbar")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(cbar_dir, exist_ok=True)

    hsi_path = os.path.join(data_root, dataset, di["info"][1])
    x_path = os.path.join(data_root, dataset, di["info"][3])
    hsi_key = di["keys"][0]
    x_key = di["keys"][1]

    x_label = "SAR" if "sar" in str(x_key).lower() else "LiDAR" if "lidar" in str(x_key).lower() else "X"

    hsi = loadmat(hsi_path)[hsi_key]  # H,W,B
    x = loadmat(x_path)[x_key]  # H,W or H,W,C

    # HSI mean (grayscale)
    hsi_mean = np.mean(hsi, axis=2)
    hsi_mean_meta = _save_gray_image_and_colorbar(
        img2d=hsi_mean,
        out_img_path=os.path.join(vis_dir, f"{dataset}_HSI_mean.png"),
        out_cbar_path=os.path.join(cbar_dir, f"{dataset}_HSI_mean_cbar.png"),
        cmap=vis_cmap,
        pmin=vis_pmin,
        pmax=vis_pmax,
        orientation=colorbar_orientation,
    )

    # HSI pseudo-RGB
    bands = _parse_bands(hsi_rgb_bands, max_bands=int(hsi.shape[2])) or _auto_hsi_rgb_bands(int(hsi.shape[2]))
    rgb = []
    band_metas = []
    for b in bands:
        img = hsi[:, :, b]
        vmin, vmax = _percentile_vmin_vmax(img, pmin=vis_pmin, pmax=vis_pmax)
        band_metas.append({"band": int(b), "vmin": float(vmin), "vmax": float(vmax)})
        rgb.append(np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0))
    hsi_rgb = np.stack(rgb, axis=2).astype(np.float32, copy=False)
    import matplotlib.pyplot as plt

    plt.imsave(os.path.join(vis_dir, f"{dataset}_HSI_rgb.png"), hsi_rgb)

    # X modality (SAR/LiDAR)
    x_metas = {}
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
    if x.ndim == 2:
        x_metas[x_label] = _save_gray_image_and_colorbar(
            img2d=x,
            out_img_path=os.path.join(vis_dir, f"{dataset}_{x_label}.png"),
            out_cbar_path=os.path.join(cbar_dir, f"{dataset}_{x_label}_cbar.png"),
            cmap=vis_cmap,
            pmin=vis_pmin,
            pmax=vis_pmax,
            orientation=colorbar_orientation,
        )
    elif x.ndim == 3:
        for ch in range(int(x.shape[2])):
            x_metas[f"{x_label}_ch{ch}"] = _save_gray_image_and_colorbar(
                img2d=x[:, :, ch],
                out_img_path=os.path.join(vis_dir, f"{dataset}_{x_label}_ch{ch}.png"),
                out_cbar_path=os.path.join(cbar_dir, f"{dataset}_{x_label}_ch{ch}_cbar.png"),
                cmap=vis_cmap,
                pmin=vis_pmin,
                pmax=vis_pmax,
                orientation=colorbar_orientation,
            )
    else:
        raise ValueError(f"Unsupported {x_label} ndim={x.ndim}")

    # Optional: mask out unlabeled region (gt==0) for paper-friendly comparison.
    if vis_mask_gt and gt_full is not None and gt_full.ndim == 2:
        import matplotlib as mpl

        mask = (gt_full == 0)
        hsi_mean_masked = np.asarray(hsi_mean).copy()
        hsi_mean_masked[mask] = np.nan

        cmap_obj = mpl.colormaps.get_cmap(vis_cmap).copy() if hasattr(mpl, "colormaps") else mpl.cm.get_cmap(vis_cmap).copy()
        cmap_obj.set_bad((0, 0, 0, 0))  # transparent background

        vmin, vmax = _percentile_vmin_vmax(hsi_mean_masked, pmin=vis_pmin, pmax=vis_pmax)
        import matplotlib.pyplot as plt

        plt.imsave(
            os.path.join(vis_dir, f"{dataset}_HSI_mean_masked.png"),
            hsi_mean_masked,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
        )
        _save_colorbar(
            os.path.join(cbar_dir, f"{dataset}_HSI_mean_masked_cbar.png"),
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            orientation=colorbar_orientation,
        )

    return {
        "hsi_path": hsi_path,
        "x_path": x_path,
        "x_label": x_label,
        "hsi_mean": hsi_mean_meta,
        "hsi_rgb": {"bands": [int(b) for b in bands], "band_metas": band_metas},
        "x": x_metas,
        "vis_dir": vis_dir,
        "colorbar_dir": cbar_dir,
    }

def _evaluate_and_save(model, model_spec, loader, device, out_dir: str, dataset: str, split: str, num_classes: int, autocast_ctx_factory=nullcontext):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

    y_true_list = []
    y_pred_list = []
    pos_list = []

    with __import__("torch").no_grad():
        for batch in loader:
            with autocast_ctx_factory():
                logits = model_spec.forward_logits(model, batch, device)
            pred = __import__("torch").argmax(logits, dim=1).detach().cpu().numpy()
            y_pred_list.append(pred)
            y_true_list.append(model_spec.get_targets(batch, device).detach().cpu().numpy())

            pos = model_spec.get_positions(batch)
            if pos is None:
                continue
            pos_list.append(pos.detach().cpu().numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    positions = np.concatenate(pos_list, axis=0) if pos_list else np.empty((0, 2), dtype=np.int32)

    confusion = confusion_matrix(y_true, y_pred, labels=list(range(int(num_classes))))
    each_acc, aa = _aa_and_each_class_accuracy(confusion)
    oa = float(accuracy_score(y_true, y_pred) * 100.0)
    kappa = float(cohen_kappa_score(y_true, y_pred) * 100.0)
    target_names = _get_target_names(dataset, num_classes)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
    )

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{dataset}_{split}_confusion.npy"), confusion)
    np.savez_compressed(
        os.path.join(out_dir, f"{dataset}_{split}_preds.npz"),
        y_true=y_true.astype(np.int64, copy=False),
        y_pred=y_pred.astype(np.int64, copy=False),
        positions_hw=positions.astype(np.int32, copy=False),
    )
    with open(os.path.join(out_dir, f"{dataset}_{split}_report.txt"), "w") as f:
        f.write("Class names:\n")
        for i, name in enumerate(target_names):
            f.write(f"  {i}: {name}\n")
        f.write("\n")
        f.write(report)
        f.write("\n")
        f.write(f"OA(%)={oa}\nAA(%)={aa*100.0}\nKappa(%)={kappa}\n")

    _save_confusion_png(
        confusion,
        os.path.join(out_dir, f"{dataset}_{split}_confusion.png"),
        title=f"{dataset} {split} confusion",
    )

    metrics = {"oa": oa, "aa": float(aa * 100.0), "kappa": kappa, "each_acc": (each_acc * 100.0).tolist()}
    return metrics

def _predict_full_map(model, model_spec, all_loader, device, gt_full: np.ndarray, num_classes: int, out_dir: str, dataset: str, autocast_ctx_factory=nullcontext):
    import torch

    H, W = int(gt_full.shape[0]), int(gt_full.shape[1])
    pred_map = np.zeros((H, W), dtype=np.int16)

    with torch.no_grad():
        for batch in all_loader:
            with autocast_ctx_factory():
                logits = model_spec.forward_logits(model, batch, device)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int16, copy=False)  # 0..C-1
            hw_t = model_spec.get_positions(batch)
            if hw_t is None:
                raise RuntimeError("positions not available in batch; cannot reconstruct full prediction map")
            hw = hw_t.detach().cpu().numpy().astype(np.int32, copy=False)
            pred_map[hw[:, 0], hw[:, 1]] = pred + 1  # 1..C

    pred_map_full = pred_map
    pred_map_masked = pred_map.copy()
    pred_map_masked[gt_full == 0] = 0

    np.save(os.path.join(out_dir, f"{dataset}_pred_map.npy"), pred_map_masked)
    np.save(os.path.join(out_dir, f"{dataset}_pred_map_full.npy"), pred_map_full)
    np.save(os.path.join(out_dir, f"{dataset}_gt_map.npy"), gt_full.astype(np.int16, copy=False))

    _save_label_map_png(
        gt_full, num_classes, os.path.join(out_dir, f"{dataset}_gt_map.png"), title=f"{dataset} GT", dataset=dataset
    )
    _save_label_map_png(
        pred_map_masked,
        num_classes,
        os.path.join(out_dir, f"{dataset}_pred_map.png"),
        title=f"{dataset} Pred",
        dataset=dataset,
    )
    _save_label_map_png(
        pred_map_full,
        num_classes,
        os.path.join(out_dir, f"{dataset}_pred_map_full.png"),
        title=f"{dataset} Pred (full)",
        dataset=dataset,
    )
    # Convenience alias: full-color prediction map (paper-like).
    _save_label_map_png(
        pred_map_full,
        num_classes,
        os.path.join(out_dir, f"{dataset}_pred_map_color.png"),
        title=f"{dataset} Pred (full)",
        dataset=dataset,
    )

    # Side-by-side
    import matplotlib.pyplot as plt

    paper_cmap = _get_paper_label_palette(dataset, num_classes=num_classes)
    fig = plt.figure(figsize=(10, 6), dpi=180)
    ax1 = fig.add_subplot(1, 2, 1)
    if paper_cmap is not None:
        ax1.imshow(_label_map_to_rgb(gt_full, paper_cmap))
    else:
        cmap = _make_label_cmap(num_classes)
        ax1.imshow(gt_full, cmap=cmap, vmin=0, vmax=int(num_classes))
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    if paper_cmap is not None:
        ax2.imshow(_label_map_to_rgb(pred_map_masked, paper_cmap))
    else:
        cmap = _make_label_cmap(num_classes)
        ax2.imshow(pred_map_masked, cmap=cmap, vmin=0, vmax=int(num_classes))
    ax2.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
    fig.savefig(os.path.join(out_dir, f"{dataset}_gt_vs_pred.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: torch. Activate the same Python environment you used for training, "
            "then re-run inference."
        ) from e

    from setting.dataLoader import get_loader
    from model.factory import get_model_spec

    parser = argparse.ArgumentParser(description="Model inference / evaluation")
    parser.add_argument("--model", type=str, default="GeoRwkvV2_MixSigmoid_WeakBeta", help="model name (see model/factory.py)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="path to .pth state_dict saved by train.py")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "trntst", "all"])
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_work", type=int, default=0)
    parser.add_argument("--useval", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1], help="use autocast in inference when CUDA is available")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="autocast dtype for CUDA inference")
    parser.add_argument("--cache_pca", type=int, default=1, choices=[0, 1], help="reuse cached PCA / standardized arrays when available")
    parser.add_argument("--return_full_hsi", type=int, default=0, choices=[0, 1], help="return full HSI patches in dataloader")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="output dir (default: ./inference/<model>-<dataset>/<ts>)",
    )
    # Default is enabled to match historical behavior of generating full-map visualizations.
    parser.add_argument(
        "--save_pred_map",
        dest="save_pred_map",
        action="store_true",
        default=True,
        help="save full prediction/GT maps (uses all_index) (default: enabled)",
    )
    parser.add_argument(
        "--no_save_pred_map",
        dest="save_pred_map",
        action="store_false",
        help="disable saving full prediction/GT maps",
    )
    parser.add_argument("--save_input_vis", action="store_true", help="save HSI/SAR(or LiDAR) whole-image visualizations + separate colorbars")
    parser.add_argument("--vis_cmap", type=str, default="gray", help="colormap for continuous images (e.g., gray, viridis)")
    parser.add_argument("--vis_pmin", type=float, default=2.0, help="lower percentile for visualization clipping")
    parser.add_argument("--vis_pmax", type=float, default=98.0, help="upper percentile for visualization clipping")
    parser.add_argument("--colorbar_orientation", type=str, default="vertical", choices=["vertical", "horizontal"])
    parser.add_argument("--hsi_rgb_bands", type=str, default="auto", help="three 0-based band indices, e.g. 10,40,100, or 'auto'")
    parser.add_argument("--vis_mask_gt", action="store_true", help="mask out gt==0 region (transparent) in some visualizations")
    args = parser.parse_args()

    model_name = args.model
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    out_dir = args.out_dir or os.path.join("inference", f"{model_name}-{args.dataset}", ts)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(int(args.amp))
    amp_dtype_str = str(args.amp_dtype).strip().lower()
    if amp_dtype_str == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        amp_dtype_str = "fp16"
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16

    def autocast_context():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    train_loader, test_loader, trntst_loader, all_loader, *_ = get_loader(
        dataset=args.dataset,
        batchsize=args.batchsize,
        num_workers=args.num_work,
        useval=args.useval,
        pin_memory=(device.type == "cuda"),
        data_root=args.data_root,
        shape_log_path=os.path.join(out_dir, f"{args.dataset}_{ts}_shapes.yaml"),
        cache_pca=bool(int(args.cache_pca)),
        return_full_hsi=bool(int(args.return_full_hsi)),
        persistent_workers=bool(int(args.num_work > 0)),
        prefetch_factor=2,
    )
    loader = _select_loader(args.split, train_loader, test_loader, trntst_loader, all_loader)

    model_spec = get_model_spec(args.model)
    model = model_spec.create(args.dataset)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    import yaml
    from scipy.io import loadmat

    with open(Path(__file__).resolve().parent / "dataset_info.yaml", "r", encoding="utf-8") as f:
        di = yaml.safe_load(f)[args.dataset]
    num_classes = int(getattr(model, "out_features", 0) or di["num_classes"])

    metrics = {}
    if args.split in {"train", "test", "trntst"}:
        metrics = _evaluate_and_save(
            model=model,
            model_spec=model_spec,
            loader=loader,
            device=device,
            out_dir=out_dir,
            dataset=args.dataset,
            split=args.split,
            num_classes=num_classes,
            autocast_ctx_factory=autocast_context,
        )
        print(
            f"[{args.dataset}:{args.split}] OA={metrics['oa']:.4f} AA={metrics['aa']:.4f} Kappa={metrics['kappa']:.4f}"
        )

    gt_full = None
    if args.save_pred_map or args.save_input_vis:
        gt_path = os.path.join(args.data_root, args.dataset, di["info"][0])
        gt_key = di["keys"][2]
        gt_full = loadmat(gt_path)[gt_key]

    vis_info = {}
    if args.save_input_vis:
        vis_info = _save_input_visualizations(
            dataset=args.dataset,
            di=di,
            data_root=args.data_root,
            out_dir=out_dir,
            vis_cmap=args.vis_cmap,
            vis_pmin=args.vis_pmin,
            vis_pmax=args.vis_pmax,
            colorbar_orientation=args.colorbar_orientation,
            hsi_rgb_bands=args.hsi_rgb_bands,
            gt_full=gt_full,
            vis_mask_gt=args.vis_mask_gt,
        )

    if args.save_pred_map and gt_full is not None:
        _predict_full_map(
            model=model,
            model_spec=model_spec,
            all_loader=all_loader,
            device=device,
            gt_full=gt_full,
            num_classes=num_classes,
            out_dir=out_dir,
            dataset=args.dataset,
            autocast_ctx_factory=autocast_context,
        )

    # Save a compact run summary
    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "split": args.split,
        "weights": os.path.abspath(args.weights),
        "device": str(device),
        "batchsize": int(args.batchsize),
        "out_dir": os.path.abspath(out_dir),
        "metrics": metrics,
        "visualization": vis_info,
    }
    try:
        import yaml

        with open(os.path.join(out_dir, "summary.yaml"), "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
