from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml


Batch = Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    create: Callable[[str], nn.Module]
    forward_logits: Callable[[nn.Module, Batch, torch.device], torch.Tensor]
    forward_train: Callable[[nn.Module, Batch, torch.device], Tuple[torch.Tensor, torch.Tensor]]
    get_targets: Callable[[Batch, torch.device], torch.Tensor]
    get_positions: Callable[[Batch], Optional[torch.Tensor]]


_REGISTRY: Dict[str, ModelSpec] = {}
_DATASET_INFO: Optional[Dict[str, dict]] = None
_GEO_RWKV_ALLOWED_DATASETS = {"Houston2013", "Houston2018", "Trento"}


def _norm(name: str) -> str:
    return str(name).strip().lower()


def register_model(spec: ModelSpec) -> None:
    key = _norm(spec.name)
    if not key:
        raise ValueError("ModelSpec.name must be non-empty")
    if key in _REGISTRY:
        raise ValueError(f"Model '{spec.name}' already registered")
    _REGISTRY[key] = spec


def get_model_spec(name: str) -> ModelSpec:
    key = _norm(name)
    if key in _REGISTRY:
        return _REGISTRY[key]
    available = ", ".join(list_models())
    raise KeyError(f"Unknown model '{name}'. Available: {available}")


def list_models() -> List[str]:
    specs = sorted(_REGISTRY.values(), key=lambda s: _norm(s.name))
    return [s.name for s in specs]


def _get_dataset_info(dataset: str) -> dict:
    global _DATASET_INFO
    if _DATASET_INFO is None:
        dataset_info_path = Path(__file__).resolve().parents[1] / "dataset_info.yaml"
        with dataset_info_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        _DATASET_INFO = loaded if isinstance(loaded, dict) else {}
    di = _DATASET_INFO.get(dataset)
    if not isinstance(di, dict):
        raise KeyError(f"Unknown dataset '{dataset}' in dataset_info.yaml")
    try:
        pca_override = int((__import__("os").environ.get("PCA_NUM_OVERRIDE", "0") or 0))
    except Exception:
        pca_override = 0
    if pca_override > 0:
        return {**di, "pca_num": int(pca_override)}
    return di


def _default_positions_from_batch(batch: Batch) -> Optional[torch.Tensor]:
    try:
        h = batch[4]
        w = batch[5]
    except Exception:
        return None
    return torch.stack([h, w], dim=1)


def _gt_from_batch(batch: Batch, device: torch.device) -> torch.Tensor:
    return batch[3].to(device, non_blocking=True)


def _validate_geo_rwkv_dataset(dataset: str) -> dict:
    if dataset not in _GEO_RWKV_ALLOWED_DATASETS:
        allowed = ", ".join(sorted(_GEO_RWKV_ALLOWED_DATASETS))
        raise ValueError(f"GeoRwkvV2 only supports LiDAR datasets: {allowed} (got '{dataset}')")
    di = _get_dataset_info(dataset)
    if int(di.get("slar_channel_num", 0)) != 1:
        raise ValueError(f"GeoRwkvV2 expects LiDAR channel=1 (got slar_channel_num={di.get('slar_channel_num')})")
    return di


def _georwkv_forward_train(model: nn.Module, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    lidar = batch[1].to(device, non_blocking=True)
    hsi_pca = batch[2].to(device, non_blocking=True)
    logits, router_loss = model(hsi_pca, lidar)
    return logits, router_loss


def _georwkv_forward_logits(model: nn.Module, batch: Batch, device: torch.device) -> torch.Tensor:
    logits, _ = _georwkv_forward_train(model, batch, device)
    return logits


def _georwkv_v2_mixsig_weakbeta_create(dataset: str) -> nn.Module:
    from model.GeoRwkvV2_MixSigmoid_WeakBeta import GeoRwkvV2_MixSigmoid_WeakBeta

    di = _validate_geo_rwkv_dataset(dataset)
    return GeoRwkvV2_MixSigmoid_WeakBeta(
        num_classes=int(di["num_classes"]),
        in_hsi=int(di.get("pca_num", 30)),
        dim=128,
        depth=3,
        patch_size=int(di.get("window_size", 11)),
        gamma=10.0,
        dropout=0.1,
        router_reg=0.01,
        use_lidar_condition=True,
        clamp_min_log_gate=-20.0,
    )


register_model(
    ModelSpec(
        name="GeoRwkvV2_MixSigmoid_WeakBeta",
        create=_georwkv_v2_mixsig_weakbeta_create,
        forward_logits=_georwkv_forward_logits,
        forward_train=_georwkv_forward_train,
        get_targets=_gt_from_batch,
        get_positions=_default_positions_from_batch,
    )
)


register_model(
    ModelSpec(
        name="GeoRwkvV2_MixSigmoid_WeakBeta_Fast",
        create=_georwkv_v2_mixsig_weakbeta_create,
        forward_logits=_georwkv_forward_logits,
        forward_train=_georwkv_forward_train,
        get_targets=_gt_from_batch,
        get_positions=_default_positions_from_batch,
    )
)
