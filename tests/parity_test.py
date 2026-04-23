from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.GeoRwkvV2_MixSigmoid_WeakBeta import GeoRwkvV2_MixSigmoid_WeakBeta as FastModel
from model.reference_slow import GeoRwkvV2_MixSigmoid_WeakBeta as RefModel


def _compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> None:
    diff = (a - b).abs().max().item()
    print(f"[{name}] max_abs_diff={diff:.8f}")
    if diff > atol:
        raise AssertionError(f"{name} mismatch: {diff} > {atol}")


def _build_models(patch_size: int):
    kwargs = dict(
        num_classes=6,
        in_hsi=30,
        dim=16,
        depth=1,
        patch_size=patch_size,
        gamma=10.0,
        dropout=0.0,
        router_reg=0.01,
        use_lidar_condition=True,
        clamp_min_log_gate=-20.0,
    )
    torch.manual_seed(0)
    ref = RefModel(**kwargs)
    fast = FastModel(**kwargs)
    fast.load_state_dict(ref.state_dict(), strict=True)
    return ref, fast


def run_eval_parity(patch_size: int) -> None:
    ref, fast = _build_models(patch_size)
    ref.eval()
    fast.eval()

    torch.manual_seed(123)
    hsi = torch.randn(4, 30, patch_size, patch_size)
    lidar = torch.randn(4, 1, patch_size, patch_size)

    with torch.no_grad():
        y_ref, r_ref = ref(hsi, lidar)
        y_fast, r_fast = fast(hsi, lidar)

    _compare_tensors(f"eval_logits_P{patch_size}", y_ref, y_fast)
    _compare_tensors(f"eval_router_P{patch_size}", r_ref, r_fast)


def run_grad_parity(patch_size: int) -> None:
    ref, fast = _build_models(patch_size)
    ref.train()
    fast.train()

    torch.manual_seed(321)
    hsi_ref = torch.randn(2, 30, patch_size, patch_size, requires_grad=True)
    lidar_ref = torch.randn(2, 1, patch_size, patch_size, requires_grad=True)
    hsi_fast = hsi_ref.detach().clone().requires_grad_(True)
    lidar_fast = lidar_ref.detach().clone().requires_grad_(True)

    y_ref, r_ref = ref(hsi_ref, lidar_ref)
    loss_ref = y_ref.square().mean() + r_ref
    loss_ref.backward()

    y_fast, r_fast = fast(hsi_fast, lidar_fast)
    loss_fast = y_fast.square().mean() + r_fast
    loss_fast.backward()

    _compare_tensors(f"grad_hsi_input_P{patch_size}", hsi_ref.grad, hsi_fast.grad)
    _compare_tensors(f"grad_lidar_input_P{patch_size}", lidar_ref.grad, lidar_fast.grad)
    _compare_tensors(f"grad_head_weight_P{patch_size}", ref.head.weight.grad, fast.head.weight.grad)
    _compare_tensors(f"grad_time_decay_P{patch_size}", ref.blocks[0].time_decay.grad, fast.blocks[0].time_decay.grad)
    _compare_tensors(f"grad_hsi_stem_P{patch_size}", ref.hsi_stem.weight.grad, fast.hsi_stem.weight.grad)


if __name__ == "__main__":
    for patch_size in (11, 13):
        run_eval_parity(patch_size)
    run_grad_parity(11)
    print("All parity checks passed.")
