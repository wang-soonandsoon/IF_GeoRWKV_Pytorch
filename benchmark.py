from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from model.GeoRwkvV2_MixSigmoid_WeakBeta import GeoRwkvV2_MixSigmoid_WeakBeta as FastModel
from model.reference_slow import GeoRwkvV2_MixSigmoid_WeakBeta as RefModel


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_ctx(device: torch.device, amp: bool, amp_dtype: str):
    if device.type != "cuda" or not amp:
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    if dtype == torch.bfloat16 and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _time_model(model, hsi, lidar, iters: int, warmup: int, amp: bool, amp_dtype: str):
    device = hsi.device
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with _autocast_ctx(device, amp, amp_dtype):
                model(hsi, lidar)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            with _autocast_ctx(device, amp, amp_dtype):
                model(hsi, lidar)
        _sync(device)
        t1 = time.perf_counter()
    return (t1 - t0) / max(iters, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reference vs optimized GeoRwkv benchmark")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--patch", type=int, default=11, choices=[11, 13])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    in_hsi = 35 if args.patch == 11 else 30
    num_classes = 15 if args.patch == 11 else 20

    kwargs = dict(
        num_classes=num_classes,
        in_hsi=in_hsi,
        dim=args.dim,
        depth=args.depth,
        patch_size=args.patch,
        gamma=10.0,
        dropout=0.0,
        router_reg=0.01,
        use_lidar_condition=True,
        clamp_min_log_gate=-20.0,
    )

    torch.manual_seed(0)
    ref = RefModel(**kwargs).to(device)
    fast = FastModel(**kwargs).to(device)
    fast.load_state_dict(ref.state_dict(), strict=True)

    hsi = torch.randn(args.batch, in_hsi, args.patch, args.patch, device=device)
    lidar = torch.randn(args.batch, 1, args.patch, args.patch, device=device)

    ref_t = _time_model(ref, hsi, lidar, args.iters, args.warmup, bool(args.amp), args.amp_dtype)
    fast_t = _time_model(fast, hsi, lidar, args.iters, args.warmup, bool(args.amp), args.amp_dtype)

    speedup = ref_t / max(fast_t, 1e-12)
    print(f"device={device}")
    print(f"ref_avg_s={ref_t:.6f}")
    print(f"fast_avg_s={fast_t:.6f}")
    print(f"speedup_x={speedup:.3f}")
