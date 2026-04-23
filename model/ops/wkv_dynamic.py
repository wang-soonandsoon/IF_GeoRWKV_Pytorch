from __future__ import annotations

import hashlib
import os
import warnings
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

_SUPPORTED_CUDA_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _neg_softplus(x: torch.Tensor) -> torch.Tensor:
    # stable negative softplus
    return -F.softplus(x)


def _wkv_dynamic_reference(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    log_gate: torch.Tensor,
) -> torch.Tensor:
    """
    Autograd-safe PyTorch reference implementation.

    For numerical stability, the recurrent states are accumulated in fp32 and the
    final output is cast back to r.dtype.
    """
    if r.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("r, k, v must all be [B, L, D]")
    if log_gate.ndim != 3 or log_gate.shape[2] != 1:
        raise ValueError("log_gate must be [B, L, 1]")
    if r.shape != k.shape or k.shape != v.shape:
        raise ValueError(f"r/k/v shape mismatch: {tuple(r.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if r.shape[:2] != log_gate.shape[:2]:
        raise ValueError(f"sequence mismatch: r={tuple(r.shape)}, log_gate={tuple(log_gate.shape)}")

    B, L, D = k.shape
    base_log_decay = _neg_softplus(time_decay.float()).view(1, D)
    time_first_f = time_first.float().view(1, D)
    log_gate_f = log_gate.float()

    aa = torch.zeros(B, D, device=k.device, dtype=torch.float32)
    bb = torch.zeros(B, D, device=k.device, dtype=torch.float32)
    pp = torch.full((B, D), -1e30, device=k.device, dtype=torch.float32)

    out = torch.empty(B, L, D, device=k.device, dtype=torch.float32)

    for t in range(L):
        kt = k[:, t, :].float()
        vt = v[:, t, :].float()
        rt = r[:, t, :].float()
        gt = log_gate_f[:, t, :]  # [B,1]

        ww = time_first_f + kt
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        denom = e1 * bb + e2 + 1e-6
        wkv = (e1 * aa + e2 * vt) / denom
        out[:, t, :] = wkv * rt

        ww2 = (base_log_decay + gt) + pp
        p2 = torch.maximum(ww2, kt)
        f1 = torch.exp(ww2 - p2)
        f2 = torch.exp(kt - p2)
        aa = f1 * aa + f2 * vt
        bb = f1 * bb + f2
        pp = p2

    return out.to(dtype=r.dtype)


def _ext_name() -> str:
    root = str(Path(__file__).resolve().parent)
    digest = hashlib.sha1(root.encode("utf-8")).hexdigest()[:8]
    return f"georwkv_wkv_dynamic_{digest}"


@lru_cache(maxsize=1)
def _load_extension():
    if os.environ.get("GEORWKV_DISABLE_CUDA_EXT", "0") == "1":
        return None
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        warnings.warn(f"[GeoRwkv] unable to import cpp_extension: {exc}")
        return None

    this_dir = Path(__file__).resolve().parent
    sources = [
        str(this_dir / "csrc" / "wkv_dynamic.cpp"),
        str(this_dir / "csrc" / "wkv_dynamic_cuda.cu"),
    ]
    build_dir = os.environ.get("GEORWKV_WKV_BUILD_DIR", str(this_dir / "csrc" / "build"))
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    verbose = os.environ.get("GEORWKV_WKV_VERBOSE_BUILD", "0") == "1"

    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]

    try:
        return load(
            name=_ext_name(),
            sources=sources,
            build_directory=build_dir,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=verbose,
        )
    except Exception as exc:  # pragma: no cover - build failure is environment-specific
        warnings.warn(
            "[GeoRwkv] failed to build/load CUDA WKV extension; falling back to the PyTorch reference path. "
            f"Reason: {exc}"
        )
        return None


def _can_use_cuda_forward(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    return (
        r.is_cuda
        and k.is_cuda
        and v.is_cuda
        and r.dtype in _SUPPORTED_CUDA_DTYPES
        and r.dtype == k.dtype == v.dtype
    )


def _cuda_forward(
    ext,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    log_gate: torch.Tensor,
) -> torch.Tensor:
    return ext.forward(
        r.contiguous(),
        k.contiguous(),
        v.contiguous(),
        time_decay.float().contiguous(),
        time_first.float().contiguous(),
        log_gate.float().contiguous(),
    )


class _WKVDynamicCudaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, time_decay, time_first, log_gate):
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("CUDA extension requested but unavailable")
        ctx.save_for_backward(r, k, v, time_decay, time_first, log_gate)
        return _cuda_forward(ext, r, k, v, time_decay, time_first, log_gate)

    @staticmethod
    def backward(ctx, grad_out):
        r, k, v, time_decay, time_first, log_gate = ctx.saved_tensors
        needs = ctx.needs_input_grad

        with torch.enable_grad():
            r_ = r.detach().requires_grad_(needs[0])
            k_ = k.detach().requires_grad_(needs[1])
            v_ = v.detach().requires_grad_(needs[2])
            td_ = time_decay.detach().requires_grad_(needs[3])
            tf_ = time_first.detach().requires_grad_(needs[4])
            lg_ = log_gate.detach().requires_grad_(needs[5])

            out = _wkv_dynamic_reference(r_, k_, v_, td_, tf_, lg_)
            grads = torch.autograd.grad(
                outputs=out,
                inputs=(r_, k_, v_, td_, tf_, lg_),
                grad_outputs=grad_out,
                allow_unused=True,
            )

        grad_r, grad_k, grad_v, grad_td, grad_tf, grad_lg = grads
        return grad_r, grad_k, grad_v, grad_td, grad_tf, grad_lg


def wkv_dynamic(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    log_gate: torch.Tensor,
) -> torch.Tensor:
    """
    Dynamic WKV dispatch.

    CUDA path:
      - fused custom CUDA forward kernel
      - autograd-safe backward via PyTorch recomputation

    Fallback path:
      - pure PyTorch reference implementation
    """
    ext = _load_extension()
    if ext is not None and _can_use_cuda_forward(r, k, v):
        wants_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in (r, k, v, time_decay, time_first, log_gate)
        )
        if wants_grad:
            return _WKVDynamicCudaAutograd.apply(r, k, v, time_decay, time_first, log_gate)
        return _cuda_forward(ext, r, k, v, time_decay, time_first, log_gate)
    return _wkv_dynamic_reference(r, k, v, time_decay, time_first, log_gate)


__all__ = [
    "wkv_dynamic",
]
