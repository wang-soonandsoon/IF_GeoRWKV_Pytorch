from __future__ import annotations

"""
GeoRwkvV2_MixSigmoid_WeakBeta (single-file, modular layout)

This file intentionally keeps the *same* module/parameter names as the original
implementation so existing checkpoints can still be loaded with `strict=True`.

This is the "final" single-path version:
  - TimeMix uses sigmoid-parameterized per-channel mixing (no clamp mode).
  - Uses LiDAR conditioning (always on for this model family).
  - Uses TokenShift2D (always on for this model family).
  - No "v-only" shortcuts; r/k/v all use time-mix.

High-level dataflow (paper-friendly):
  HSI/LiDAR -> Stem (1x1 conv) -> N x GeoRwkvBlockV2 -> Center readout -> Linear head

Inside each GeoRwkvBlockV2:
  TokenShift2D (optional) -> Spiral flatten -> LN
    -> Router (alpha) decides mix of:
       A) FastMLP(x)            (content path)
       B) Bi-directional WKV    (token mixing path, LiDAR-gated)
    -> RWKVChannelMix           (channel mixing path)
  -> Un-spiral back to 2D grid
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_spiral_indices(patch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate spiral indices from center outwards for a P x P grid.

    Returns:
      spiral_idx: [L] row-major flatten indices ordered by spiral
      inv_spiral_idx: [L] inverse mapping so that x[:, inv_spiral_idx] restores row-major order
    """
    P = int(patch_size)
    if P % 2 != 1:
        raise ValueError(f"patch_size must be odd (got {P})")
    cy = cx = P // 2

    visited = torch.zeros(P, P, dtype=torch.bool)
    coords = []

    # Right, Down, Left, Up (clockwise)
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_idx = 0

    y, x = cy, cx
    coords.append((y, x))
    visited[y, x] = True

    for _ in range(P * P - 1):
        ny = y + dirs[(dir_idx + 1) % 4][0]
        nx = x + dirs[(dir_idx + 1) % 4][1]
        if 0 <= ny < P and 0 <= nx < P and not visited[ny, nx]:
            dir_idx = (dir_idx + 1) % 4

        y = y + dirs[dir_idx][0]
        x = x + dirs[dir_idx][1]
        coords.append((y, x))
        visited[y, x] = True

    coords_t = torch.tensor(coords, dtype=torch.long)  # [L,2] (y,x)
    spiral_idx = coords_t[:, 0] * P + coords_t[:, 1]
    inv_spiral_idx = torch.empty_like(spiral_idx)
    inv_spiral_idx[spiral_idx] = torch.arange(spiral_idx.numel(), dtype=torch.long)
    return spiral_idx, inv_spiral_idx


def log_gate_from_lidar_seq(lidar_seq: torch.Tensor, gamma: float = 10.0, clamp_min: float = -20.0) -> torch.Tensor:
    """
    Step-wise log-gate from LiDAR height sequence.

    lidar_seq: [B, L, 1] (spiral-ordered heights)
    return:   [B, L, 1] log_gate <= 0
    """
    diff = torch.zeros_like(lidar_seq)
    diff[:, 1:, :] = (lidar_seq[:, 1:, :] - lidar_seq[:, :-1, :]).abs()
    return (-float(gamma) * diff).clamp(min=float(clamp_min), max=0.0)


def inverse_softplus(x: float) -> float:
    # y = softplus(z) => z = log(exp(y)-1)
    return math.log(math.exp(float(x)) - 1.0)


def inverse_sigmoid(p: float) -> float:
    p = float(p)
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def time_shift_1d(x: torch.Tensor) -> torch.Tensor:
    """
    RWKV time_shift (shift along sequence length).
    x: [B, L, D]
    return: [B, L, D] where out[:,0]=0 and out[:,t]=x[:,t-1]
    """
    return torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)


def lerp_mix(x: torch.Tensor, xx: torch.Tensor, mix: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    x,xx: [B,L,D]
    mix:  [D] or [1,1,D], intended in [0,1]
    return: x * mix + xx * (1-mix)
    """
    if mix.dim() == 1:
        mix = mix.view(1, 1, -1)
    if clamp:
        mix = mix.clamp(0.0, 1.0)
    return x * mix + xx * (1.0 - mix)


def grid_to_spiral_seq(x_grid: torch.Tensor, spiral_idx: torch.Tensor) -> torch.Tensor:
    """
    Convert a 2D grid (channels-first) into a spiral-ordered sequence.

    x_grid:    [B, D, P, P]
    spiral_idx:[L] where L = P*P (row-major indices in spiral order)
    return:    [B, L, D]
    """
    B, D, P, _ = x_grid.shape
    L = P * P
    x_row = x_grid.permute(0, 2, 3, 1).reshape(B, L, D)
    return x_row[:, spiral_idx, :]


def spiral_seq_to_grid(x_spiral: torch.Tensor, inv_spiral_idx: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Restore spiral-ordered sequence back to 2D grid (channels-first).

    x_spiral:       [B, L, D]
    inv_spiral_idx: [L] inverse mapping so x[:, inv_spiral_idx] restores row-major
    patch_size:     P
    return:         [B, D, P, P]
    """
    P = int(patch_size)
    B, L, D = x_spiral.shape
    x_row = x_spiral[:, inv_spiral_idx, :]
    return x_row.reshape(B, P, P, D).permute(0, 3, 1, 2)


def _wkv_dynamic_impl(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    log_gate: torch.Tensor,
) -> torch.Tensor:
    """
    RWKV-style WKV recurrence with dynamic log-gate.

    r,k,v:     [B, L, D]
    log_gate:  [B, L, 1] (<=0), step-wise barrier from LiDAR
    time_decay,time_first: [D]
    """
    B, L, D = k.shape
    base_log_decay = -F.softplus(time_decay)  # [D] <=0

    aa = torch.zeros(B, D, device=k.device, dtype=k.dtype)
    bb = torch.zeros(B, D, device=k.device, dtype=k.dtype)
    pp = torch.full((B, D), -1e38, device=k.device, dtype=k.dtype)

    out = torch.zeros_like(k)

    for t in range(L):
        kt = k[:, t, :]
        vt = v[:, t, :]
        rt = r[:, t, :]
        gt = log_gate[:, t, :]  # [B,1]

        ww = time_first + kt
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2 + 1e-6)
        out[:, t, :] = wkv * rt

        ww2 = (base_log_decay + gt) + pp  # [B,D]
        p2 = torch.maximum(ww2, kt)
        e1 = torch.exp(ww2 - p2)
        e2 = torch.exp(kt - p2)
        aa = e1 * aa + e2 * vt
        bb = e1 * bb + e2
        pp = p2

    return out


try:
    wkv_dynamic = torch.jit.script(_wkv_dynamic_impl)
except Exception:
    wkv_dynamic = _wkv_dynamic_impl


class TokenShift2D(nn.Module):
    """
    Vision-RWKV style TokenShift:
    split channels into 5 groups: stay / up / down / left / right (shift by 1 pixel).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        split = D // 5
        g0 = D - 4 * split  # remainder goes to "stay"

        out = torch.zeros_like(x)
        out[:, :g0] = x[:, :g0]

        if split > 0:
            out[:, g0 : g0 + split, 1:, :] = x[:, g0 : g0 + split, :-1, :]
            out[:, g0 + split : g0 + 2 * split, :-1, :] = x[:, g0 + split : g0 + 2 * split, 1:, :]
            out[:, g0 + 2 * split : g0 + 3 * split, :, 1:] = x[:, g0 + 2 * split : g0 + 3 * split, :, :-1]
            out[:, g0 + 3 * split : g0 + 4 * split, :, :-1] = x[:, g0 + 3 * split : g0 + 4 * split, :, 1:]

        return out


class RWKVChannelMix(nn.Module):
    """
    RWKV-style ChannelMix (FFN):
      xx = time_shift(x)
      xk = lerp(x, xx, time_mix_k)
      xr = lerp(x, xx, time_mix_r)
      k = key(xk)
      k = GELU(k)
      v = value(k)
      out = sigmoid(receptance(xr)) * v
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 4.0,
        dropout: float = 0.0,
        rwkv_zero_init: bool = True,
        receptance_zero_init: bool = True,
        receptance_init_std: float = 1e-3,
    ):
        super().__init__()
        self.dim = int(dim)
        hidden = int(self.dim * float(hidden_mult))

        self.time_mix_k = nn.Parameter(torch.full((self.dim,), 0.5))
        self.time_mix_r = nn.Parameter(torch.full((self.dim,), 0.5))

        self.key = nn.Linear(self.dim, hidden, bias=False)
        self.value = nn.Linear(hidden, self.dim, bias=False)
        self.receptance = nn.Linear(self.dim, self.dim, bias=False)

        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        if rwkv_zero_init:
            nn.init.zeros_(self.value.weight)
            if receptance_zero_init:
                nn.init.zeros_(self.receptance.weight)
            else:
                nn.init.normal_(self.receptance.weight, mean=0.0, std=float(receptance_init_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = time_shift_1d(x)
        xk = lerp_mix(x, xx, self.time_mix_k)
        xr = lerp_mix(x, xx, self.time_mix_r)

        k = self.key(xk)
        k = F.gelu(k)
        v = self.value(self.drop(k))

        r = torch.sigmoid(self.receptance(xr))
        return r * v


class GeoRwkvBlockV2(nn.Module):
    """
    GeoRwkvBlock with RWKV-style TimeMix (time_mix + time_shift) and ChannelMix (GELU + gate + time_mix).
    Keeps TokenShift2D / Spiral / LiDAR gate / Bi-WKV / router structure.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        router_hidden: int = 64,
        cmix_hidden_mult: float = 4.0,
        gate_beta_init: float = 0.2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.shift = TokenShift2D(dim)

        self.ln_x = nn.LayerNorm(dim)
        self.ln_l = nn.LayerNorm(dim)

        # Path A: fast MLP
        self.fast_mlp = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
        )

        # Path B: RWKV TimeMix (sigmoid-parameterized time_mix for r/k/v)
        self.time_mix_k = nn.Parameter(torch.full((dim,), inverse_sigmoid(0.9)))
        self.time_mix_v = nn.Parameter(torch.full((dim,), inverse_sigmoid(0.7)))
        self.time_mix_r = nn.Parameter(torch.full((dim,), inverse_sigmoid(0.9)))

        self.time_decay = nn.Parameter(torch.zeros(dim))
        self.time_first = nn.Parameter(torch.zeros(dim))

        self.r_x = nn.Linear(dim, dim, bias=False)
        self.k_x = nn.Linear(dim, dim, bias=False)
        self.v_x = nn.Linear(dim, dim, bias=False)

        self.r_l = nn.Linear(dim, dim, bias=False)
        self.k_l = nn.Linear(dim, dim, bias=False)

        self.out = nn.Linear(dim, dim, bias=False)

        # RWKV-style ChannelMix (replaces generic FFN)
        self.ln_cmix = nn.LayerNorm(dim)
        self.channel_mix = RWKVChannelMix(
            dim=dim,
            hidden_mult=cmix_hidden_mult,
            dropout=dropout,
            rwkv_zero_init=True,
            receptance_zero_init=True,
        )

        # Soft router
        self.router = nn.Sequential(
            nn.Linear(dim * 2 + 3, router_hidden),
            nn.Tanh(),
            nn.Linear(router_hidden, 1),
            nn.Sigmoid(),
        )

        self.gate_scale_param = nn.Parameter(torch.tensor(inverse_softplus(gate_beta_init), dtype=torch.float32))

        # Keep the initialization consistent with the historically best config:
        # rwkv_zero_init=True and zero_init_mode="all".
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.k_x.weight)
        nn.init.zeros_(self.r_x.weight)

    def _make_rkv(self, x_ln: torch.Tensor, l_ln: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xx = time_shift_1d(x_ln)
        mk = torch.sigmoid(self.time_mix_k).view(1, 1, -1)
        mv = torch.sigmoid(self.time_mix_v).view(1, 1, -1)
        mr = torch.sigmoid(self.time_mix_r).view(1, 1, -1)

        xk = x_ln * mk + xx * (1.0 - mk)
        xr = x_ln * mr + xx * (1.0 - mr)
        xv = x_ln * mv + xx * (1.0 - mv)

        r = self.r_x(xr)
        k = self.k_x(xk)
        v = self.v_x(xv)
        r = r + self.r_l(l_ln)
        k = k + self.k_l(l_ln)

        r = torch.sigmoid(r)
        return r, k, v

    def _router_alpha(self, x_ln: torch.Tensor, l_ln: torch.Tensor, gate_stats: torch.Tensor) -> torch.Tensor:
        """
        Router weight alpha in [0,1] (broadcasted to tokens).

        x_ln,l_ln: [B,L,D]
        gate_stats:[B,3]
        return:    [B,1,1]
        """
        x_pool = x_ln.mean(1)
        l_pool = l_ln.mean(1)
        return self.router(torch.cat([x_pool, l_pool, gate_stats], dim=1)).unsqueeze(1)

    def _wkv_direction(self, x_ln: torch.Tensor, l_ln: torch.Tensor, log_gate: torch.Tensor) -> torch.Tensor:
        r, k, v = self._make_rkv(x_ln, l_ln)
        return wkv_dynamic(r, k, v, self.time_decay, self.time_first, log_gate)

    def forward(
        self,
        x_grid: torch.Tensor,
        l_grid: torch.Tensor,
        log_gate_fwd: torch.Tensor,
        log_gate_bwd: torch.Tensor,
        gate_stats: torch.Tensor,
        spiral_idx: torch.Tensor,
        inv_spiral_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, P, _ = x_grid.shape

        x_grid = self.shift(x_grid)

        x = grid_to_spiral_seq(x_grid, spiral_idx)
        l = grid_to_spiral_seq(l_grid, spiral_idx)

        x_ln = self.ln_x(x)
        l_ln = self.ln_l(l)

        alpha = self._router_alpha(x_ln, l_ln, gate_stats)  # [B,1,1]

        out_fast = self.fast_mlp(x_ln)

        gate_scale = F.softplus(self.gate_scale_param)  # beta > 0
        lg_f = log_gate_fwd * gate_scale
        lg_b = log_gate_bwd * gate_scale

        y_fwd = self._wkv_direction(x_ln, l_ln, lg_f)
        y_bwd = self._wkv_direction(x_ln.flip(1), l_ln.flip(1), lg_b).flip(1)

        out_fusion = self.out(y_fwd + y_bwd)

        x = x + alpha * out_fast + (1.0 - alpha) * out_fusion
        x = x + self.channel_mix(self.ln_cmix(x))

        return spiral_seq_to_grid(x, inv_spiral_idx, P), alpha.mean()


class GeoRwkvV2_MixSigmoid_WeakBeta(nn.Module):
    """
    Fully standalone implementation of the GeoRwkvV2 MixSigmoid + WeakBeta variant.

    This module intentionally does NOT import or depend on `model/GeoRwkvV2.py`.
    """

    def __init__(
        self,
        num_classes: int,
        in_hsi: int = 30,
        dim: int = 128,
        depth: int = 3,
        patch_size: int = 11,
        gamma: float = 10.0,
        dropout: float = 0.1,
        router_reg: float = 0.01,
        use_lidar_condition: bool = True,
        clamp_min_log_gate: float = -20.0,
    ):
        super().__init__()
        if not bool(use_lidar_condition):
            raise ValueError("GeoRwkvV2_MixSigmoid_WeakBeta requires LiDAR conditioning (use_lidar_condition=True).")
        self.out_features = int(num_classes)
        self.in_hsi = int(in_hsi)
        self.dim = int(dim)
        self.depth = int(depth)
        self.P = int(patch_size)
        self.gamma = float(gamma)
        self.dropout = float(dropout)
        self.router_reg = float(router_reg)
        self.use_lidar_condition = True
        self.clamp_min_log_gate = float(clamp_min_log_gate)

        spiral_idx, inv_spiral_idx = generate_spiral_indices(self.P)
        self.register_buffer("spiral_idx", spiral_idx)
        self.register_buffer("inv_spiral_idx", inv_spiral_idx)

        self.hsi_stem = nn.Conv2d(self.in_hsi, self.dim, kernel_size=1, bias=False)
        self.lid_stem = nn.Conv2d(1, self.dim, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList(
            [
                GeoRwkvBlockV2(
                    self.dim,
                    dropout=self.dropout,
                    cmix_hidden_mult=4.0,
                    gate_beta_init=0.2,
                )
                for _ in range(self.depth)
            ]
        )

        self.head_ln = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.out_features)

    def _build_lidar_gates(self, lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build LiDAR-conditioned gates (forward/backward) and their summary statistics.

        lidar: [B,1,P,P]
        return:
          log_gate_fwd: [B,L,1]
          log_gate_bwd: [B,L,1]
          gate_stats:   [B,3] = [mean, min, std] of log_gate_fwd over tokens
        """
        lidar_sp = grid_to_spiral_seq(lidar, self.spiral_idx)  # [B,L,1]
        log_gate_fwd = log_gate_from_lidar_seq(lidar_sp, gamma=self.gamma, clamp_min=self.clamp_min_log_gate)
        log_gate_bwd = log_gate_from_lidar_seq(
            lidar_sp.flip(1), gamma=self.gamma, clamp_min=self.clamp_min_log_gate
        )

        gate_mean = log_gate_fwd.mean(1)
        gate_min = log_gate_fwd.min(1)[0]
        gate_std = log_gate_fwd.std(1)
        gate_stats = torch.cat([gate_mean, gate_min, gate_std], dim=1)  # [B,3]
        return log_gate_fwd, log_gate_bwd, gate_stats

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, P, _ = hsi.shape
        if P != self.P:
            raise ValueError(f"patch_size mismatch: got {P}, expected {self.P}")
        if lidar.shape[1] != 1:
            raise ValueError(f"GeoRwkvV2_MixSigmoid_WeakBeta expects lidar with 1 channel, got {tuple(lidar.shape)}")

        x_grid = self.hsi_stem(hsi)
        l_grid = self.lid_stem(lidar)

        log_gate_fwd, log_gate_bwd, gate_stats = self._build_lidar_gates(lidar)

        router_loss = torch.zeros((), device=hsi.device, dtype=hsi.dtype)
        for blk in self.blocks:
            x_grid, alpha_mean = blk(
                x_grid,
                l_grid,
                log_gate_fwd,
                log_gate_bwd,
                gate_stats,
                self.spiral_idx,
                self.inv_spiral_idx,
            )
            router_loss = router_loss + (alpha_mean - 0.5) ** 2
        router_loss = router_loss * self.router_reg

        cy = cx = P // 2
        feat = x_grid[:, :, cy, cx]
        feat = self.head_ln(feat)
        logits = self.head(feat)
        return logits, router_loss

__all__ = [
    "GeoRwkvV2_MixSigmoid_WeakBeta",
    "GeoRwkvBlockV2",
    "RWKVChannelMix",
    "TokenShift2D",
    "generate_spiral_indices",
    "inverse_softplus",
    "log_gate_from_lidar_seq",
    "lerp_mix",
    "grid_to_spiral_seq",
    "spiral_seq_to_grid",
    "time_shift_1d",
    "wkv_dynamic",
]
