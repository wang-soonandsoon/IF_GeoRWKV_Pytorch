from __future__ import annotations

"""
Optimized GeoRwkvV2_MixSigmoid_WeakBeta

What changed relative to the original bundle implementation:
  1) The model is now sequence-native after the stem:
       grid -> spiral sequence happens once, not once per block.
  2) TokenShift is executed directly in spiral-sequence space through
     precomputed gather maps that are exactly equivalent to the original
     2D shift + spiral flatten.
  3) LiDAR sequence is computed once and reused by all blocks.
  4) Dynamic WKV dispatches to a custom CUDA forward kernel when available,
     while preserving an autograd-safe PyTorch reference fallback.
  5) The parameter/module names of the learnable parts are kept unchanged so
     existing checkpoints can still be loaded with strict=True.

Notes:
  - New helper buffers (spiral indices / shift maps) are registered as
    non-persistent buffers, so they do not enter the state_dict.
  - Output semantics and training interface remain unchanged:
      logits, router_loss = model(hsi_pca, lidar)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ops.wkv_dynamic import wkv_dynamic


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


def _build_spiral_shift_maps(
    patch_size: int,
    spiral_idx: torch.Tensor,
    inv_spiral_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Precompute gather maps that make sequence-space TokenShift exactly match
    the original 2D TokenShift2D followed by spiral flatten.

    Each target token position corresponds to a row-major location rm_t.
    We compute the source row-major location under the 2D shift and map it
    back to a spiral position using inv_spiral_idx.
    """
    P = int(patch_size)
    target_rm = spiral_idx.clone().long()  # [L]
    ys = target_rm // P
    xs = target_rm % P

    def _map(src_y: torch.Tensor, src_x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_y = torch.where(valid, src_y, torch.zeros_like(src_y))
        safe_x = torch.where(valid, src_x, torch.zeros_like(src_x))
        src_rm = safe_y * P + safe_x
        src_sp = inv_spiral_idx[src_rm]
        return src_sp.long(), valid.view(1, -1, 1).bool()

    idx_from_up, valid_from_up = _map(ys - 1, xs, ys > 0)
    idx_from_down, valid_from_down = _map(ys + 1, xs, ys < (P - 1))
    idx_from_left, valid_from_left = _map(ys, xs - 1, xs > 0)
    idx_from_right, valid_from_right = _map(ys, xs + 1, xs < (P - 1))

    return {
        "idx_from_up": idx_from_up,
        "valid_from_up": valid_from_up,
        "idx_from_down": idx_from_down,
        "valid_from_down": valid_from_down,
        "idx_from_left": idx_from_left,
        "valid_from_left": valid_from_left,
        "idx_from_right": idx_from_right,
        "valid_from_right": valid_from_right,
    }


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


class TokenShift2D(nn.Module):
    """
    Original compatibility implementation.
    Accepts grid input [B,D,H,W].

    Kept for backwards utility imports and for parity checks.
    The optimized model internally uses TokenShiftSpiral instead.
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


class TokenShiftSpiral(nn.Module):
    """
    Sequence-native equivalent of TokenShift2D + spiral flatten.

    Input/output: [B, L, D]
    Channel split is identical to the original 2D version.
    """

    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.dim = int(dim)
        spiral_idx, inv_spiral_idx = generate_spiral_indices(int(patch_size))
        maps = _build_spiral_shift_maps(int(patch_size), spiral_idx, inv_spiral_idx)
        for name, value in maps.items():
            self.register_buffer(name, value, persistent=False)

    def _shift_group(self, x: torch.Tensor, seq_idx: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        gathered = x.index_select(1, seq_idx)
        return gathered.masked_fill(~valid_mask, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TokenShiftSpiral expects [B,L,D], got shape={tuple(x.shape)}")
        B, L, D = x.shape
        split = D // 5
        g0 = D - 4 * split

        out = torch.zeros_like(x)
        out[:, :, :g0] = x[:, :, :g0]

        if split > 0:
            ch1 = slice(g0, g0 + split)
            ch2 = slice(g0 + split, g0 + 2 * split)
            ch3 = slice(g0 + 2 * split, g0 + 3 * split)
            ch4 = slice(g0 + 3 * split, g0 + 4 * split)

            out[:, :, ch1] = self._shift_group(x[:, :, ch1], self.idx_from_up, self.valid_from_up)
            out[:, :, ch2] = self._shift_group(x[:, :, ch2], self.idx_from_down, self.valid_from_down)
            out[:, :, ch3] = self._shift_group(x[:, :, ch3], self.idx_from_left, self.valid_from_left)
            out[:, :, ch4] = self._shift_group(x[:, :, ch4], self.idx_from_right, self.valid_from_right)

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
    Optimized GeoRwkvBlock:
      - sequence-native throughout the block
      - exact TokenShift behavior via precomputed spiral-space gathers
      - learnable parameter/module names preserved for checkpoint compatibility
    """

    def __init__(
        self,
        dim: int,
        patch_size: int,
        dropout: float = 0.1,
        router_hidden: int = 64,
        cmix_hidden_mult: float = 4.0,
        gate_beta_init: float = 0.2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.shift = TokenShiftSpiral(dim=dim, patch_size=patch_size)

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
        x_seq: torch.Tensor,
        l_seq: torch.Tensor,
        log_gate_fwd: torch.Tensor,
        log_gate_bwd: torch.Tensor,
        gate_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.shift(x_seq)

        x_ln = self.ln_x(x_seq)
        l_ln = self.ln_l(l_seq)

        alpha = self._router_alpha(x_ln, l_ln, gate_stats)  # [B,1,1]

        out_fast = self.fast_mlp(x_ln)

        gate_scale = F.softplus(self.gate_scale_param)  # beta > 0
        lg_f = log_gate_fwd * gate_scale
        lg_b = log_gate_bwd * gate_scale

        y_fwd = self._wkv_direction(x_ln, l_ln, lg_f)
        y_bwd = self._wkv_direction(x_ln.flip(1), l_ln.flip(1), lg_b).flip(1)

        out_fusion = self.out(y_fwd + y_bwd)

        x_seq = x_seq + alpha * out_fast + (1.0 - alpha) * out_fusion
        x_seq = x_seq + self.channel_mix(self.ln_cmix(x_seq))

        return x_seq, alpha.mean()


class GeoRwkvV2_MixSigmoid_WeakBeta(nn.Module):
    """
    Sequence-native optimized implementation of GeoRwkvV2 MixSigmoid + WeakBeta.

    This keeps the learnable module / parameter names compatible with the
    original standalone bundle while changing the internal dataflow for speed.
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
        self.center_token_index = 0  # center is first token in the generated spiral

        self.hsi_stem = nn.Conv2d(self.in_hsi, self.dim, kernel_size=1, bias=False)
        self.lid_stem = nn.Conv2d(1, self.dim, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList(
            [
                GeoRwkvBlockV2(
                    self.dim,
                    patch_size=self.P,
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

        x_seq = grid_to_spiral_seq(self.hsi_stem(hsi), self.spiral_idx)
        l_seq = grid_to_spiral_seq(self.lid_stem(lidar), self.spiral_idx)

        log_gate_fwd, log_gate_bwd, gate_stats = self._build_lidar_gates(lidar)

        router_loss = torch.zeros((), device=hsi.device, dtype=x_seq.dtype)
        for blk in self.blocks:
            x_seq, alpha_mean = blk(
                x_seq,
                l_seq,
                log_gate_fwd,
                log_gate_bwd,
                gate_stats,
            )
            router_loss = router_loss + (alpha_mean - 0.5) ** 2
        router_loss = router_loss * self.router_reg

        feat = x_seq[:, self.center_token_index, :]
        feat = self.head_ln(feat)
        logits = self.head(feat)
        return logits, router_loss


__all__ = [
    "GeoRwkvV2_MixSigmoid_WeakBeta",
    "GeoRwkvBlockV2",
    "RWKVChannelMix",
    "TokenShift2D",
    "TokenShiftSpiral",
    "generate_spiral_indices",
    "inverse_softplus",
    "log_gate_from_lidar_seq",
    "lerp_mix",
    "grid_to_spiral_seq",
    "spiral_seq_to_grid",
    "time_shift_1d",
    "wkv_dynamic",
]
