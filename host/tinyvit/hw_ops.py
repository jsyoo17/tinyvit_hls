# host/tinyvit/hw_ops.py
"""
Hardware-oriented Conv2D operators for TinyViT HLS project.

This file contains:
  - Simple reference conv:     hw_conv2d_nchw, HWConv2d
  - Explicit im2col (full):    im2col_nchw_explicit  (for debugging)
  - Tiled streaming im2col:    im2col_nchw_tile
  - Explicit systolic GEMM:    systolic_gemm_explicit
  - Streaming conv2d:          conv2d_nchw_streaming_im2col

All functions assume NCHW layout and NO dilation.
"""

from __future__ import annotations
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert int or (int, int) to (int, int)."""
    if isinstance(x, tuple):
        return x
    return (x, x)


def linear_to_2d(l: int, W_out: int) -> Tuple[int, int]:
    """Convert linear index l in [0, H_out*W_out) to (oh, ow)."""
    oh = l // W_out
    ow = l % W_out
    return oh, ow


# ---------------------------------------------------------------------------
# 1. Simple reference conv: full im2col + matmul
# ---------------------------------------------------------------------------

def hw_conv2d_nchw(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
) -> torch.Tensor:
    """
    Pure-tensor Conv2D implementation (NCHW) using full im2col + matmul.

    Equivalent to torch.nn.Conv2d(dilation=1) with:
        - input:  x      (N, C_in, H_in, W_in)
        - weight: (C_out, C_in/groups, K_h, K_w)
        - bias:   (C_out,) or None

    This is the simplest "hardware-style" reference: it explicitly exposes
    the im2col + GEMM structure, but still materializes the full im2col
    matrix in memory.

    This is great as a **golden reference** for:
        - correctness checks
        - matching nn.Conv2d exactly
    """
    from torch.nn.functional import unfold

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_per_g, K_h, K_w = weight.shape

    stride_h, stride_w = _to_2tuple(stride)
    pad_h, pad_w       = _to_2tuple(padding)

    assert C_in % groups == 0, "C_in must be divisible by groups"
    assert C_out % groups == 0, "C_out must be divisible by groups"

    C_in_g  = C_in // groups
    C_out_g = C_out // groups

    # Output spatial size
    H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

    outs = []

    for g in range(groups):
        # Slice input channels for this group
        x_g = x[:, g * C_in_g:(g + 1) * C_in_g, :, :]  # (N, C_in_g, H_in, W_in)

        # im2col: (N, C_in_g*K_h*K_w, L)
        x_unf = unfold(
            x_g,
            kernel_size=(K_h, K_w),
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w),
        )  # (N, K, L) where K = C_in_g*K_h*K_w

        # Weight for this group
        w_g = weight[g * C_out_g:(g + 1) * C_out_g, :, :, :]  # (C_out_g, C_in_g, K_h, K_w)
        w_g = w_g.view(C_out_g, -1)  # (C_out_g, K)

        # Matmul per batch: (N, C_out_g, L)
        out_g = torch.matmul(w_g, x_unf)  # (N, C_out_g, L)
        outs.append(out_g)

    # Concatenate groups: (N, C_out, L)
    out = torch.cat(outs, dim=1)

    # Reshape to (N, C_out, H_out, W_out)
    out = out.view(N, C_out, H_out, W_out)

    # Add bias
    if bias is not None:
        out = out + bias.view(1, C_out, 1, 1)

    return out


class HWConv2d(nn.Module):
    """
    Hardware-style Conv2D nn.Module wrapper around hw_conv2d_nchw.

    - No nn.Conv2d/F.conv2d in forward.
    - No dilation support (dilation is assumed 1).
    - Groups supported.

    Useful for:
        - plugging into your TinyViT SW model
        - copying weights from nn.Conv2d and comparing outputs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = _to_2tuple(kernel_size)
        self.stride       = _to_2tuple(stride)
        self.padding      = _to_2tuple(padding)
        self.groups       = groups

        k_h, k_w = self.kernel_size
        c_in_per_g = in_channels // groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, c_in_per_g, k_h, k_w)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Init similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * k_h * k_w / groups
            bound = 1.0 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "HWConv2d":
        """Create HWConv2d from an existing nn.Conv2d (dilation must be 1)."""
        assert conv.dilation == (1, 1), "HWConv2d does not support dilation!"

        hw = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        with torch.no_grad():
            hw.weight.copy_(conv.weight)
            if hw.bias is not None and conv.bias is not None:
                hw.bias.copy_(conv.bias)
        return hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hw_conv2d_nchw(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )


# ---------------------------------------------------------------------------
# 2. Explicit full im2col (debugging / reference, not for speed)
# ---------------------------------------------------------------------------

def im2col_nchw_explicit(
    x: torch.Tensor,  # (N, C, H, W)
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> torch.Tensor:
    """
    Explicit im2col implementation (no torch.unfold).

    Args:
        x: (N, C, H, W)
        kernel_size: (K_h, K_w)
        stride: (stride_h, stride_w)
        padding: (pad_h, pad_w)

    Returns:
        cols: (N, C*K_h*K_w, L) where L = H_out * W_out
    """
    N, C, H, W = x.shape
    K_h, K_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    H_out = (H + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W + 2 * pad_w - K_w) // stride_w + 1
    L = H_out * W_out
    K = C * K_h * K_w

    cols = torch.zeros(N, K, L, dtype=x.dtype, device=x.device)

    for n in range(N):
        col_idx = 0
        for oh in range(H_out):
            for ow in range(W_out):
                in_h0 = oh * stride_h - pad_h
                in_w0 = ow * stride_w - pad_w

                flat_idx = 0
                for c in range(C):
                    for kh in range(K_h):
                        for kw in range(K_w):
                            ih = in_h0 + kh
                            iw = in_w0 + kw
                            if (0 <= ih < H) and (0 <= iw < W):
                                v = x[n, c, ih, iw]
                            else:
                                v = 0.0
                            cols[n, flat_idx, col_idx] = v
                            flat_idx += 1

                col_idx += 1

    return cols


# ---------------------------------------------------------------------------
# 3. Streaming / tiled im2col: build only a subset of columns at a time
# ---------------------------------------------------------------------------

def im2col_nchw_tile(
    x_n: torch.Tensor,          # (C_in, H_in, W_in) for a single batch
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    H_out: int,
    W_out: int,
    l_start: int,
    tile_L: int,
) -> torch.Tensor:
    """
    Build a *tile* of the im2col matrix for a single batch.

    Inputs:
        x_n:        (C_in, H_in, W_in)
        kernel_size: (K_h, K_w)
        stride:     (stride_h, stride_w)
        padding:    (pad_h, pad_w)
        H_out, W_out: output spatial dims
        l_start:    starting linear location index
        tile_L:     number of locations to generate (max)

    Output:
        B_tile: (C_in*K_h*K_w, L_tile_eff) where
                L_tile_eff = min(tile_L, total_L - l_start)
    """
    C_in, H_in, W_in = x_n.shape
    K_h, K_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    L_total = H_out * W_out
    L_tile_eff = min(tile_L, L_total - l_start)
    K = C_in * K_h * K_w

    B_tile = torch.zeros(K, L_tile_eff, dtype=x_n.dtype, device=x_n.device)

    for t in range(L_tile_eff):
        l = l_start + t
        oh, ow = linear_to_2d(l, W_out)

        in_h0 = oh * stride_h - pad_h
        in_w0 = ow * stride_w - pad_w

        flat_idx = 0
        for c in range(C_in):
            for kh in range(K_h):
                for kw in range(K_w):
                    ih = in_h0 + kh
                    iw = in_w0 + kw
                    if (0 <= ih < H_in) and (0 <= iw < W_in):
                        v = x_n[c, ih, iw]
                    else:
                        v = 0.0
                    B_tile[flat_idx, t] = v
                    flat_idx += 1

    return B_tile


# ---------------------------------------------------------------------------
# 4. Explicit systolic-style GEMM (output-stationary)
# ---------------------------------------------------------------------------

def systolic_gemm_explicit(
    A: torch.Tensor,  # (M, K) weight matrix
    B: torch.Tensor,  # (K, N) activation tile
    P: int,           # PE rows  (subset of M)
    Q: int,           # PE cols  (subset of N)
    tile_k: int = 16,
) -> torch.Tensor:
    """
    Output-stationary systolic-like GEMM:
        C = A @ B

    Shape:
        A: (M, K)
        B: (K, N)
        C: (M, N)

    P × Q = PE array size:
        - P rows of PEs, each handling one output-row index
        - Q cols of PEs, each handling one output-col index

    Implementation:
        - Tiles over M, N using P, Q
        - Scans K in chunks of tile_k
        - Each tile uses a local C_tile accumulator
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "GEMM dimension mismatch"

    C = torch.zeros(M, N, dtype=A.dtype, device=A.device)

    for m0 in range(0, M, P):
        for n0 in range(0, N, Q):
            m_max = min(m0 + P, M)
            n_max = min(n0 + Q, N)
            p_eff = m_max - m0
            q_eff = n_max - n0

            # Local accumulators for this tile (P × Q PEs)
            C_tile = torch.zeros(p_eff, q_eff, dtype=A.dtype, device=A.device)

            for k0 in range(0, K, tile_k):
                k_max = min(k0 + tile_k, K)
                k_eff = k_max - k0

                for kk in range(k_eff):
                    k_idx = k0 + kk

                    # Broadcast A[m0+i, k_idx] along PE row i
                    # Broadcast B[k_idx, n0+j] along PE col j
                    for i in range(p_eff):
                        a_val = A[m0 + i, k_idx]
                        for j in range(q_eff):
                            b_val = B[k_idx, n0 + j]
                            C_tile[i, j] += a_val * b_val

            # Write back tile
            C[m0:m_max, n0:n_max] = C_tile

    return C


# ---------------------------------------------------------------------------
# 5. Streaming Conv2D: tiled im2col + systolic GEMM
# ---------------------------------------------------------------------------

def conv2d_nchw_streaming_im2col(
    x: torch.Tensor,        # (N, C_in, H_in, W_in)
    weight: torch.Tensor,   # (C_out, C_in/groups, K_h, K_w)
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    max_L_tile: int = 64,   # max spatial locations per tile
    P: int = 8,             # PE rows
    Q: int = 8,             # PE cols
    tile_k: int = 16,
) -> torch.Tensor:
    """
    Streaming / tiled Conv2D via:
        - im2col_nchw_tile (build activation tiles)
        - systolic_gemm_explicit (GEMM core)
        - writeback to NCHW output

    Inputs:
        x:      (N, C_in, H_in, W_in)
        weight: (C_out, C_in/groups, K_h, K_w)
        bias:   (C_out,) or None
        stride, padding: int or (h, w)
        groups: standard or depthwise conv style

    Tuning knobs:
        max_L_tile: controls im2col tile size (K × max_L_tile buffer)
        P, Q:       systolic PE array shape
        tile_k:     K dimension chunking

    This function is the closest software analogue to a hardware conv engine
    that:
        - streams patches from external memory
        - uses a systolic/GEMM-like core
        - writes partial outputs back to memory
    """
    stride_h, stride_w = _to_2tuple(stride)
    pad_h, pad_w       = _to_2tuple(padding)

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_per_g, K_h, K_w = weight.shape

    assert C_in % groups == 0, "C_in must be divisible by groups"
    assert C_out % groups == 0, "C_out must be divisible by groups"

    C_in_g  = C_in // groups
    C_out_g = C_out // groups

    # Output spatial dims
    H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1
    L_total = H_out * W_out

    # Output tensor
    y = torch.zeros(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)

    # Iterate over groups
    for g in range(groups):
        # Weight for this group: (C_out_g, C_in_g, K_h, K_w) → (M, K)
        W_g = weight[g * C_out_g:(g + 1) * C_out_g, :, :, :].reshape(C_out_g, -1)

        if bias is not None:
            bias_g = bias[g * C_out_g:(g + 1) * C_out_g]  # (C_out_g,)
        else:
            bias_g = None

        # Iterate over batch
        for n_idx in range(N):
            x_n = x[n_idx, :, :, :]  # (C_in, H_in, W_in)

            l_start = 0
            while l_start < L_total:
                L_tile_eff = min(max_L_tile, L_total - l_start)

                # 1) Build im2col tile for this batch (all channels)
                B_tile_full = im2col_nchw_tile(
                    x_n,
                    kernel_size=(K_h, K_w),
                    stride=(stride_h, stride_w),
                    padding=(pad_h, pad_w),
                    H_out=H_out,
                    W_out=W_out,
                    l_start=l_start,
                    tile_L=L_tile_eff,
                )  # (C_in*K_h*K_w, L_tile_eff)

                # 2) Slice channels corresponding to this group
                group_offset = g * C_in_g * K_h * K_w
                B_tile = B_tile_full[group_offset:group_offset + C_in_g * K_h * K_w, :]

                # 3) GEMM via systolic core: Y_tile = W_g @ B_tile
                Y_tile = systolic_gemm_explicit(
                    A=W_g,
                    B=B_tile,
                    P=P,
                    Q=Q,
                    tile_k=tile_k,
                )  # (C_out_g, L_tile_eff)

                # 4) Add bias
                if bias_g is not None:
                    Y_tile = Y_tile + bias_g.view(-1, 1)

                # 5) Write back into y
                for t in range(L_tile_eff):
                    l = l_start + t
                    oh, ow = linear_to_2d(l, W_out)
                    y[n_idx, g * C_out_g:(g + 1) * C_out_g, oh, ow] = Y_tile[:, t]

                l_start += L_tile_eff

    return y
