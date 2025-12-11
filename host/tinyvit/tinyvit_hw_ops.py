# sw/tinyvit_hw_ops.py
from __future__ import annotations
from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F

def _to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def hw_conv2d_nchw(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
) -> torch.Tensor:
    """
    Pure tensor implementation of Conv2D (NCHW) with NO dilation.
    Exactly matches PyTorch Conv2d when dilation=1.
    """
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_per_g, K_h, K_w = weight.shape

    stride_h, stride_w = _to_2tuple(stride)
    pad_h, pad_w       = _to_2tuple(padding)

    assert C_in % groups == 0
    assert C_out % groups == 0

    C_in_g  = C_in // groups
    C_out_g = C_out // groups

    outputs = []

    for g in range(groups):
        # (N, C_in_g, H_in, W_in)
        x_g = x[:, g*C_in_g:(g+1)*C_in_g, :, :]

        # im2col (NO dilation now)
        x_unf = F.unfold(
            x_g,
            kernel_size=(K_h, K_w),
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w),
        )  # (N, C_in_g*K_h*K_w, L)

        # Select weights of group
        w_g = weight[g*C_out_g:(g+1)*C_out_g, :, :, :]
        w_g = w_g.view(C_out_g, -1)  # (C_out_g, C_in_g*K_h*K_w)

        # Convolution = matmul (N, C_out_g, L)
        out_g = torch.matmul(w_g, x_unf)

        outputs.append(out_g)

    # concatenate groups
    out = torch.cat(outputs, dim=1)  # (N, C_out, L)

    # Compute h_out / w_out (no dilation)
    H_out = (H_in + 2*pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2*pad_w - K_w) // stride_w + 1

    out = out.view(N, C_out, H_out, W_out)

    # Add bias
    if bias is not None:
        out = out + bias.view(1, C_out, 1, 1)

    return out
