# host/tests/test_hw_ops_conv2d.py
"""
Tests for hardware-style Conv2D ops in host.tinyvit.hw_ops.

Run from repo root:
    (.venv) python -m host.tests.test_hw_ops_conv2d

Or directly:
    (.venv) python host/tests/test_hw_ops_conv2d.py
"""

import os
import sys

# ---------------------------------------------------------------------
# Ensure repo root is on sys.path when running as a script
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from host.tinyvit.hw_ops import (
    HWConv2d,
    hw_conv2d_nchw,
    conv2d_nchw_streaming_im2col,
)


def compare_conv_vs_hwconv(
    in_channels=8,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
    groups=1,
    bias=True,
    H=32,
    W=32,
):
    torch.manual_seed(0)

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=1,
    )

    hw_conv = HWConv2d.from_conv(conv)

    x = torch.randn(2, in_channels, H, W)

    y_ref = conv(x)
    y_hw  = hw_conv(x)

    max_diff = (y_ref - y_hw).abs().max().item()
    mean_diff = (y_ref - y_hw).abs().mean().item()

    print(f"[HWConv2d] in={in_channels}, out={out_channels}, k={kernel_size}, "
          f"stride={stride}, pad={padding}, groups={groups}, bias={bias}")
    print(f"  max_diff  = {max_diff:.3e}")
    print(f"  mean_diff = {mean_diff:.3e}")
    print("-" * 60)


def compare_conv_vs_streaming(
    in_channels=8,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
    groups=1,
    bias=True,
    H=16,
    W=16,
    max_L_tile=16,
    P=4,
    Q=4,
    tile_k=8,
):
    torch.manual_seed(1)

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=1,
    )

    x = torch.randn(1, in_channels, H, W)

    y_ref = conv(x)
    y_hw  = conv2d_nchw_streaming_im2col(
        x,
        conv.weight,
        conv.bias,
        stride=stride,
        padding=padding,
        groups=groups,
        max_L_tile=max_L_tile,
        P=P,
        Q=Q,
        tile_k=tile_k,
    )

    max_diff = (y_ref - y_hw).abs().max().item()
    mean_diff = (y_ref - y_hw).abs().mean().item()

    print(f"[StreamingConv] in={in_channels}, out={out_channels}, k={kernel_size}, "
          f"stride={stride}, pad={padding}, groups={groups}, "
          f"max_L_tile={max_L_tile}, P={P}, Q={Q}, tile_k={tile_k}")
    print(f"  max_diff  = {max_diff:.3e}")
    print(f"  mean_diff = {mean_diff:.3e}")
    print("-" * 60)


def main():
    print("=== Test 1: HWConv2d vs nn.Conv2d (full im2col) ===")
    compare_conv_vs_hwconv()
    compare_conv_vs_hwconv(in_channels=16, out_channels=16, groups=16)  # depthwise
    compare_conv_vs_hwconv(in_channels=8, out_channels=4, kernel_size=1, padding=0)

    print("\n=== Test 2: Streaming Conv vs nn.Conv2d ===")
    compare_conv_vs_streaming()
    compare_conv_vs_streaming(
        in_channels=16, out_channels=16, groups=16,
        kernel_size=3, stride=1, padding=1,
        H=16, W=16,
        max_L_tile=8,
        P=4, Q=4, tile_k=8,
    )
    compare_conv_vs_streaming(
        in_channels=8, out_channels=8,
        kernel_size=1, stride=2, padding=0,
        H=15, W=15,
        max_L_tile=10,
        P=2, Q=5, tile_k=4,
    )


if __name__ == "__main__":
    main()
