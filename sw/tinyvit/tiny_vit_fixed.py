# forked from TinyViT:
# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

def float2fixed(value, num_bits=32, num_int=16):
    """
    Mimic float32 -> fixed-point conversion by rounding and clamping.
    NOTE: Input/output are still float tensors; this just emulates quantization.
    """
    MAX_VAL = 2**(num_int - 1) - 1 / (2**(num_bits - num_int))
    MIN_VAL = -2**(num_int - 1)

    scale = 2 ** (num_bits - num_int - 1)
    fixed_val = torch.clamp((value * scale).round() / scale, MIN_VAL, MAX_VAL)
    return fixed_val



class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Conv2dFixed(nn.Conv2d):
    """Conv2d layer with fixed-point quantization emulation."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        num_bits=32,
        num_int=16,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.num_bits = num_bits
        self.num_int = num_int

    def forward(self, x):
        # Quantize input
        x_q = float2fixed(x, num_bits=self.num_bits, num_int=self.num_int)

        # Quantize weights and bias
        w_q = float2fixed(self.weight, num_bits=self.num_bits, num_int=self.num_int)
        if self.bias is not None:
            b_q = float2fixed(self.bias, num_bits=self.num_bits, num_int=self.num_int)
        else:
            b_q = None

        # Do convolution in (emulated) fixed-point
        y = F.conv2d(
            x_q,
            w_q,
            b_q,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Quantize output
        y_q = float2fixed(y, num_bits=self.num_bits, num_int=self.num_int)
        return y_q

class BatchNorm2dFixed(nn.BatchNorm2d):
    """BatchNorm2d layer with fixed-point quantization emulation."""
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        num_bits=32,
        num_int=16,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self.num_bits = num_bits
        self.num_int = num_int

    def forward(self, x):
        # We mainly care about inference behavior (eval mode) for FPGA
        if self.training:
            # For now, fall back to standard BN in training
            x_q = float2fixed(x, self.num_bits, self.num_int)
            y = super().forward(x_q)
            return float2fixed(y, self.num_bits, self.num_int)

        # ---- Inference-style BN with quantized everything ----
        x_q = float2fixed(x, self.num_bits, self.num_int)

        if self.track_running_stats:
            running_mean_q = float2fixed(self.running_mean, self.num_bits, self.num_int)
            running_var_q = float2fixed(self.running_var, self.num_bits, self.num_int)
        else:
            running_mean_q = x_q.mean(dim=(0, 2, 3))
            running_var_q = x_q.var(dim=(0, 2, 3), unbiased=False)

        if self.affine:
            weight_q = float2fixed(self.weight, self.num_bits, self.num_int)
            bias_q = float2fixed(self.bias, self.num_bits, self.num_int)
        else:
            weight_q = torch.ones_like(running_mean_q)
            bias_q = torch.zeros_like(running_mean_q)

        # reshape for broadcasting: [C] -> [1, C, 1, 1]
        mean = running_mean_q.view(1, -1, 1, 1)
        var = running_var_q.view(1, -1, 1, 1)
        gamma = weight_q.view(1, -1, 1, 1)
        beta = bias_q.view(1, -1, 1, 1)

        x_norm = (x_q - mean) / torch.sqrt(var + self.eps)
        y = gamma * x_norm + beta

        y_q = float2fixed(y, self.num_bits, self.num_int)
        return y_q

class GELUFixed(nn.Module):
    """GELU activation with fixed-point quantization emulation."""
    def __init__(self, num_bits=32, num_int=16):
        super().__init__()
        self.act = nn.GELU()
        self.num_bits = num_bits
        self.num_int = num_int

    def forward(self, x):
        # Quantize input
        x_q = float2fixed(x, num_bits=self.num_bits, num_int=self.num_int)
        # Apply standard GELU in float
        y = self.act(x_q)
        # Quantize output
        y_q = float2fixed(y, num_bits=self.num_bits, num_int=self.num_int)
        return y_q

class Conv2dBNFixed(nn.Module):
    """
    Conv + BatchNorm block with fixed-point quantization emulation.

    Typically built from an existing (conv, bn) pair, e.g. a timm ConvNorm:
        src = model.patch_embed.conv1   # ConvNorm with .conv and .bn
        fixed = Conv2dBNFixed(src.conv, src.bn, num_bits=32, num_int=16)
    """
    def __init__(self, conv_src: nn.Conv2d, bn_src: nn.BatchNorm2d,
                 num_bits: int = 32, num_int: int = 16):
        super().__init__()

        # ---- Conv (fixed) ----
        self.conv = Conv2dFixed(
            in_channels=conv_src.in_channels,
            out_channels=conv_src.out_channels,
            kernel_size=conv_src.kernel_size,
            stride=conv_src.stride,
            padding=conv_src.padding,
            dilation=conv_src.dilation,
            groups=conv_src.groups,
            bias=(conv_src.bias is not None),
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            self.conv.weight.copy_(conv_src.weight)
            if conv_src.bias is not None and self.conv.bias is not None:
                self.conv.bias.copy_(conv_src.bias)

        # ---- BN (fixed) ----
        self.bn = BatchNorm2dFixed(
            num_features=bn_src.num_features,
            eps=bn_src.eps,
            momentum=bn_src.momentum,
            affine=bn_src.affine,
            track_running_stats=bn_src.track_running_stats,
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            if bn_src.affine:
                self.bn.weight.copy_(bn_src.weight)
                self.bn.bias.copy_(bn_src.bias)
            if bn_src.track_running_stats:
                self.bn.running_mean.copy_(bn_src.running_mean)
                self.bn.running_var.copy_(bn_src.running_var)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class PatchEmbedFixed(nn.Module):
    """
    Fixed-point emulation of TinyViT patch embedding.

    Expects a source patch_embed module that has:
        - conv1: ConvNorm-like (with .conv and .bn)
        - act:   activation (e.g., GELU)
        - conv2: ConvNorm-like (with .conv and .bn)

    Example usage:
        timm_model = timm.create_model("tiny_vit_5m_224", pretrained=True)
        pe_src = timm_model.patch_embed
        pe_fixed = PatchEmbedFixed(pe_src, img_size=224, num_bits=32, num_int=16)
    """
    def __init__(
        self,
        patch_embed_src: nn.Module,
        img_size: int | tuple[int, int] | None = None,
        num_bits: int = 32,
        num_int: int = 16,
    ):
        super().__init__()

        # Extract original modules
        convbn1_src = patch_embed_src.conv1  # e.g., ConvNorm
        act_src     = patch_embed_src.act
        convbn2_src = patch_embed_src.conv2

        # Build fixed-point equivalents
        self.conv1 = Conv2dBNFixed(convbn1_src.conv, convbn1_src.bn,
                                   num_bits=num_bits, num_int=num_int)
        self.act   = GELUFixed(num_bits=num_bits, num_int=num_int)
        self.conv2 = Conv2dBNFixed(convbn2_src.conv, convbn2_src.bn,
                                   num_bits=num_bits, num_int=num_int)

        # Stride (TinyViT patch embed downsamples by 4)
        self.stride = getattr(patch_embed_src, "stride", 4)

        # Optional: keep TinyViT-style attributes if img_size is provided
        if img_size is not None:
            if isinstance(img_size, int):
                H = W = img_size
            else:
                H, W = img_size
            self.patches_resolution = (H // self.stride, W // self.stride)
            self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x  # (B, C_out, H/4, W/4) for TinyViT 5M with img_size=224

class MBConvFixed(nn.Module):
    """
    Fixed-point emulation of TinyViT MBConv block.

    Built directly from an existing MBConv:
        - Conv2d_BN -> Conv2dBNFixed (conv+bn quantized)
        - GELU      -> GELUFixed     (activation quantized)
        - DropPath  -> reused as-is

    Example:
        mb_src = tinyvit.layers[0].blocks[0]   # original MBConv
        mb_fx  = MBConvFixed(mb_src, num_bits=32, num_int=16)
    """
    def __init__(
        self,
        mbconv_src: nn.Module,
        num_bits: int = 32,
        num_int: int = 16,
        use_fixed_act: bool = True,
    ):
        super().__init__()

        # ---- Conv1: Conv2d_BN -> Conv2dBNFixed ----
        # mbconv_src.conv1 is Conv2d_BN with .c and .bn
        self.conv1 = Conv2dBNFixed(
            mbconv_src.conv1.conv,
            mbconv_src.conv1.bn,
            num_bits=num_bits,
            num_int=num_int,
        )

        # ---- Act1 ----
        self.act1 = GELUFixed(num_bits=num_bits, num_int=num_int) if use_fixed_act else mbconv_src.act1

        # ---- Conv2: depthwise Conv2d_BN -> Conv2dBNFixed ----
        self.conv2 = Conv2dBNFixed(
            mbconv_src.conv2.conv,
            mbconv_src.conv2.bn,
            num_bits=num_bits,
            num_int=num_int,
        )
        self.act2 = GELUFixed(num_bits=num_bits, num_int=num_int) if use_fixed_act else mbconv_src.act2

        # ---- Conv3: Conv2d_BN -> Conv2dBNFixed ----
        # in original: bn_weight_init=0.0 at init, but we copy trained weights
        self.conv3 = Conv2dBNFixed(
            mbconv_src.conv3.conv,
            mbconv_src.conv3.bn,
            num_bits=num_bits,
            num_int=num_int,
        )

        # ---- Final act ----
        self.act3 = GELUFixed(num_bits=num_bits, num_int=num_int) if use_fixed_act else mbconv_src.act3

        # ---- DropPath: keep exactly the same module ----
        # for tiny_vit_5m_224 this is Identity (drop_path_rate=0.0)
        self.drop_path = mbconv_src.drop_path

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x = x + shortcut
        x = self.act3(x)

        return x

class MBConvFixedConvOnly(nn.Module):
    """
    MBConv variant where ONLY the Conv weights are run through Conv2dFixed.
    BN and GELU stay as original float modules.

    This isolates the effect of Conv quantization.
    """
    def __init__(self, mbconv_src: nn.Module, num_bits: int = 32, num_int: int = 16):
        super().__init__()

        # ---- Conv1: Conv2d -> Conv2dFixed, BN + act unchanged ----
        self.conv1 = Conv2dFixed(
            in_channels=mbconv_src.conv1.conv.in_channels,
            out_channels=mbconv_src.conv1.conv.out_channels,
            kernel_size=mbconv_src.conv1.conv.kernel_size,
            stride=mbconv_src.conv1.conv.stride,
            padding=mbconv_src.conv1.conv.padding,
            dilation=mbconv_src.conv1.conv.dilation,
            groups=mbconv_src.conv1.conv.groups,
            bias=(mbconv_src.conv1.conv.bias is not None),
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            self.conv1.weight.copy_(mbconv_src.conv1.conv.weight)
            if self.conv1.bias is not None and mbconv_src.conv1.conv.bias is not None:
                self.conv1.bias.copy_(mbconv_src.conv1.conv.bias)

        self.bn1 = mbconv_src.conv1.bn           # float BN as-is
        self.act1 = mbconv_src.act1              # float GELU as-is

        # ---- Conv2 (depthwise) ----
        self.conv2 = Conv2dFixed(
            in_channels=mbconv_src.conv2.conv.in_channels,
            out_channels=mbconv_src.conv2.conv.out_channels,
            kernel_size=mbconv_src.conv2.conv.kernel_size,
            stride=mbconv_src.conv2.conv.stride,
            padding=mbconv_src.conv2.conv.padding,
            dilation=mbconv_src.conv2.conv.dilation,
            groups=mbconv_src.conv2.conv.groups,
            bias=(mbconv_src.conv2.conv.bias is not None),
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            self.conv2.weight.copy_(mbconv_src.conv2.conv.weight)
            if self.conv2.bias is not None and mbconv_src.conv2.conv.bias is not None:
                self.conv2.bias.copy_(mbconv_src.conv2.conv.bias)

        self.bn2 = mbconv_src.conv2.bn
        self.act2 = mbconv_src.act2

        # ---- Conv3 ----
        self.conv3 = Conv2dFixed(
            in_channels=mbconv_src.conv3.conv.in_channels,
            out_channels=mbconv_src.conv3.conv.out_channels,
            kernel_size=mbconv_src.conv3.conv.kernel_size,
            stride=mbconv_src.conv3.conv.stride,
            padding=mbconv_src.conv3.conv.padding,
            dilation=mbconv_src.conv3.conv.dilation,
            groups=mbconv_src.conv3.conv.groups,
            bias=(mbconv_src.conv3.conv.bias is not None),
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            self.conv3.weight.copy_(mbconv_src.conv3.conv.weight)
            if self.conv3.bias is not None and mbconv_src.conv3.conv.bias is not None:
                self.conv3.bias.copy_(mbconv_src.conv3.conv.bias)

        self.bn3 = mbconv_src.conv3.bn
        self.act3 = mbconv_src.act3

        self.drop_path = mbconv_src.drop_path  # Identity for 5M

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.drop_path(x)
        x = x + shortcut
        x = self.act3(x)
        return x


class MBConvFixedBNOnly(nn.Module):
    """
    MBConv variant where ONLY BatchNorm is replaced by BatchNorm2dFixed.
    Conv and GELU stay in float.

    This isolates the effect of BN quantization.
    """
    def __init__(self, mbconv_src: nn.Module, num_bits: int = 32, num_int: int = 16):
        super().__init__()

        # Use the same conv modules (shared weights OK for inference tests)
        self.conv1 = mbconv_src.conv1.conv
        self.bn1 = BatchNorm2dFixed(
            num_features=mbconv_src.conv1.bn.num_features,
            eps=mbconv_src.conv1.bn.eps,
            momentum=mbconv_src.conv1.bn.momentum,
            affine=mbconv_src.conv1.bn.affine,
            track_running_stats=mbconv_src.conv1.bn.track_running_stats,
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            if self.bn1.affine:
                self.bn1.weight.copy_(mbconv_src.conv1.bn.weight)
                self.bn1.bias.copy_(mbconv_src.conv1.bn.bias)
            if self.bn1.track_running_stats:
                self.bn1.running_mean.copy_(mbconv_src.conv1.bn.running_mean)
                self.bn1.running_var.copy_(mbconv_src.conv1.bn.running_var)

        self.act1 = mbconv_src.act1  # float GELU

        # Conv2 (depthwise)
        self.conv2 = mbconv_src.conv2.conv
        self.bn2 = BatchNorm2dFixed(
            num_features=mbconv_src.conv2.bn.num_features,
            eps=mbconv_src.conv2.bn.eps,
            momentum=mbconv_src.conv2.bn.momentum,
            affine=mbconv_src.conv2.bn.affine,
            track_running_stats=mbconv_src.conv2.bn.track_running_stats,
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            if self.bn2.affine:
                self.bn2.weight.copy_(mbconv_src.conv2.bn.weight)
                self.bn2.bias.copy_(mbconv_src.conv2.bn.bias)
            if self.bn2.track_running_stats:
                self.bn2.running_mean.copy_(mbconv_src.conv2.bn.running_mean)
                self.bn2.running_var.copy_(mbconv_src.conv2.bn.running_var)

        self.act2 = mbconv_src.act2

        # Conv3
        self.conv3 = mbconv_src.conv3.conv
        self.bn3 = BatchNorm2dFixed(
            num_features=mbconv_src.conv3.bn.num_features,
            eps=mbconv_src.conv3.bn.eps,
            momentum=mbconv_src.conv3.bn.momentum,
            affine=mbconv_src.conv3.bn.affine,
            track_running_stats=mbconv_src.conv3.bn.track_running_stats,
            num_bits=num_bits,
            num_int=num_int,
        )
        with torch.no_grad():
            if self.bn3.affine:
                self.bn3.weight.copy_(mbconv_src.conv3.bn.weight)
                self.bn3.bias.copy_(mbconv_src.conv3.bn.bias)
            if self.bn3.track_running_stats:
                self.bn3.running_mean.copy_(mbconv_src.conv3.bn.running_mean)
                self.bn3.running_var.copy_(mbconv_src.conv3.bn.running_var)

        self.act3 = mbconv_src.act3  # float GELU

        self.drop_path = mbconv_src.drop_path

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.drop_path(x)
        x = x + shortcut
        x = self.act3(x)
        return x


class MBConvFixedActOnly(nn.Module):
    """
    MBConv variant where ONLY the GELU activations are quantized (GELUFixed).
    Conv and BN stay in float.

    This isolates the effect of activation quantization.
    """
    def __init__(self, mbconv_src: nn.Module, num_bits: int = 32, num_int: int = 16):
        super().__init__()

        # conv+bn as original
        self.conv1 = mbconv_src.conv1.conv
        self.bn1   = mbconv_src.conv1.bn
        self.act1  = GELUFixed(num_bits=num_bits, num_int=num_int)

        self.conv2 = mbconv_src.conv2.conv
        self.bn2   = mbconv_src.conv2.bn
        self.act2  = GELUFixed(num_bits=num_bits, num_int=num_int)

        self.conv3 = mbconv_src.conv3.conv
        self.bn3   = mbconv_src.conv3.bn
        self.act3  = GELUFixed(num_bits=num_bits, num_int=num_int)

        self.drop_path = mbconv_src.drop_path

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.drop_path(x)
        x = x + shortcut
        x = self.act3(x)
        return x

class PatchMergingFixed(nn.Module):
    """
    Fixed-point emulation of TinyViT PatchMerging used as stage downsample.

    This matches the timm TinyViT PatchMerging behavior in your model:
    input:  (B, C, H, W)
    output: (B, C_out, H', W')
    """
    def __init__(
        self,
        pm_src: nn.Module,
        num_bits: int = 32,
        num_int: int = 16,
        use_fixed_act: bool = True,
    ):
        super().__init__()

        # ---- Conv1 (ConvNorm) -> Conv2dBNFixed ----
        self.conv1 = Conv2dBNFixed(
            pm_src.conv1.conv,
            pm_src.conv1.bn,
            num_bits=num_bits,
            num_int=num_int,
        )
        self.act1 = GELUFixed(num_bits=num_bits, num_int=num_int) if use_fixed_act else pm_src.act1

        # ---- Conv2 (depthwise ConvNorm) -> Conv2dBNFixed ----
        self.conv2 = Conv2dBNFixed(
            pm_src.conv2.conv,
            pm_src.conv2.bn,
            num_bits=num_bits,
            num_int=num_int,
        )
        self.act2 = GELUFixed(num_bits=num_bits, num_int=num_int) if use_fixed_act else pm_src.act2

        # ---- Conv3 (ConvNorm) -> Conv2dBNFixed ----
        self.conv3 = Conv2dBNFixed(
            pm_src.conv3.conv,
            pm_src.conv3.bn,
            num_bits=num_bits,
            num_int=num_int,
        )

        # (optional metadata)
        self.input_resolution = getattr(pm_src, "input_resolution", None)
        self.dim = getattr(pm_src, "dim", None)
        self.out_dim = getattr(pm_src, "out_dim", None)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        # IMPORTANT: match timm PatchMerging output: keep 4D
        # (B, C_out, H', W')
        return x

class TinyVitBlockFixed(nn.Module):
    """
    Fixed-point emulation wrapper for timm.models.tiny_vit.TinyVitBlock.

    Assumes input is NHWC: (B, H, W, C), which matches the original TinyVitBlock.
    We keep all inner modules (LayerNorm, Attention, MLP) as-is, but we:
      - quantize input to fixed point
      - quantize outputs of norm1, attn, residual adds, norm2, mlp

    This is a first step: signal-level fixed-point emulation without rewriting Attention/MLP.
    """

    def __init__(
        self,
        blk_src: nn.Module,
        num_bits: int = 32,
        num_int: int = 16,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.num_int = num_int

        # Copy over submodules by reference (weights are shared with blk_src)
        self.norm1 = blk_src.norm1
        self.attn = blk_src.attn
        self.drop_path = blk_src.drop_path
        self.norm2 = blk_src.norm2
        self.mlp = blk_src.mlp

        # Optional metadata (nice to have)
        for attr in ["input_resolution", "dim", "num_heads", "window_size"]:
            if hasattr(blk_src, attr):
                setattr(self, attr, getattr(blk_src, attr))

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        """Helper: apply float2fixed with this block's bit config."""
        return float2fixed(x, num_bits=self.num_bits, num_int=self.num_int)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C) in NHWC layout.
        Returns same shape/layout.
        """
        # Quantize input
        x_q = self._q(x)
        shortcut = x_q

        # ---- Attention branch ----
        y = self.norm1(x_q)   # LayerNorm in float
        y = self._q(y)

        y = self.attn(y)      # Attention in float, expects NHWC-compatible last dim = C
        y = self._q(y)

        y = self.drop_path(y)
        x_q = shortcut + y
        x_q = self._q(x_q)

        # ---- MLP branch ----
        z = self.norm2(x_q)
        z = self._q(z)

        z = self.mlp(z)       # MLP in float (Linear + GELU + Linear)
        z = self._q(z)

        x_q = x_q + self.drop_path(z)
        x_q = self._q(x_q)

        return x_q
