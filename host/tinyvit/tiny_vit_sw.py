# host/tinyvit/tiny_vit_sw.py

#!/usr/bin/env python3
"""
tiny_vit_sw.py

Software-accurate TinyViT-5M (dist_in22k_ft_in1k) implementation,
structured to match timm's TinyVit module names so that we can
load the pretrained state_dict and compare block-by-block.

This is the "_sw" version:
- Clear, readable PyTorch
- Same math as timm TinyViT
- No training / features / registry glue
"""

from __future__ import annotations

from typing import Tuple, List, Optional, Type

import itertools
import os
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import LayerNorm2d, NormMlpClassifierHead, use_fused_attn, calculate_drop_path_rates


# -----------------------------------------------------------------------------
# Basic Conv + BN block (ConvNorm) – matches timm TinyViT ConvNorm
# -----------------------------------------------------------------------------

class ConvNormSW(nn.Sequential):
    """Conv2d + BatchNorm2d as in TinyViT ConvNorm.

    Names:
      .conv
      .bn
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, ks, stride, pad, dilation, groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chs)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0.0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """Fuse Conv+BN into a single Conv2d (not used yet in SW tests)."""
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5

        fused = nn.Conv2d(
            in_channels=w.size(1) * c.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2:],
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            bias=True,
        )
        fused.weight.data.copy_(w)
        fused.bias.data.copy_(b)
        return fused


# -----------------------------------------------------------------------------
# Patch Embedding – same names as timm: patch_embed.conv1, .act, .conv2
# -----------------------------------------------------------------------------

class PatchEmbedSW(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.stride = 4
        self.conv1 = ConvNormSW(in_chs, out_chs // 2, ks=3, stride=2, pad=1)
        self.act = act_layer()
        self.conv2 = ConvNormSW(out_chs // 2, out_chs, ks=3, stride=2, pad=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        # -> [B, out_chs, 56, 56]
        return x


# -----------------------------------------------------------------------------
# MBConv + ConvLayer (stage 0)
# -----------------------------------------------------------------------------

class MBConvSW(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        expand_ratio: float,
        act_layer: Type[nn.Module],
        drop_path: float = 0.0,  # kept for shape parity; drop_path_rate=0 for 5M
    ):
        super().__init__()
        mid_chs = int(in_chs * expand_ratio)
        self.conv1 = ConvNormSW(in_chs, mid_chs, ks=1)
        self.act1 = act_layer()
        self.conv2 = ConvNormSW(mid_chs, mid_chs, ks=3, stride=1, pad=1, groups=mid_chs)
        self.act2 = act_layer()
        self.conv3 = ConvNormSW(mid_chs, out_chs, ks=1, bn_weight_init=0.0)
        self.act3 = act_layer()
        # In TinyViT-5M, drop_path_rate = 0, so this is effectively Identity.
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ConvLayerSW(nn.Module):
    """Stage 0 in TinyViT-5M: stack of MBConv blocks, no downsampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        act_layer: Type[nn.Module] = nn.GELU,
        drop_path: float | List[float] = 0.0,
        conv_expand_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        blocks = []
        for i in range(depth):
            dp_i = drop_path[i] if isinstance(drop_path, list) else drop_path
            blocks.append(
                MBConvSW(
                    in_chs=dim,
                    out_chs=dim,
                    expand_ratio=conv_expand_ratio,
                    act_layer=act_layer,
                    drop_path=dp_i,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim, H, W]
        x = self.blocks(x)
        return x


# -----------------------------------------------------------------------------
# Norm + MLP (token MLP in TinyVitBlock)
# -----------------------------------------------------------------------------

class NormMlpSW(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -----------------------------------------------------------------------------
# Attention with relative position bias (windowed)
# -----------------------------------------------------------------------------

class AttentionSW(nn.Module):
    """TinyViT attention with relative position biases (software version).

    Names match timm:
      .norm
      .qkv
      .proj
      .attention_biases
      .attention_bias_idxs (buffer)
    """

    fused_attn: torch.jit.Final[bool]
    attention_bias_cache: dict

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: int = 1,
        resolution: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn_ratio = attn_ratio
        self.val_dim = int(attn_ratio * key_dim)
        self.out_dim = self.val_dim * num_heads
        self.scale = key_dim ** -0.5
        self.resolution = resolution
        self.fused_attn = use_fused_attn()

        self.norm = nn.LayerNorm(dim)
        # output dim: num_heads * (val_dim + 2*key_dim)
        self.qkv = nn.Linear(dim, num_heads * (self.val_dim + 2 * key_dim))
        self.proj = nn.Linear(self.out_dim, dim)

        # Relative position bias (same construction as original TinyViT)
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs: List[int] = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs",
            torch.tensor(idxs, dtype=torch.long).view(N, N),
            persistent=False,
        )
        self.attention_bias_cache = {}

    @torch.no_grad()
    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[
                    :, self.attention_bias_idxs
                ]
            return self.attention_bias_cache[device_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        attn_bias = self.get_attention_biases(x.device)  # [H, N, N]
        B, N, _ = x.shape

        x = self.norm(x)
        qkv = self.qkv(x)  # [B, N, H*(val_dim + 2*key_dim)]

        # [B, N, H, val+2*key] -> split into q,k,v
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.val_dim],
            dim=3,
        )
        # -> [B, H, N, d]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)   # [B, H, N, N]
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            x = attn @ v                     # [B, H, N, val_dim]

        x = x.transpose(1, 2).reshape(B, N, self.out_dim)  # [B, N, H*val_dim]
        x = self.proj(x)                                   # [B, N, dim]
        return x


# -----------------------------------------------------------------------------
# TinyViTBlock – window attention + local conv + MLP
# -----------------------------------------------------------------------------

class TinyVitBlockSW(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert dim % num_heads == 0
        head_dim = dim // num_heads
        window_resolution = (window_size, window_size)

        self.attn = AttentionSW(
            dim=dim,
            key_dim=head_dim,
            num_heads=num_heads,
            attn_ratio=1,
            resolution=window_resolution,
        )
        self.drop_path1 = nn.Identity()  # drop_path_rate set to 0 for 5M

        self.mlp = NormMlpSW(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = nn.Identity()

        pad = local_conv_size // 2
        self.local_conv = ConvNormSW(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        L = H * W

        shortcut = x

        if H == self.window_size and W == self.window_size:
            # Single window
            x = x.reshape(B, L, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)
        else:
            # pad to multiples of window_size and do windowed attention
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # (C, W, H) order in pad args

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size

            # BCHW-style window partition in BHWC
            x = x.view(
                B, nH, self.window_size, nW, self.window_size, C
            ).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )

            x = self.attn(x)

            x = x.view(
                B, nH, nW, self.window_size, self.window_size, C
            ).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

        x = shortcut + self.drop_path1(x)

        # local depthwise conv
        x = x.permute(0, 3, 1, 2)             # BHWC -> BCHW
        x = self.local_conv(x)                # [B, C, H, W]
        x = x.reshape(B, C, L).transpose(1, 2)  # [B, L, C]

        x = x + self.drop_path2(self.mlp(x))
        x = x.view(B, H, W, C)
        return x


# -----------------------------------------------------------------------------
# PatchMerging (downsample between stages) and stage wrapper
# -----------------------------------------------------------------------------

class PatchMergingSW(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.conv1 = ConvNormSW(dim, out_dim, ks=1, stride=1, pad=0)
        self.act1 = act_layer()
        self.conv2 = ConvNormSW(out_dim, out_dim, ks=3, stride=2, pad=1, groups=out_dim)
        self.act2 = act_layer()
        self.conv3 = ConvNormSW(out_dim, out_dim, ks=1, stride=1, pad=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class TinyVitStageSW(nn.Module):
    """TinyViTStage for stages 1–3 (downsample + attention blocks)."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float | List[float] = 0.0,
        local_conv_size: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.depth = depth
        self.out_dim = out_dim

        self.downsample = PatchMergingSW(dim=dim, out_dim=out_dim, act_layer=act_layer)

        blocks = []
        for i in range(depth):
            dp_i = drop_path[i] if isinstance(drop_path, list) else drop_path
            blocks.append(
                TinyVitBlockSW(
                    dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dp_i,
                    local_conv_size=local_conv_size,
                    act_layer=act_layer,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return x


# -----------------------------------------------------------------------------
# Full TinyViT-5M SW model (backbone + classifier head)
# -----------------------------------------------------------------------------

class TinyVitSWSmall(nn.Module):
    """TinyViT-5M (tiny_vit_5m_224.dist_in22k_ft_in1k) software model."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        embed_dims: Tuple[int, ...] = (64, 128, 160, 320),
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (2, 4, 5, 10),
        window_sizes: Tuple[int, ...] = (7, 7, 14, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,  # 0.0 for 5M
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        self.mlp_ratio = mlp_ratio

        # patch embedding
        self.patch_embed = PatchEmbedSW(
            in_chs=in_chans,
            out_chs=embed_dims[0],
            act_layer=act_layer,
        )

        # stochastic depth schedule (all zeros for 5M, but keep code for parity)
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths))

        # build stages[0..3]
        self.stages = nn.Sequential()
        stride = self.patch_embed.stride
        prev_dim = embed_dims[0]
        self.feature_info = []  # not used, but kept for parity

        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                stage = ConvLayerSW(
                    dim=prev_dim,
                    depth=depths[stage_idx],
                    act_layer=act_layer,
                    drop_path=dpr[: depths[stage_idx]],
                    conv_expand_ratio=mbconv_expand_ratio,
                )
            else:
                out_dim = embed_dims[stage_idx]
                dp_slice = dpr[
                    sum(depths[:stage_idx]) : sum(depths[: stage_idx + 1])
                ]
                stage = TinyVitStageSW(
                    dim=embed_dims[stage_idx - 1],
                    out_dim=out_dim,
                    depth=depths[stage_idx],
                    num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    drop_path=dp_slice,
                    local_conv_size=local_conv_size,
                    act_layer=act_layer,
                )
                prev_dim = out_dim
                stride *= 2

            self.stages.append(stage)
            self.feature_info.append(
                dict(num_chs=prev_dim, reduction=stride, module=f"stages.{stage_idx}")
            )

        # classifier head – use the same NormMlpClassifierHead + LayerNorm2d(eps=1e-5)
        self.num_features = embed_dims[-1]
        norm_layer_cf = partial(LayerNorm2d, eps=1e-5)
        self.head = NormMlpClassifierHead(
            in_features=self.num_features,
            num_classes=num_classes,
            pool_type=global_pool,
            norm_layer=norm_layer_cf,
        )

        # weight init for linear layers (same idea as original TinyVit)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------
    # For SW tests
    # ------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)    # [B, 64, 56, 56]
        x = self.stages(x)         # [B, 320, 7, 7]
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# -----------------------------------------------------------------------------
# Helper: create SW model and load timm state_dict
# -----------------------------------------------------------------------------

def create_tiny_vit_5m_sw(
    models_dir: Path | str = "data/models",
    state_dict_fname: str = "tiny_vit_5m_224_dist_in22k_ft_in1k_state_dict.pth",
    device: torch.device | str = "cpu",
) -> TinyVitSWSmall:
    """Create TinyVitSWSmall and load pretrained timm state_dict (strict=False)."""
    from timm import create_model as timm_create_model

    device = torch.device(device)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir.resolve())
    os.environ["HF_HUB_CACHE"] = str(models_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(models_dir.resolve())

    safe_name = "tiny_vit_5m_224.dist_in22k_ft_in1k"
    state_path = models_dir / state_dict_fname

    if state_path.exists():
        print(f"[SW] Using models directory: {models_dir}")
        print(f"[SW] Found saved TinyViT state_dict at {state_path}")
        # create timm model and ensure state_dict matches
        timm_model = timm_create_model(
            "tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=False
        )
        sd = torch.load(state_path, map_location="cpu")
        m = timm_model.load_state_dict(sd, strict=False)
        print(
            f"[SW] timm TinyViT load_state_dict(strict=False): "
            f"missing={len(m.missing_keys)}, unexpected={len(m.unexpected_keys)}"
        )
    else:
        print("[SW] Downloading TinyViT pretrained weights via timm/HF...")
        timm_model = timm_create_model(
            "tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True
        )
        sd = timm_model.state_dict()
        torch.save(sd, state_path)
        print(f"[SW] Saved state_dict to {state_path}")

    # create SW model and load same state_dict (strict=False, we allow any harmless extras)
    sw_model = TinyVitSWSmall().to(device)
    m2 = sw_model.load_state_dict(sd, strict=False)
    print("[SW] load_state_dict(strict=False)")
    print(f"[SW]   Model params not loaded (missing_keys): {len(m2.missing_keys)}")
    print(f"[SW]   Extra state_dict entries not used (unexpected_keys): {len(m2.unexpected_keys)}")

    return sw_model


if __name__ == "__main__":
    # quick smoke test on random input
    dev = torch.device("cpu")
    models_dir = Path("data/models")
    model_sw = create_tiny_vit_5m_sw(models_dir=models_dir, device=dev)
    x = torch.randn(1, 3, 224, 224, device=dev)
    with torch.no_grad():
        feats = model_sw.forward_features(x)
        logits = model_sw(x)
    print("[SW] forward_features shape:", tuple(feats.shape))
    print("[SW] logits shape          :", tuple(logits.shape))
