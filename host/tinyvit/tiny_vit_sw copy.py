#!/usr/bin/env python3
"""
tiny_vit_sw.py

Simplified TinyViT-5M software reference, currently implementing:

  - ConvNormSW (Conv2d + BatchNorm2d, unfused)
  - PatchEmbedSW (first stem)
  - MBConvSW + ConvLayerSW (stage 0)
  - PatchMergingSW + TinyVitBlockSW + TinyVitStageSW (stage 1, attention)
  - TinyVit5MSW (patch_embed + stage0 + stage1)

Target model: tiny_vit_5m_224.dist_in22k_ft_in1k (timm)

We keep module / parameter names compatible with timm so that the official
state_dict can be loaded directly and per-module outputs can be compared.
"""

from pathlib import Path
from typing import Type, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import use_fused_attn


__all__ = [
    "ConvNormSW",
    "PatchEmbedSW",
    "MBConvSW",
    "ConvLayerSW",
    "NormMlpSW",
    "AttentionSW",
    "TinyVitBlockSW",
    "PatchMergingSW",
    "TinyVitStageSW",
    "TinyVit5MSW",
    "create_tiny_vit_5m_sw",
]


# -------------------------------------------------------------------------
# Basic blocks (software reference, unfused Conv+BN)
# -------------------------------------------------------------------------

class ConvNormSW(nn.Module):
    """
    Software version of Conv + BatchNorm used in TinyViT.

    Structure:
      self.conv : nn.Conv2d
      self.bn   : nn.BatchNorm2d

    Names match timm:
      patch_embed.conv1.conv.weight
      patch_embed.conv1.bn.weight
      ...
      stages.0.blocks.0.conv1.conv.weight
      stages.0.blocks.0.conv1.bn.weight
      ...
      stages.1.downsample.conv1.conv.weight
      ...
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
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class PatchEmbedSW(nn.Module):
    """
    Software version of TinyViT patch embedding for 224x224 inputs.

    For TinyViT-5M:
      embed_dims[0] = 64
      conv1: 3  -> 32   (ks=3, stride=2, pad=1)
      conv2: 32 -> 64   (ks=3, stride=2, pad=1)

    Layout:
      self.conv1 : ConvNormSW
      self.act   : GELU
      self.conv2 : ConvNormSW
    """

    def __init__(
        self,
        in_chs: int = 3,
        embed_dim: int = 64,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.stride = 4
        mid_chs = embed_dim // 2

        self.conv1 = ConvNormSW(in_chs, mid_chs, ks=3, stride=2, pad=1)
        self.act = act_layer()
        self.conv2 = ConvNormSW(mid_chs, embed_dim, ks=3, stride=2, pad=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        x = self.conv1(x)      # [B, mid_chs, 112, 112]
        x = self.act(x)
        x = self.conv2(x)      # [B, embed_dim, 56, 56]
        return x


class MBConvSW(nn.Module):
    """
    Software version of TinyViT MBConv block (used in stage 0 ConvLayer).

    Mirrors the timm TinyViT MBConv structure:

      conv1: 1x1 ConvNorm (in_chs -> mid_chs)
      act1
      conv2: 3x3 depthwise ConvNorm (mid_chs -> mid_chs, groups=mid_chs)
      act2
      conv3: 1x1 ConvNorm (mid_chs -> out_chs)
      residual add
      act3

    DropPath exists in the original, but for TinyViT-5M the drop_path_rate is 0,
    so we omit it here to keep the SW model minimal.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        expand_ratio: float,
        act_layer: Type[nn.Module],
    ):
        super().__init__()
        mid_chs = int(in_chs * expand_ratio)

        # 1x1 pointwise
        self.conv1 = ConvNormSW(in_chs, mid_chs, ks=1, stride=1, pad=0)
        self.act1 = act_layer()

        # 3x3 depthwise
        self.conv2 = ConvNormSW(
            mid_chs,
            mid_chs,
            ks=3,
            stride=1,
            pad=1,
            groups=mid_chs,
        )
        self.act2 = act_layer()

        # 1x1 pointwise (out)
        self.conv3 = ConvNormSW(mid_chs, out_chs, ks=1, stride=1, pad=0)
        self.act3 = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = x + shortcut
        x = self.act3(x)

        return x


class ConvLayerSW(nn.Module):
    """
    Software version of TinyViT ConvLayer (stage 0).

    For TinyViT-5M:
      dim = 64
      depth = 2
      conv_expand_ratio = 4.0

    Names (to match timm):
      stages.0.blocks.0
      stages.0.blocks.1
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        act_layer: Type[nn.Module],
        conv_expand_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        blocks = []
        for i in range(depth):
            blocks.append(
                MBConvSW(
                    in_chs=dim,
                    out_chs=dim,
                    expand_ratio=conv_expand_ratio,
                    act_layer=act_layer,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# -------------------------------------------------------------------------
# Stage 1 components: NormMlp, Attention, TinyVitBlock, PatchMerging, TinyVitStage
# -------------------------------------------------------------------------

class NormMlpSW(nn.Module):
    """
    Software version of TinyViT MLP with pre-norm.

    Params / names match timm's NormMlp:

      .norm
      .fc1
      .act
      .drop1
      .fc2
      .drop2
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttentionSW(nn.Module):
    """
    Software version of TinyViT attention with relative position biases.

    Names / params match timm Attention:

      .norm
      .qkv
      .proj
      .attention_biases
      .attention_bias_idxs (buffer)
    """

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
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.out_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio
        self.resolution = resolution
        self.fused_attn = use_fused_attn()

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, num_heads * (self.val_dim + 2 * key_dim))
        self.proj = nn.Linear(self.out_dim, dim)

        # Relative position bias indices
        points = [(i, j) for i in range(resolution[0]) for j in range(resolution[1])]
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
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
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        attn_bias = self.get_attention_biases(x.device)  # [num_heads, N, N]
        B, N, _ = x.shape

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.val_dim],
            dim=3,
        )
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)     # [B, num_heads, N, N]
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)
        return x


class TinyVitBlockSW(nn.Module):
    """
    Software version of TinyViT block (attention + local conv + MLP).

    Structure / names match timm TinyVitBlock:

      .attn  (Attention)
      .mlp   (NormMlp)
      .local_conv (ConvNormSW)
      .drop_path1 (Identity here)
      .drop_path2 (Identity here)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
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
        # DropPath is 0 for TinyViT-5M, so Identity
        self.drop_path1 = nn.Identity()

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = NormMlpSW(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=0.0,
        )
        self.drop_path2 = nn.Identity()

        pad = local_conv_size // 2
        self.local_conv = ConvNormSW(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        L = H * W

        shortcut = x

        if H == self.window_size and W == self.window_size:
            x = x.reshape(B, L, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)
        else:
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size

            # window partition
            x = x.view(
                B, nH, self.window_size,
                nW, self.window_size, C
            ).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )

            x = self.attn(x)

            # window reverse
            x = x.view(
                B, nH, nW, self.window_size, self.window_size, C
            ).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

        x = shortcut + self.drop_path1(x)

        # Local conv branch
        x = x.permute(0, 3, 1, 2)             # BHWC -> BCHW
        x = self.local_conv(x)
        x = x.reshape(B, C, L).transpose(1, 2) # BCHW -> BLC

        x = x + self.drop_path2(self.mlp(x))   # BLC
        return x.view(B, H, W, C)


class PatchMergingSW(nn.Module):
    """
    Software version of TinyViT PatchMerging (downsample) layer used at the
    beginning of stages 1, 2, 3.

    Names / structure match timm PatchMerging:

      .conv1, .act1, .conv2, .act2, .conv3
    """

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
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class TinyVitStageSW(nn.Module):
    """
    Software version of TinyViT stage (for stages 1, 2, 3).

    For TinyViT-5M stage 1:
      dim       = 64
      out_dim   = 128
      depth     = 2
      num_heads = 4
      window_size = 7
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        local_conv_size: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.depth = depth
        self.out_dim = out_dim

        # Downsample (PatchMerging)
        self.downsample = PatchMergingSW(
            dim=dim,
            out_dim=out_dim,
            act_layer=act_layer,
        )

        # Attention blocks (TinyVitBlock)
        blocks = []
        for i in range(depth):
            blocks.append(
                TinyVitBlockSW(
                    dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    local_conv_size=local_conv_size,
                    act_layer=act_layer,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        x = self.downsample(x)   # -> [B, out_dim, H/2, W/2]
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return x


# -------------------------------------------------------------------------
# Top-level TinyVit5MSW wrapper (patch_embed + ConvLayer stage0 + TinyVitStage stage1)
# -------------------------------------------------------------------------

class TinyVit5MSW(nn.Module):
    """
    Simplified TinyViT-5M software reference model.

    Currently implemented:
      - patch_embed: PatchEmbedSW
      - stages[0]  : ConvLayerSW (MBConv stack)
      - stages[1]  : TinyVitStageSW (PatchMerging + TinyVitBlockSW stack)

    forward_features(x) returns the output after stage1:

      x0 = patch_embed(x)
      x1 = stages[0](x0)
      x2 = stages[1](x1)
      return x2
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims=(64, 128, 160, 320),
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 5, 10),
        window_sizes=(7, 7, 14, 7),
        act_layer: Type[nn.Module] = nn.GELU,
        device: str = "cpu",
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.window_sizes = window_sizes

        # Patch embed
        self.patch_embed = PatchEmbedSW(
            in_chs=in_chans,
            embed_dim=embed_dims[0],
            act_layer=act_layer,
        )

        # Stage 0 = ConvLayer (MBConv)
        stage0 = ConvLayerSW(
            dim=embed_dims[0],
            depth=depths[0],
            act_layer=act_layer,
            conv_expand_ratio=4.0,
        )

        # Stage 1 = TinyVitStage (PatchMerging + TinyVitBlock)
        stage1 = TinyVitStageSW(
            dim=embed_dims[0],
            out_dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_sizes[1],
            mlp_ratio=4.0,
            local_conv_size=3,
            act_layer=act_layer,
        )

        # Stages as Sequential to match "stages.0", "stages.1" naming
        self.stages = nn.Sequential(
            stage0,
            stage1,
        )

        self.to(device)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        x = self.patch_embed(x)   # [B, 64, 56, 56]
        x = self.stages[0](x)     # [B, 64, 56, 56]
        x = self.stages[1](x)     # [B, 128, 28, 28]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


# -------------------------------------------------------------------------
# Helper: load timm TinyViT state_dict and create SW model
# -------------------------------------------------------------------------

def _load_tinyvit_state_dict(
    model_name: str,
    models_dir: Path,
) -> dict:
    """
    Load (or download + cache) timm TinyViT state_dict.

    Uses same HF cache environment variables as your other tools.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    models_dir = models_dir.resolve()

    print("[SW] Using models directory:", models_dir)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    os.environ["XDG_CACHE_HOME"] = str(models_dir)

    safe_name = model_name.replace(".", "_")
    state_path = models_dir / f"{safe_name}_state_dict.pth"

    if state_path.exists():
        print(f"[SW] Found saved TinyViT state_dict at {state_path}")
        state = torch.load(state_path, map_location="cpu")
    else:
        print("[SW] Downloading TinyViT pretrained weights via timm/HF...")
        model_full = timm.create_model(model_name, pretrained=True)
        state = model_full.state_dict()
        try:
            torch.save(state, state_path)
            print(f"[SW] Saved state_dict to {state_path}")
        except Exception as e:
            print(f"[SW][WARN] Failed to save state_dict: {e}")

    return state


def create_tiny_vit_5m_sw(
    model_name: str = "tiny_vit_5m_224.dist_in22k_ft_in1k",
    models_dir: str = "data/models",
    device: str = "cpu",
) -> TinyVit5MSW:
    """
    Factory function to build TinyVit5MSW and load timm TinyViT-5M weights
    (for the parts that exist in this SW model: patch_embed + stage0 + stage1).

    This uses strict=False so extra keys from later stages / head are ignored.
    """
    models_dir_path = Path(models_dir)
    state_dict = _load_tinyvit_state_dict(model_name, models_dir_path)

    model_sw = TinyVit5MSW(
        in_chans=3,
        embed_dims=(64, 128, 160, 320),
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 5, 10),
        window_sizes=(7, 7, 14, 7),
        act_layer=nn.GELU,
        device=device,
    )

    incompatible = model_sw.load_state_dict(state_dict, strict=False)
    missing = incompatible.missing_keys
    unexpected = incompatible.unexpected_keys

    print("[SW] load_state_dict(strict=False)")
    print(f"[SW]   Model params not loaded (missing_keys): {len(missing)}")
    print(f"[SW]   Extra state_dict entries not used (unexpected_keys): {len(unexpected)}")

    return model_sw


# -------------------------------------------------------------------------
# Quick smoke test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cpu"
    model = create_tiny_vit_5m_sw(device=device)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        y0 = model.patch_embed(x)
        y1 = model.stages[0](y0)
        y2 = model.stages[1](y1)
        y_feat = model.forward_features(x)

    print("[SW] patch_embed output shape:", tuple(y0.shape))   # (1, 64, 56, 56)
    print("[SW] stage0 output shape     :", tuple(y1.shape))   # (1, 64, 56, 56)
    print("[SW] stage1 output shape     :", tuple(y2.shape))   # (1, 128, 28, 28)
    print("[SW] features output shape   :", tuple(y_feat.shape))
