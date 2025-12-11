#!/usr/bin/env python3
"""
Deeper BN debug for TinyViT stage0 MBConv:

- Same state_dict loaded into timm + SW.
- Capture conv3 output (BN input).
- Run:
    - timm BN forward
    - SW BN forward
    - manual F.batch_norm with ref BN params
- Compare all three.

Run from repo root:
    python host/tinyvit/debug_mbconv_bn_math.py
"""

from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from tiny_vit_sw import TinyVitSWSmall  # adjust import path if needed


MODEL_NAME = "tiny_vit_5m_224.dist_in22k_ft_in1k"
SAFE_NAME = MODEL_NAME.replace(".", "_")
DEFAULT_MODELS_DIR = Path("data/models")


def load_shared_state_dict(models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    models_dir = models_dir.resolve()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    os.environ["XDG_CACHE_HOME"] = str(models_dir)

    state_path = models_dir / f"{SAFE_NAME}_state_dict.pth"

    if not state_path.exists():
        print(f"[DBG] state_dict not found at {state_path}, downloading via timm...")
        ref = timm.create_model(MODEL_NAME, pretrained=True)
        torch.save(ref.state_dict(), state_path)
        print(f"[DBG] Saved state_dict to {state_path}")

    print(f"[DBG] Loading shared state_dict from {state_path}")
    sd = torch.load(state_path, map_location="cpu")
    return sd, state_path


def tensor_diff(a: torch.Tensor, b: torch.Tensor, name: str):
    a = a.detach().cpu()
    b = b.detach().cpu()
    diff = (a - b).abs()
    print(
        f"  {name:15s}: "
        f"max={diff.max().item():.6e} "
        f"mean={diff.mean().item():.6e} "
        f"shape={tuple(a.shape)}"
    )


def main():
    device = torch.device("cpu")
    models_dir = DEFAULT_MODELS_DIR

    # 1) Load shared state_dict
    sd, state_path = load_shared_state_dict(models_dir)

    # 2) Build timm + SW models with same state_dict
    print("[DBG] Building timm reference model...")
    timm_model = timm.create_model(MODEL_NAME, pretrained=False)
    res_ref = timm_model.load_state_dict(sd, strict=False)
    print("[timm] missing_keys   :", len(res_ref.missing_keys))
    print("[timm] unexpected_keys:", len(res_ref.unexpected_keys))

    print("[DBG] Building SW model...")
    sw_model = TinyVitSWSmall()
    res_sw = sw_model.load_state_dict(sd, strict=False)
    print("[sw  ] missing_keys   :", len(res_sw.missing_keys))
    print("[sw  ] unexpected_keys:", len(res_sw.unexpected_keys))

    timm_model.to(device).eval()
    sw_model.to(device).eval()
    torch.set_grad_enabled(False)

    # 3) Grab BN modules
    name_to_mod_ref = dict(timm_model.named_modules())
    name_to_mod_sw = dict(sw_model.named_modules())

    target_conv3 = "stages.0.blocks.0.conv3"
    target_bn = target_conv3 + ".bn"
    target_conv = target_conv3 + ".conv"

    print("\n[DBG] Checking that target modules exist:")
    for name in [target_conv3, target_conv, target_bn]:
        print(f"  {name} in timm_model? {name in name_to_mod_ref}")
        print(f"  {name} in sw_model?   {name in name_to_mod_sw}")

    if target_bn not in name_to_mod_ref or target_bn not in name_to_mod_sw:
        print("[ERR] BN module not found under expected name. Aborting.")
        return

    bn_ref: nn.BatchNorm2d = name_to_mod_ref[target_bn]
    bn_sw: nn.BatchNorm2d = name_to_mod_sw[target_bn]

    print("\n[DBG] BN types:")
    print(f"  type(bn_ref) = {type(bn_ref)}")
    print(f"  type(bn_sw)  = {type(bn_sw)}")

    print("\n[DBG] BN state_dict key summary:")
    sd_ref_bn = bn_ref.state_dict()
    sd_sw_bn = bn_sw.state_dict()
    print(f"  bn_ref.state_dict keys: {list(sd_ref_bn.keys())}")
    print(f"  bn_sw.state_dict  keys: {list(sd_sw_bn.keys())}")

    print("\n[DBG] num_batches_tracked + dtypes:")
    nbt_ref = sd_ref_bn.get("num_batches_tracked", None)
    nbt_sw = sd_sw_bn.get("num_batches_tracked", None)
    print(f"  bn_ref.num_batches_tracked: {nbt_ref}")
    print(f"  bn_sw.num_batches_tracked : {nbt_sw}")
    print(f"  bn_ref.weight.dtype       : {bn_ref.weight.dtype}")
    print(f"  bn_sw.weight.dtype        : {bn_sw.weight.dtype}")

    print("\n[DBG] BN parameter diffs (timm vs sw):")
    tensor_diff(bn_ref.weight.data,       bn_sw.weight.data,       "weight")
    tensor_diff(bn_ref.bias.data,         bn_sw.bias.data,         "bias")
    tensor_diff(bn_ref.running_mean.data, bn_sw.running_mean.data, "running_mean")
    tensor_diff(bn_ref.running_var.data,  bn_sw.running_var.data,  "running_var")

    # 4) Capture conv3_out (BN input) and BN outputs
    captured_ref = {}
    captured_sw = {}

    def make_conv_hook(store_dict, tag):
        def hook(m, inp, out):
            store_dict[tag] = out.detach().cpu()
        return hook

    def make_bn_hook(store_dict, tag_out):
        def hook(m, inp, out):
            # we explicitly capture the BN output here
            store_dict[tag_out] = out.detach().cpu()
        return hook

    conv_ref = name_to_mod_ref[target_conv]
    conv_sw = name_to_mod_sw[target_conv]

    h_ref = []
    h_sw = []

    h_ref.append(conv_ref.register_forward_hook(make_conv_hook(captured_ref, "conv3_out")))
    h_ref.append(bn_ref.register_forward_hook(make_bn_hook(captured_ref, "bn_out")))

    h_sw.append(conv_sw.register_forward_hook(make_conv_hook(captured_sw, "conv3_out")))
    h_sw.append(bn_sw.register_forward_hook(make_bn_hook(captured_sw, "bn_out")))

    # 5) Forward pass with fixed input
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224, device=device)

    print("\n[DBG] Running forward pass...")
    _ = timm_model(x)
    _ = sw_model(x)

    for h in h_ref + h_sw:
        h.remove()

    conv_in_ref = captured_ref["conv3_out"]
    conv_in_sw  = captured_sw["conv3_out"]

    print("\n[DBG] conv3_out (BN input) diffs:")
    tensor_diff(conv_in_ref, conv_in_sw, "conv3_out")

    bn_out_ref = captured_ref["bn_out"]
    bn_out_sw  = captured_sw["bn_out"]

    print("\n[DBG] BN forward output diffs (module outputs):")
    tensor_diff(bn_out_ref, bn_out_sw, "bn_out")

    # 6) Manual BN using ref BN parameters
    print("\n[DBG] Manual F.batch_norm using bn_ref params...")
    # Ensure contiguous + correct dtype
    xin = conv_in_ref.to(device=device, dtype=bn_ref.weight.dtype)

    manual = F.batch_norm(
        xin,
        running_mean=bn_ref.running_mean,
        running_var=bn_ref.running_var,
        weight=bn_ref.weight,
        bias=bn_ref.bias,
        training=False,      # eval-mode behavior: use running stats
        momentum=bn_ref.momentum,
        eps=bn_ref.eps,
    )

    manual = manual.detach().cpu()
    bn_out_ref_cpu = bn_out_ref.detach().cpu()
    bn_out_sw_cpu  = bn_out_sw.detach().cpu()

    print("\n[DBG] Compare manual vs timm BN, manual vs SW BN:")
    tensor_diff(manual, bn_out_ref_cpu, "manual_vs_timm_bn")
    tensor_diff(manual, bn_out_sw_cpu,  "manual_vs_sw_bn")

    print("\n[DBG] Done. Key things to look at:")
    print("  - If manual == timm and != SW: SW BN is doing something different.")
    print("  - If manual == SW and != timm: timm path is special-cased somehow.")
    print("  - If manual != both: then something about shapes/dtypes/usage is off.\n")


if __name__ == "__main__":
    main()
