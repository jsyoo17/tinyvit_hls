#!/usr/bin/env python3
"""
Debug script for TinyViT SW vs timm around stages.0.blocks.0.conv3.bn.

Run from repo root:
    python host/tinyvit/debug_mbconv_bn.py
"""

from pathlib import Path
import os

import torch
import torch.nn as nn
import timm

from tiny_vit_sw import TinyVitSWSmall  # adjust if path is different


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


def print_key_stats(tag: str, load_result):
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys
    print(f"[{tag}] load_state_dict(strict=False)")
    print(f"  missing_keys   : {len(missing)}")
    if missing:
        print("    (first few)", missing[:10])
    print(f"  unexpected_keys: {len(unexpected)}")
    if unexpected:
        print("    (first few)", unexpected[:10])


def tensor_diff(a: torch.Tensor, b: torch.Tensor, name: str):
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

    # 2) Build timm + SW models and load the SAME state_dict into both
    print("[DBG] Building timm reference model...")
    timm_model = timm.create_model(MODEL_NAME, pretrained=False)
    res_ref = timm_model.load_state_dict(sd, strict=False)
    print_key_stats("timm", res_ref)

    print("[DBG] Building SW model...")
    sw_model = TinyVitSWSmall()
    res_sw = sw_model.load_state_dict(sd, strict=False)
    print_key_stats("sw  ", res_sw)

    timm_model.to(device)
    sw_model.to(device)
    timm_model.eval()
    sw_model.eval()
    torch.set_grad_enabled(False)

    # 3) Look up the problematic modules
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
        print("[ERR] One of the BN modules was not found. Check module names.")
        print("      Available names containing 'stages.0.blocks.0.conv3' in timm:")
        for n in name_to_mod_ref:
            if "stages.0.blocks.0.conv3" in n:
                print("        timm:", n)
        print("      Available names containing 'stages.0.blocks.0.conv3' in sw:")
        for n in name_to_mod_sw:
            if "stages.0.blocks.0.conv3" in n:
                print("        sw  :", n)
        return

    bn_ref: nn.BatchNorm2d = name_to_mod_ref[target_bn]
    bn_sw: nn.BatchNorm2d = name_to_mod_sw[target_bn]

    print("\n[DBG] BatchNorm2d config comparison:")
    print(f"  bn_ref.training         = {bn_ref.training}")
    print(f"  bn_sw.training          = {bn_sw.training}")
    print(f"  bn_ref.track_running    = {bn_ref.track_running_stats}")
    print(f"  bn_sw.track_running     = {bn_sw.track_running_stats}")
    print(f"  bn_ref.eps              = {bn_ref.eps}")
    print(f"  bn_sw.eps               = {bn_sw.eps}")
    print(f"  bn_ref.momentum         = {bn_ref.momentum}")
    print(f"  bn_sw.momentum          = {bn_sw.momentum}")

    print("\n[DBG] BN parameter diffs (timm vs sw):")
    tensor_diff(bn_ref.weight.data,        bn_sw.weight.data,        "weight")
    tensor_diff(bn_ref.bias.data,          bn_sw.bias.data,          "bias")
    tensor_diff(bn_ref.running_mean.data,  bn_sw.running_mean.data,  "running_mean")
    tensor_diff(bn_ref.running_var.data,   bn_sw.running_var.data,   "running_var")

    # 4) Forward pass hooks to inspect conv3 + bn IO
    captured_ref = {}
    captured_sw = {}

    def make_conv_hook(store_dict, tag):
        def hook(m, inp, out):
            store_dict[tag] = out.detach().cpu()
        return hook

    def make_bn_hook(store_dict, tag_in, tag_out):
        def hook(m, inp, out):
            # inp is a tuple; inp[0] is the tensor
            store_dict[tag_in] = inp[0].detach().cpu()
            store_dict[tag_out] = out.detach().cpu()
        return hook

    # Register hooks
    h_ref = []
    h_sw = []

    conv_ref = name_to_mod_ref[target_conv]
    conv_sw = name_to_mod_sw[target_conv]
    bn_ref_mod = bn_ref
    bn_sw_mod = bn_sw

    h_ref.append(conv_ref.register_forward_hook(make_conv_hook(captured_ref, "conv3_out")))
    h_ref.append(bn_ref_mod.register_forward_hook(make_bn_hook(captured_ref, "bn_in", "bn_out")))

    h_sw.append(conv_sw.register_forward_hook(make_conv_hook(captured_sw, "conv3_out")))
    h_sw.append(bn_sw_mod.register_forward_hook(make_bn_hook(captured_sw, "bn_in", "bn_out")))

    # 5) Run a forward pass with a fixed random input (same seed for reproducibility)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224, device=device)

    print("\n[DBG] Running forward pass to capture conv3/bn activations...")
    _ = timm_model(x)
    _ = sw_model(x)

    # Remove hooks
    for h in h_ref + h_sw:
        h.remove()

    print("\n[DBG] Activation diffs at stages.0.blocks.0.conv3 / bn")
    for key in ["conv3_out", "bn_in", "bn_out"]:
        if key not in captured_ref or key not in captured_sw:
            print(f"  [WARN] Missing captured tensor for key='{key}'")
            continue
        tensor_diff(captured_ref[key], captured_sw[key], key)

    print("\n[DBG] Done. Use the above diffs to see if the problem is:")
    print("  - state_dict not loading BN stats into SW (non-zero running_mean diff), or")
    print("  - something else in the forward path around conv3/bn.\n")


if __name__ == "__main__":
    main()
