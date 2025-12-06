#!/usr/bin/env python3
"""
dump_tinyvit_params.py

example usage in powershell:
  # selected modules
  python sw/tools/dump_tinyvit_params.py `
    --modules patch_embed.conv1.conv stages.0.blocks.0.local_conv

  # all Conv2d + Linear modules
  python sw/tools/dump_tinyvit_params.py `
    --all

Dump selected TinyViT-5M parameters to:
    data/models/tinyvit_5m_224_dist_in22k_ft_in1k_params/

For each module path (e.g. 'patch_embed.conv1.conv'), it will write:
    <module_path_with_underscores>_weight.bin
    <module_path_with_underscores>_bias.bin   (if bias exists)

Binary formats:

Conv2d weight:
    int32 out_ch, int32 in_ch, int32 kH, int32 kW
    float32 data[out_ch * in_ch * kH * kW]

Bias (Conv2d, Linear):
    int32 out_ch
    float32 data[out_ch]

Linear weight:
    int32 out_features, int32 in_features
    float32 data[out_features * in_features]
"""

import os
from pathlib import Path
from functools import reduce
import argparse

import numpy as np
import torch
import timm


def get_module_by_path(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    parts = module_path.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


def save_conv2d_params(base_dir: Path, module_path: str, conv: torch.nn.Conv2d):
    w = conv.weight.detach().cpu().contiguous()  # [out_ch, in_ch, kH, kW]
    assert w.ndim == 4
    out_ch, in_ch, kH, kW = w.shape

    flat_name = module_path.replace(".", "_")
    weight_path = base_dir / f"{flat_name}_weight.bin"

    header = np.array([out_ch, in_ch, kH, kW], dtype=np.int32)
    data = w.numpy().astype(np.float32).ravel()

    os.makedirs(base_dir, exist_ok=True)
    with open(weight_path, "wb") as f:
        header.tofile(f)
        data.tofile(f)
    print(f"[INFO] Saved Conv2d weight {w.shape} -> {weight_path}")

    if conv.bias is not None:
        b = conv.bias.detach().cpu().contiguous()  # [out_ch]
        bias_path = base_dir / f"{flat_name}_bias.bin"
        header_b = np.array([out_ch], dtype=np.int32)
        data_b = b.numpy().astype(np.float32).ravel()
        with open(bias_path, "wb") as f:
            header_b.tofile(f)
            data_b.tofile(f)
        print(f"[INFO] Saved Conv2d bias {b.shape} -> {bias_path}")
    else:
        print(f"[INFO] Conv2d '{module_path}' has no bias; skipping bias file.")


def save_linear_params(base_dir: Path, module_path: str, lin: torch.nn.Linear):
    w = lin.weight.detach().cpu().contiguous()  # [out_features, in_features]
    assert w.ndim == 2
    out_f, in_f = w.shape

    flat_name = module_path.replace(".", "_")
    weight_path = base_dir / f"{flat_name}_weight.bin"

    header = np.array([out_f, in_f], dtype=np.int32)
    data = w.numpy().astype(np.float32).ravel()

    os.makedirs(base_dir, exist_ok=True)
    with open(weight_path, "wb") as f:
        header.tofile(f)
        data.tofile(f)
    print(f"[INFO] Saved Linear weight {w.shape} -> {weight_path}")

    if lin.bias is not None:
        b = lin.bias.detach().cpu().contiguous()  # [out_features]
        bias_path = base_dir / f"{flat_name}_bias.bin"
        header_b = np.array([out_f], dtype=np.int32)
        data_b = b.numpy().astype(np.float32).ravel()
        with open(bias_path, "wb") as f:
            header_b.tofile(f)
            data_b.tofile(f)
        print(f"[INFO] Saved Linear bias {b.shape} -> {bias_path}")
    else:
        print(f"[INFO] Linear '{module_path}' has no bias; skipping bias file.")


def load_tinyvit(model_name: str, models_dir: Path) -> torch.nn.Module:
    models_dir.mkdir(parents=True, exist_ok=True)
    models_dir = models_dir.resolve()

    print("[INFO] Using models directory:", models_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    os.environ["XDG_CACHE_HOME"] = str(models_dir)

    safe_name = model_name.replace(".", "_")
    state_path = models_dir / f"{safe_name}_state_dict.pth"

    if state_path.exists():
        print(f"[INFO] Found saved state_dict: {state_path}")
        model = timm.create_model(model_name, pretrained=False)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        print("[INFO] Downloading pretrained TinyViT weights via timm...")
        model = timm.create_model(model_name, pretrained=True)
        try:
            torch.save(model.state_dict(), state_path)
            print(f"[INFO] Saved state_dict to {state_path}")
        except Exception as e:
            print(f"[WARN] Could not save state_dict: {e}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Dump TinyViT-5M parameters for selected modules.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="tiny_vit_5m_224.dist_in22k_ft_in1k",
        help="timm model name",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory to cache state_dict / HF downloads",
    )
    parser.add_argument(
        "--params-dir",
        type=str,
        default="data/models/tinyvit_5m_224_dist_in22k_ft_in1k_params",
        help="Directory to save parameter .bin files",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=["patch_embed.conv1.conv"],
        help=(
            "List of dotted module paths to dump, "
            "e.g. 'patch_embed.conv1.conv', 'stages.0.blocks.0.local_conv', "
            "or a single value 'all' together with --all."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Dump parameters for all Conv2d and Linear modules.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    params_dir = Path(args.params_dir)

    model = load_tinyvit(args.model_name, models_dir)

    # Decide if we're dumping all modules
    dump_all = args.all or (len(args.modules) == 1 and args.modules[0].lower() == "all")

    if dump_all:
        print("[INFO] --all / 'all' selected: dumping all Conv2d + Linear modules.")
        for name, mod in model.named_modules():
            if not name:
                continue  # skip root
            if isinstance(mod, torch.nn.Conv2d):
                print(f"[INFO] Conv2d module: {name}")
                save_conv2d_params(params_dir, name, mod)
            elif isinstance(mod, torch.nn.Linear):
                print(f"[INFO] Linear module: {name}")
                save_linear_params(params_dir, name, mod)
        print("[DONE] Dumped all Conv2d + Linear module parameters.")
        return

    # Otherwise, use explicit module list
    for module_path in args.modules:
        print(f"[INFO] Processing module: {module_path}")
        mod = get_module_by_path(model, module_path)

        if isinstance(mod, torch.nn.Conv2d):
            save_conv2d_params(params_dir, module_path, mod)
        elif isinstance(mod, torch.nn.Linear):
            save_linear_params(params_dir, module_path, mod)
        else:
            print(f"[WARN] Module '{module_path}' is type {type(mod)}, "
                  f"not Conv2d or Linear. Skipping.")


if __name__ == "__main__":
    main()
