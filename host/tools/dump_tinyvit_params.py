#!/usr/bin/env python3
"""
dump_tinyvit_params.py

example usage in powershell:
  # selected modules
  python host/tools/dump_tinyvit_params.py `
    --modules patch_embed.conv1.conv stages.0.blocks.0.local_conv

  # all Conv2d + Linear + BatchNorm2d + LayerNorm modules
  python host/tools/dump_tinyvit_params.py --all

Dump selected TinyViT-5M parameters to:
    data/models/tinyvit_5m_224_dist_in22k_ft_in1k_params/

For each module path (e.g. 'patch_embed.conv1.conv'), it will write:

  Conv2d:
    <module_path_with_underscores>_weight.bin
    <module_path_with_underscores>_bias.bin           (if bias exists)

  Linear:
    <module_path_with_underscores>_weight.bin
    <module_path_with_underscores>_bias.bin           (if bias exists)

  BatchNorm2d:
    <module_path_with_underscores>_weight.bin         (gamma)
    <module_path_with_underscores>_bias.bin           (beta)
    <module_path_with_underscores>_running_mean.bin
    <module_path_with_underscores>_running_var.bin

  LayerNorm:
    <module_path_with_underscores>_weight.bin         (gamma)
    <module_path_with_underscores>_bias.bin           (beta)

and a JSON manifest:
    tinyvit_params_manifest.json

Binary formats:

Conv2d weight:
    int32 out_ch, int32 in_ch, int32 kH, int32 kW
    float32 data[out_ch * in_ch * kH * kW]

Linear weight:
    int32 out_features, int32 in_features
    float32 data[out_features * in_features]

1D params (bias, BN weight, running stats, LN weight/bias, etc.):
    int32 length
    float32 data[length]
"""

import os
import json
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import timm


def get_module_by_path(model: nn.Module, module_path: str) -> nn.Module:
    parts = module_path.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


# -----------------------------
# Low-level save helpers
# -----------------------------

def save_1d_param(base_dir: Path, flat_name: str, suffix: str, tensor: torch.Tensor):
    """
    Save a 1D parameter tensor (or flattenable) as:
        int32 length
        float32 data[length]

    Returns (filename, shape_list).
    """
    t = tensor.detach().cpu().contiguous()
    arr = t.view(-1).numpy().astype(np.float32)
    length = arr.size

    fname = f"{flat_name}_{suffix}.bin"
    fpath = base_dir / fname

    os.makedirs(base_dir, exist_ok=True)
    header = np.array([length], dtype=np.int32)

    with open(fpath, "wb") as f:
        header.tofile(f)
        arr.tofile(f)

    print(f"[INFO] Saved 1D param '{suffix}' shape {list(t.shape)} -> {fpath}")
    return fname, list(t.shape)


def save_conv2d_params(base_dir: Path, module_path: str, conv: nn.Conv2d):
    w = conv.weight.detach().cpu().contiguous()  # [out_ch, in_ch, kH, kW]
    assert w.ndim == 4
    out_ch, in_ch, kH, kW = w.shape

    flat_name = module_path.replace(".", "_")
    weight_fname = f"{flat_name}_weight.bin"
    weight_path = base_dir / weight_fname

    header = np.array([out_ch, in_ch, kH, kW], dtype=np.int32)
    data = w.numpy().astype(np.float32).ravel()

    os.makedirs(base_dir, exist_ok=True)
    with open(weight_path, "wb") as f:
        header.tofile(f)
        data.tofile(f)
    print(f"[INFO] Saved Conv2d weight {w.shape} -> {weight_path}")

    bias_fname = None
    bias_shape = None
    if conv.bias is not None:
        b = conv.bias.detach().cpu().contiguous()  # [out_ch]
        bias_fname, bias_shape = save_1d_param(base_dir, flat_name, "bias", b)
    else:
        print(f"[INFO] Conv2d '{module_path}' has no bias; skipping bias file.")

    return {
        "type": "Conv2d",
        "weight_file": weight_fname,
        "bias_file": bias_fname,
        "weight_shape": [out_ch, in_ch, kH, kW],
        "bias_shape": bias_shape,
    }


def save_linear_params(base_dir: Path, module_path: str, lin: nn.Linear):
    w = lin.weight.detach().cpu().contiguous()  # [out_features, in_features]
    assert w.ndim == 2
    out_f, in_f = w.shape

    flat_name = module_path.replace(".", "_")
    weight_fname = f"{flat_name}_weight.bin"
    weight_path = base_dir / weight_fname

    header = np.array([out_f, in_f], dtype=np.int32)
    data = w.numpy().astype(np.float32).ravel()

    os.makedirs(base_dir, exist_ok=True)
    with open(weight_path, "wb") as f:
        header.tofile(f)
        data.tofile(f)
    print(f"[INFO] Saved Linear weight {w.shape} -> {weight_path}")

    bias_fname = None
    bias_shape = None
    if lin.bias is not None:
        b = lin.bias.detach().cpu().contiguous()  # [out_features]
        bias_fname, bias_shape = save_1d_param(base_dir, flat_name, "bias", b)
    else:
        print(f"[INFO] Linear '{module_path}' has no bias; skipping bias file.")

    return {
        "type": "Linear",
        "weight_file": weight_fname,
        "bias_file": bias_fname,
        "weight_shape": [out_f, in_f],
        "bias_shape": bias_shape,
    }


def save_batchnorm2d_params(base_dir: Path, module_path: str, bn: nn.BatchNorm2d):
    """
    Save BatchNorm2d params used at inference:
        weight (gamma), bias (beta), running_mean, running_var, eps
    All as 1D param files.
    """
    flat_name = module_path.replace(".", "_")

    num_features = bn.num_features

    weight_file = bias_file = rm_file = rv_file = None
    weight_shape = bias_shape = rm_shape = rv_shape = None

    if bn.weight is not None:
        weight_file, weight_shape = save_1d_param(base_dir, flat_name, "weight", bn.weight)
    else:
        print(f"[INFO] BatchNorm2d '{module_path}' has no weight.")

    if bn.bias is not None:
        bias_file, bias_shape = save_1d_param(base_dir, flat_name, "bias", bn.bias)
    else:
        print(f"[INFO] BatchNorm2d '{module_path}' has no bias.")

    if bn.running_mean is not None:
        rm_file, rm_shape = save_1d_param(base_dir, flat_name, "running_mean", bn.running_mean)
    else:
        print(f"[WARN] BatchNorm2d '{module_path}' has no running_mean (training mode?).")

    if bn.running_var is not None:
        rv_file, rv_shape = save_1d_param(base_dir, flat_name, "running_var", bn.running_var)
    else:
        print(f"[WARN] BatchNorm2d '{module_path}' has no running_var (training mode?).")

    return {
        "type": "BatchNorm2d",
        "num_features": num_features,
        "weight_file": weight_file,
        "bias_file": bias_file,
        "running_mean_file": rm_file,
        "running_var_file": rv_file,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "running_mean_shape": rm_shape,
        "running_var_shape": rv_shape,
        "eps": float(bn.eps),
    }


def save_layernorm_params(base_dir: Path, module_path: str, ln: nn.LayerNorm):
    """
    Save LayerNorm params for inference:
        weight (gamma), bias (beta)
    """
    flat_name = module_path.replace(".", "_")

    weight_file = bias_file = None
    weight_shape = bias_shape = None

    if ln.weight is not None:
        weight_file, weight_shape = save_1d_param(base_dir, flat_name, "weight", ln.weight)
    else:
        print(f"[INFO] LayerNorm '{module_path}' has no weight.")

    if ln.bias is not None:
        bias_file, bias_shape = save_1d_param(base_dir, flat_name, "bias", ln.bias)
    else:
        print(f"[INFO] LayerNorm '{module_path}' has no bias.")

    return {
        "type": "LayerNorm",
        "normalized_shape": list(ln.normalized_shape),
        "weight_file": weight_file,
        "bias_file": bias_file,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "eps": float(ln.eps),
    }


# -----------------------------
# Model loading
# -----------------------------

def load_tinyvit(model_name: str, models_dir: Path) -> nn.Module:
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


# -----------------------------
# Main
# -----------------------------

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
        help="Dump parameters for all Conv2d, Linear, BatchNorm2d and LayerNorm modules.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    params_dir = Path(args.params_dir)

    model = load_tinyvit(args.model_name, models_dir)

    dump_all = args.all or (len(args.modules) == 1 and args.modules[0].lower() == "all")

    manifest = {
        "model_name": args.model_name,
        "dtype": "float32",
        "params_dir": str(params_dir),
        "modules": {},
    }

    if dump_all:
        print("[INFO] --all / 'all' selected: dumping Conv2d, Linear, BatchNorm2d, LayerNorm.")
        for name, mod in model.named_modules():
            if not name:
                continue  # skip root

            if isinstance(mod, nn.Conv2d):
                print(f"[INFO] Conv2d module: {name}")
                info = save_conv2d_params(params_dir, name, mod)
                manifest["modules"][name] = info

            elif isinstance(mod, nn.Linear):
                print(f"[INFO] Linear module: {name}")
                info = save_linear_params(params_dir, name, mod)
                manifest["modules"][name] = info

            elif isinstance(mod, nn.BatchNorm2d):
                print(f"[INFO] BatchNorm2d module: {name}")
                info = save_batchnorm2d_params(params_dir, name, mod)
                manifest["modules"][name] = info

            elif isinstance(mod, nn.LayerNorm):
                print(f"[INFO] LayerNorm module: {name}")
                info = save_layernorm_params(params_dir, name, mod)
                manifest["modules"][name] = info

        print("[DONE] Dumped all supported module parameters.")
    else:
        for module_path in args.modules:
            print(f"[INFO] Processing module: {module_path}")
            mod = get_module_by_path(model, module_path)

            if isinstance(mod, nn.Conv2d):
                info = save_conv2d_params(params_dir, module_path, mod)
                manifest["modules"][module_path] = info

            elif isinstance(mod, nn.Linear):
                info = save_linear_params(params_dir, module_path, mod)
                manifest["modules"][module_path] = info

            elif isinstance(mod, nn.BatchNorm2d):
                info = save_batchnorm2d_params(params_dir, module_path, mod)
                manifest["modules"][module_path] = info

            elif isinstance(mod, nn.LayerNorm):
                info = save_layernorm_params(params_dir, module_path, mod)
                manifest["modules"][module_path] = info

            else:
                print(f"[WARN] Module '{module_path}' is type {type(mod)}, "
                      f"not Conv2d/Linear/BatchNorm2d/LayerNorm. Skipping.")

    # Save manifest JSON
    os.makedirs(params_dir, exist_ok=True)
    manifest_path = params_dir / "tinyvit_params_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved params manifest to {manifest_path}")

    print("[DONE] Parameter dump complete.")


if __name__ == "__main__":
    main()
