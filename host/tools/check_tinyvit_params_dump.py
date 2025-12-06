#!/usr/bin/env python3
"""
check_tinyvit_params_dump.py

Verify that the dumped TinyViT parameters in:
    data/models/tinyvit_5m_224_dist_in22k_ft_in1k_params/

match the actual PyTorch model parameters from timm.

It reads:
    tinyvit_params_manifest.json

and for each module entry, reloads the corresponding Conv2d / Linear /
BatchNorm2d / LayerNorm parameters from .bin files and compares them
against the live model parameters.

Example usage (from project root):

  python host/tools/check_tinyvit_params_dump.py

With custom tolerance / dirs:

  python host/tools/check_tinyvit_params_dump.py ^
    --tol 0.0 ^
    --model-name tiny_vit_5m_224.dist_in22k_ft_in1k ^
    --models-dir data/models ^
    --params-dir data/models/tinyvit_5m_224_dist_in22k_ft_in1k_params
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm


# -----------------------------
# Helpers: load model & modules
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


def get_module_by_path(model: nn.Module, module_path: str) -> nn.Module:
    parts = module_path.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


# -----------------------------
# Helpers: read binary params
# -----------------------------

def load_conv2d_weight(path: Path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        if header.size != 4:
            raise RuntimeError(f"[ERROR] Failed to read Conv2d header from {path}")
        out_ch, in_ch, kH, kW = header.tolist()
        data = np.fromfile(f, dtype=np.float32)
    if data.size != out_ch * in_ch * kH * kW:
        raise RuntimeError(
            f"[ERROR] Conv2d data size mismatch in {path}: "
            f"got {data.size}, expected {out_ch*in_ch*kH*kW}"
        )
    return data.reshape(out_ch, in_ch, kH, kW)


def load_linear_weight(path: Path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise RuntimeError(f"[ERROR] Failed to read Linear header from {path}")
        out_f, in_f = header.tolist()
        data = np.fromfile(f, dtype=np.float32)
    if data.size != out_f * in_f:
        raise RuntimeError(
            f"[ERROR] Linear data size mismatch in {path}: "
            f"got {data.size}, expected {out_f*in_f}"
        )
    return data.reshape(out_f, in_f)


def load_1d_param(path: Path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=1)
        if header.size != 1:
            raise RuntimeError(f"[ERROR] Failed to read 1D header from {path}")
        length = int(header[0])
        data = np.fromfile(f, dtype=np.float32)
    if data.size != length:
        raise RuntimeError(
            f"[ERROR] 1D data size mismatch in {path}: "
            f"got {data.size}, expected {length}"
        )
    return data  # shape [length]


def compare_arrays(name: str, arr_dump: np.ndarray, arr_ref: np.ndarray, tol: float):
    if arr_dump.shape != arr_ref.shape:
        print(f"[FAIL] {name}: shape mismatch dump{arr_dump.shape} vs ref{arr_ref.shape}")
        return False, float("inf")

    diff = arr_dump.astype(np.float64) - arr_ref.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0

    if max_abs > tol:
        print(f"[FAIL] {name}: max_abs_diff={max_abs:.8g} > tol={tol}")
        return False, max_abs
    else:
        print(f"[PASS] {name}: max_abs_diff={max_abs:.8g} <= tol={tol}")
        return True, max_abs


# -----------------------------
# Main checking logic
# -----------------------------

def check_params(
    model: nn.Module,
    manifest_path: Path,
    tol: float = 0.0,
) -> bool:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    params_dir = Path(manifest["params_dir"])
    model_name_manifest = manifest.get("model_name", "<unknown>")
    print(f"[INFO] Manifest model_name: {model_name_manifest}")
    print(f"[INFO] Params dir from manifest: {params_dir}")

    all_ok = True
    global_max_abs = 0.0

    for module_path, info in manifest["modules"].items():
        mtype = info.get("type", "Unknown")
        mod = get_module_by_path(model, module_path)
        print(f"\n[MODULE] {module_path} (type={mtype})")

        # Conv2d
        if mtype == "Conv2d" and isinstance(mod, nn.Conv2d):
            w_file = info["weight_file"]
            w_path = params_dir / w_file
            w_dump = load_conv2d_weight(w_path)
            w_ref = mod.weight.detach().cpu().numpy()
            ok_w, max_w = compare_arrays(module_path + ".weight", w_dump, w_ref, tol)
            all_ok &= ok_w
            global_max_abs = max(global_max_abs, max_w)

            b_file = info.get("bias_file")
            if b_file is not None and mod.bias is not None:
                b_path = params_dir / b_file
                b_dump = load_1d_param(b_path)
                b_ref = mod.bias.detach().cpu().view(-1).numpy()
                ok_b, max_b = compare_arrays(module_path + ".bias", b_dump, b_ref, tol)
                all_ok &= ok_b
                global_max_abs = max(global_max_abs, max_b)
            elif mod.bias is not None:
                print(f"[WARN] {module_path}.bias exists in model but no bias_file in manifest.")
                all_ok = False

        # Linear
        elif mtype == "Linear" and isinstance(mod, nn.Linear):
            w_file = info["weight_file"]
            w_path = params_dir / w_file
            w_dump = load_linear_weight(w_path)
            w_ref = mod.weight.detach().cpu().numpy()
            ok_w, max_w = compare_arrays(module_path + ".weight", w_dump, w_ref, tol)
            all_ok &= ok_w
            global_max_abs = max(global_max_abs, max_w)

            b_file = info.get("bias_file")
            if b_file is not None and mod.bias is not None:
                b_path = params_dir / b_file
                b_dump = load_1d_param(b_path)
                b_ref = mod.bias.detach().cpu().view(-1).numpy()
                ok_b, max_b = compare_arrays(module_path + ".bias", b_dump, b_ref, tol)
                all_ok &= ok_b
                global_max_abs = max(global_max_abs, max_b)
            elif mod.bias is not None:
                print(f"[WARN] {module_path}.bias exists in model but no bias_file in manifest.")
                all_ok = False

        # BatchNorm2d
        elif mtype == "BatchNorm2d" and isinstance(mod, nn.BatchNorm2d):
            # weight
            w_file = info.get("weight_file")
            if w_file is not None and mod.weight is not None:
                w_path = params_dir / w_file
                w_dump = load_1d_param(w_path)
                w_ref = mod.weight.detach().cpu().view(-1).numpy()
                ok_w, max_w = compare_arrays(module_path + ".weight", w_dump, w_ref, tol)
                all_ok &= ok_w
                global_max_abs = max(global_max_abs, max_w)

            # bias
            b_file = info.get("bias_file")
            if b_file is not None and mod.bias is not None:
                b_path = params_dir / b_file
                b_dump = load_1d_param(b_path)
                b_ref = mod.bias.detach().cpu().view(-1).numpy()
                ok_b, max_b = compare_arrays(module_path + ".bias", b_dump, b_ref, tol)
                all_ok &= ok_b
                global_max_abs = max(global_max_abs, max_b)

            # running_mean
            rm_file = info.get("running_mean_file")
            if rm_file is not None and mod.running_mean is not None:
                rm_path = params_dir / rm_file
                rm_dump = load_1d_param(rm_path)
                rm_ref = mod.running_mean.detach().cpu().view(-1).numpy()
                ok_rm, max_rm = compare_arrays(module_path + ".running_mean", rm_dump, rm_ref, tol)
                all_ok &= ok_rm
                global_max_abs = max(global_max_abs, max_rm)

            # running_var
            rv_file = info.get("running_var_file")
            if rv_file is not None and mod.running_var is not None:
                rv_path = params_dir / rv_file
                rv_dump = load_1d_param(rv_path)
                rv_ref = mod.running_var.detach().cpu().view(-1).numpy()
                ok_rv, max_rv = compare_arrays(module_path + ".running_var", rv_dump, rv_ref, tol)
                all_ok &= ok_rv
                global_max_abs = max(global_max_abs, max_rv)

            # eps
            eps_manifest = float(info.get("eps", mod.eps))
            eps_ref = float(mod.eps)
            eps_diff = abs(eps_manifest - eps_ref)
            if eps_diff > 0:
                print(f"[WARN] {module_path}.eps mismatch: manifest={eps_manifest} ref={eps_ref}")
            else:
                print(f"[PASS] {module_path}.eps matches: {eps_ref}")

        # LayerNorm
        elif mtype == "LayerNorm" and isinstance(mod, nn.LayerNorm):
            # weight
            w_file = info.get("weight_file")
            if w_file is not None and mod.weight is not None:
                w_path = params_dir / w_file
                w_dump = load_1d_param(w_path)
                w_ref = mod.weight.detach().cpu().view(-1).numpy()
                ok_w, max_w = compare_arrays(module_path + ".weight", w_dump, w_ref, tol)
                all_ok &= ok_w
                global_max_abs = max(global_max_abs, max_w)

            # bias
            b_file = info.get("bias_file")
            if b_file is not None and mod.bias is not None:
                b_path = params_dir / b_file
                b_dump = load_1d_param(b_path)
                b_ref = mod.bias.detach().cpu().view(-1).numpy()
                ok_b, max_b = compare_arrays(module_path + ".bias", b_dump, b_ref, tol)
                all_ok &= ok_b
                global_max_abs = max(global_max_abs, max_b)

            # eps
            eps_manifest = float(info.get("eps", mod.eps))
            eps_ref = float(mod.eps)
            eps_diff = abs(eps_manifest - eps_ref)
            if eps_diff > 0:
                print(f"[WARN] {module_path}.eps mismatch: manifest={eps_manifest} ref={eps_ref}")
            else:
                print(f"[PASS] {module_path}.eps matches: {eps_ref}")

            # normalized_shape
            ns_manifest = info.get("normalized_shape")
            ns_ref = list(mod.normalized_shape)
            if ns_manifest is not None and ns_manifest != ns_ref:
                print(f"[WARN] {module_path}.normalized_shape mismatch: "
                      f"manifest={ns_manifest} ref={ns_ref}")
            else:
                print(f"[PASS] {module_path}.normalized_shape matches: {ns_ref}")

        else:
            print(f"[WARN] Skipping module '{module_path}' type={mtype}, "
                  f"actual_type={type(mod)}")

    print("\n========== SUMMARY ==========")
    print(f"Global max_abs_diff over all checked params: {global_max_abs:.8g}")
    print(f"All params within tol={tol}? {'YES' if all_ok else 'NO'}")
    print("================================")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Check TinyViT parameter dumps against model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="tiny_vit_5m_224.dist_in22k_ft_in1k",
        help="timm model name (must match what you used for dumping).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory where TinyViT state_dict is cached.",
    )
    parser.add_argument(
        "--params-dir",
        type=str,
        default="data/models/tinyvit_5m_224_dist_in22k_ft_in1k_params",
        help="Directory where parameter .bin files and manifest live.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to tinyvit_params_manifest.json; "
             "if omitted, uses <params-dir>/tinyvit_params_manifest.json",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Max allowed absolute difference per param (default: 0.0 for exact match).",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    params_dir = Path(args.params_dir)
    manifest_path = Path(args.manifest) if args.manifest is not None else params_dir / "tinyvit_params_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"[ERROR] Manifest not found: {manifest_path}")

    model = load_tinyvit(args.model_name, models_dir)
    ok = check_params(model, manifest_path, tol=args.tol)

    if not ok:
        # Non-zero exit code if mismatch
        raise SystemExit(1)


if __name__ == "__main__":
    main()
