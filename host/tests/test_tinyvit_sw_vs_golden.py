# host/tests/test_tinyvit_sw_vs_golden.py

#!/usr/bin/env python3
"""
test_tinyvit_sw_vs_golden.py

Compare TinyVitSWSmall (_sw) activations against pre-dumped golden
activations from dump_tinyvit_activation.py.

Expected layout (for N=50):

  data/test_vectors/input_50/
      input_50.bin
      activations_manifest_50.json
      <module_name_flat>_output_50_golden.bin
      ...

Binary format (4D):
  int32 N, C, H, W
  float32 data[N*C*H*W]

Usage examples:

  # Compare ALL golden activations
  python host/tests/test_tinyvit_sw_vs_golden.py

  # Compare only a few modules
  python host/tests/test_tinyvit_sw_vs_golden.py \
      --modules patch_embed.conv1.conv stages.1.blocks.0.local_conv

  # Explicit N (if you dump a different count later)
  python host/tests/test_tinyvit_sw_vs_golden.py --num-images 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

# allow "from tinyvit.tiny_vit_sw import create_tiny_vit_5m_sw"
THIS_FILE = Path(__file__).resolve()
HOST_DIR = THIS_FILE.parents[1]
PROJ_ROOT = HOST_DIR.parent
if str(HOST_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(HOST_DIR))

from tinyvit.tiny_vit_sw import create_tiny_vit_5m_sw  # type: ignore


# ---------------------------------------------------------------------
# Binary tensor loader (4D NCHW)
# ---------------------------------------------------------------------

def load_tensor_bin_4d(path: Path) -> torch.Tensor:
    """
    Load [N, C, H, W] tensor from .bin:

        int32 N, C, H, W
        float32 data[N*C*H*W]
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        if header.size != 4:
            raise RuntimeError(f"Failed to read header from {path}")
        N, C, H, W = header.tolist()
        data = np.fromfile(f, dtype=np.float32)

    expected = N * C * H * W
    if data.size != expected:
        raise RuntimeError(
            f"Data size mismatch in {path}: got {data.size}, expected {expected}"
        )
    arr = data.reshape(N, C, H, W)
    return torch.from_numpy(arr)


def load_input_tensor(path: Path) -> torch.Tensor:
    return load_tensor_bin_4d(path)


# ---------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------

def compare_tensors(name: str, y_gold: torch.Tensor, y_sw: torch.Tensor) -> Dict[str, Any]:
    if y_gold.shape != y_sw.shape:
        raise RuntimeError(
            f"Shape mismatch for {name}: golden {tuple(y_gold.shape)} vs sw {tuple(y_sw.shape)}"
        )

    diff = (y_gold - y_sw).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    tol = 1e-5

    print(f"[COMPARE] {name}")
    print(f"  golden shape: {tuple(y_gold.shape)}")
    print(f"  sw     shape: {tuple(y_sw.shape)}")
    print(f"  max_abs_diff  = {max_diff}")
    print(f"  mean_abs_diff = {mean_diff}")
    if max_diff > tol:
        print(f"  WARNING: max_abs_diff > {tol} (= {max_diff})")
    else:
        print(f"  OK: within tolerance {tol}")
    print()

    return {
        "name": name,
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "ok": bool(max_diff <= tol),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare TinyVitSWSmall activations vs golden dumps."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="N used in input_N.bin and activations_manifest_N.json (default: 50).",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/test_vectors",
        help="Root directory containing input_N/ subfolders.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory containing TinyViT state_dict.",
    )
    parser.add_argument(
        "--state-dict-fname",
        type=str,
        default="tiny_vit_5m_224_dist_in22k_ft_in1k_state_dict.pth",
        help="State dict filename under models-dir.",
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        default=None,
        help=(
            "List of dotted module paths to compare, e.g. "
            "'patch_embed.conv1.conv stages.1.blocks.0.local_conv'. "
            "If omitted and --all not given, defaults to ALL modules in manifest."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare ALL modules listed in activations manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (e.g. 'cpu' or 'cuda').",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(">>")
    print(f"[INFO] Project root   : {PROJ_ROOT}")
    print(f"[INFO] Using device   : {device}")
    print(f"[INFO] input_root     : {args.input_root}")
    print(f"[INFO] models_dir     : {args.models_dir}")
    print(f"[INFO] num_images (N) : {args.num_images}")
    print()

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    tv_root = PROJ_ROOT / args.input_root
    input_dir = tv_root / f"input_{args.num_images}"
    input_path = input_dir / f"input_{args.num_images}.bin"
    manifest_path = input_dir / f"activations_manifest_{args.num_images}.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"[INFO] Using input tensor file      : {input_path}")
    print(f"[INFO] Using activations manifest  : {manifest_path}")
    print()

    # ---------------------------------------------------------
    # Load manifest
    # ---------------------------------------------------------
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tensors_list = manifest.get("tensors", [])
    if not tensors_list:
        raise RuntimeError(f"No 'tensors' entries found in manifest {manifest_path}")

    # Map: module_name -> entry
    manifest_by_module: Dict[str, Dict[str, Any]] = {}
    for entry in tensors_list:
        mod_name = entry["module"]
        manifest_by_module[mod_name] = entry

    # Determine which modules to test
    if args.all or args.modules is None:
        modules_to_test = sorted(manifest_by_module.keys())
        print(f"[INFO] Comparing ALL modules from manifest ({len(modules_to_test)} entries).")
    else:
        modules_to_test = []
        for mname in args.modules:
            if mname not in manifest_by_module:
                print(f"[WARN] Requested module '{mname}' not in manifest; skipping.")
            else:
                modules_to_test.append(mname)
        if not modules_to_test:
            raise RuntimeError("No valid module names remain after filtering by manifest.")
        print(f"[INFO] Comparing selected modules ({len(modules_to_test)}):")
        for m in modules_to_test:
            print(f"    - {m}")
    print()

    # ---------------------------------------------------------
    # Load input tensor (ALL N images)
    # ---------------------------------------------------------
    x = load_input_tensor(input_path)  # [N, 3, 224, 224]
    x = x.to(device)
    print(f"[INFO] Loaded input tensor with shape {tuple(x.shape)}")
    print(f"[INFO] Using ALL {x.shape[0]} images for comparison.\n")

    # ---------------------------------------------------------
    # Create SW model & load weights
    # ---------------------------------------------------------
    models_dir = PROJ_ROOT / args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = str(models_dir.resolve())
    os.environ["HF_HUB_CACHE"] = str(models_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(models_dir.resolve())

    sw_model = create_tiny_vit_5m_sw(
        models_dir=models_dir,
        state_dict_fname=args.state_dict_fname,
        device=device,
    )
    sw_model.eval()
    print("[INFO] TinyVitSWSmall created and set to eval().\n")

    # ---------------------------------------------------------
    # Register forward hooks for selected modules
    # ---------------------------------------------------------
    name_to_module = dict(sw_model.named_modules())
    captured: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            captured[name] = out.detach().cpu()
        return hook_fn

    handles = []
    for mod_name in modules_to_test:
        if mod_name not in name_to_module:
            raise KeyError(
                f"Module '{mod_name}' not found in sw_model.named_modules()."
            )
        m = name_to_module[mod_name]
        h = m.register_forward_hook(make_hook(mod_name))
        handles.append(h)

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    with torch.no_grad():
        _ = sw_model(x)

    for h in handles:
        h.remove()

    if not captured:
        raise RuntimeError("No activations captured; hooks may not have fired.")

    # ---------------------------------------------------------
    # Compare each selected module
    # ---------------------------------------------------------
    results: List[Dict[str, Any]] = []

    for mod_name in modules_to_test:
        entry = manifest_by_module[mod_name]
        fname = entry["file"]
        shape = entry.get("shape", None)

        golden_path = input_dir / fname
        if not golden_path.exists():
            print(f"[WARN] Golden file for '{mod_name}' not found at {golden_path}; skipping.")
            continue

        y_gold = load_tensor_bin_4d(golden_path)  # [N, C, H, W]
        y_sw = captured.get(mod_name, None)
        if y_sw is None:
            print(f"[WARN] No SW activation captured for '{mod_name}'; skipping.")
            continue

        # sanity check: shape from manifest vs golden vs captured
        if shape is not None:
            if list(y_gold.shape) != list(shape):
                print(
                    f"[WARN] Manifest shape {shape} does not match golden tensor shape {tuple(y_gold.shape)} for '{mod_name}'."
                )
        # Move y_sw to CPU if not already
        y_sw_cpu = y_sw.cpu()

        res = compare_tensors(mod_name, y_gold, y_sw_cpu)
        results.append(res)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("========== TinyViT SW vs GOLDEN SUMMARY ==========")
    if not results:
        print("No comparisons were successfully performed.")
    else:
        all_ok = True
        for r in results:
            status = "OK" if r["ok"] else "FAIL"
            print(
                f"[SUMMARY] {r['name']:<40} "
                f"status={status:<4} "
                f"max_abs_diff={r['max_abs_diff']:.6e} "
                f"mean_abs_diff={r['mean_abs_diff']:.6e}"
            )
            if not r["ok"]:
                all_ok = False
        print("-----------------------------------------------")
        if all_ok:
            print("[SUMMARY] All compared modules PASSED within tolerance.")
        else:
            print("[SUMMARY] Some modules FAILED; see details above.")
    print("===============================================")


if __name__ == "__main__":
    main()
